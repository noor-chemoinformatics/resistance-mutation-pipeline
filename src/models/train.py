"""
Train a classifier to predict drug resistance from structural features.

Uses Leave-One-Protein-Out cross-validation to test generalization
across different therapeutic targets.
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
)

FEATURE_COLS = [
    "position",
    "sasa",
    "distance_to_ligand",
    "is_gatekeeper",
    "hydrophobicity_change",
    "volume_change",
    "charge_change",
    "polarity_change",
    "blosum62_score",
    "protein_EGFR",
    "protein_ABL1",
    "protein_ALK",
    "protein_KRAS",
    "protein_BRAF",
]

def load_features(path="data/processed/features.csv"):
    """Load the feature table and clean missing values."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} mutations from {path}")

    # Keep only rows where structural features were found
    df_struct = df[df["residue_found"] == 1].copy()
    print(f"Kept {len(df_struct)} rows with structural features")

    # Fill missing numeric values with median
    for col in [
        "sasa", "distance_to_ligand", "hydrophobicity_change",
        "volume_change", "charge_change", "polarity_change", "blosum62_score",
    ]:
        if col not in df_struct.columns:
            df_struct[col] = 0
            print(f"  {col}: MISSING — filled with 0")
            continue
        n_missing = df_struct[col].isna().sum()
        median = df_struct[col].median()
        df_struct[col] = df_struct[col].fillna(median)
        print(f"  {col}: filled {n_missing} NaN with {median:.3f}")
    # Ensure all feature columns exist
    for col in FEATURE_COLS:
        if col not in df_struct.columns:
            df_struct[col] = 0
            print(f"  Warning: {col} missing; filled with 0")

    return df_struct


def train_and_evaluate(df):
    """Train RandomForest with Leave-One-Protein-Out CV."""
    X = df[FEATURE_COLS].values
    y = df["resistance_label"].values
    groups = df["protein"].values

    print(f"\nX shape: {X.shape}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Groups: {sorted(set(groups))}")

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    logo = LeaveOneGroupOut()

    print("\n=== Leave-One-Protein-Out Cross-Validation ===")
    for train_idx, test_idx in logo.split(X, y, groups):
        train_proteins = sorted(set(groups[train_idx]))
        test_proteins = sorted(set(groups[test_idx]))
        print(f"  Train on {train_proteins} → test on {test_proteins}")

    # Cross-validated predictions
    y_pred = cross_val_predict(clf, X, y, groups=groups, cv=logo, method="predict")
    y_prob = cross_val_predict(
        clf, X, y, groups=groups, cv=logo, method="predict_proba"
    )[:, 1]

    print("\n=== Classification Report ===")
    print(classification_report(y, y_pred, zero_division=0))

    print(f"Accuracy: {accuracy_score(y, y_pred):.3f}")
    try:
        print(f"AUC: {roc_auc_score(y, y_prob):.3f}")
    except ValueError:
        print("AUC: cannot compute")

    print("\nConfusion matrix:")
    cm = confusion_matrix(y, y_pred)
    print(cm)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"\nSensitivity: {sens:.3f}")
        print(f"Specificity: {spec:.3f}")
        print(f"False positives: {fp}")
        print(f"False negatives: {fn}")

    # Train final model on all data
    clf.fit(X, y)

    print("\n=== Feature Importance ===")
    for name, imp in sorted(
        zip(FEATURE_COLS, clf.feature_importances_),
        key=lambda x: -x[1],
    ):
        print(f"  {name:25s} {imp:.4f}")

    os.makedirs("results", exist_ok=True)
    joblib.dump(clf, "results/model.pkl")
    print("\nModel saved: results/model.pkl")

    df_out = df.copy()
    df_out["y_pred"] = y_pred
    df_out["y_prob"] = y_prob
    df_out.to_csv("results/predictions.csv", index=False)
    print("Predictions saved: results/predictions.csv")

    return clf, y_pred, y_prob


def main():
    df = load_features()
    train_and_evaluate(df)


if __name__ == "__main__":
    main()
