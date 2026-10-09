"""
Generate plots and figures for the model results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    classification_report,
)


FEATURE_COLS = [
    "position", "sasa", "distance_to_ligand", "is_gatekeeper",
    "hydrophobicity_change", "volume_change", "charge_change",
    "polarity_change", "blosum62_score",
    "protein_EGFR", "protein_ABL1", "protein_ALK",
    "protein_KRAS", "protein_BRAF",
]


def plot_confusion(y_true, y_pred, output_path):
    """Confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Sensitive (0)", "Resistant (1)"],
        yticklabels=["Sensitive (0)", "Resistant (1)"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix — Leave-One-Protein-Out")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_roc(y_true, y_prob, output_path):
    """ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}", color="darkorange", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Leave-One-Protein-Out")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_feature_importance(clf, output_path):
    """Feature importance bar chart."""
    importances = clf.feature_importances_
    indices = np.argsort(importances)

    plt.figure(figsize=(8, 6))
    plt.barh(
        [FEATURE_COLS[i] for i in indices],
        importances[indices],
        color="steelblue",
    )
    plt.xlabel("Importance")
    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_per_protein(df, output_path):
    """Bar chart of accuracy per protein."""
    df = df.copy()
    df["correct"] = (df["y_pred"] == df["resistance_label"]).astype(int)
    per_protein = df.groupby("protein").agg(
        accuracy=("correct", "mean"),
        n=("correct", "size"),
    )

    plt.figure(figsize=(7, 5))
    bars = plt.bar(per_protein.index, per_protein["accuracy"], color="seagreen")
    plt.xlabel("Protein")
    plt.ylabel("Accuracy")
    plt.title("Per-Protein Accuracy (Leave-One-Protein-Out)")
    plt.ylim(0, 1)
    for bar, n in zip(bars, per_protein["n"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"n={n}",
            ha="center",
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    import joblib

    os.makedirs("results/figures", exist_ok=True)

    df = pd.read_csv("results/predictions.csv")
    y_true = df["resistance_label"].values
    y_pred = df["y_pred"].values
    y_prob = df["y_prob"].values

    plot_confusion(y_true, y_pred, "results/figures/confusion_matrix.png")
    plot_roc(y_true, y_prob, "results/figures/roc_curve.png")

    clf = joblib.load("results/model.pkl")
    plot_feature_importance(clf, "results/figures/feature_importance.png")
    plot_per_protein(df, "results/figures/per_protein_accuracy.png")

    print("\nAll figures saved to results/figures/")


if __name__ == "__main__":
    main()
