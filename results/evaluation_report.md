# Evaluation Report: Drug-Resistance Mutation Prediction

## Dataset

- **Source**: cBioPortal (TCGA studies) + published clinical literature
- **Total unique mutations**: 188
- **Mutations with structural features**: 176
- **Proteins**: EGFR, ABL1, ALK, KRAS, BRAF
- **Class distribution**: 141 resistant (1) / 47 sensitive (0)

## Method

- **Classifier**: Random Forest (300 trees, max_depth=6, class_weight='balanced')
- **Validation**: Leave-One-Protein-Out cross-validation (trains on 4 proteins, tests on the 5th)
- **Features** (14 total):
  - Structural: SASA, distance to ligand, position, gatekeeper flag
  - Chemistry: hydrophobicity change, volume change, charge change, polarity change, BLOSUM62 score
  - One-hot: protein identity

## Results

| Metric | Value |
|--------|-------|
| Accuracy | 0.812 |
| AUC | 0.765 |
| Sensitivity (resistant recall) | 0.964 |
| Specificity (sensitive recall) | 0.282 |
| False positives | 28 |
| False negatives | 5 |

## Feature Importance

Top features driving predictions:

1. Position (0.215)
2. Distance to ligand (0.209)
3. SASA (0.187)
4. Volume change (0.108)
5. Hydrophobicity change (0.084)

## Limitations

1. **Class imbalance**: 141 resistant vs 47 sensitive. The model defaults toward the majority class, driving specificity low (0.282).
2. **Small dataset**: 176 mutations across 5 proteins. More data would improve generalization.
3. **Label noise**: Some resistance labels come from clinical literature with varying evidence levels.
4. **Structural coverage**: 12 of 188 mutations lacked residues in PDB structures and were excluded.

## Recommendations

- **For resistance screening**: the model is reliable (96% sensitivity). Use it to flag likely-resistant mutations.
- **For sensitivity calls**: DO NOT rely on this model alone (28% specificity). Combine with clinical evidence.
- **Future work**: add more sensitive mutations, use SMOTE for oversampling, or try gradient boosting (XGBoost).

## Files

- `results/model.pkl` — trained Random Forest
- `results/predictions.csv` — per-mutation predictions and probabilities
- `results/figures/confusion_matrix.png`
- `results/figures/roc_curve.png`
- `results/figures/feature_importance.png`
- `results/figures/per_protein_accuracy.png`
