import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
Interactive demo: predict drug resistance for a mutation.

Real feature computation — loads the PDB structure and computes
SASA and distance-to-ligand for the mutated residue.
"""

import joblib
import numpy as np
import streamlit as st

from src.features.parse_mutations import parse_mutation, hydrophobicity_change
from src.features.build_features import (
    PROTEIN_PDB,
    GATEKEEPERS,
    AA_VOLUME,
    AA_CHARGE,
    AA_POLAR,
    BLOSUM62,
    load_structure_with_sasa,
    get_residue,
    distance_to_ligand,
)


st.set_page_config(page_title="Resistance Predictor", page_icon="🧬")
st.title("🧬 Drug-Resistance Mutation Predictor")
st.write(
    "Enter a protein and mutation to estimate whether it will cause "
    "resistance to targeted cancer therapy."
)

PROTEINS = ["EGFR", "ABL1", "ALK", "KRAS", "BRAF"]


@st.cache_resource
def load_model():
    return joblib.load("results/model.pkl")


@st.cache_resource
def load_structures():
    """Load all PDB structures once and cache them."""
    models = {}
    for protein, (pdb_path, chain_id) in PROTEIN_PDB.items():
        try:
            models[protein] = (load_structure_with_sasa(pdb_path), chain_id)
        except Exception as e:
            st.warning(f"Could not load structure for {protein}: {e}")
    return models


def build_feature_vector(protein, mutation, structures):
    """Compute the real 14-dimensional feature vector."""
    parsed = parse_mutation(mutation)
    if parsed is None:
        return None, "Could not parse mutation. Use format like T790M."

    pos = parsed["position"]
    wt = parsed["wildtype_aa"]
    mut = parsed["mutant_aa"]

    # Get structure
    if protein not in structures:
        return None, f"No structure available for {protein}"

    model, chain_id = structures[protein]

    # Structural features
    residue = get_residue(model, chain_id, pos)
    sasa = residue.sasa if residue is not None and hasattr(residue, "sasa") else 18.2
    dist = distance_to_ligand(model, chain_id, pos)
    if dist is None:
        dist = 6.6

    # Curated
    gatekeeper = 1 if GATEKEEPERS.get(protein) == pos else 0

    # Chemistry
    hydro = hydrophobicity_change(wt, mut)
    if hydro is None:
        hydro = 0.0

    vol_change = 0.0
    charge_change = 0.0
    polarity_change = 0.0
    blosum = -3
    if wt and mut:
        vw = AA_VOLUME.get(wt, 0)
        vm = AA_VOLUME.get(mut, 0)
        vol_change = vm - vw
        charge_change = AA_CHARGE.get(mut, 0) - AA_CHARGE.get(wt, 0)
        polarity_change = AA_POLAR.get(mut, 0) - AA_POLAR.get(wt, 0)
        blosum = BLOSUM62.get((wt, mut), BLOSUM62.get((mut, wt), -3))

    features = {
        "position": pos,
        "sasa": sasa,
        "distance_to_ligand": dist,
        "is_gatekeeper": gatekeeper,
        "hydrophobicity_change": hydro,
        "volume_change": vol_change,
        "charge_change": charge_change,
        "polarity_change": polarity_change,
        "blosum62_score": blosum,
        "protein_EGFR": 1 if protein == "EGFR" else 0,
        "protein_ABL1": 1 if protein == "ABL1" else 0,
        "protein_ALK": 1 if protein == "ALK" else 0,
        "protein_KRAS": 1 if protein == "KRAS" else 0,
        "protein_BRAF": 1 if protein == "BRAF" else 0,
    }

    debug = {
        "Position": pos,
        "Residue found": "Yes" if residue is not None else "No",
        "SASA (Å²)": f"{sasa:.2f}",
        "Distance to ligand (Å)": f"{dist:.2f}" if dist else "N/A",
        "Gatekeeper": "Yes" if gatekeeper else "No",
        "Volume change (Å³)": f"{vol_change:.1f}",
        "BLOSUM62": blosum,
    }

    return list(features.values()), debug


protein = st.selectbox("Protein", PROTEINS)
mutation = st.text_input("Mutation (e.g., T790M, G12C, V600E)", value="T790M")

if st.button("Predict"):
    structures = load_structures()
    feats, debug_or_error = build_feature_vector(
        protein, mutation.strip().upper(), structures
    )

    if feats is None:
        st.error(debug_or_error)
    else:
        clf = load_model()
        X = np.array([feats])
        prob = clf.predict_proba(X)[0][1]
        pred = "Resistant" if prob > 0.5 else "Sensitive"

        if pred == "Resistant":
            st.error(f"**Prediction: {pred}**")
        else:
            st.success(f"**Prediction: {pred}**")

        st.metric("Probability of resistance", f"{prob:.3f}")

        # Show computed features for transparency
        with st.expander("Computed features"):
            for k, v in debug_or_error.items():
                st.write(f"**{k}**: {v}")

        st.info(
            "**Disclaimer**: This is a research prototype. "
            "Do not use for clinical decisions."
        )
