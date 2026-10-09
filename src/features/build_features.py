"""
Build a features table for ML from curated mutations.

Combines:
- Parsed mutation info (position, wildtype/mutant amino acids)
- Structural features from PDB (SASA, distance to ligand)
- Chemistry change features (volume, charge, polarity, BLOSUM62)
- Curated features (gatekeeper flag)
- One-hot protein encoding
"""

import os
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley

from src.features.parse_mutations import (
    parse_mutation, hydrophobicity_change
)


# Map protein -> PDB file and chain
PROTEIN_PDB = {
    "EGFR": ("data/structures/4I22.pdb", "A"),
    "ABL1": ("data/structures/2HYY.pdb", "A"),
    "ALK":  ("data/structures/4MKC.pdb", "A"),
    "KRAS": ("data/structures/6OIM.pdb", "A"),
    "BRAF": ("data/structures/4MNE.pdb", "B"),
}


# Known gatekeeper residues
GATEKEEPERS = {
    "EGFR": 790,
    "ABL1": 315,
    "ALK":  1196,
    "KRAS": None,
    "BRAF": None,
}


# ---------------- Amino acid property tables ----------------

# Volume (cubic Angstroms)
AA_VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
    "Q": 143.8, "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7,
    "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
    "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
}

# Charge at pH 7 (simplified)
AA_CHARGE = {
    "A": 0, "R": 1, "N": 0, "D": -1, "C": 0,
    "Q": 0, "E": -1, "G": 0, "H": 0.1, "I": 0,
    "L": 0, "K": 1, "M": 0, "F": 0, "P": 0,
    "S": 0, "T": 0, "W": 0, "Y": 0, "V": 0,
}

# Polarity (1 = polar, 0 = nonpolar)
AA_POLAR = {
    "A": 0, "R": 1, "N": 1, "D": 1, "C": 0,
    "Q": 1, "E": 1, "G": 0, "H": 1, "I": 0,
    "L": 0, "K": 1, "M": 0, "F": 0, "P": 0,
    "S": 1, "T": 1, "W": 0, "Y": 1, "V": 0,
}

# BLOSUM62 substitution matrix (subset — common pairs only)
BLOSUM62 = {
    ("A", "A"): 4, ("A", "R"): -1, ("A", "N"): -2, ("A", "D"): -2, ("A", "C"): 0,
    ("A", "Q"): -1, ("A", "E"): -1, ("A", "G"): 0, ("A", "H"): -2, ("A", "I"): -1,
    ("A", "L"): -1, ("A", "K"): -1, ("A", "M"): -1, ("A", "F"): -2, ("A", "P"): -1,
    ("A", "S"): 1, ("A", "T"): 0, ("A", "W"): -3, ("A", "Y"): -2, ("A", "V"): 0,
    ("R", "R"): 5, ("R", "N"): 0, ("R", "D"): -2, ("R", "C"): -3,
    ("R", "Q"): 1, ("R", "E"): 0, ("R", "G"): -2, ("R", "H"): 0, ("R", "I"): -3,
    ("R", "L"): -2, ("R", "K"): 2, ("R", "M"): -1, ("R", "F"): -3, ("R", "P"): -2,
    ("R", "S"): -1, ("R", "T"): -1, ("R", "W"): -3, ("R", "Y"): -2, ("R", "V"): -3,
    ("N", "N"): 6, ("N", "D"): 1, ("N", "C"): -3,
    ("N", "Q"): 0, ("N", "E"): 0, ("N", "G"): 0, ("N", "H"): 1, ("N", "I"): -3,
    ("N", "L"): -3, ("N", "K"): 0, ("N", "M"): -2, ("N", "F"): -3, ("N", "P"): -2,
    ("N", "S"): 1, ("N", "T"): 0, ("N", "W"): -4, ("N", "Y"): -2, ("N", "V"): -3,
    ("D", "D"): 6, ("D", "C"): -3,
    ("D", "Q"): 0, ("D", "E"): 2, ("D", "G"): -1, ("D", "H"): -1, ("D", "I"): -3,
    ("D", "L"): -4, ("D", "K"): -1, ("D", "M"): -3, ("D", "F"): -3, ("D", "P"): -1,
    ("D", "S"): 0, ("D", "T"): -1, ("D", "W"): -4, ("D", "Y"): -3, ("D", "V"): -3,
    ("C", "C"): 9,
    ("C", "Q"): -3, ("C", "E"): -3, ("C", "G"): -3, ("C", "H"): -3, ("C", "I"): -1,
    ("C", "L"): -1, ("C", "K"): -3, ("C", "M"): -1, ("C", "F"): -2, ("C", "P"): -3,
    ("C", "S"): -1, ("C", "T"): -1, ("C", "W"): -2, ("C", "Y"): -2, ("C", "V"): -1,
    ("Q", "Q"): 5, ("Q", "E"): 2, ("Q", "G"): -2, ("Q", "H"): 0, ("Q", "I"): -3,
    ("Q", "L"): -2, ("Q", "K"): 1, ("Q", "M"): 0, ("Q", "F"): -3, ("Q", "P"): -1,
    ("Q", "S"): 0, ("Q", "T"): -1, ("Q", "W"): -2, ("Q", "Y"): -1, ("Q", "V"): -2,
    ("E", "E"): 5, ("E", "G"): -2, ("E", "H"): 0, ("E", "I"): -3,
    ("E", "L"): -3, ("E", "K"): 1, ("E", "M"): -2, ("E", "F"): -3, ("E", "P"): -1,
    ("E", "S"): 0, ("E", "T"): -1, ("E", "W"): -3, ("E", "Y"): -2, ("E", "V"): -2,
    ("G", "G"): 6, ("G", "H"): -2, ("G", "I"): -4, ("G", "L"): -4, ("G", "K"): -2,
    ("G", "M"): -3, ("G", "F"): -3, ("G", "P"): -2, ("G", "S"): 0, ("G", "T"): -2,
    ("G", "W"): -2, ("G", "Y"): -3, ("G", "V"): -3,
    ("H", "H"): 8, ("H", "I"): -3, ("H", "L"): -3, ("H", "K"): -1, ("H", "M"): -2,
    ("H", "F"): -1, ("H", "P"): -2, ("H", "S"): -1, ("H", "T"): -2, ("H", "W"): -2,
    ("H", "Y"): 2, ("H", "V"): -3,
    ("I", "I"): 4, ("I", "L"): 2, ("I", "K"): -3, ("I", "M"): 1, ("I", "F"): 0,
    ("I", "P"): -3, ("I", "S"): -2, ("I", "T"): -1, ("I", "W"): -3, ("I", "Y"): -1,
    ("I", "V"): 3,
    ("L", "L"): 4, ("L", "K"): -2, ("L", "M"): 2, ("L", "F"): 0, ("L", "P"): -3,
    ("L", "S"): -2, ("L", "T"): -1, ("L", "W"): -2, ("L", "Y"): -1, ("L", "V"): 1,
    ("K", "K"): 5, ("K", "M"): -1, ("K", "F"): -3, ("K", "P"): -1, ("K", "S"): 0,
    ("K", "T"): -1, ("K", "W"): -3, ("K", "Y"): -2, ("K", "V"): -2,
    ("M", "M"): 5, ("M", "F"): 0, ("M", "P"): -2, ("M", "S"): -1, ("M", "T"): -1,
    ("M", "W"): -1, ("M", "Y"): -1, ("M", "V"): 1,
    ("F", "F"): 6, ("F", "P"): -4, ("F", "S"): -2, ("F", "T"): -2, ("F", "W"): 1,
    ("F", "Y"): 3, ("F", "V"): -1,
    ("P", "P"): 7, ("P", "S"): -1, ("P", "T"): -1, ("P", "W"): -4, ("P", "Y"): -3,
    ("P", "V"): -2,
    ("S", "S"): 4, ("S", "T"): 1, ("S", "W"): -3, ("S", "Y"): -2, ("S", "V"): -2,
    ("T", "T"): 5, ("T", "W"): -2, ("T", "Y"): -2, ("T", "V"): 0,
    ("W", "W"): 11, ("W", "Y"): 2, ("W", "V"): -3,
    ("Y", "Y"): 7, ("Y", "V"): -1,
    ("V", "V"): 4,
}


def substitution_features(wt, mut):
    """Compute chemistry-change features for a substitution."""
    if wt is None or mut is None:
        return {
            "volume_change": None,
            "charge_change": None,
            "polarity_change": None,
            "blosum62_score": None,
        }
    vol_wt = AA_VOLUME.get(wt)
    vol_mut = AA_VOLUME.get(mut)
    vol_change = (vol_mut - vol_wt) if vol_wt is not None and vol_mut is not None else None

    # BLOSUM62 is symmetric
    blosum = BLOSUM62.get((wt, mut), BLOSUM62.get((mut, wt), -3))

    return {
        "volume_change": vol_change,
        "charge_change": AA_CHARGE.get(mut, 0) - AA_CHARGE.get(wt, 0),
        "polarity_change": AA_POLAR.get(mut, 0) - AA_POLAR.get(wt, 0),
        "blosum62_score": blosum,
    }


# ---------------- Structure handling ----------------

def load_structure_with_sasa(pdb_path):
    """Load a PDB and precompute SASA for all residues."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("p", pdb_path)
    model = structure[0]

    sr = ShrakeRupley()
    sr.compute(model, level="R")
    return model


def get_residue(model, chain_id, residue_number):
    """Fetch a residue by chain and number."""
    if chain_id not in [c.id for c in model]:
        chain_id = [c.id for c in model][0]
    for residue in model[chain_id]:
        if residue.id[1] == residue_number:
            return residue
    return None


def distance_to_ligand(model, chain_id, residue_number):
    """Min distance from residue atoms to any hetero-atom (ligand)."""
    ligand_atoms = []
    for res in model.get_residues():
        if res.id[0] != " " and res.id[0] != "W":
            ligand_atoms.extend(res.get_atoms())

    if not ligand_atoms:
        return None

    residue = get_residue(model, chain_id, residue_number)
    if residue is None:
        return None

    min_dist = float("inf")
    for ta in residue.get_atoms():
        for la in ligand_atoms:
            d = ta - la
            if d < min_dist:
                min_dist = d
    return min_dist


def build_row(protein, mutation, label, model, chain_id):
    """Build one feature row."""
    parsed = parse_mutation(mutation)
    if parsed is None:
        return None

    pos = parsed["position"]
    wt = parsed["wildtype_aa"]
    mut = parsed["mutant_aa"]

    residue = get_residue(model, chain_id, pos)
    sasa = residue.sasa if residue is not None and hasattr(residue, "sasa") else None
    dist = distance_to_ligand(model, chain_id, pos)
    gatekeeper = 1 if GATEKEEPERS.get(protein) == pos else 0
    hydro = hydrophobicity_change(wt, mut)
    chem = substitution_features(wt, mut)

    return {
        "protein": protein,
        "mutation": mutation,
        "resistance_label": label,
        "position": pos,
        "wildtype_aa": wt,
        "mutant_aa": mut,
        "mutation_type": parsed["mutation_type"],
        "sasa": sasa,
        "distance_to_ligand": dist,
        "is_gatekeeper": gatekeeper,
        "hydrophobicity_change": hydro,
        "volume_change": chem["volume_change"],
        "charge_change": chem["charge_change"],
        "polarity_change": chem["polarity_change"],
        "blosum62_score": chem["blosum62_score"],
        "residue_found": 1 if residue is not None else 0,
    }


def main():
    input_path = "data/processed/curated_mutations.csv"
    output_path = "data/processed/features.csv"

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} mutations")

    rows = []
    models = {}

    for protein, group in df.groupby("protein"):
        if protein not in PROTEIN_PDB:
            print(f"Skipping {protein}: no PDB mapping")
            continue

        pdb_path, chain_id = PROTEIN_PDB[protein]
        if not os.path.exists(pdb_path):
            print(f"Skipping {protein}: missing {pdb_path}")
            continue

        if protein not in models:
            print(f"Loading structure for {protein}...")
            models[protein] = load_structure_with_sasa(pdb_path)

        model = models[protein]
        for _, row in group.iterrows():
            r = build_row(protein, row["mutation"],
                          row["resistance_label"], model, chain_id)
            if r is not None:
                rows.append(r)

    features = pd.DataFrame(rows)
    print(f"\nBuilt features for {len(features)} mutations")
    if len(features) > 0:
        print(f"Residue found: {features['residue_found'].sum()} of {len(features)}")

    # One-hot protein
    for p in PROTEIN_PDB.keys():
        features[f"protein_{p}"] = (features["protein"] == p).astype(int)

    features.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
