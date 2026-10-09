"""
Parse mutation strings like 'T790M' into structured fields.
"""

import re


AA_MAP = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

HYDRO = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}


def parse_mutation(mutation_str):
    """Parse X123Y or X123_Y456del into components."""
    mutation_str = str(mutation_str).strip()
    if not mutation_str:
        return None

    # Simple missense
    m = re.fullmatch(r"([A-Z])(\d+)([A-Z])", mutation_str)
    if m:
        wt, pos, mut = m.groups()
        return {"wildtype_aa": wt, "position": int(pos),
                "mutant_aa": mut, "mutation_type": "missense"}

    # Deletion
    m = re.fullmatch(r"([A-Z])(\d+)(?:_([A-Z])(\d+))?del", mutation_str)
    if m:
        wt, pos, _, _ = m.groups()
        return {"wildtype_aa": wt, "position": int(pos),
                "mutant_aa": None, "mutation_type": "deletion"}

    # Delins
    m = re.fullmatch(r"([A-Z])(\d+)(?:_([A-Z])(\d+))?delins([A-Z]+)", mutation_str)
    if m:
        wt, pos, _, _, _ = m.groups()
        return {"wildtype_aa": wt, "position": int(pos),
                "mutant_aa": None, "mutation_type": "delins"}

    return None


def hydrophobicity_change(wt, mut):
    if wt is None or mut is None:
        return None
    a = HYDRO.get(wt)
    b = HYDRO.get(mut)
    return None if a is None or b is None else b - a


if __name__ == "__main__":
    for t in ["T790M", "L858R", "E746_A750del", "G12C", "Q61H"]:
        print(f"{t:20s} -> {parse_mutation(t)}")
