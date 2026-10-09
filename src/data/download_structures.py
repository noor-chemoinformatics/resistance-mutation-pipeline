"""
Download PDB structures for each target protein.
"""

import os
import requests


STRUCTURES = {
    "4I22": "EGFR",
    "2HYY": "ABL1",
    "4MKC": "ALK",
    "6OIM": "KRAS",
    "4MNE": "BRAF",
}


def download_pdb(pdb_id, output_dir="data/structures"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{pdb_id}.pdb")
    if os.path.exists(path):
        print(f"  {pdb_id}: already exists")
        return path

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"  Downloading {pdb_id}...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(path, "w") as f:
        f.write(r.text)
    print(f"  Saved {path}")
    return path


def main():
    for pdb_id, protein in STRUCTURES.items():
        print(f"{protein}:")
        try:
            download_pdb(pdb_id)
        except Exception as e:
            print(f"  FAILED: {e}")


if __name__ == "__main__":
    main()
