"""
Collect cancer mutations from the cBioPortal public API.

Uses the POST /mutations/fetch endpoint, which works reliably
across all TCGA studies.

API docs: https://www.cbioportal.org/api/swagger-ui.html

Output: data/raw/cbioportal_mutations.csv
"""

import os
import time
import requests
import pandas as pd


BASE_URL = "https://www.cbioportal.org/api"

# Genes we want, with their Entrez gene IDs
GENE_IDS = {
    "EGFR": 1956,
    "ABL1": 25,
    "ALK": 238,
    "KRAS": 3845,
    "BRAF": 673,
}

# TCGA studies relevant to these genes
STUDIES = [
    "luad_tcga",       # Lung adenocarcinoma
    "laml_tcga",       # Acute myeloid leukemia
    "skcm_tcga",       # Skin cutaneous melanoma
    "coadread_tcga",   # Colorectal adenocarcinoma
]


def get_mutation_profile_id(study_id):
    """Return the mutation molecular profile ID for a study."""
    url = f"{BASE_URL}/studies/{study_id}/molecular-profiles"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    for p in r.json():
        if p.get("molecularAlterationType") == "MUTATION_EXTENDED":
            return p["molecularProfileId"]
    raise ValueError(f"No mutation profile for {study_id}")


def get_sequenced_sample_list_id(study_id):
    """Return the best available sample list ID for mutations."""
    url = f"{BASE_URL}/studies/{study_id}/sample-lists"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    ids = [s["sampleListId"] for s in r.json()]

    # Preference order: sequenced > all
    for suffix in ("_sequenced", "_all"):
        for sid in ids:
            if sid.endswith(suffix):
                return sid

    if ids:
        return ids[0]
    raise ValueError(f"No sample lists for {study_id}")


def fetch_mutations(profile_id, sample_list_id):
    """Fetch mutations via the POST /mutations/fetch endpoint."""
    url = f"{BASE_URL}/molecular-profiles/{profile_id}/mutations/fetch"

    payload = {
        "sampleListId": sample_list_id,
        "entrezGeneIds": list(GENE_IDS.values()),
    }
    headers = {"Content-Type": "application/json"}

    r = requests.post(url, json=payload, headers=headers, timeout=180)
    r.raise_for_status()
    return r.json()


def mutations_to_dataframe(mutations):
    """Convert raw mutation records to a tidy DataFrame."""
    # Build a reverse lookup: entrez ID -> gene symbol
    id_to_gene = {v: k for k, v in GENE_IDS.items()}

    rows = []
    for m in mutations:
        entrez_id = m.get("entrezGeneId")
        gene_symbol = id_to_gene.get(entrez_id, "")

        rows.append({
            "protein": gene_symbol,
            "mutation": m.get("proteinChange", ""),
            "mutation_type": m.get("mutationType", ""),
            "sample_id": m.get("sampleId", ""),
            "study_id": m.get("studyId", ""),
            "chr": m.get("chr", ""),
            "start_position": m.get("startPosition", ""),
            "end_position": m.get("endPosition", ""),
            "ref_allele": m.get("referenceAllele", ""),
            "alt_allele": m.get("variantAllele", ""),
            "entrez_gene_id": entrez_id if entrez_id else "",
        })
    return pd.DataFrame(rows)    
def main():
    os.makedirs("data/raw", exist_ok=True)
    all_frames = []

    for study_id in STUDIES:
        print(f"\n=== Study: {study_id} ===")
        try:
            profile_id = get_mutation_profile_id(study_id)
            sample_list_id = get_sequenced_sample_list_id(study_id)
            print(f"  Profile: {profile_id}")
            print(f"  Sample list: {sample_list_id}")

            mutations = fetch_mutations(profile_id, sample_list_id)
            print(f"  Fetched {len(mutations)} mutations")
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        if not mutations:
            print("  No mutations returned.")
            continue

        df = mutations_to_dataframe(mutations)

        # Keep only the genes we care about
        df = df[df["protein"].isin(GENE_IDS.keys())]
        print(f"  After filtering to target genes: {len(df)} mutations")

        if not df.empty:
            print(f"  Per gene: {df['protein'].value_counts().to_dict()}")
            all_frames.append(df)

        time.sleep(1)

    if not all_frames:
        print("\nNo mutations collected.")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["protein", "mutation", "sample_id", "study_id"]
    )
    combined = combined[combined["mutation"].astype(str).str.len() > 0]

    output_path = "data/raw/cbioportal_mutations.csv"
    combined.to_csv(output_path, index=False)

    print("\n=== Summary ===")
    print(f"Total unique mutations saved: {len(combined)}")
    print(f"Output: {output_path}")
    print("\nMutations per protein:")
    print(combined["protein"].value_counts())


if __name__ == "__main__":
    main()
