"""
Curate the mutation dataset with resistance labels.

Sources:
- Published clinical literature
- MdrDB (Drug Resistance Mutation Database)
- OncoKB clinical annotations

Label convention:
  1 = resistant (drug no longer works)
  0 = sensitive (drug still works, or mutation is a sensitizing one)

This curated set contains ONLY mutations with well-documented clinical
evidence. It is intentionally small and high-confidence.
"""

import os
import pandas as pd


KNOWN_LABELS = {
    # ==================================================
    # EGFR — verified from clinical literature
    # ==================================================
    ("EGFR", "L858R"): 0,
    ("EGFR", "L861Q"): 0,
    ("EGFR", "G719S"): 0,
    ("EGFR", "G719A"): 0,
    ("EGFR", "G719C"): 0,
    ("EGFR", "S768I"): 0,
    ("EGFR", "E746_A750del"): 0,
    ("EGFR", "E746_T751del"): 0,
    ("EGFR", "L747_P753del"): 0,
    ("EGFR", "L747_T751del"): 0,
    ("EGFR", "L747_A750del"): 0,
    ("EGFR", "L747_S752del"): 0,
    ("EGFR", "L747_E749del"): 0,
    ("EGFR", "A750_E758del"): 0,
    ("EGFR", "S752_I759del"): 0,
    ("EGFR", "R831H"): 0,
    ("EGFR", "A859T"): 0,
    ("EGFR", "E866K"): 0,
    ("EGFR", "L861R"): 0,

    ("EGFR", "T790M"): 1,
    ("EGFR", "C797S"): 1,
    ("EGFR", "C797G"): 1,
    ("EGFR", "L718Q"): 1,
    ("EGFR", "L718V"): 1,
    ("EGFR", "G724S"): 1,
    ("EGFR", "L792H"): 1,
    ("EGFR", "L792F"): 1,
    ("EGFR", "G796S"): 1,
    ("EGFR", "G796R"): 1,
    ("EGFR", "G796D"): 1,
    ("EGFR", "T854A"): 1,
    ("EGFR", "L747S"): 1,
    ("EGFR", "D761Y"): 1,
    ("EGFR", "V843I"): 1,
    ("EGFR", "G810S"): 1,
    ("EGFR", "G810D"): 1,
    ("EGFR", "G810A"): 1,

    # ==================================================
    # ABL1 — verified
    # ==================================================
    ("ABL1", "T315I"): 1,
    ("ABL1", "T315A"): 1,
    ("ABL1", "T315M"): 1,
    ("ABL1", "T315V"): 1,
    ("ABL1", "E255K"): 1,
    ("ABL1", "E255V"): 1,
    ("ABL1", "Y253H"): 1,
    ("ABL1", "Y253F"): 1,
    ("ABL1", "Y253C"): 1,
    ("ABL1", "F359V"): 1,
    ("ABL1", "F359C"): 1,
    ("ABL1", "F359I"): 1,
    ("ABL1", "H396R"): 1,
    ("ABL1", "H396P"): 1,
    ("ABL1", "H396A"): 1,
    ("ABL1", "M351T"): 1,
    ("ABL1", "M351V"): 1,
    ("ABL1", "G250E"): 1,
    ("ABL1", "G250R"): 1,
    ("ABL1", "V299L"): 1,
    ("ABL1", "F317L"): 1,
    ("ABL1", "F317I"): 1,
    ("ABL1", "F317V"): 1,
    ("ABL1", "F317C"): 1,
    ("ABL1", "F311L"): 1,
    ("ABL1", "E282K"): 1,
    ("ABL1", "L248V"): 1,
    ("ABL1", "L248R"): 1,
    ("ABL1", "Q252H"): 1,
    ("ABL1", "Q252R"): 1,
    ("ABL1", "D276G"): 1,
    ("ABL1", "E279K"): 1,
    ("ABL1", "E292V"): 1,
    ("ABL1", "L387M"): 1,
    ("ABL1", "L387F"): 1,
    ("ABL1", "V289A"): 1,
    ("ABL1", "V289F"): 1,
    ("ABL1", "E355G"): 1,
    ("ABL1", "E355D"): 1,
    ("ABL1", "A380T"): 0,
    ("ABL1", "A365V"): 0,
    ("ABL1", "A344V"): 0,
    ("ABL1", "K247R"): 0,
    ("ABL1", "E258D"): 0,
    ("ABL1", "P309S"): 0,

    # ==================================================
    # ALK — verified
    # ==================================================
    ("ALK", "G1202R"): 1,
    ("ALK", "G1202del"): 1,
    ("ALK", "L1196M"): 1,
    ("ALK", "C1156Y"): 1,
    ("ALK", "C1156F"): 1,
    ("ALK", "F1174L"): 1,
    ("ALK", "F1174C"): 1,
    ("ALK", "F1174V"): 1,
    ("ALK", "F1174S"): 1,
    ("ALK", "L1152R"): 1,
    ("ALK", "L1152P"): 1,
    ("ALK", "S1206Y"): 1,
    ("ALK", "S1206C"): 1,
    ("ALK", "G1269A"): 1,
    ("ALK", "G1269S"): 1,
    ("ALK", "I1171T"): 1,
    ("ALK", "I1171N"): 1,
    ("ALK", "I1171S"): 1,
    ("ALK", "V1180L"): 1,
    ("ALK", "I1151T"): 1,
    ("ALK", "T1151M"): 1,
    ("ALK", "E1210K"): 1,
    ("ALK", "D1203N"): 1,
    ("ALK", "F1245C"): 1,
    ("ALK", "F1245V"): 1,
    ("ALK", "F1245L"): 1,

    ("ALK", "L1198F"): 0,
    ("ALK", "R1275Q"): 0,
    ("ALK", "R1275L"): 0,
    ("ALK", "M1160I"): 0,
    ("ALK", "A1200V"): 0,
    ("ALK", "A348T"): 0,
    ("ALK", "V349F"): 0,
    ("ALK", "V349L"): 0,
    ("ALK", "D1091N"): 0,
    ("ALK", "T1151K"): 0,
    ("ALK", "I1170N"): 0,
    ("ALK", "K1062M"): 0,
    ("ALK", "I1268V"): 0,

    # ==================================================
    # KRAS — verified
    # ==================================================
    ("KRAS", "G12C"): 0,
    ("KRAS", "G12D"): 1,
    ("KRAS", "G12V"): 1,
    ("KRAS", "G12A"): 1,
    ("KRAS", "G12S"): 1,
    ("KRAS", "G12R"): 1,
    ("KRAS", "G12F"): 1,
    ("KRAS", "G12I"): 1,
    ("KRAS", "G12L"): 1,
    ("KRAS", "G13D"): 1,
    ("KRAS", "G13C"): 1,
    ("KRAS", "G13V"): 1,
    ("KRAS", "G13A"): 1,
    ("KRAS", "G13S"): 1,
    ("KRAS", "G13R"): 1,
    ("KRAS", "G13E"): 1,
    ("KRAS", "Q61H"): 1,
    ("KRAS", "Q61K"): 1,
    ("KRAS", "Q61R"): 1,
    ("KRAS", "Q61L"): 1,
    ("KRAS", "Q61E"): 1,
    ("KRAS", "Q61P"): 1,
    ("KRAS", "K117N"): 1,
    ("KRAS", "A146T"): 1,
    ("KRAS", "A146V"): 1,
    ("KRAS", "A146P"): 1,
    ("KRAS", "V14I"): 1,

    # ==================================================
    # BRAF — verified
    # ==================================================
    ("BRAF", "V600E"): 0,
    ("BRAF", "V600K"): 0,
    ("BRAF", "V600R"): 0,
    ("BRAF", "V600D"): 0,
    ("BRAF", "V600G"): 0,
    ("BRAF", "V600_K601delinsE"): 0,
    ("BRAF", "V600_K601delinsD"): 0,
    ("BRAF", "V600delinsDLAT"): 0,

    ("BRAF", "V600M"): 1,
    ("BRAF", "V600L"): 1,
    ("BRAF", "G469A"): 1,
    ("BRAF", "G469V"): 1,
    ("BRAF", "G469E"): 1,
    ("BRAF", "G469R"): 1,
    ("BRAF", "G469S"): 1,
    ("BRAF", "K601E"): 1,
    ("BRAF", "K601N"): 1,
    ("BRAF", "K601T"): 1,
    ("BRAF", "L597V"): 1,
    ("BRAF", "L597R"): 1,
    ("BRAF", "L597Q"): 1,
    ("BRAF", "D594G"): 1,
    ("BRAF", "D594N"): 1,
    ("BRAF", "D594V"): 1,
    ("BRAF", "D594A"): 1,
    ("BRAF", "D594H"): 1,
    ("BRAF", "G466V"): 1,
    ("BRAF", "G466E"): 1,
    ("BRAF", "G466A"): 1,
    ("BRAF", "G466R"): 1,
    ("BRAF", "N581S"): 1,
    ("BRAF", "N581I"): 1,
    ("BRAF", "N581Y"): 1,
    ("BRAF", "G596R"): 1,
    ("BRAF", "G596D"): 1,
    ("BRAF", "F595L"): 1,
    ("BRAF", "T599I"): 1,
    ("BRAF", "L505H"): 1,
    ("BRAF", "R462I"): 1,
    ("BRAF", "I463S"): 1,
}


def label_mutation(protein, mutation):
    return KNOWN_LABELS.get((protein, mutation))


def curate(cbioportal_path, output_path):
    """Load cBioPortal mutations, add labels, expand with curated entries."""
    df = pd.read_csv(cbioportal_path)
    print(f"Loaded {len(df)} mutations from cBioPortal")

    df["resistance_label"] = [
        label_mutation(row["protein"], row["mutation"])
        for _, row in df.iterrows()
    ]
    labeled = df.dropna(subset=["resistance_label"]).copy()
    labeled["resistance_label"] = labeled["resistance_label"].astype(int)

    print(f"Labeled cBioPortal mutations: {len(labeled)}")
    unique_cbio = labeled[["protein", "mutation"]].drop_duplicates().shape[0]
    print(f"Unique (protein, mutation) from cBioPortal: {unique_cbio}")

    existing_keys = set(zip(labeled["protein"], labeled["mutation"]))
    extra_rows = []
    for (protein, mutation), label in KNOWN_LABELS.items():
        if (protein, mutation) not in existing_keys:
            extra_rows.append({
                "protein": protein,
                "mutation": mutation,
                "mutation_type": "",
                "sample_id": "curated",
                "study_id": "literature",
                "chr": "",
                "start_position": "",
                "end_position": "",
                "ref_allele": "",
                "alt_allele": "",
                "entrez_gene_id": "",
                "resistance_label": label,
            })

    extra_df = pd.DataFrame(extra_rows)
    print(f"Added {len(extra_df)} curated mutations from literature/MdrDB")

    combined = pd.concat([labeled, extra_df], ignore_index=True)
    dedup = combined.drop_duplicates(subset=["protein", "mutation"], keep="first")

    print(f"\nTotal unique mutations: {len(dedup)}")
    print("\nClass distribution:")
    print(dedup["resistance_label"].value_counts())
    print("\nPer protein per class:")
    print(dedup.groupby(["protein", "resistance_label"]).size().unstack(fill_value=0))

    os.makedirs("data/processed", exist_ok=True)
    dedup.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    curate(
        cbioportal_path="data/raw/cbioportal_mutations.csv",
        output_path="data/processed/curated_mutations.csv",
    )
