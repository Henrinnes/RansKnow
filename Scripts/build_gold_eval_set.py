"""
RansKnow Phase 0.3 -- gold evaluation subset sampler.

Draws a stratified sample of videos for human annotation, held out from
all model training in every later phase. Stratifies on three axes:
  - Channel_Type   (Vendor / Conference / Independent / DFIR / Government / Media)
  - Year bucket    (<=2022 / 2023-2024 / 2025-2026)
  - Family bucket  (0 / 1 / 2+ families currently detected)

Output is an annotation-ready spreadsheet with blank columns for two
independent annotators to fill in by hand -- this script does NOT
generate labels, only the sampling and the template.

Usage:
    python3 Scripts/build_gold_eval_set.py --n 180
"""

import argparse
from pathlib import Path

import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
FEATURES = ROOT / "outputs" / "Knowledge_Agent_Features_1034.csv"
OUT      = ROOT / "outputs" / "Gold_Eval_Set_Annotation_Template.xlsx"


def year_bucket(y):
    if pd.isna(y):
        return "unknown"
    y = int(y)
    if y <= 2022:
        return "<=2022"
    if y <= 2024:
        return "2023-2024"
    return "2025-2026"


def family_bucket(n):
    if n == 0:
        return "0"
    if n == 1:
        return "1"
    return "2+"


def main(n_target: int, seed: int):
    df = pd.read_csv(FEATURES)
    df["year_bucket"] = df["Year"].apply(year_bucket)
    df["family_bucket"] = df["Family_Count"].apply(family_bucket)
    df["stratum"] = df["Channel_Type"] + " | " + df["year_bucket"] + " | " + df["family_bucket"]

    strata = df["stratum"].value_counts()
    n_strata = len(strata)
    per_stratum = max(1, n_target // n_strata)

    print(f"{n_strata} strata, targeting ~{per_stratum} videos each for n={n_target}\n")

    sampled = []
    rng = pd.Series(range(len(df)))  # placeholder to keep pandas sample reproducible via seed
    for stratum, group in df.groupby("stratum"):
        k = min(per_stratum, len(group))
        sampled.append(group.sample(k, random_state=seed))

    sample_df = pd.concat(sampled).sort_values(["Channel_Type", "Year", "Video_ID"]).reset_index(drop=True)

    # Top up to n_target if strata were too thin, from whatever's left.
    if len(sample_df) < n_target:
        remaining = df[~df["Video_ID"].isin(sample_df["Video_ID"])]
        top_up = remaining.sample(min(n_target - len(sample_df), len(remaining)), random_state=seed)
        sample_df = pd.concat([sample_df, top_up]).reset_index(drop=True)

    print(f"Sampled {len(sample_df)} videos")
    print(sample_df["Channel_Type"].value_counts().to_string())

    template = sample_df[[
        "Video_ID", "Channel_ID", "Channel_Name", "Channel_Type",
        "Video_Title", "YouTube_URL", "Year", "Transcript_Path",
        "Family_Count", "Family_List", "Dominant_Tactic", "Platform_Signal",
    ]].copy()
    template = template.rename(columns={
        "Family_Count":   "KA_Family_Count",
        "Family_List":    "KA_Family_List",
        "Dominant_Tactic": "KA_Dominant_Tactic",
        "Platform_Signal": "KA_Platform_Signal",
    })

    # Blank columns for each of two independent annotators, plus adjudication.
    for who in ("Annotator1", "Annotator2", "Adjudicated"):
        template[f"{who}_Ransomware_Relevant_YN"] = ""
        template[f"{who}_Family"] = ""
        template[f"{who}_Dominant_Tactic"] = ""
        template[f"{who}_Platform"] = ""
        template[f"{who}_Notes"] = ""

    template.to_excel(OUT, index=False)
    print(f"\nWrote annotation template -> {OUT.relative_to(ROOT)}")
    print("KA_* columns are the current Knowledge Agent output, shown for reference only --")
    print("annotators should watch/skim each transcript and fill in their own columns blind")
    print("to the KA_* values where practical.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the RansKnow gold evaluation subset")
    parser.add_argument("--n", type=int, default=180, help="Target sample size (default: 180)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.n, args.seed)
