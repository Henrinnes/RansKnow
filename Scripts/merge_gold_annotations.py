"""
RansKnow Phase 0.3 -- merge annotator exports into the master gold-set file.

Takes the JSON files exported by two independent annotators from the
annotation tool (outputs/Gold_Annotation_Tool.html) and writes their
answers into Annotator1_*/Annotator2_* columns of
Gold_Eval_Set_Annotation_Template.xlsx. Adjudicated_* columns are left
for manual reconciliation of disagreements -- not filled automatically,
since resolving them is exactly the judgment call this process exists to
capture.

Also prints Cohen's kappa per field (Relevant, Dominant_Tactic, Platform)
over videos both annotators actually completed, and a raw agreement rate
for Family (free text, so kappa isn't a clean fit -- exact string match
after lowercasing/stripping is the crude-but-honest metric here).

Usage:
    python3 Scripts/merge_gold_annotations.py \
        --annotator1 ransknow_gold_henry_2026-08-15.json \
        --annotator2 ransknow_gold_someone_else_2026-08-16.json
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent
GOLD_XLSX = ROOT / "outputs" / "Gold_Eval_Set_Annotation_Template.xlsx"
BACKUP_DIR = ROOT / "outputs" / "gold_backups"

ANNOTATOR_COLS = [
    "Ransomware_Relevant_YN", "Family", "Dominant_Tactic", "Platform", "Notes",
]


def load_export(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload.get("annotations", payload), payload.get("annotator", path.stem)


def fill_columns(df: pd.DataFrame, annotations: dict, who: str) -> None:
    # Force object dtype first -- these columns start out all-NaN (float64
    # by default), and assigning strings into a float64 column is a
    # deprecated pandas pattern that's noisy today and an error in a
    # future pandas version.
    for suffix in ANNOTATOR_COLS:
        col = f"{who}_{suffix}"
        if df[col].dtype != object:
            df[col] = df[col].astype(object)

    for vid, a in annotations.items():
        mask = df["Video_ID"] == vid
        if not mask.any():
            print(f"  [WARN] {vid} not found in gold set -- skipping")
            continue
        df.loc[mask, f"{who}_Ransomware_Relevant_YN"] = a.get("relevant", "")
        df.loc[mask, f"{who}_Family"] = a.get("family", "")
        df.loc[mask, f"{who}_Dominant_Tactic"] = a.get("dominant_tactic", "")
        df.loc[mask, f"{who}_Platform"] = a.get("platform", "")
        df.loc[mask, f"{who}_Notes"] = a.get("notes", "")


def agreement_report(df: pd.DataFrame) -> None:
    both = df[
        (df["Annotator1_Ransomware_Relevant_YN"].fillna("") != "") &
        (df["Annotator2_Ransomware_Relevant_YN"].fillna("") != "")
    ]
    print(f"\nVideos completed by both annotators: {len(both)} / {len(df)}")
    if len(both) < 2:
        print("Not enough overlap yet for agreement stats.")
        return

    for field, col in [
        ("Ransomware-relevant", "Ransomware_Relevant_YN"),
        ("Dominant tactic", "Dominant_Tactic"),
        ("Platform", "Platform"),
    ]:
        a1 = both[f"Annotator1_{col}"].fillna("")
        a2 = both[f"Annotator2_{col}"].fillna("")
        kappa = cohen_kappa_score(a1, a2)
        agree = (a1 == a2).mean()
        print(f"  {field:22} kappa={kappa:.3f}  raw agreement={agree:.0%}")

    f1 = both["Annotator1_Family"].fillna("").str.strip().str.lower()
    f2 = both["Annotator2_Family"].fillna("").str.strip().str.lower()
    print(f"  {'Family (exact match)':22} raw agreement={(f1 == f2).mean():.0%}  "
          f"(free text -- treat as a lower bound, not a real kappa)")


def main():
    parser = argparse.ArgumentParser(description="Merge gold-set annotator exports")
    parser.add_argument("--annotator1", required=True, help="Path to annotator 1's exported JSON")
    parser.add_argument("--annotator2", help="Path to annotator 2's exported JSON (optional)")
    args = parser.parse_args()

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = BACKUP_DIR / f"{GOLD_XLSX.stem}_{time.strftime('%Y%m%d_%H%M%S')}.xlsx"
    shutil.copy2(GOLD_XLSX, backup_path)
    print(f"Backed up current file -> {backup_path.relative_to(ROOT)}")

    df = pd.read_excel(GOLD_XLSX)

    ann1, name1 = load_export(Path(args.annotator1))
    print(f"Annotator1 = {name1} ({len(ann1)} annotations)")
    fill_columns(df, ann1, "Annotator1")

    if args.annotator2:
        ann2, name2 = load_export(Path(args.annotator2))
        print(f"Annotator2 = {name2} ({len(ann2)} annotations)")
        fill_columns(df, ann2, "Annotator2")

    df.to_excel(GOLD_XLSX, index=False)
    print(f"\nWrote {GOLD_XLSX.relative_to(ROOT)}")

    if args.annotator2:
        agreement_report(df)
    print("\nAdjudicated_* columns left blank -- resolve disagreements manually, "
          "that judgment call is the point of having two annotators.")


if __name__ == "__main__":
    main()
