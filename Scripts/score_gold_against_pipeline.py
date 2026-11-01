"""
RansKnow Phase 0.3 -- score the Knowledge Agent against the human gold set.

Reads Gold_Eval_Set_Annotation_Template.xlsx after
Scripts/merge_gold_annotations.py has filled in the annotator columns, and
reports how well the pipeline's labels match human judgment, with Wilson 95%
intervals. A field's gold value is the Adjudicated_* column if filled,
otherwise the annotators' value where both agree. Unresolved disagreements
are counted and left out, not guessed.

This script generates no labels. It reports nothing until humans have
annotated the set.

The sample is stratified with roughly equal per-stratum targets, so
family-positive videos are over-represented relative to the corpus. These
figures describe the sample and should not be quoted as corpus-wide rates.

Usage:
    python3 Scripts/score_gold_against_pipeline.py
    python3 Scripts/score_gold_against_pipeline.py --gold path.xlsx --out scores.json
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from knowledge_agent import FAMILY_LIST, _load_aliases  # noqa: E402

GOLD_XLSX = ROOT / "outputs" / "Gold_Eval_Set_Annotation_Template.xlsx"
FEATURES = ROOT / "outputs" / "Knowledge_Agent_Features_1034.csv"


def wilson(k: int, n: int, z: float = 1.96) -> dict:
    if n == 0:
        return {"k": 0, "n": 0, "rate": None, "ci95": None}
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return {"k": k, "n": n, "rate": round(p, 3),
            "ci95": [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]}


def _clean(value) -> str:
    return "" if pd.isna(value) else str(value).strip()


def resolve(row: pd.Series, field: str):
    """Returns (gold value, status): adjudicated, agreed, disagreement, or missing."""
    adjudicated = _clean(row[f"Adjudicated_{field}"])
    if adjudicated:
        return adjudicated, "adjudicated"
    a1, a2 = _clean(row[f"Annotator1_{field}"]), _clean(row[f"Annotator2_{field}"])
    if not a1 and not a2:
        return "", "missing"
    if a1.lower() == a2.lower():
        return a1, "agreed"
    return "", "disagreement"


def family_set(text: str, alias_map: dict, unmapped: set) -> set:
    """Maps free-text family names onto the pipeline's canonical names."""
    names = set()
    for part in re.split(r"[,;/]", text):
        part = part.strip().lower()
        if not part or part in {"none", "n/a", "na"}:
            continue
        if part in alias_map:
            names.add(alias_map[part])
        else:
            unmapped.add(part)
            names.add(part)
    return names


def score_families(rows: list, alias_map: dict) -> dict:
    """Pair-level precision and recall over (video, family). A blank gold
    field means the annotators named no family, so it counts as an empty set
    only where they said the video had been reviewed (relevance answered)."""
    tp, fp, fn = 0, 0, 0
    per_family = defaultdict(lambda: [0, 0])
    unmapped: set = set()
    for ka_text, gold_text in rows:
        ka = family_set(ka_text, alias_map, set())
        gold = family_set(gold_text, alias_map, unmapped)
        tp += len(ka & gold)
        fp += len(ka - gold)
        fn += len(gold - ka)
        for fam in ka:
            per_family[fam][1] += 1
            per_family[fam][0] += fam in gold
    return {
        "videos": len(rows),
        "precision": wilson(tp, tp + fp),
        "recall": wilson(tp, tp + fn),
        "precision_per_family": {f: wilson(k, n) for f, (k, n) in sorted(per_family.items())},
        "gold_names_not_in_alias_list": sorted(unmapped),
    }


def score_single_label(gold_df: pd.DataFrame, field: str, ka_col: str, skip: set) -> dict:
    agree, total, excluded = 0, 0, 0
    status = defaultdict(int)
    for _, row in gold_df.iterrows():
        gold, how = resolve(row, field)
        status[how] += 1
        if not gold:
            continue
        if gold in skip:
            excluded += 1
            continue
        total += 1
        agree += gold.lower() == _clean(row[ka_col]).lower()
    return {"agreement": wilson(agree, total), "excluded": excluded, "status": dict(status)}


def main(gold_path: Path, out_path: Path = None) -> None:
    gold = pd.read_excel(gold_path)
    features = pd.read_csv(FEATURES)
    alias_map = {k.lower(): v for k, v in _load_aliases(FAMILY_LIST).items()}
    for name in set(alias_map.values()):
        alias_map.setdefault(name.lower(), name)

    ka_relevant = dict(zip(
        features["Video_ID"],
        (features["Family_Count"] > 0) | (features["Tactic_Impact"] > 0),
    ))

    tp, fp, fn, relevance_status = 0, 0, 0, defaultdict(int)
    family_rows = []
    for _, row in gold.iterrows():
        answer, how = resolve(row, "Ransomware_Relevant_YN")
        relevance_status[how] += 1
        if how in {"missing", "disagreement"}:
            continue
        family, family_how = resolve(row, "Family")
        if family_how != "disagreement":
            family_rows.append((_clean(row["KA_Family_List"]), family))
        if answer == "Uncertain":
            continue
        ka = bool(ka_relevant[row["Video_ID"]])
        tp += ka and answer == "Yes"
        fp += ka and answer == "No"
        fn += (not ka) and answer == "Yes"

    report = {
        "n_sampled": len(gold),
        "relevance": {
            "precision": wilson(tp, tp + fp),
            "recall": wilson(tp, tp + fn),
            "status": dict(relevance_status),
        },
        "family": score_families(family_rows, alias_map),
        "dominant_tactic": score_single_label(gold, "Dominant_Tactic", "KA_Dominant_Tactic", set()),
        "platform": score_single_label(gold, "Platform", "KA_Platform_Signal", {"Multiple"}),
        "note": "Stratified sample figures, not corpus-wide rates.",
    }
    print(json.dumps(report, indent=2))
    if out_path:
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score Knowledge Agent labels against the gold set")
    parser.add_argument("--gold", type=Path, default=GOLD_XLSX)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    main(args.gold, args.out)
