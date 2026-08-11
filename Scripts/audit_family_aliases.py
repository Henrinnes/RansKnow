"""
RansKnow Phase 0.2 — family alias collision audit.

Flags aliases in the family coverage list that are themselves common
English words (dictionary collisions), then re-extracts family mentions
for those aliases only, requiring a disambiguating ransomware-context
term within a token window. Reports a before/after count table.

Usage:
    python3 Scripts/audit_family_aliases.py
"""

import json
import re
from pathlib import Path

import pandas as pd

ROOT        = Path(__file__).resolve().parent.parent
TRANSCRIPTS = ROOT / "transcripts"
FAMILY_LIST = ROOT / "rubrics" / "Ransomware_Family_Coverage_List.xlsx"
FEATURES    = ROOT / "outputs" / "Knowledge_Agent_Features_1034.csv"
DICT_WORDS  = Path("/usr/share/dict/words")

# Disambiguating context required near a flagged (dictionary-word) alias
# for it to count as a real family mention.
CONTEXT_TERMS = [
    "ransomware", "ransom", "encrypt", "decrypt", "gang", "group",
    "extortion", "victim", "breach", "malware", "attack", "leak",
    "affiliate", "threat actor", "cyber", "hacker", "incident",
    "compromise", "double extortion", "raas",
]
WINDOW_TOKENS = 15


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def load_alias_map():
    df = pd.read_excel(FAMILY_LIST).fillna("")
    alias_map = {}
    for _, row in df.iterrows():
        fam = str(row.get("Ransomware_Family_Name", "")).strip()
        if not fam:
            continue
        alias_map[fam.lower()] = fam
        for alias in str(row.get("Alias_Names", "")).split(","):
            alias = alias.strip()
            if alias and alias != "—":
                alias_map[alias.lower()] = fam
    return alias_map


def load_dict_words():
    words = set()
    with open(DICT_WORDS, encoding="utf-8", errors="ignore") as f:
        for line in f:
            words.add(line.strip().lower())
    return words


def flag_collisions(alias_map, dict_words):
    flagged = {}
    for alias, fam in alias_map.items():
        if " " in alias:
            continue  # multi-word aliases are effectively self-disambiguating
        if len(alias) > 2 and alias in dict_words:
            flagged[alias] = fam
    return flagged


def find_matches(text: str, pattern: re.Pattern):
    return [m.start() for m in pattern.finditer(text)]


def has_context(text: str, pos: int, window_tokens: int) -> bool:
    start = max(0, pos - window_tokens * 8)   # ~8 chars/token estimate, generous
    end = min(len(text), pos + window_tokens * 8)
    span = text[start:end]
    return any(term in span for term in CONTEXT_TERMS)


def main():
    alias_map = load_alias_map()
    dict_words = load_dict_words()
    flagged = flag_collisions(alias_map, dict_words)

    print(f"Aliases loaded: {len(alias_map)}")
    print(f"Flagged as dictionary-word collisions: {len(flagged)}")
    for a, fam in sorted(flagged.items()):
        print(f"  {a!r:16} -> {fam}")
    print()

    df = pd.read_csv(FEATURES)
    paths = {p.name.replace(".txt", ""): p for p in TRANSCRIPTS.rglob("V*.txt")}

    flagged_pats = {a: re.compile(rf"\b{re.escape(a)}\b") for a in flagged}

    rows = []
    for _, r in df.iterrows():
        vid = r["Video_ID"]
        p = paths.get(vid)
        if p is None:
            continue
        text = _norm(p.read_text(encoding="utf-8", errors="ignore"))

        before_hit = {}
        after_hit = {}
        for alias, fam in flagged.items():
            positions = find_matches(text, flagged_pats[alias])
            if not positions:
                continue
            before_hit[fam] = before_hit.get(fam, 0) + len(positions)
            kept = sum(1 for pos in positions if has_context(text, pos, WINDOW_TOKENS))
            if kept:
                after_hit[fam] = after_hit.get(fam, 0) + kept

        if before_hit or after_hit:
            rows.append({
                "Video_ID": vid,
                "before_families": ", ".join(sorted(before_hit)),
                "after_families": ", ".join(sorted(after_hit)),
                "lost": ", ".join(sorted(set(before_hit) - set(after_hit))),
            })

    audit_df = pd.DataFrame(rows)
    out_path = ROOT / "outputs" / "Family_Alias_Audit.csv"
    audit_df.to_csv(out_path, index=False)

    print(f"Videos with >=1 flagged-alias hit: {len(audit_df)}")
    print(f"Wrote per-video audit trail -> {out_path.relative_to(ROOT)}\n")

    # Before/after per-family counts
    print(f"{'Family':<14}{'Before (any hit)':>18}{'After (context-gated)':>24}{'Dropped':>10}")
    for fam in sorted(set(flagged.values())):
        b = (audit_df["before_families"].str.split(", ").apply(lambda L: fam in L if isinstance(L, list) else False)).sum()
        a = (audit_df["after_families"].str.split(", ").apply(lambda L: fam in L if isinstance(L, list) else False)).sum()
        print(f"{fam:<14}{b:>18}{a:>24}{b - a:>10}")


if __name__ == "__main__":
    main()
