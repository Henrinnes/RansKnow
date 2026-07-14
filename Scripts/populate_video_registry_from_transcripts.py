import re
import os
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

import pandas as pd


# -----------------------------
# CONFIG
# -----------------------------
FAMILY_LIST_XLSX = "Ransomware_Family_Coverage_List.xlsx"   # your file
VIDEO_REGISTRY_XLSX = "Video_Registry_Template.xlsx"       # your file
TRANSCRIPTS_DIR = "transcripts"                            # folder of V0001.txt etc.
OUTPUT_XLSX = "Video_Registry_Populated.xlsx"

# Minimum hit difference to call a “primary” family confidently
PRIMARY_FAMILY_MARGIN = 1

# ATT&CK-lite keyword mapping (expand as you like)
TTP_KEYWORDS = {
    "InitialAccess": [
        r"\bphish\w*\b", r"\bphishing\b", r"\bmalspam\b", r"\brdp\b", r"\bexposed rdp\b",
        r"\bdrive-by\b", r"\bexploit\w*\b", r"\bvulnerability\b", r"\b0-day\b", r"\bcredential\b",
        r"\bpassword spray\b", r"\bbrute force\b",
    ],
    "Execution": [
        r"\bpowershell\b", r"\bcmd\.exe\b", r"\bwscript\b", r"\bcscript\b", r"\brundll32\b",
        r"\bhta\b", r"\bmshta\b", r"\bmacro\b",
    ],
    "Persistence": [
        r"\bscheduled task\b", r"\bregistry run key\b", r"\bstartup folder\b",
        r"\bservice\b", r"\bautostart\b",
    ],
    "PrivilegeEscalation": [
        r"\bprivilege escalation\b", r"\buac\b", r"\btoken\b", r"\bcredential dumping\b", r"\bmimikatz\b"
    ],
    "LateralMovement": [
        r"\bpsexec\b", r"\bwmi\b", r"\brdp\b", r"\bremote service\b", r"\bsmb\b",
        r"\bcobalt strike\b", r"\bbeacon\b", r"\bwinrm\b",
    ],
    "Discovery": [
        r"\bnet\s+view\b", r"\bipconfig\b", r"\bwhoami\b", r"\bad\s+enumeration\b",
        r"\bactive directory\b", r"\bdomain controller\b", r"\bbloodhound\b",
    ],
    "DefenseEvasion": [
        r"\bdisable antivirus\b", r"\bdefender\b", r"\bedr\b", r"\bamsi\b",
        r"\bobfusc\w*\b", r"\btamper\b",
    ],
    "CredentialAccess": [
        r"\bmimikatz\b", r"\bcredential dumping\b", r"\blsass\b", r"\bhashdump\b",
    ],
    "Collection": [
        r"\bcollect\w*\b", r"\bstage\w*\b", r"\barchive\b", r"\b7-zip\b", r"\bwinrar\b",
    ],
    "Exfiltration": [
        r"\bexfil\w*\b", r"\bdata theft\b", r"\bleak site\b", r"\bdouble extortion\b",
        r"\bmeganz\b", r"\brclone\b", r"\bsftp\b", r"\bftp\b", r"\btor\b",
    ],
    "Impact": [
        r"\bencrypt\w*\b", r"\bransom note\b", r"\bshadow copy\b", r"\bvss\b",
        r"\bdelete backups\b", r"\bimpact\b", r"\bextortion\b",
    ],
}

PLATFORM_KEYWORDS = {
    "ESXi": [r"\besxi\b", r"\bvmware\b", r"\bvcenter\b"],
    "Linux": [r"\blinux\b", r"\bubuntu\b", r"\bdebian\b", r"\bcentos\b", r"\bred hat\b", r"\brhel\b"],
    "Windows": [r"\bwindows\b", r"\bactive directory\b", r"\bdomain controller\b", r"\bntfs\b", r"\blsass\b"],
}

DEPTH_SIGNALS = {
    "High": [
        r"\b(ttp|tactics|techniques|procedures)\b", r"\bmitre\b", r"\batt&ck\b",
        r"\bforensic\w*\b", r"\btelemetry\b", r"\bevidence\b", r"\blog(s)?\b",
        r"\bioc(s)?\b", r"\byara\b", r"\bsigma\b", r"\bpcap\b", r"\bmemory dump\b",
        r"\bprocess injection\b", r"\bapi call\b", r"\breverse engineering\b"
    ],
    "Medium": [
        r"\bincident response\b", r"\bkill chain\b", r"\bintrusion\b",
        r"\blateral movement\b", r"\bprivilege escalation\b", r"\bdata exfiltration\b",
    ],
}


# -----------------------------
# HELPERS
# -----------------------------
def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text


def compile_patterns(patterns: List[str]) -> List[re.Pattern]:
    return [re.compile(p, flags=re.IGNORECASE) for p in patterns]


def count_hits(text: str, compiled: List[re.Pattern]) -> int:
    return sum(len(p.findall(text)) for p in compiled)


def detect_platform(text: str) -> str:
    hits = {}
    for plat, pats in PLATFORM_KEYWORDS.items():
        hits[plat] = count_hits(text, compile_patterns(pats))

    chosen = [p for p, h in hits.items() if h > 0]
    if not chosen:
        return "Mixed"  # unknown → treat as Mixed to avoid wrong specificity
    if len(chosen) == 1:
        return chosen[0]
    return "Mixed"


def estimate_depth(text: str) -> str:
    high = count_hits(text, compile_patterns(DEPTH_SIGNALS["High"]))
    med = count_hits(text, compile_patterns(DEPTH_SIGNALS["Medium"]))

    # simple heuristic
    if high >= 6:
        return "High"
    if (high >= 2) or (med >= 4):
        return "Medium"
    return "Low"


def extract_ttps(text: str) -> List[str]:
    ttp_scores = {}
    for ttp, pats in TTP_KEYWORDS.items():
        ttp_scores[ttp] = count_hits(text, compile_patterns(pats))

    # Keep TTPs with at least 2 hits, order by score desc
    chosen = [k for k, v in sorted(ttp_scores.items(), key=lambda x: x[1], reverse=True) if v >= 2]
    return chosen[:6]  # cap to keep field tidy


def load_family_aliases(family_xlsx: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Returns:
      alias_to_family: maps any alias -> canonical family name
      family_to_tier: canonical family -> tier (Core/Secondary/LongTail)
    """
    df = pd.read_excel(family_xlsx, sheet_name=0)
    df = df.fillna("")

    alias_to_family = {}
    family_to_tier = {}

    for _, row in df.iterrows():
        fam = str(row["Ransomware_Family_Name"]).strip()
        tier = str(row["Coverage_Tier (Core/Secondary/LongTail)"]).strip()
        aliases = str(row.get("Alias_Names", "")).strip()

        if not fam:
            continue
        family_to_tier[fam] = tier

        # include canonical itself as alias
        alias_to_family[fam.lower()] = fam

        if aliases and aliases != "—":
            for a in [x.strip() for x in aliases.split(",")]:
                if a:
                    alias_to_family[a.lower()] = fam

    return alias_to_family, family_to_tier


def detect_families(text: str, alias_to_family: Dict[str, str]) -> Counter:
    """
    Counts mentions of families based on alias occurrences.
    Uses word-boundary-ish matching; handles things like "revil" and "lockbit 3.0".
    """
    counts = Counter()
    for alias_lower, canonical in alias_to_family.items():
        # escape regex special chars
        alias_re = re.escape(alias_lower)
        # allow minor spacing variations
        alias_re = alias_re.replace(r"\ ", r"\s+")
        pattern = re.compile(rf"\b{alias_re}\b", flags=re.IGNORECASE)
        hits = len(pattern.findall(text))
        if hits:
            counts[canonical] += hits
    return counts


def choose_primary_secondary(fam_counts: Counter) -> Tuple[str, str]:
    """
    Returns (primary, secondary_csv)
    """
    if not fam_counts:
        return ("", "")

    ranked = fam_counts.most_common()
    primary, primary_hits = ranked[0]

    secondary = []
    for fam, hits in ranked[1:]:
        secondary.append(fam)

    # Optional: if primary not clearly dominant, leave primary blank and treat as multi-family mention
    if len(ranked) > 1:
        second_hits = ranked[1][1]
        if primary_hits - second_hits < PRIMARY_FAMILY_MARGIN:
            # ambiguous
            # choose primary anyway but keep secondary list — you can revise manually
            pass

    return (primary, ", ".join(secondary))


# -----------------------------
# MAIN
# -----------------------------
def main():
    alias_to_family, family_to_tier = load_family_aliases(FAMILY_LIST_XLSX)

    reg = pd.read_excel(VIDEO_REGISTRY_XLSX, sheet_name=0)
    reg = reg.fillna("")

    transcripts_path = Path(TRANSCRIPTS_DIR)
    if not transcripts_path.exists():
        raise FileNotFoundError(f"Transcripts folder not found: {TRANSCRIPTS_DIR}")

    # Iterate rows
    updates = []
    for idx, row in reg.iterrows():
        vid = str(row.get("Video_ID", "")).strip()
        if not vid:
            updates.append({})
            continue

        tfile = transcripts_path / f"{vid}.txt"
        if not tfile.exists():
            # No transcript yet; keep empty
            updates.append({})
            continue

        text = normalize_text(tfile.read_text(encoding="utf-8", errors="ignore"))

        fam_counts = detect_families(text, alias_to_family)
        primary, secondary = choose_primary_secondary(fam_counts)

        tier = family_to_tier.get(primary, "")
        ttps = extract_ttps(text)
        platform = detect_platform(text)
        depth = estimate_depth(text)

        updates.append({
            "Transcript_Available (Yes/No)": "Yes",
            "Primary_Ransomware_Family": primary,
            "Secondary_Families_Mentioned": secondary,
            "Primary_TTPs_Mentioned": ", ".join(ttps),
            "Platform (Windows/Linux/ESXi/Mixed)": platform,
            "Estimated_Technical_Depth (Low/Medium/High)": depth,
            "Family_Coverage_Tag (Core/Secondary/LongTail)": tier,
        })

    # Apply updates
    upd_df = pd.DataFrame(updates)
    for col in upd_df.columns:
        if col in reg.columns:
            reg[col] = reg[col].where(reg[col].astype(str).str.strip() != "", upd_df[col])
        else:
            reg[col] = upd_df[col]

    reg.to_excel(OUTPUT_XLSX, index=False)
    print(f"Saved populated registry to: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
