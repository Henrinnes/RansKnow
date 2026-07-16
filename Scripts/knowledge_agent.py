"""
RansKnow Knowledge Agent — feature extraction pipeline.

Extracts structured features from every transcript with real content:
  - Ransomware family mentions (alias list)
  - MITRE ATT&CK tactic mention counts (10 tactics)
  - Offensive tool mention counts (8 tools)
  - Platform signals (Windows / Linux / ESXi)
  - Metadata: Year, DurationSeconds, Transcript_Provider

Usage:
    python3 Scripts/knowledge_agent.py
    python3 Scripts/knowledge_agent.py --out outputs/Knowledge_Agent_Features_718.csv
    python3 Scripts/knowledge_agent.py --only-missing  # append new videos to existing CSV
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ROOT        = Path(__file__).resolve().parent.parent
TRANSCRIPTS = ROOT / "transcripts"
FAMILY_LIST = ROOT / "rubrics" / "Ransomware_Family_Coverage_List.xlsx"
DEFAULT_OUT = ROOT / "outputs" / "Knowledge_Agent_Features_718.csv"

# ── Tactic patterns ───────────────────────────────────────────────────────────

TACTICS: Dict[str, List[str]] = {
    "Initial_Access": [
        r"\bphish",             r"\bmalspam\b",         r"\bvishing\b",
        r"\brdp\b",             r"\bbrute.?force\b",    r"\bpassword.?spray\b",
        r"\bexploit",           r"\bvulnerab",          r"\b0.day\b",
        r"\bdrive.?by\b",       r"\bcredential.?stuffing\b",
        r"\bspear.?phish",      r"\bwatering.?hole\b",
    ],
    "Execution": [
        r"\bpowershell\b",      r"\bcmd\.exe\b",        r"\bwscript\b",
        r"\bcscript\b",         r"\bhta\b",             r"\bmshta\b",
        r"\bmacro\b",           r"\bvba\b",             r"\bexecution\b",
        r"\bscriptlet\b",       r"\bregsvr32\b",        r"\brundll32\b",
    ],
    "Persistence": [
        r"\bscheduled.?task\b", r"\brun.?key\b",        r"\bstartup\b",
        r"\bautorun\b",         r"\bservice.?install",  r"\bpersistence\b",
        r"\bboot.?sector\b",    r"\blogon.?script\b",
    ],
    "Privilege_Escalation": [
        r"\bprivilege.?escalat", r"\buac\b",            r"\btoken.?impersonat",
        r"\bbypass.?uac\b",     r"\blocal.?admin\b",    r"\bsystem.?privilege\b",
        r"\bprivilege\b",
    ],
    "Credential_Access": [
        r"\bmimikatz\b",        r"\bmimi.?katz\b",      r"\blsass\b",
        r"\bcredential.?dump",  r"\bntlm\b",            r"\bkerberoast",
        r"\bpass.?the.?hash\b", r"\bhashcat\b",         r"\bcredential.?harvest",
        r"\bsecrets.?dump\b",   r"\bntds\.dit\b",
    ],
    "Lateral_Movement": [
        r"\bpsexec\b",          r"\bps.?exec\b",        r"\bwmi\b",
        r"\bwinrm\b",           r"\bsmbexec\b",         r"\blateral.?movement\b",
        r"\bremote.?exec",      r"\bpass.?the.?hash\b", r"\bimpacket\b",
    ],
    "Discovery": [
        r"\bnetwork.?scan\b",   r"\bad.?enumerat",      r"\bbloodhound\b",
        r"\bnmap\b",            r"\bping.?sweep\b",     r"\breconnaissance\b",
        r"\bdiscovery\b",       r"\benumerat",          r"\bsharpview\b",
        r"\badrecon\b",
    ],
    "Command_and_Control": [
        r"\bcobalt.?strike\b",  r"\bbeacon\b",          r"\breverse.?shell\b",
        r"\bc2\b",              r"\bc&c\b",             r"\bcommand.?and.?control\b",
        r"\brat\b",             r"\bremote.?access.?tool\b",
        r"\bmetasploit\b",      r"\bsliver\b",          r"\bbrute.?ratel\b",
    ],
    "Exfiltration": [
        r"\bexfil",             r"\brclone\b",          r"\bmega\.?nz\b",
        r"\bmeganz\b",          r"\bdouble.?extortion\b", r"\bleak.?site\b",
        r"\bdata.?theft\b",     r"\bwinscp\b",          r"\bftp\b",
        r"\bsftp\b",
    ],
    "Impact": [
        r"\bencrypt",           r"\bransom.?note\b",    r"\bshadow.?cop",
        r"\bvss\b",             r"\bdata.?destruct",    r"\bwiper\b",
        r"\blocked.?file\b",    r"\bdecrypt",           r"\bransom\b",
        r"\bfile.?encrypt",     r"\bdestroj",
    ],
}

# ── Tool patterns ─────────────────────────────────────────────────────────────

TOOLS: Dict[str, List[str]] = {
    "Cobalt_Strike":  [r"\bcobalt.?strike\b",   r"\bcs.?beacon\b"],
    "Mimikatz":       [r"\bmimikatz\b",          r"\bmimi.?katz\b",      r"\bmimikats\b"],
    "PsExec":         [r"\bpsexec\b",            r"\bps.?exec\b",        r"\bsysinternals.{0,20}exec\b"],
    "Rclone":         [r"\brclone\b"],
    "MegaNZ":         [r"\bmega\.?nz\b",         r"\bmeganz\b",          r"\bmega\s+upload\b"],
    "AnyDesk":        [r"\banydesk\b",           r"\bany.?desk\b"],
    "TeamViewer":     [r"\bteamviewer\b",        r"\bteam.?viewer\b"],
    "BloodHound":     [r"\bbloodhound\b",        r"\bblood.?hound\b",    r"\bsharpview\b"],
}

# ── Platform patterns ─────────────────────────────────────────────────────────

PLATFORMS: Dict[str, List[str]] = {
    "Windows": [
        r"\bwindows\b",         r"\bactive.?directory\b", r"\bdomain.?controller\b",
        r"\blsass\b",           r"\bntfs\b",              r"\bwindows.?server\b",
        r"\bsystem32\b",        r"\bregistry\b",
    ],
    "Linux": [
        r"\blinux\b",           r"\bubuntu\b",            r"\bdebian\b",
        r"\bcentos\b",          r"\brhel\b",              r"\bred.?hat\b",
        r"\bbash\b",            r"\bcron\b",              r"\bext4\b",
    ],
    "ESXi": [
        r"\besxi\b",            r"\bvmware\b",            r"\bvcenter\b",
        r"\bhypervisor\b",      r"\bvirtual.?machine\b",  r"\bvm\b",
        r"\bvmdk\b",
    ],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def _count(text: str, patterns: List[str]) -> int:
    return sum(len(re.findall(p, text, flags=re.IGNORECASE)) for p in patterns)


def _load_aliases(path: Path) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    if not path.exists():
        print(f"  [WARN] Family alias file not found: {path}")
        return alias_map
    df = pd.read_excel(path).fillna("")
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


def _find_families(text: str, alias_map: Dict[str, str]) -> List[str]:
    found = set()
    for alias, canon in alias_map.items():
        pat = re.escape(alias).replace(r"\ ", r"\s+")
        if re.search(rf"\b{pat}\b", text, flags=re.IGNORECASE):
            found.add(canon)
    return sorted(found)


def extract(vid: str, txt_path: Path, meta: dict, alias_map: Dict[str, str]) -> dict:
    text = _norm(txt_path.read_text(encoding="utf-8", errors="ignore"))

    families     = _find_families(text, alias_map)
    tactic_cnts  = {t: _count(text, pats) for t, pats in TACTICS.items()}
    tool_cnts    = {t: _count(text, pats) for t, pats in TOOLS.items()}
    plat_cnts    = {p: _count(text, pats) for p, pats in PLATFORMS.items()}

    dom_tactic   = max(tactic_cnts, key=tactic_cnts.get) if any(tactic_cnts.values()) else None
    plat_signal  = max(plat_cnts,   key=plat_cnts.get)   if any(plat_cnts.values())   else None

    yt_id = meta.get("YouTube_Video_ID", "")

    return {
        "Video_ID":            vid,
        "Channel_ID":          meta.get("Channel_ID", ""),
        "Channel_Name":        meta.get("Channel_Name", ""),
        "Video_Title":         meta.get("Video_Title", ""),
        "YouTube_URL":         f"https://www.youtube.com/watch?v={yt_id}" if yt_id else "",
        "Transcript_Path":     str(txt_path),
        "Year":                meta.get("Year"),
        "DurationSeconds":     meta.get("DurationSeconds"),
        "Transcript_Provider": meta.get("Transcript_Provider") or "youtube_api",

        "Family_Count": len(families),
        "Family_List":  ", ".join(families) if families else None,

        "Tool_Total_Mentions": sum(tool_cnts.values()),
        "Tool_List":           ", ".join(k for k, v in tool_cnts.items() if v > 0) or None,

        "Tactic_Total_Mentions": sum(tactic_cnts.values()),
        "Dominant_Tactic":       dom_tactic,
        "Platform_Signal":       plat_signal,

        **{f"Tactic_{k}": v for k, v in tactic_cnts.items()},
        **{f"Tool_{k}":   v for k, v in tool_cnts.items()},
        **{f"Platform_{k}": v for k, v in plat_cnts.items()},
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run(only_missing: bool, out_path: Path) -> None:
    alias_map = _load_aliases(FAMILY_LIST)
    print(f"Family aliases loaded: {len(alias_map)}")

    existing_ids: set = set()
    if only_missing and out_path.exists():
        existing_ids = set(pd.read_csv(out_path)["Video_ID"].astype(str))
        print(f"Existing CSV: {len(existing_ids)} rows — will skip those")

    todo: List[Tuple[str, Path, dict]] = []
    skipped = 0

    for meta_file in sorted(TRANSCRIPTS.rglob("*.meta.json")):
        vid = meta_file.name.replace(".meta.json", "")
        txt_path = meta_file.with_name(vid + ".txt")
        if not txt_path.exists():
            continue
        if txt_path.read_text(encoding="utf-8", errors="ignore").strip().startswith("[NO TRANSCRIPT AVAILABLE]"):
            continue
        if only_missing and vid in existing_ids:
            skipped += 1
            continue
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        todo.append((vid, txt_path, meta))

    print(f"Videos to process: {len(todo)}  |  Skipped (already in CSV): {skipped}\n")

    records = []
    for i, (vid, txt_path, meta) in enumerate(todo, 1):
        records.append(extract(vid, txt_path, meta, alias_map))
        if i % 100 == 0 or i == len(todo):
            print(f"  [{i}/{len(todo)}] {vid}", flush=True)

    new_df = pd.DataFrame(records)

    if only_missing and out_path.exists():
        existing_df = pd.read_csv(out_path)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.sort_values("Video_ID").reset_index(drop=True)
        combined.to_csv(out_path, index=False)
        combined.to_excel(out_path.with_suffix(".xlsx"), index=False)
        print(f"\nAppended {len(new_df)} rows → total {len(combined)} in {out_path.name}")
        _print_summary(combined)
    else:
        new_df = new_df.sort_values("Video_ID").reset_index(drop=True)
        new_df.to_csv(out_path, index=False)
        new_df.to_excel(out_path.with_suffix(".xlsx"), index=False)
        print(f"\nSaved {len(new_df)} rows → {out_path.name}")
        _print_summary(new_df)


def _print_summary(df: pd.DataFrame) -> None:
    total = len(df)
    print(f"\n{'─'*50}")
    print(f"Total videos:          {total}")
    print(f"With family detected:  {(df['Family_Count'] > 0).sum()} ({(df['Family_Count'] > 0).mean():.0%})")
    print(f"With tool detected:    {(df['Tool_Total_Mentions'] > 0).sum()}")

    tool_cols = [c for c in df.columns if c.startswith("Tool_") and c not in ("Tool_Total_Mentions", "Tool_List")]
    print("\nTool mention counts:")
    print(df[tool_cols].sum().sort_values(ascending=False).to_string())

    plat_cols = [c for c in df.columns if c.startswith("Platform_") and c != "Platform_Signal"]
    print("\nPlatform mention counts:")
    print(df[plat_cols].sum().sort_values(ascending=False).to_string())

    tactic_cols = [c for c in df.columns if c.startswith("Tactic_") and c not in ("Tactic_Total_Mentions", "Dominant_Tactic")]
    print("\nTop tactic mentions:")
    print(df[tactic_cols].sum().sort_values(ascending=False).head(5).to_string())

    if "Family_List" in df.columns:
        print("\nTop ransomware families:")
        fams = df["Family_List"].dropna().str.split(", ").explode()
        fams = fams[fams != ""]
        print(fams.value_counts().head(10).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RansKnow Knowledge Agent — feature extraction")
    parser.add_argument("--only-missing", action="store_true",
                        help="Append only videos not already in the output CSV")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help=f"Output CSV path (default: {DEFAULT_OUT.name})")
    args = parser.parse_args()
    run(only_missing=args.only_missing, out_path=Path(args.out))
