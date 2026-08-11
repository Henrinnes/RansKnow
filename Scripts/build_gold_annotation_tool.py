"""
RansKnow Phase 0.3 -- build the gold-set annotation tool.

Regenerates outputs/Gold_Annotation_Tool.html: a self-contained,
single-file annotation interface for the 180-video gold evaluation set
(all transcript text embedded, no server/network needed once built).
Two annotators run this independently -- see Scripts/merge_gold_annotations.py
to combine their exports afterward.

Rebuild this whenever Gold_Eval_Set_Annotation_Template.xlsx changes
(e.g. re-sampling with a different seed or size).

Usage:
    python3 Scripts/build_gold_annotation_tool.py
"""

import json
from pathlib import Path

import pandas as pd

ROOT       = Path(__file__).resolve().parent.parent
GOLD_XLSX  = ROOT / "outputs" / "Gold_Eval_Set_Annotation_Template.xlsx"
FAMILY_LIST = ROOT / "rubrics" / "Ransomware_Family_Coverage_List.xlsx"
TEMPLATE   = Path(__file__).resolve().parent / "gold_annotation_tool_template.html"
FONT_DIR   = Path(__file__).resolve().parent / "assets"
OUT        = ROOT / "outputs" / "Gold_Annotation_Tool.html"


def build_data_payload() -> dict:
    gold = pd.read_excel(GOLD_XLSX)
    fam_df = pd.read_excel(FAMILY_LIST)
    known_families = sorted(fam_df["Ransomware_Family_Name"].dropna().unique().tolist())

    records = []
    for _, r in gold.iterrows():
        txt_path = Path(r["Transcript_Path"])
        text = (txt_path.read_text(encoding="utf-8", errors="ignore")
                if txt_path.exists() else "[TRANSCRIPT NOT FOUND]")
        records.append({
            "video_id":          r["Video_ID"],
            "channel_id":        r["Channel_ID"],
            "channel_name":      r["Channel_Name"],
            "channel_type":      r["Channel_Type"],
            "title":             r["Video_Title"],
            "youtube_url":       r["YouTube_URL"],
            "year":              None if pd.isna(r["Year"]) else int(r["Year"]),
            "transcript":        text,
            "ka_family_count":   None if pd.isna(r["KA_Family_Count"]) else int(r["KA_Family_Count"]),
            "ka_family_list":    None if pd.isna(r["KA_Family_List"]) else r["KA_Family_List"],
            "ka_dominant_tactic": None if pd.isna(r["KA_Dominant_Tactic"]) else r["KA_Dominant_Tactic"],
            "ka_platform_signal": None if pd.isna(r["KA_Platform_Signal"]) else r["KA_Platform_Signal"],
        })

    return {"known_families": known_families, "videos": records}


def main():
    payload = build_data_payload()
    data_str = json.dumps(payload)
    # Prevent premature </script> termination if any transcript text
    # happens to contain that literal substring.
    data_str = data_str.replace("</", "<\\/")

    tmpl = TEMPLATE.read_text(encoding="utf-8")
    normal = (FONT_DIR / "newsreader-normal.b64").read_text().strip()
    italic = (FONT_DIR / "newsreader-italic.b64").read_text().strip()

    out = (tmpl
           .replace("__FONT_NORMAL__", normal)
           .replace("__FONT_ITALIC__", italic)
           .replace("__DATA_JSON__", data_str))

    OUT.write_text(out, encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024 / 1024:.2f} MB, "
          f"{len(payload['videos'])} videos)")


if __name__ == "__main__":
    main()
