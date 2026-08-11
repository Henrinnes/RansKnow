"""
Export knowledge_agent.py's regex patterns/alias map as JSON, for the
JS reimplementation used in the inference demo. Single source of truth
stays in knowledge_agent.py -- this script just serializes it so the
demo can't silently drift from what the real pipeline does.

Usage:
    python3 Scripts/export_rulebased_patterns.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import knowledge_agent as ka  # noqa: E402

OUT = ROOT.parent / "outputs" / "rulebased_patterns.json"


def main():
    alias_map = ka._load_aliases(ka.FAMILY_LIST)

    bundle = {
        "alias_map": alias_map,                          # alias (lowercase) -> canonical family
        "ambiguous_aliases": sorted(ka.AMBIGUOUS_ALIASES),
        "context_terms": ka.CONTEXT_TERMS,
        "context_window_chars": ka.CONTEXT_WINDOW_CHARS,
        "tactics": ka.TACTICS,     # tactic -> [regex patterns]
        "tools": ka.TOOLS,         # tool -> [regex patterns]
        "platforms": ka.PLATFORMS, # platform -> [regex patterns]
    }

    OUT.write_text(json.dumps(bundle, indent=2))
    print(f"Wrote {OUT.relative_to(ROOT.parent)}")
    print(f"  aliases: {len(alias_map)}, ambiguous: {len(ka.AMBIGUOUS_ALIASES)}")
    print(f"  tactics: {len(ka.TACTICS)}, tools: {len(ka.TOOLS)}, platforms: {len(ka.PLATFORMS)}")


if __name__ == "__main__":
    main()
