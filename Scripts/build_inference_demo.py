"""
RansKnow inference demo -- build script.

Assembles outputs/Inference_Demo.html from:
  - Scripts/inference_demo_template.html  (page structure/CSS/JS shell)
  - Scripts/assets/ka_engine.js           (TF-IDF + rule-based JS engine,
                                            numerically verified against
                                            the Python pipeline)
  - Scripts/assets/demo_examples.json     (sample transcripts)
  - Scripts/assets/newsreader-*.b64       (fonts)
  - outputs/rulebased_patterns.json       (from export_rulebased_patterns.py)
  - outputs/demo_models.json              (from export_demo_models.py)

Re-run export_rulebased_patterns.py / export_demo_models.py first if
knowledge_agent.py's patterns or the trained models have changed, then
this script to rebuild the page.

Usage:
    python3 Scripts/build_inference_demo.py
"""

from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
SCRIPTS   = Path(__file__).resolve().parent
ASSETS    = SCRIPTS / "assets"
TEMPLATE  = SCRIPTS / "inference_demo_template.html"
OUT       = ROOT / "outputs" / "Inference_Demo.html"


def safe(s: str) -> str:
    """Prevent premature </script> termination if embedded data/text
    happens to contain that literal substring."""
    return s.replace("</", "<\\/")


def main():
    tmpl = TEMPLATE.read_text(encoding="utf-8")

    normal = (ASSETS / "newsreader-normal.b64").read_text().strip()
    italic = (ASSETS / "newsreader-italic.b64").read_text().strip()

    rule_json = (ROOT / "outputs" / "rulebased_patterns.json").read_text(encoding="utf-8")
    model_json = (ROOT / "outputs" / "demo_models.json").read_text(encoding="utf-8")
    examples_json = (ASSETS / "demo_examples.json").read_text(encoding="utf-8")

    ka_engine = (ASSETS / "ka_engine.js").read_text(encoding="utf-8")
    ka_engine = ka_engine.split("if (typeof module")[0]  # strip Node-only export block

    out = (tmpl
           .replace("__FONT_NORMAL__", normal)
           .replace("__FONT_ITALIC__", italic)
           .replace("__RULE_DATA_JSON__", safe(rule_json))
           .replace("__MODEL_DATA_JSON__", safe(model_json))
           .replace("__EXAMPLES_JSON__", safe(examples_json))
           .replace("__KA_ENGINE_JS__", ka_engine))

    OUT.write_text(out, encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
