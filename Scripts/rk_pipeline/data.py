"""Canonical dataset loader for the RansKnow experiment pipeline."""

from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_CSV = ROOT / "outputs" / "Knowledge_Agent_Features_1034.csv"


@lru_cache(maxsize=1)
def load_features() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_CSV)
    return df.reset_index(drop=True)


@lru_cache(maxsize=4096)
def load_transcript(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8", errors="ignore")
