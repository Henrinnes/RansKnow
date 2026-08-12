"""
RansKnow Phase 5.1 -- length confound check.

Two things, not one, because they answer different questions:

1. Correlation between transcript word count and raw detection counts.
   Answers "is longer-video-equals-more-mentions a real pattern in this
   corpus" (yes, strongly for tactics; weakly for families; not at all
   for tools).

2. Re-score the structured-feature dominant_tactic classifier (the
   strongest Phase 1 result, 0.74-0.81 macro-F1) with length-normalized
   features (mentions per 1000 words) instead of raw counts. Answers
   "is that strong score partly the model exploiting video length as a
   shortcut" -- a *different* question from (1), and the one that
   actually matters for whether the Phase 1 result is trustworthy.

   Note what this does NOT test: re-deriving Dominant_Tactic itself
   (the label) from normalized rates instead of raw counts. That's
   mathematically guaranteed to never change anything -- dividing all
   10 tactic counts for one video by that video's own word count can't
   change which one is largest (argmax is scale-invariant per row).
   Checked this directly: 0/1034 labels changed under that transform.
   The confound can only bite as a *feature* for a downstream model,
   not in how the label itself was constructed.

Usage:
    python3 Scripts/phase5_length_confound.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rk_pipeline.data import load_features  # noqa: E402
from rk_pipeline.features import STRUCTURED_EXCLUDE  # noqa: E402
from rk_pipeline.models import _logreg_pipeline  # noqa: E402
from rk_pipeline.tasks import register_tasks, TASKS  # noqa: E402

OUT = ROOT / "outputs" / "phase5_length_confound.json"


def word_counts_for(df: pd.DataFrame) -> np.ndarray:
    return np.array([
        len(Path(p).read_text(encoding="utf-8", errors="ignore").split())
        for p in df["Transcript_Path"]
    ])


def correlation_check(df: pd.DataFrame, word_count: np.ndarray) -> dict:
    results = {}
    for col in ["Family_Count", "Tactic_Total_Mentions", "Tool_Total_Mentions"]:
        r, p = pearsonr(word_count, df[col])
        rho, ps = spearmanr(word_count, df[col])
        results[col] = {"pearson_r": r, "pearson_p": p, "spearman_rho": rho, "spearman_p": ps}
    return results


def cv_macro_f1(X: np.ndarray, y: np.ndarray, seed: int = 0, n_splits: int = 5) -> float:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train_idx, test_idx in skf.split(X, y):
            pipe = _logreg_pipeline()
            pipe.fit(X[train_idx], y[train_idx])
            pred = pipe.predict(X[test_idx])
            labels = np.unique(y[test_idx])
            scores.append(f1_score(y[test_idx], pred, labels=labels, average="macro", zero_division=0))
    return float(np.mean(scores))


def shortcut_check(df: pd.DataFrame, word_count: np.ndarray) -> dict:
    register_tasks(df)
    task = TASKS.get("dominant_tactic")
    y = task.build_y(df)

    cols = [c for c in df.columns if c not in STRUCTURED_EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]
    X_raw = df[cols].fillna(0).to_numpy(dtype=float)

    X_norm = X_raw.copy()
    for i, c in enumerate(cols):
        if c != "Year":
            X_norm[:, i] = X_norm[:, i] / np.maximum(word_count, 1) * 1000

    return {
        "raw_macro_f1": cv_macro_f1(X_raw, y),
        "length_normalized_macro_f1": cv_macro_f1(X_norm, y),
    }


def main():
    df = load_features()
    word_count = word_counts_for(df)

    print("=== 1. Correlation: word count vs raw detection counts ===")
    corr = correlation_check(df, word_count)
    for col, stats in corr.items():
        print(f"  {col:24} pearson r={stats['pearson_r']:.3f} (p={stats['pearson_p']:.1e})  "
              f"spearman rho={stats['spearman_rho']:.3f}")

    print("\n=== 2. Shortcut check: does dominant_tactic's structured-feature score "
          "survive length normalization? ===")
    shortcut = shortcut_check(df, word_count)
    print(f"  raw counts:            {shortcut['raw_macro_f1']:.3f} macro-F1")
    print(f"  length-normalized:     {shortcut['length_normalized_macro_f1']:.3f} macro-F1")
    delta = shortcut["raw_macro_f1"] - shortcut["length_normalized_macro_f1"]
    print(f"  delta:                 {delta:+.3f} "
          f"({'small -- not a length-exploitation artifact' if abs(delta) < 0.03 else 'notable -- investigate further'})")

    import json
    OUT.write_text(json.dumps({"correlation": corr, "shortcut_check": shortcut}, indent=2))
    print(f"\nWrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
