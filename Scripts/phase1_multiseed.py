"""
RansKnow Phase 1 -- multi-seed variance for the best (feature, model)
combo per task (Table 3 / results.tex).

Checklist item 7 (statistical significance) originally answered
answerNo{}: every macro-F1 in the paper was a single-seed point
estimate. This closes that gap for the number that actually appears in
the results table -- not the full 180-combination grid, which would be
expensive to repeat 5x for marginal benefit given the paper only reports
the best cell per task.

Only stratified_random varies with a seed: channel_grouped (GroupKFold)
and temporal (fixed pre/post-2024 cutoff) are both deterministic single
partitions with no random component (rk_pipeline/splits.py), so
seed-to-seed variance is not a meaningful question for those two
protocols and isn't computed for them. Model random_state (RF/GBM) is
held fixed at 0 throughout (rk_pipeline/models.py, unchanged) -- this
specifically isolates variance from which videos land in train vs. test
under 5 different stratified partitions, not full stochastic-training
variance, and is reported as such rather than overclaimed.

Usage:
    python3 Scripts/phase1_multiseed.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rk_pipeline.data import load_features  # noqa: E402
from rk_pipeline.features import FEATURES  # noqa: E402
from rk_pipeline.models import MODELS  # noqa: E402
from rk_pipeline.runner import _run_one_fold  # noqa: E402
from rk_pipeline.splits import _stratified_random  # noqa: E402
from rk_pipeline.tasks import register_tasks, TASKS  # noqa: E402

OUT = ROOT / "outputs" / "experiments" / "phase1_multiseed.json"
SEEDS = [0, 1, 2, 3, 4]

# Best (feature, model) per task from Table 3. family's true best combo
# (embedding/logreg) needs the dedicated .venv-embeddings interpreter,
# same documented fallback used everywhere else in this project when
# run from the default environment.
BEST_COMBO = {
    "relevance": ("tfidf", "gbm"),
    "dominant_tactic": ("structured", "gbm"),
    "platform": ("tfidf", "gbm"),
    "tool": ("tfidf", "logreg"),
    "family": ("tfidf", "logreg"),
}


def run_seed(task, feature_name, model_name, df, seed):
    y_full = task.build_y(df)
    model_spec = MODELS.get(model_name)
    fold_scores = []
    for train_idx, test_idx in _stratified_random(df, y_full, n_splits=5, seed=seed):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        y_train, y_test = y_full[train_idx], y_full[test_idx]
        s = _run_one_fold(task, model_spec, feature_name, train_df, test_df, y_train, y_test)
        fold_scores.append(s["macro_f1"])
    return float(np.mean(fold_scores))


def main():
    df = load_features()
    register_tasks(df)

    results = {}
    for task_name, (feature_name, model_name) in BEST_COMBO.items():
        task = TASKS.get(task_name)
        print(f"\n=== {task_name} ({feature_name}/{model_name}) ===")
        seed_scores = []
        for seed in SEEDS:
            score = run_seed(task, feature_name, model_name, df, seed)
            seed_scores.append(score)
            print(f"  seed={seed}: macro_f1={score:.4f}")
        mean, std = float(np.mean(seed_scores)), float(np.std(seed_scores, ddof=1))
        print(f"  mean={mean:.4f}  std={std:.4f}")
        results[task_name] = {
            "feature": feature_name, "model": model_name,
            "seeds": SEEDS, "macro_f1_per_seed": seed_scores,
            "mean": mean, "std": std,
        }
        note = "tfidf/logreg fallback -- true best combo needs .venv-embeddings" if task_name == "family" else None
        if note:
            results[task_name]["note"] = note

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
