"""
RansKnow Phase 5.1 -- channel-identity leakage check.

Phase 1's channel_grouped split already showed a gap vs stratified_random
CV (e.g. family|tfidf|logreg: 0.075 stratified vs 0.048 channel_grouped)
-- the plan's whole reason for including that split. This asks a follow-up
question: how much of that gap is explained by the model literally
reading the channel's own name in the transcript (46% of a 200-video
sample mention their own channel name), versus deeper stylistic leakage
(recurring phrasing, topic focus, presenter habits) that masking a name
string can't remove?

Masks occurrences of each video's own channel name (case-insensitive,
plus its first word alone for multi-word names, e.g. "CrowdStrike" from
"CrowdStrike Falcon") with a neutral placeholder, rebuilds TF-IDF on the
masked corpus, and re-scores logreg under both split protocols.

Usage:
    python3 Scripts/phase5_channel_leakage.py
"""

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rk_pipeline.data import load_features, load_transcript  # noqa: E402
from rk_pipeline.models import MODELS  # noqa: E402
from rk_pipeline.tasks import register_tasks, TaskSpec, TASKS  # noqa: E402

OUT = ROOT / "outputs" / "phase5_channel_leakage.json"


def mask_channel_name(text: str, channel_name: str) -> str:
    name = channel_name.strip()
    variants = {name}
    words = name.split()
    if len(words) > 1 and len(words[0]) > 3:
        variants.add(words[0])
    masked = text
    for v in variants:
        pat = re.compile(r"\b" + re.escape(v) + r"\b", re.IGNORECASE)
        masked = pat.sub("[CHANNEL]", masked)
    return masked


def build_tfidf(texts, max_features=3000):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2)
    return vec.fit_transform(texts)


def cv_macro_f1_stratified(X, y, task: TaskSpec, seed=0, n_splits=5):
    def _stratify_key(y):
        return np.clip(y.sum(axis=1), 0, 2) if y.ndim == 2 else y
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return _run_folds(skf.split(X, _stratify_key(y)), X, y, task)


def cv_macro_f1_grouped(X, y, groups, task: TaskSpec, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    return _run_folds(gkf.split(X, groups=groups), X, y, task)


def _run_folds(fold_iter, X, y, task: TaskSpec):
    scores = []
    logreg_spec = MODELS.get("logreg")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train_idx, test_idx in fold_iter:
            pipe = logreg_spec.build(task.label_type, task)
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            if y.ndim == 2:
                support = y_test.sum(axis=0)
                present = np.where(support > 0)[0]
                if len(present) == 0:
                    continue
                scores.append(f1_score(y_test[:, present], pred[:, present], average="macro", zero_division=0))
            else:
                labels = np.unique(y_test)
                scores.append(f1_score(y_test, pred, labels=labels, average="macro", zero_division=0))
    return float(np.mean(scores))


def main():
    df = load_features()
    register_tasks(df)

    print("Masking channel names in all transcripts...")
    raw_texts = [load_transcript(p) for p in df["Transcript_Path"]]
    masked_texts = [
        mask_channel_name(t, name) for t, name in zip(raw_texts, df["Channel_Name"])
    ]
    n_masked = sum(1 for r, m in zip(raw_texts, masked_texts) if r != m)
    print(f"  {n_masked}/{len(df)} transcripts had at least one mask applied")

    results = {}
    for task_name in ["family", "dominant_tactic"]:
        task = TASKS.get(task_name)
        y = task.build_y(df)

        X_raw = build_tfidf(raw_texts)
        X_masked = build_tfidf(masked_texts)

        strat_raw = cv_macro_f1_stratified(X_raw, y, task)
        strat_masked = cv_macro_f1_stratified(X_masked, y, task)
        group_raw = cv_macro_f1_grouped(X_raw, y, df["Channel_ID"].to_numpy(), task)
        group_masked = cv_macro_f1_grouped(X_masked, y, df["Channel_ID"].to_numpy(), task)

        gap_raw = strat_raw - group_raw
        gap_masked = strat_masked - group_masked

        results[task_name] = {
            "stratified_raw": strat_raw, "stratified_masked": strat_masked,
            "channel_grouped_raw": group_raw, "channel_grouped_masked": group_masked,
            "gap_raw": gap_raw, "gap_masked": gap_masked,
        }

        print(f"\n=== {task_name} ===")
        print(f"  stratified_random:  raw={strat_raw:.3f}  masked={strat_masked:.3f}")
        print(f"  channel_grouped:    raw={group_raw:.3f}  masked={group_masked:.3f}")
        print(f"  gap (strat - grouped): raw={gap_raw:.3f}  masked={gap_masked:.3f}  "
              f"({'gap shrank -- some leakage was literal channel naming' if gap_masked < gap_raw - 0.01 else 'gap persists -- leakage is deeper than channel naming'})")

    OUT.write_text(json.dumps({"n_masked": n_masked, "n_total": len(df), "tasks": results}, indent=2))
    print(f"\nWrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
