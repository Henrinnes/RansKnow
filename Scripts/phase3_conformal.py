"""
RansKnow Phase 3 -- conformal prediction, against weak labels.

Scope note (paper Section 4.3 / sec:gold-scope, same caveat as Phase 2):
split conformal prediction's distribution-free coverage guarantee is a
statement about exchangeability with the calibration set, not about
label correctness. It holds formally even when calibrated against weak
labels, because the guarantee is "the true *weak* label falls in the
predicted set (1-alpha) of the time," not "the true real-world label
does." Reported as such throughout.

Method: split conformal with the standard LAC (least-ambiguous-set)
nonconformity score, s(x, y) = 1 - p_model(y | x), at a 90% target
coverage (alpha=0.10), with the finite-sample-corrected calibration
quantile ceil((n_calib+1)*(1-alpha)) / n_calib.

- dominant_tactic, platform: multiclass LAC conformal directly on the
  task's full class vocabulary.
- family, tool: no native multiclass structure to conformalize (they're
  independent per-class binary classifiers, see rk_pipeline/models.py),
  so each class gets its own binary conformal calibration and its own
  coverage/set-size, averaged for a task-level summary.

5 repeated random 60/20/20 (train/calib/test) stratified splits, results
averaged across repeats -- a single arbitrary calibration split is not
a reliable enough number to report on its own at this sample size.

Conditional-coverage stress test: pools test-fold predictions across
the 5 repeats and reports coverage restricted to two slices where the
corpus's own composition might break the marginal guarantee --
post-2025 videos (Year>=2026, the newest slice, further out than the
temporal split's 2024 boundary used elsewhere) and conference-channel
videos (Channel_Type=='Conference'). The gold-eval slice from the
original design is no longer available (Section sec:gold-scope) -- these
two are what the corpus itself offers as out-of-calibration-distribution
subgroups. Coverage collapse on either is itself the result, not a bug
to fix, and feeds into Phase 4's use of this same nonconformity score
as one of four OOD-detection signals.

Usage:
    python3 Scripts/phase3_conformal.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rk_pipeline.data import load_features  # noqa: E402
from rk_pipeline.features import FEATURES  # noqa: E402
from rk_pipeline.models import MODELS  # noqa: E402
from rk_pipeline.tasks import register_tasks, TASKS  # noqa: E402

OUT = ROOT / "outputs" / "phase3_conformal.json"
ALPHA = 0.10  # 90% target coverage
N_REPEATS = 5

# Same best (feature, model) per task as Phase 2. family's true best
# combo (embedding/logreg) needs the dedicated .venv-embeddings
# interpreter and isn't importable here -- tfidf/logreg used as a
# documented fallback, same choice Phase 2 made for its RF-entropy
# signal on this task.
COMBO = {
    "dominant_tactic": ("structured", "gbm"),
    "platform": ("tfidf", "gbm"),
    "family": ("tfidf", "logreg"),
    "tool": ("tfidf", "logreg"),
}


def _stratify_key(y):
    return np.clip(y.sum(axis=1), 0, 2) if y.ndim == 2 else y


def _safe_stratify_key(key, min_count=4):
    """StratifiedShuffleSplit needs >=2 members per class in the SMALLER
    of the two resulting splits; dominant_tactic has classes as small as
    4 total occurrences (Credential_Access), which a second 50/50 split
    of an already-halved subset can push below that floor. Classes below
    min_count are pooled into a shared '_RARE_' bucket for stratification
    purposes only -- this keeps the split balanced on the classes that
    actually have enough members to balance, rather than crashing or
    (the alternative) silently dropping stratification for every class."""
    key = np.asarray(key)
    counts = pd.Series(key).value_counts()
    rare = set(counts[counts < min_count].index)
    if not rare:
        return key
    return np.array(["_RARE_" if k in rare else k for k in key], dtype=object)


def _lac_qhat(calib_scores, alpha):
    n = len(calib_scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(calib_scores, level, method="higher"))


def _multiclass_conformal_repeat(task, feature_name, model_name, df, seed):
    y_full = task.build_y(df)
    idx = np.arange(len(df))
    key = _stratify_key(y_full)

    train_idx, rest_idx = train_test_split(idx, test_size=0.4, random_state=seed,
                                            stratify=_safe_stratify_key(key))
    rest_key = _safe_stratify_key(key[rest_idx])
    calib_idx, test_idx = train_test_split(rest_idx, test_size=0.5, random_state=seed, stratify=rest_key)

    train_df, calib_df, test_df = df.iloc[train_idx].reset_index(drop=True), \
        df.iloc[calib_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
    y_train = y_full[train_idx]

    builder = FEATURES.get(feature_name)()
    X_train = builder.fit_transform(train_df)
    X_calib = builder.transform(calib_df)
    X_test = builder.transform(test_df)

    est = MODELS.get(model_name).build(task.label_type, task)
    est.fit(X_train, y_train)

    classes = list(est.steps[-1][1].classes_) if hasattr(est, "steps") else list(est.classes_)

    def _score(proba_row, true_label):
        p_true = proba_row[classes.index(true_label)] if true_label in classes else 0.0
        return 1 - p_true

    calib_proba = est.predict_proba(X_calib)
    y_calib = y_full[calib_idx]
    calib_scores = np.array([_score(calib_proba[i], y_calib[i]) for i in range(len(y_calib))])
    qhat = _lac_qhat(calib_scores, ALPHA)

    test_proba = est.predict_proba(X_test)
    y_test = y_full[test_idx]
    pred_sets = [
        {classes[j] for j in range(len(classes)) if test_proba[i, j] >= 1 - qhat}
        for i in range(len(test_idx))
    ]
    covered = np.array([y_test[i] in pred_sets[i] for i in range(len(test_idx))])
    set_sizes = np.array([len(s) for s in pred_sets])

    return {
        "qhat": qhat,
        "test_idx": test_idx,
        "covered": covered,
        "set_size": set_sizes,
        "n_test": len(test_idx),
    }


def _multilabel_conformal_repeat(task, feature_name, model_name, df, seed):
    y_full = task.build_y(df)  # (n, n_classes) binary
    idx = np.arange(len(df))
    key = _stratify_key(y_full)

    train_idx, rest_idx = train_test_split(idx, test_size=0.4, random_state=seed,
                                            stratify=_safe_stratify_key(key))
    rest_key = _safe_stratify_key(key[rest_idx])
    calib_idx, test_idx = train_test_split(rest_idx, test_size=0.5, random_state=seed, stratify=rest_key)

    train_df, calib_df, test_df = df.iloc[train_idx].reset_index(drop=True), \
        df.iloc[calib_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
    y_train = y_full[train_idx]

    builder = FEATURES.get(feature_name)()
    X_train = builder.fit_transform(train_df)
    X_calib = builder.transform(calib_df)
    X_test = builder.transform(test_df)

    est = MODELS.get(model_name).build(task.label_type, task)
    est.fit(X_train, y_train)

    y_calib = y_full[calib_idx]
    y_test = y_full[test_idx]
    n_classes = y_full.shape[1]

    per_label_covered = np.zeros((len(test_idx), n_classes), dtype=bool)
    per_label_set_size = np.zeros((len(test_idx), n_classes), dtype=int)
    qhats = np.zeros(n_classes)

    for c in range(n_classes):
        wrapped = est.estimators_[c]
        p_calib_1 = wrapped.predict_proba(X_calib)[:, 1]
        true_c = y_calib[:, c]
        scores = np.where(true_c == 1, 1 - p_calib_1, p_calib_1)  # 1 - p(true class)
        qhat = _lac_qhat(scores, ALPHA)
        qhats[c] = qhat

        p_test_1 = wrapped.predict_proba(X_test)[:, 1]
        include_1 = p_test_1 >= 1 - qhat          # class "1" survives if score(1) = 1-p1 <= qhat
        include_0 = (1 - p_test_1) >= 1 - qhat    # class "0" survives if score(0) = p1 <= qhat
        true_c_test = y_test[:, c]
        covered = np.where(true_c_test == 1, include_1, include_0)
        set_size = include_0.astype(int) + include_1.astype(int)

        per_label_covered[:, c] = covered
        per_label_set_size[:, c] = set_size

    return {
        "qhats": qhats.tolist(),
        "test_idx": test_idx,
        "covered": per_label_covered,   # (n_test, n_classes)
        "set_size": per_label_set_size,
        "n_test": len(test_idx),
        "classes": task.classes,
    }


def _slice_coverage(df, test_idx, covered_flat, slice_mask_fn):
    """covered_flat: 1D boolean array aligned to test_idx (already
    flattened across classes for multilabel, or as-is for multiclass)."""
    rows = df.iloc[test_idx]
    mask = slice_mask_fn(rows).to_numpy()
    if mask.sum() == 0:
        return {"n": 0, "coverage": None}
    return {"n": int(mask.sum()), "coverage": float(covered_flat[mask].mean())}


def run_multiclass_task(task_name, feature_name, model_name, df):
    task = TASKS.get(task_name)
    print(f"\n=== {task_name} ({feature_name}/{model_name}) -- multiclass LAC conformal, "
          f"target coverage {1 - ALPHA:.0%} ===")

    all_qhat, all_cov, all_size = [], [], []
    pooled_test_idx, pooled_covered = [], []

    for r in range(N_REPEATS):
        res = _multiclass_conformal_repeat(task, feature_name, model_name, df, seed=r)
        all_qhat.append(res["qhat"])
        all_cov.append(float(res["covered"].mean()))
        all_size.append(float(res["set_size"].mean()))
        pooled_test_idx.append(res["test_idx"])
        pooled_covered.append(res["covered"])
        print(f"  repeat {r}: qhat={res['qhat']:.3f}  coverage={res['covered'].mean():.3f}  "
              f"avg_set_size={res['set_size'].mean():.2f}/{len(task.classes)}")

    pooled_test_idx = np.concatenate(pooled_test_idx)
    pooled_covered = np.concatenate(pooled_covered)

    post2025 = _slice_coverage(df, pooled_test_idx, pooled_covered, lambda r: r["Year"] >= 2026)
    conference = _slice_coverage(df, pooled_test_idx, pooled_covered, lambda r: r["Channel_Type"] == "Conference")

    print(f"  Marginal coverage (pooled, {len(pooled_covered)} test predictions): {pooled_covered.mean():.3f}")
    print(f"  Slice: Year>=2026        n={post2025['n']:4d}  coverage={post2025['coverage']}")
    print(f"  Slice: Channel=Conference n={conference['n']:4d}  coverage={conference['coverage']}")

    return {
        "feature": feature_name, "model": model_name, "n_classes": len(task.classes),
        "target_coverage": 1 - ALPHA,
        "per_repeat": {"qhat": all_qhat, "coverage": all_cov, "avg_set_size": all_size},
        "marginal_coverage_pooled": float(pooled_covered.mean()),
        "n_pooled_test": int(len(pooled_covered)),
        "slice_post2025": post2025,
        "slice_conference": conference,
    }


def run_multilabel_task(task_name, feature_name, model_name, df):
    task = TASKS.get(task_name)
    print(f"\n=== {task_name} ({feature_name}/{model_name}) -- per-label binary LAC conformal, "
          f"target coverage {1 - ALPHA:.0%} ===")

    per_repeat_cov, per_repeat_size = [], []
    pooled_test_idx, pooled_covered_flat = [], []

    for r in range(N_REPEATS):
        res = _multilabel_conformal_repeat(task, feature_name, model_name, df, seed=r)
        mean_cov = float(res["covered"].mean())
        mean_size = float(res["set_size"].mean())
        per_repeat_cov.append(mean_cov)
        per_repeat_size.append(mean_size)
        # Flatten per-sample: a sample "covered" if ALL its labels' sets covered the truth
        # (conservative, joint notion) -- reported alongside the per-label-pair average.
        sample_covered = res["covered"].all(axis=1)
        pooled_test_idx.append(np.repeat(res["test_idx"], 1))
        pooled_covered_flat.append(sample_covered)
        print(f"  repeat {r}: mean per-label coverage={mean_cov:.3f}  "
              f"mean per-label set_size={mean_size:.2f}/2  "
              f"all-labels-jointly-covered={sample_covered.mean():.3f}")

    pooled_test_idx = np.concatenate(pooled_test_idx)
    pooled_covered_flat = np.concatenate(pooled_covered_flat)

    post2025 = _slice_coverage(df, pooled_test_idx, pooled_covered_flat, lambda r: r["Year"] >= 2026)
    conference = _slice_coverage(df, pooled_test_idx, pooled_covered_flat, lambda r: r["Channel_Type"] == "Conference")

    print(f"  Marginal all-labels-jointly-covered rate (pooled, n={len(pooled_covered_flat)}): "
          f"{pooled_covered_flat.mean():.3f}")
    print(f"  Slice: Year>=2026        n={post2025['n']:4d}  coverage={post2025['coverage']}")
    print(f"  Slice: Channel=Conference n={conference['n']:4d}  coverage={conference['coverage']}")

    return {
        "feature": feature_name, "model": model_name, "n_classes": len(task.classes),
        "target_coverage_per_label": 1 - ALPHA,
        "per_repeat": {"mean_per_label_coverage": per_repeat_cov, "mean_per_label_set_size": per_repeat_size},
        "marginal_all_labels_jointly_covered_pooled": float(pooled_covered_flat.mean()),
        "n_pooled_test": int(len(pooled_covered_flat)),
        "slice_post2025_joint": post2025,
        "slice_conference_joint": conference,
        "note": "family's (feature, model) is a tfidf/logreg fallback -- the task's true "
                "best combo (embedding/logreg, see Phase 1/2) needs the dedicated "
                ".venv-embeddings interpreter, not importable here.",
    }


def main():
    df = load_features()
    register_tasks(df)

    results = {}
    for task_name in ["dominant_tactic", "platform"]:
        feature_name, model_name = COMBO[task_name]
        results[task_name] = run_multiclass_task(task_name, feature_name, model_name, df)

    for task_name in ["family", "tool"]:
        feature_name, model_name = COMBO[task_name]
        results[task_name] = run_multilabel_task(task_name, feature_name, model_name, df)

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
