"""
RansKnow Phase 2 -- uncertainty and calibration, against weak labels.

Scope note (see paper Section 4.3 / sec:gold-scope): there is no
independently verified ground truth for this corpus. Everything below
answers a narrower, still-honest question: does a model's predicted
confidence track its *agreement with the Knowledge Agent extraction
pipeline* -- not whether that pipeline is correct. A model that is
well-calibrated against weak labels is a model whose confidence is a
useful signal for flagging likely pipeline disagreements for review; it
says nothing about real-world accuracy on its own.

Two independent uncertainty signals, per task, using each task's
best-scoring (feature, model) combination from the Phase 1 baseline
sweep (outputs/experiments/phase1_*.csv):

1. Predicted-probability calibration -- reliability diagram (10 equal-
   width confidence bins) + Expected Calibration Error (ECE), computed
   from 5-fold stratified out-of-fold predictions (never train==test).
   Multiclass tasks (dominant_tactic, platform, relevance) use the
   top-class probability as "confidence". Multilabel tasks (family,
   tool) are flattened to one confidence/correctness pair per
   (sample, label) with confidence = max(p, 1-p).

   Headline metric: "confident disagreement rate" -- among predictions
   with confidence > 90%, what fraction disagree with the weak label.
   This is the closest available analogue to a "confidently wrong" rate
   without ground truth to be actually wrong against.

2. Random Forest tree-vote entropy -- a free, model-internal uncertainty
   proxy that needs no separate calibration: for each test prediction,
   the normalized entropy of the individual trees' votes (0 = every tree
   agrees, 1 = maximally split). Reported as mean entropy conditional on
   agreeing/disagreeing with the weak label, plus how well entropy alone
   separates the two (AUROC of entropy as a disagreement detector).
   Uses Random Forest specifically (not necessarily each task's best
   model) because tree votes are the whole point of the proxy.

Usage:
    python3 Scripts/phase2_calibration.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rk_pipeline.data import load_features  # noqa: E402
from rk_pipeline.features import FEATURES  # noqa: E402
from rk_pipeline.models import MODELS  # noqa: E402
from rk_pipeline.splits import SPLITS  # noqa: E402
from rk_pipeline.tasks import register_tasks, TASKS  # noqa: E402

OUT = ROOT / "outputs" / "phase2_calibration.json"
N_BINS = 10

# Best (feature, model) per task from the Phase 1 sweep
# (outputs/experiments/phase1_*.csv, stratified_random, rule_based_ka
# excluded as circular). Matches the results table in the paper.
BEST_COMBO = {
    "relevance": ("tfidf", "gbm"),
    "dominant_tactic": ("structured", "gbm"),
    "platform": ("tfidf", "gbm"),
    "tool": ("tfidf", "logreg"),
    "family": ("embedding", "logreg"),
}

# family/embedding requires the dedicated torch venv; running that combo
# from the default interpreter raises ImportError inside FEATURES.get.
# Skipped here (not silently) and reported in the output.
SKIP_COMBOS = {("embedding",)}


def _oof_proba(task, feature_name, model_name, df):
    """5-fold stratified out-of-fold predict_proba, aligned to df's
    original row order. Returns (y_true_per_row, proba_list_per_row,
    classes) where proba_list_per_row[i] is the class-probability vector
    for row i under whichever fold held it out."""
    y_full = task.build_y(df)
    split_spec = SPLITS.get("stratified_random")
    model_spec = MODELS.get(model_name)

    n = len(df)
    out_proba = [None] * n
    classes_ref = None

    for train_idx, test_idx in split_spec.make_splits(df, y_full):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        y_train = y_full[train_idx]

        builder = FEATURES.get(feature_name)()
        X_train = builder.fit_transform(train_df)
        X_test = builder.transform(test_df)

        est = model_spec.build(task.label_type, task)
        est.fit(X_train, y_train)

        if task.label_type == "multilabel":
            # MultiOutputClassifier of per-class SingleClassSafeClassifier
            proba = np.stack(
                [e.predict_proba(X_test)[:, 1] for e in est.estimators_], axis=1
            )  # (n_test, n_classes) -- P(label=1) per class
        else:
            proba = est.predict_proba(X_test)
            fold_classes = est.steps[-1][1].classes_ if hasattr(est, "steps") else est.classes_
            if classes_ref is None:
                classes_ref = list(task.classes)
            # Realign fold's class ordering (may drop absent classes) onto
            # the task's full class list so probability vectors are
            # comparable/concatenable across folds.
            full = np.zeros((proba.shape[0], len(classes_ref)))
            for j, c in enumerate(fold_classes):
                full[:, classes_ref.index(c)] = proba[:, j]
            proba = full

        for local_i, global_i in enumerate(test_idx):
            out_proba[global_i] = proba[local_i]

    return y_full, np.array(out_proba), classes_ref


def _reliability_multiclass(y_true, proba, classes):
    top_idx = proba.argmax(axis=1)
    confidence = proba.max(axis=1)
    pred_class = np.array([classes[i] for i in top_idx])
    correct = (pred_class == y_true)
    return confidence, correct


def _reliability_multilabel(y_true, proba):
    # Flatten to one (confidence, correct) pair per (sample, label).
    p = proba.reshape(-1)
    y = y_true.reshape(-1)
    pred = (p >= 0.5).astype(int)
    confidence = np.maximum(p, 1 - p)
    correct = (pred == y)
    return confidence, correct


def _ece_and_bins(confidence, correct, n_bins=N_BINS):
    edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    ece = 0.0
    n = len(confidence)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lo) & (confidence < hi) if hi < 1 else (confidence >= lo) & (confidence <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            bins.append({"lo": float(lo), "hi": float(hi), "n": 0, "accuracy": None, "avg_confidence": None})
            continue
        acc = float(correct[mask].mean())
        avg_conf = float(confidence[mask].mean())
        ece += (cnt / n) * abs(acc - avg_conf)
        bins.append({"lo": float(lo), "hi": float(hi), "n": cnt, "accuracy": acc, "avg_confidence": avg_conf})
    return float(ece), bins


def _confident_disagreement(confidence, correct, threshold=0.9):
    mask = confidence > threshold
    n_confident = int(mask.sum())
    if n_confident == 0:
        return {"threshold": threshold, "n_confident": 0, "n_confident_wrong": 0,
                "rate_of_all": 0.0, "rate_within_confident": None}
    n_wrong = int((~correct[mask]).sum())
    return {
        "threshold": threshold,
        "n_confident": n_confident,
        "n_confident_wrong": n_wrong,
        "rate_of_all": n_wrong / len(confidence),
        "rate_within_confident": n_wrong / n_confident,
    }


def _rf_tree_vote_entropy(task, feature_name, df):
    """5-fold OOF: for each held-out row, normalized entropy of the
    RandomForest's individual tree votes, plus whether the RF's own
    (majority-vote) prediction agreed with the weak label."""
    y_full = task.build_y(df)
    split_spec = SPLITS.get("stratified_random")
    rf_spec = MODELS.get("random_forest")

    n = len(df)
    out_entropy = np.full(n, np.nan)
    out_correct = np.zeros(n, dtype=bool)

    for train_idx, test_idx in split_spec.make_splits(df, y_full):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        y_train, y_test = y_full[train_idx], y_full[test_idx]

        builder = FEATURES.get(feature_name)()
        X_train = builder.fit_transform(train_df)
        X_test = builder.transform(test_df)

        est = rf_spec.build(task.label_type, task)
        est.fit(X_train, y_train)

        if task.label_type == "multilabel":
            # Per-class binary RF; average normalized tree-vote entropy
            # across labels for each sample, majority-vote correctness
            # likewise averaged (fraction of labels predicted correctly).
            n_test = X_test.shape[0]
            ent_per_label = []
            correct_per_label = []
            for cls_i, wrapped in enumerate(est.estimators_):
                rf = wrapped.estimator_
                y_true_c = y_test[:, cls_i]
                if rf is None:  # fold had a single class for this label
                    ent_per_label.append(np.zeros(n_test))
                    pred_c = np.full(n_test, wrapped._constant)
                else:
                    votes = np.stack([t.predict(X_test) for t in rf.estimators_], axis=1)
                    p1 = votes.mean(axis=1)
                    h = -(p1 * np.log(np.clip(p1, 1e-12, 1)) + (1 - p1) * np.log(np.clip(1 - p1, 1e-12, 1)))
                    ent_per_label.append(h / np.log(2))
                    pred_c = (p1 >= 0.5).astype(int)
                correct_per_label.append(pred_c == y_true_c)
            ent = np.mean(np.stack(ent_per_label, axis=1), axis=1)
            corr = np.mean(np.stack(correct_per_label, axis=1), axis=1) >= 0.5
        else:
            rf = est  # unwrapped plain RandomForestClassifier
            # Individual trees in a fitted RandomForestClassifier are fit on
            # sklearn's internally label-encoded y, so tree.predict() returns
            # integer class indices (0..k-1), not the original string labels
            # -- comparing those directly against y_test's strings silently
            # produces 0% agreement for every sample. Map back through
            # rf.classes_ before comparing.
            raw_votes = np.stack([t.predict(X_test) for t in rf.estimators_], axis=1).astype(int)
            classes_seen = rf.classes_
            votes = classes_seen[raw_votes]
            n_test = votes.shape[0]
            ent = np.zeros(n_test)
            pred = np.empty(n_test, dtype=object) if y_test.dtype == object else np.empty(n_test, dtype=votes.dtype)
            k = len(classes_seen)
            for i in range(n_test):
                vals, counts = np.unique(votes[i], return_counts=True)
                p = counts / counts.sum()
                h = -(p * np.log(np.clip(p, 1e-12, 1))).sum()
                ent[i] = h / np.log(k) if k > 1 else 0.0
                pred[i] = vals[counts.argmax()]
            corr = (pred == y_test)

        for local_i, global_i in enumerate(test_idx):
            out_entropy[global_i] = ent[local_i]
            out_correct[global_i] = corr[local_i]

    return out_entropy, out_correct


def main():
    df = load_features()
    register_tasks(df)

    results = {}
    for task_name, (feature_name, model_name) in BEST_COMBO.items():
        print(f"\n=== {task_name} (best combo: {feature_name}/{model_name}) ===")
        task = TASKS.get(task_name)

        if feature_name == "embedding":
            print("  Skipping predict_proba calibration for the embedding representation: "
                  "requires the dedicated .venv-embeddings interpreter, not importable here. "
                  "Reported as a known gap, not silently omitted.")
            calib_block = {"skipped": True,
                            "reason": "embedding feature requires .venv-embeddings; "
                                      "not run from the default interpreter"}
        else:
            y_true, proba, classes_ref = _oof_proba(task, feature_name, model_name, df)
            if task.label_type == "multilabel":
                confidence, correct = _reliability_multilabel(y_true, proba)
            else:
                confidence, correct = _reliability_multiclass(y_true, proba, classes_ref)

            ece, bins = _ece_and_bins(confidence, correct)
            conf_disagree = _confident_disagreement(confidence, correct)
            print(f"  ECE={ece:.4f}  n={len(confidence)}  "
                  f"confident(>90%)_disagreement_rate_of_all={conf_disagree['rate_of_all']:.3f}")
            calib_block = {
                "feature": feature_name, "model": model_name,
                "n_predictions": int(len(confidence)),
                "ece": ece, "reliability_bins": bins,
                "confident_disagreement": conf_disagree,
            }

        # RF tree-vote entropy always runs (RF works fine on tfidf/structured;
        # for the family task -- whose best combo is embedding -- this still
        # gives a usable entropy signal on tfidf as a fallback representation).
        rf_feature = feature_name if feature_name != "embedding" else "tfidf"
        entropy, rf_correct = _rf_tree_vote_entropy(task, rf_feature, df)
        mean_ent_correct = float(np.nanmean(entropy[rf_correct])) if rf_correct.any() else None
        mean_ent_wrong = float(np.nanmean(entropy[~rf_correct])) if (~rf_correct).any() else None
        try:
            auroc = float(roc_auc_score((~rf_correct).astype(int), entropy))
        except ValueError:
            auroc = None
        print(f"  RF tree-vote entropy (feature={rf_feature}): "
              f"mean|agree={mean_ent_correct}  mean|disagree={mean_ent_wrong}  "
              f"AUROC(entropy detects disagreement)={auroc}")

        results[task_name] = {
            "calibration": calib_block,
            "rf_tree_vote_entropy": {
                "feature": rf_feature,
                "mean_entropy_agree_weak_label": mean_ent_correct,
                "mean_entropy_disagree_weak_label": mean_ent_wrong,
                "auroc_entropy_detects_disagreement": auroc,
                "n": int(len(entropy)),
            },
        }

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
