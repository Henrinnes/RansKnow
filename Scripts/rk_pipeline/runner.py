"""Config-driven experiment runner: task x feature x model x split combos."""

import json
import subprocess
import time
from pathlib import Path

import pandas as pd

from . import metrics as _metrics
from .data import load_features
from .registry import FEATURES, MODELS, SPLITS, TASKS
from .tasks import register_tasks

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "outputs" / "experiments"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _run_one_fold(task, model_spec, feature_name, train_df, test_df, y_train, y_test):
    if model_spec.requires_raw_df:
        X_train, X_test = train_df, test_df
    else:
        builder = FEATURES.get(feature_name)()
        X_train = builder.fit_transform(train_df)
        X_test = builder.transform(test_df)

    est = model_spec.build(task.label_type, task)
    est.fit(X_train, y_train)
    y_pred = est.predict(X_test)

    if task.label_type == "multilabel":
        return _metrics.score_multilabel(y_test, y_pred)
    return _metrics.score_multiclass(y_test, y_pred)


def run(task_names, feature_names, model_names, split_names, out_name="results", verbose=True):
    df = load_features()
    register_tasks(df)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for task_name in task_names:
        task = TASKS.get(task_name)
        y_full = task.build_y(df)

        for feature_name in feature_names:
            for model_name in model_names:
                model_spec = MODELS.get(model_name)

                for split_name in split_names:
                    split_spec = SPLITS.get(split_name)
                    fold_scores = []

                    for fold_i, (train_idx, test_idx) in enumerate(split_spec.make_splits(df, y_full)):
                        train_df = df.iloc[train_idx].reset_index(drop=True)
                        test_df = df.iloc[test_idx].reset_index(drop=True)
                        y_train, y_test = y_full[train_idx], y_full[test_idx]

                        s = _run_one_fold(task, model_spec, feature_name, train_df, test_df, y_train, y_test)
                        s["fold"] = fold_i
                        s["n_train"] = len(train_idx)
                        s["n_test"] = len(test_idx)
                        fold_scores.append(s)

                    fold_df = pd.DataFrame(fold_scores)
                    agg = fold_df.drop(columns=["fold"]).mean().to_dict()
                    agg.update({
                        "task": task_name,
                        "feature": feature_name,
                        "model": model_name,
                        "split": split_name,
                        "n_folds": len(fold_scores),
                    })
                    rows.append(agg)
                    if verbose:
                        print(f"[{task_name:16} | {feature_name:10} | {model_name:14} | {split_name:17}] "
                              f"macro_f1={agg.get('macro_f1', float('nan')):.3f}")

    results = pd.DataFrame(rows)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{out_name}_{ts}.csv"
    results.to_csv(out_path, index=False)

    meta = {
        "git_commit": _git_commit(),
        "timestamp": ts,
        "n_rows_data": len(df),
        "task_names": task_names,
        "feature_names": feature_names,
        "model_names": model_names,
        "split_names": split_names,
    }
    (RESULTS_DIR / f"{out_name}_{ts}.meta.json").write_text(json.dumps(meta, indent=2))

    if verbose:
        print(f"\nWrote {out_path.relative_to(ROOT)}")
    return results, out_path
