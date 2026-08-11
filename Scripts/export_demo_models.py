"""
RansKnow inference-demo model export.

Fits a single shared TfidfVectorizer on all 1034 transcripts, then fits
LogReg and SVM per task on the FULL dataset (no held-out split -- this
produces a demo artifact, not an evaluation; Phase 1's CV results in
outputs/experiments/ are the actual accuracy claims). Exports everything
needed to reproduce predict_proba/decision_function in JS: the TF-IDF
vocabulary + idf weights, and per task/model/class the fitted
StandardScaler.scale_ + LogisticRegression/LinearSVC coef_/intercept_.

Family and Tool are multilabel (MultiOutputClassifier wrapping one
SingleClassSafeClassifier per class), so each class's pipeline is fit
independently and exported separately -- not a shared weight matrix like
the plain multiclass tasks.

Usage:
    python3 Scripts/export_demo_models.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rk_pipeline.data import load_features, load_transcript  # noqa: E402
from rk_pipeline.models import _logreg_pipeline, _svm_pipeline, SingleClassSafeClassifier  # noqa: E402
from rk_pipeline.tasks import register_tasks, TASKS  # noqa: E402
from sklearn.multioutput import MultiOutputClassifier  # noqa: E402

OUT = ROOT.parent / "outputs" / "demo_models.json"


def export_binary_pipeline(pipeline) -> dict:
    """A fitted Pipeline(StandardScaler(with_mean=False) -> LogReg/LinearSVC) --
    always exactly these 2 steps, by construction in rk_pipeline.models."""
    scaler = pipeline.steps[0][1]
    clf = pipeline.steps[-1][1]
    return {
        "scale": scaler.scale_.tolist(),
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "classes": [str(c) for c in clf.classes_],
    }


def export_multiclass_task(df, task) -> dict:
    y = task.build_y(df)
    result = {}
    for model_name, factory in [("logreg", _logreg_pipeline), ("svm", _svm_pipeline)]:
        pipe = factory()
        pipe.fit(TFIDF_MATRIX, y)
        result[model_name] = export_binary_pipeline(pipe)
    return result


def export_multilabel_task(df, task) -> dict:
    y = task.build_y(df)  # (n_samples, n_classes) binary indicator
    result = {}
    for model_name, factory in [("logreg", _logreg_pipeline), ("svm", _svm_pipeline)]:
        moc = MultiOutputClassifier(SingleClassSafeClassifier(factory()), n_jobs=-1)
        moc.fit(TFIDF_MATRIX, y)
        per_class = []
        for i, cls_name in enumerate(task.classes):
            safe_clf = moc.estimators_[i]
            if safe_clf.estimator_ is None:
                per_class.append({"constant": safe_clf._constant, "class_name": cls_name})
            else:
                exported = export_binary_pipeline(safe_clf.estimator_)
                exported["class_name"] = cls_name
                per_class.append(exported)
        result[model_name] = per_class
    return result


def main():
    global TFIDF_MATRIX

    df = load_features()
    register_tasks(df)

    print("Fitting shared TF-IDF vectorizer on all transcripts...")
    vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)
    texts = [load_transcript(p) for p in df["Transcript_Path"]]
    TFIDF_MATRIX = vec.fit_transform(texts)
    print(f"  vocab size: {len(vec.vocabulary_)}")

    bundle = {
        "vocabulary": {term: int(idx) for term, idx in vec.vocabulary_.items()},
        "idf": vec.idf_.tolist(),
        "tasks": {},
    }

    for task_name in ["family", "dominant_tactic", "platform", "tool", "relevance"]:
        task = TASKS.get(task_name)
        print(f"Fitting {task_name} ({task.label_type}, {len(task.classes)} classes)...")
        if task.label_type == "multilabel":
            bundle["tasks"][task_name] = {
                "label_type": "multilabel",
                "classes": task.classes,
                "models": export_multilabel_task(df, task),
            }
        else:
            bundle["tasks"][task_name] = {
                "label_type": "multiclass",
                "classes": task.classes,
                "models": export_multiclass_task(df, task),
            }

    OUT.write_text(json.dumps(bundle))
    print(f"\nWrote {OUT.relative_to(ROOT.parent)} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
