"""
Task-appropriate metrics.

macro_f1 is computed over classes actually present in that fold's
y_true, not the full class vocabulary. With 21 long-tailed family
classes and 5-fold CV, most folds simply don't contain every class --
scoring an absent class as F1=0 (sklearn's default with zero_division=0
over the full label set) isn't measuring model quality, it's measuring
how unlucky the fold split was. macro_f1_all_classes is kept alongside
it so the gap between the two is visible rather than hidden.
"""

import numpy as np
from sklearn.metrics import f1_score


def score_multilabel(y_true, y_pred) -> dict:
    support = y_true.sum(axis=0)
    present = np.where(support > 0)[0]

    macro_f1_present = (
        f1_score(y_true[:, present], y_pred[:, present], average="macro", zero_division=0)
        if len(present) else float("nan")
    )
    return {
        "macro_f1": macro_f1_present,
        "macro_f1_all_classes": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "n_classes_present": len(present),
    }


def score_multiclass(y_true, y_pred) -> dict:
    labels = np.unique(y_true)
    return {
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "n_classes_present": len(labels),
    }
