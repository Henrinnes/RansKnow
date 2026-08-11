"""
Task label builders.

Class vocabularies (which families exist, which tactics exist) are
data-dependent, so tasks are registered lazily via register_tasks(df)
once the dataset is loaded, rather than at import time.
"""

from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from .registry import TASKS


@dataclass
class TaskSpec:
    name: str
    label_type: str          # "multilabel" | "multiclass"
    classes: List[str]
    build_y: Callable[[pd.DataFrame], np.ndarray]
    describe: str = ""


def _family_classes(df: pd.DataFrame, min_class_count: int = 2) -> List[str]:
    """Families with fewer than min_class_count total occurrences can't
    appear in both a train and a test split under any CV protocol, and
    can't be oversampled (SMOTE/ADASYN/CTGAN all need >=2 real examples
    to interpolate or model a distribution from). Excluded from the
    quantitative task rather than silently scored as a phantom class --
    printed so the exclusion is visible, not hidden."""
    fams = df["Family_List"].dropna().str.split(", ").explode()
    fams = fams[fams != ""]
    counts = fams.value_counts()
    kept = sorted(counts[counts >= min_class_count].index.tolist())
    excluded = counts[counts < min_class_count]
    if len(excluded):
        print(f"[family task] excluding {len(excluded)} classes with <{min_class_count} "
              f"occurrences (no CV split or oversampling method can use them): "
              + ", ".join(f"{k}({v})" for k, v in excluded.items()))
    return kept


def _build_family_y(df: pd.DataFrame, classes: List[str]) -> np.ndarray:
    lists = df["Family_List"].fillna("").apply(
        lambda s: [f for f in s.split(", ") if f] if s else []
    )
    mlb = MultiLabelBinarizer(classes=classes)
    return mlb.fit_transform(lists)


def _tactic_classes(df: pd.DataFrame) -> List[str]:
    vals = sorted(df["Dominant_Tactic"].dropna().unique().tolist())
    return vals + ["None"]


def _build_tactic_y(df: pd.DataFrame, classes: List[str]) -> np.ndarray:
    return df["Dominant_Tactic"].fillna("None").to_numpy()


def register_tasks(df: pd.DataFrame) -> None:
    fam_classes = _family_classes(df)
    if "family" not in TASKS.names():
        TASKS.register("family")(TaskSpec(
            name="family",
            label_type="multilabel",
            classes=fam_classes,
            build_y=lambda d: _build_family_y(d, fam_classes),
            describe=f"Ransomware family, multi-label, {len(fam_classes)} classes",
        ))

    tactic_classes = _tactic_classes(df)
    if "dominant_tactic" not in TASKS.names():
        TASKS.register("dominant_tactic")(TaskSpec(
            name="dominant_tactic",
            label_type="multiclass",
            classes=tactic_classes,
            build_y=lambda d: _build_tactic_y(d, tactic_classes),
            describe=f"Dominant MITRE tactic, multi-class, {len(tactic_classes)} classes",
        ))
