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


def _platform_classes(df: pd.DataFrame) -> List[str]:
    vals = sorted(df["Platform_Signal"].dropna().unique().tolist())
    return vals + ["None"]


def _build_platform_y(df: pd.DataFrame, classes: List[str]) -> np.ndarray:
    return df["Platform_Signal"].fillna("None").to_numpy()


def _tool_classes(df: pd.DataFrame, min_class_count: int = 2) -> List[str]:
    """Same reasoning as _family_classes -- Tool_MegaNZ has zero
    occurrences in the whole dataset (checked: 0/1034), which is even
    more degenerate than family's rarest classes. Excluded here rather
    than silently producing an always-0 column."""
    tools = df["Tool_List"].dropna().str.split(", ").explode()
    tools = tools[tools != ""]
    counts = tools.value_counts()
    kept = sorted(counts[counts >= min_class_count].index.tolist())
    excluded = counts[counts < min_class_count]
    all_possible = {c.replace("Tool_", "") for c in df.columns if c.startswith("Tool_")
                    and c not in ("Tool_Total_Mentions", "Tool_List")}
    never_seen = all_possible - set(counts.index)
    if len(excluded) or never_seen:
        parts = [f"{k}({v})" for k, v in excluded.items()] + [f"{k}(0)" for k in never_seen]
        print(f"[tool task] excluding {len(parts)} classes with <{min_class_count} "
              f"occurrences: " + ", ".join(parts))
    return kept


def _build_tool_y(df: pd.DataFrame, classes: List[str]) -> np.ndarray:
    lists = df["Tool_List"].fillna("").apply(
        lambda s: [t for t in s.split(", ") if t] if s else []
    )
    mlb = MultiLabelBinarizer(classes=classes)
    return mlb.fit_transform(lists)


def _build_relevance_y(df: pd.DataFrame, classes: List[str]) -> np.ndarray:
    """Task E from the plan: a cheap binary screen for 'is this video even
    about ransomware', useful as the actual bottleneck if the fetch
    pipeline is ever pointed at new channels without hand-reviewing every
    keyword hit. Not a column that exists in the CSV -- derived as
    Family_Count > 0 OR Tactic_Impact > 0, per the plan's task table."""
    relevant = (df["Family_Count"] > 0) | (df["Tactic_Impact"] > 0)
    return relevant.map({True: "Relevant", False: "NotRelevant"}).to_numpy()


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

    platform_classes = _platform_classes(df)
    if "platform" not in TASKS.names():
        TASKS.register("platform")(TaskSpec(
            name="platform",
            label_type="multiclass",
            classes=platform_classes,
            build_y=lambda d: _build_platform_y(d, platform_classes),
            describe=f"Platform signal, multi-class, {len(platform_classes)} classes",
        ))

    tool_classes = _tool_classes(df)
    if "tool" not in TASKS.names():
        TASKS.register("tool")(TaskSpec(
            name="tool",
            label_type="multilabel",
            classes=tool_classes,
            build_y=lambda d: _build_tool_y(d, tool_classes),
            describe=f"Offensive tool mentions, multi-label, {len(tool_classes)} classes",
        ))

    relevance_classes = ["NotRelevant", "Relevant"]
    if "relevance" not in TASKS.names():
        TASKS.register("relevance")(TaskSpec(
            name="relevance",
            label_type="multiclass",
            classes=relevance_classes,
            build_y=lambda d: _build_relevance_y(d, relevance_classes),
            describe="Ransomware-relevance screen, binary (derived: Family_Count>0 OR Tactic_Impact>0)",
        ))
