"""
CV split protocols -- the three Phase 1 validation strategies, run in
parallel on purpose:

- stratified_random : the standard number, and the one most likely to be
  optimistic.
- channel_grouped    : no Channel_ID appears in both train and test. With
  up to 20 videos per channel and only ~92 channels, a model can otherwise
  learn "this is CrowdStrike's narration style" instead of the content.
  The gap between this score and stratified_random *is* a result.
- temporal           : train on pre-2024, test on 2024+. 79% of the corpus
  is 2024 or later, so this is the honest generalization check.
"""

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

from .registry import SPLITS


@dataclass
class SplitSpec:
    name: str
    make_splits: Callable[[pd.DataFrame, np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]
    describe: str = ""


def _stratify_key(y: np.ndarray) -> np.ndarray:
    """StratifiedKFold needs a 1D target. For multilabel y (2D), stratify
    on the label-cardinality bucket instead (0 / 1 / 2+ labels)."""
    if y.ndim == 2:
        return np.clip(y.sum(axis=1), 0, 2)
    return y


def _stratified_random(df: pd.DataFrame, y: np.ndarray, n_splits=5, seed=0):
    key = _stratify_key(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(df, key))


def _channel_grouped(df: pd.DataFrame, y: np.ndarray, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(df, groups=df["Channel_ID"]))


def _temporal(df: pd.DataFrame, y: np.ndarray, cutoff_year=2024):
    train_idx = np.where(df["Year"] < cutoff_year)[0]
    test_idx = np.where(df["Year"] >= cutoff_year)[0]
    return [(train_idx, test_idx)]


SPLITS.register("stratified_random")(SplitSpec(
    name="stratified_random",
    make_splits=_stratified_random,
    describe="5-fold stratified random CV",
))
SPLITS.register("channel_grouped")(SplitSpec(
    name="channel_grouped",
    make_splits=_channel_grouped,
    describe="5-fold GroupKFold by Channel_ID -- catches channel-style leakage",
))
SPLITS.register("temporal")(SplitSpec(
    name="temporal",
    make_splits=_temporal,
    describe="Single split: train pre-2024, test 2024+",
))
