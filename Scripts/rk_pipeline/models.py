"""
Model registry -- a common fit/predict interface across the rule-based
Knowledge Agent baseline and the sklearn/LightGBM model ladder.

Every build() factory has the signature (label_type, task) -> estimator,
even though most models ignore `task` -- only the rule-based baseline
needs it (to know which regex logic to re-run).
"""

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import lightgbm as lgb

from .registry import MODELS

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import knowledge_agent as ka  # noqa: E402  (Scripts/knowledge_agent.py, path-injected above)


@dataclass
class ModelSpec:
    name: str
    build: Callable[[str, Any], Any]
    describe: str = ""
    requires_raw_df: bool = False


class SingleClassSafeClassifier(BaseEstimator, ClassifierMixin):
    """Wraps a binary classifier and falls back to predicting the single
    observed class when a training fold contains only one class.

    This is the normal case, not an edge case, for several of the 21
    family labels: under channel_grouped or temporal splits, a rare
    family (single digits of total occurrences) can easily have zero
    positive examples in a given fold's training data, and plain
    LogisticRegression/etc. raise rather than degrade gracefully.
    """

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, X, y):
        classes = np.unique(y)
        if len(classes) < 2:
            self._constant = int(classes[0]) if len(classes) else 0
            self.estimator_ = None
        else:
            self._constant = None
            self.estimator_ = clone(self.base_estimator).fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        n = X.shape[0]
        if self.estimator_ is None:
            return np.full(n, self._constant)
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        n = X.shape[0]
        if self.estimator_ is None:
            proba = np.zeros((n, 2))
            proba[:, self._constant] = 1.0
            return proba
        return self.estimator_.predict_proba(X)


def _wrap_multilabel(factory, label_type: str):
    if label_type != "multilabel":
        return factory()
    # Parallelize across the (up to 21) independent per-class fits, not
    # inside each one -- RandomForest/LightGBM's own n_jobs=-1 nested
    # inside a per-class loop spins up a full-core thread pool for each
    # sub-second fit, and that spawn overhead dominated a real run here
    # (structured-only smoke test didn't finish in 100s on 32 cores until
    # this was fixed).
    return MultiOutputClassifier(SingleClassSafeClassifier(factory()), n_jobs=-1)


def _logreg_pipeline():
    # with_mean=False (not centering) so this also works unmodified on
    # sparse TF-IDF input, not just dense structured features. The
    # scaling itself -- not the centering -- is what actually fixes
    # convergence: unscaled DurationSeconds (300-17,618) sitting next to
    # single-digit tactic counts made lbfgs take ~1.4s per tiny 800-row
    # fit instead of a few milliseconds.
    return make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )


MODELS.register("logreg")(ModelSpec(
    name="logreg",
    build=lambda label_type, task: _wrap_multilabel(_logreg_pipeline, label_type),
    describe="Logistic Regression baseline (scaled features, see module comment)",
))

def _svm_pipeline():
    # LinearSVC, not kernel SVC -- O(n) rather than O(n^2 - n^3), and the
    # standard choice for TF-IDF specifically (this is the "SVM" most text
    # classification baselines mean). No predict_proba out of the box;
    # if Phase 2 calibration needs probabilities from this model, wrap it
    # in CalibratedClassifierCV rather than switching to kernel SVC.
    return make_pipeline(
        StandardScaler(with_mean=False),
        LinearSVC(class_weight="balanced", max_iter=5000),
    )


MODELS.register("svm")(ModelSpec(
    name="svm",
    build=lambda label_type, task: _wrap_multilabel(_svm_pipeline, label_type),
    describe="Linear SVM (LinearSVC) -- standard TF-IDF baseline; no predict_proba, "
              "wrap in CalibratedClassifierCV before Phase 2",
))

MODELS.register("random_forest")(ModelSpec(
    name="random_forest",
    build=lambda label_type, task: _wrap_multilabel(
        lambda: RandomForestClassifier(
            n_estimators=150, max_depth=20, class_weight="balanced_subsample",
            random_state=0, n_jobs=1),
        label_type),
    describe="Random Forest -- also the cheapest source of an uncertainty "
              "signal for Phase 2 (tree-vote entropy)",
))

MODELS.register("gbm")(ModelSpec(
    name="gbm",
    build=lambda label_type, task: _wrap_multilabel(
        lambda: lgb.LGBMClassifier(n_estimators=300, random_state=0, verbosity=-1,
                                    is_unbalance=True, n_jobs=1),
        label_type),
    describe="LightGBM gradient-boosted trees (is_unbalance=True)",
))


@lru_cache(maxsize=4096)
def _ka_text(rel_path: str) -> str:
    return ka._norm((ka.ROOT / rel_path).read_text(encoding="utf-8", errors="ignore"))


@lru_cache(maxsize=4096)
def _ka_families(rel_path: str) -> frozenset:
    alias_map = ka._load_aliases(ka.FAMILY_LIST)
    return frozenset(ka._find_families(_ka_text(rel_path), alias_map))


@lru_cache(maxsize=4096)
def _ka_tactic_counts(rel_path: str) -> dict:
    # NOT sorted -- lru_cache only requires the argument to be hashable,
    # not the return value, so there's no reason to round-trip through a
    # sorted tuple. Preserving ka.TACTICS's own iteration order matters:
    # max(cnts, key=cnts.get) breaks ties by first-seen-in-iteration-order,
    # and a sorted-by-name reordering picked different winners than
    # knowledge_agent.py's own extract() on tied counts -- caught by the
    # rule-based baseline scoring 0.92 instead of a tautological 1.000
    # against its own labels once the "tool"/"platform" tasks started
    # exercising this path with different fold splits than "family" had.
    text = _ka_text(rel_path)
    return {t: ka._count(text, pats) for t, pats in ka.TACTICS.items()}


def _ka_dominant_tactic(rel_path: str) -> str:
    cnts = _ka_tactic_counts(rel_path)
    return max(cnts, key=cnts.get) if any(cnts.values()) else "None"


@lru_cache(maxsize=4096)
def _ka_platform_signal(rel_path: str) -> str:
    text = _ka_text(rel_path)
    cnts = {p: ka._count(text, pats) for p, pats in ka.PLATFORMS.items()}
    return max(cnts, key=cnts.get) if any(cnts.values()) else "None"


@lru_cache(maxsize=4096)
def _ka_tools(rel_path: str) -> frozenset:
    text = _ka_text(rel_path)
    cnts = {t: ka._count(text, pats) for t, pats in ka.TOOLS.items()}
    return frozenset(k for k, v in cnts.items() if v > 0)


def _ka_relevant(rel_path: str) -> str:
    fams = _ka_families(rel_path)
    tactic_impact = dict(_ka_tactic_counts(rel_path)).get("Impact", 0)
    return "Relevant" if (len(fams) > 0 or tactic_impact > 0) else "NotRelevant"


class RuleBasedKAClassifier:
    """Zero-training baseline: re-runs the Knowledge Agent's own regex
    extraction on the transcript text. This is baseline 0, not a strawman
    -- but its score against the *current* CSV labels is circular (those
    labels came from this exact logic) and will read as near-perfect for
    that reason alone. Only meaningful once scored against the Phase 0.3
    human-annotated gold set.

    Predictions are memoized per Transcript_Path at module level (see
    _ka_families/_ka_dominant_tactic above): the same document gets
    re-evaluated across every CV fold in every split protocol, and the
    regex tactic-counting in particular (~100 pattern scans per document)
    measured at ~69ms/doc uncached -- for the family+dominant_tactic
    smoke test across 3 split protocols that was the dominant cost in the
    whole run, not the trained models.
    """

    def __init__(self, label_type: str, task):
        self.label_type = label_type
        self.task = task

    def fit(self, X, y=None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.task.name == "family":
            rows = []
            for rel_path in X["Transcript_Path"]:
                fams = _ka_families(rel_path)
                rows.append([1 if c in fams else 0 for c in self.task.classes])
            return np.array(rows)

        if self.task.name == "dominant_tactic":
            return np.array([_ka_dominant_tactic(rel_path) for rel_path in X["Transcript_Path"]])

        if self.task.name == "platform":
            return np.array([_ka_platform_signal(rel_path) for rel_path in X["Transcript_Path"]])

        if self.task.name == "tool":
            rows = []
            for rel_path in X["Transcript_Path"]:
                tools = _ka_tools(rel_path)
                rows.append([1 if c in tools else 0 for c in self.task.classes])
            return np.array(rows)

        if self.task.name == "relevance":
            return np.array([_ka_relevant(rel_path) for rel_path in X["Transcript_Path"]])

        raise NotImplementedError(f"Rule-based baseline not implemented for task '{self.task.name}'")


MODELS.register("rule_based_ka")(ModelSpec(
    name="rule_based_ka",
    build=lambda label_type, task: RuleBasedKAClassifier(label_type, task),
    describe="Baseline 0 -- the existing regex-based Knowledge Agent, wrapped to the same interface",
    requires_raw_df=True,
))
