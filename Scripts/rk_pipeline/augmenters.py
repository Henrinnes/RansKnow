"""
Class-balancing / augmentation strategies for the long-tailed family task.

Three rungs, in the order they should actually be tried:

1. class_weight="balanced" (already wired into logreg/random_forest;
   gbm uses is_unbalance=True) -- free, no synthetic data, always on.
2. SMOTE / ADASYN (this module) -- nearest-neighbor interpolation over
   the *structured* feature representation only. Needs >=2 real examples
   per class; k_neighbors is adapted down per class rather than left at
   imblearn's default of 5, which would simply error on any class with
   fewer than 6 real examples -- and several family classes here have
   exactly 5 or 6 (DarkSide=5, Maze=5, BlackCat=6, Black Basta=6,
   Cactus=6).
3. CTGAN -- registered as a documented stub, not installed by default.
   A conditional GAN needs hundreds of real rows per class to learn a
   non-degenerate distribution; several classes here have single digits.
   Fed 5-6 real examples, CTGAN is more likely to produce near-duplicate
   synthetic rows than genuinely novel ones -- which would look like it
   "worked" on a naive before/after check without generalizing. Worth
   running only as an explicit ablation against SMOTE ("does a deep
   generative augmenter beat classical interpolation here"), not as the
   default augmentation path, and only after `pip install ctgan` (not a
   current dependency).

Augmentation must only ever touch the *training* fold inside CV --
never the test fold, and never anything derived from transcript text
(TF-IDF/embedding representations, or the rule-based KA baseline, which
all need real text that SMOTE/CTGAN over structured features can't
produce).
"""

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .registry import Registry

AUGMENTERS: Registry = Registry("augmenter")


@dataclass
class AugmenterSpec:
    name: str
    build: Callable[[], Any]
    describe: str = ""
    structured_only: bool = True


class SmoteAdaptiveK:
    """SMOTE wrapper that lowers k_neighbors per minority class instead
    of erroring when a class has fewer than the default 6 examples
    needed for k_neighbors=5. Falls back to simple duplication (k=1,
    equivalent to nearest-neighbor replication with no interpolation)
    for classes with only 2-3 real examples, since SMOTE's interpolation
    geometry is barely meaningful there anyway.
    """

    def __init__(self, random_state: int = 0):
        self.random_state = random_state

    def fit_resample(self, X: np.ndarray, y: np.ndarray):
        from imblearn.over_sampling import SMOTE

        classes, counts = np.unique(y, return_counts=True)
        min_count = counts.min()
        if min_count < 2:
            raise ValueError(
                f"Cannot oversample: a class has only {min_count} example(s). "
                "Classes with a single example should already be excluded "
                "upstream (see tasks._family_classes min_class_count)."
            )
        k = max(1, min(5, int(min_count) - 1))
        smote = SMOTE(k_neighbors=k, random_state=self.random_state)
        return smote.fit_resample(X, y)


AUGMENTERS.register("smote")(AugmenterSpec(
    name="smote",
    build=lambda: SmoteAdaptiveK(),
    describe="SMOTE with per-fold adaptive k_neighbors, structured features only",
))


def _ctgan_not_installed(*_args, **_kwargs):
    raise NotImplementedError(
        "CTGAN is not installed (`pip install ctgan`) and isn't the "
        "recommended default here -- see module docstring: several family "
        "classes have only 5-6 real examples, too few for a GAN to learn a "
        "non-degenerate distribution from. Install it and treat this as an "
        "explicit ablation against 'smote', not a replacement for it."
    )


AUGMENTERS.register("ctgan")(AugmenterSpec(
    name="ctgan",
    build=_ctgan_not_installed,
    describe="Conditional GAN augmentation -- not installed, ablation-only, see docstring",
))
