"""
Explainability hooks -- advanced-technique slot alongside Phases 2-4
(uncertainty, conformal prediction, OOD detection) in the experimental
plan.

Counterfactual explanations are the flagship use case here and are
specifically motivated by Phase 0.2's alias-collision finding: for a
text classifier trained on transcripts, a minimal-word-removal
counterfactual ("what's the smallest edit that flips this prediction
from Family=Play to Family=None") directly tests whether the model
learned real ransomware vocabulary or picked up the same kind of
spurious-word correlation the regex baseline did (Play/Hive/Cactus
colliding with ordinary English). A model whose counterfactuals for
Play predictions bottom out on removing the word "play" itself, with no
other family-specific vocabulary in the minimal edit set, would be
strong independent evidence that Finding A's noise propagated into the
model rather than being filtered out by training.

Not implemented yet -- this is a registered stub so the framework has a
slot for it once Phase 1 baselines exist to explain. Candidate approach
for text counterfactuals: greedy/beam word-removal search (mask each
token, keep removals that move predicted probability toward the target
class, repeat until the prediction flips or a removal budget is hit) --
a simpler, dependency-free alternative to MiCE/Polyjuice-style generative
counterfactual methods, appropriate given this is a diagnostic tool
rather than the paper's primary contribution.
"""

from dataclasses import dataclass
from typing import Any, Callable

from .registry import EXPLAINERS


@dataclass
class ExplainerSpec:
    name: str
    build: Callable[[Any], Any]
    describe: str = ""


def _not_implemented(*_args, **_kwargs):
    raise NotImplementedError(
        "Counterfactual explainer not implemented yet -- needs a fitted "
        "Phase 1 model to explain. Planned approach: greedy word-removal "
        "search over transcript tokens, tracking predicted-probability "
        "movement toward the counterfactual target class. See module "
        "docstring for why this is prioritized (Phase 0.2 alias-collision "
        "cross-check)."
    )


EXPLAINERS.register("counterfactual_word_removal")(ExplainerSpec(
    name="counterfactual_word_removal",
    build=_not_implemented,
    describe="Greedy word-removal counterfactuals for text classifiers -- "
              "TODO, see module docstring",
))
