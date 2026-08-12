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

import re
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from .registry import EXPLAINERS


@dataclass
class ExplainerSpec:
    name: str
    build: Callable[[Any], Any]
    describe: str = ""


@dataclass
class CounterfactualResult:
    original_prob: float
    final_prob: float
    flipped: bool           # crossed below 0.5
    removed_words: List[str]
    trajectory: List[float]  # probability after each removal


def word_removal_counterfactual(
    text: str,
    target_class: str,
    predict_proba_fn: Callable[[str], float],
    max_removals: int = 25,
    flip_threshold: float = 0.5,
    candidate_vocab: Optional[set] = None,
) -> CounterfactualResult:
    """Greedy word-*type* removal counterfactual search.

    Operates on unique lowercase word types, not raw token positions --
    each step removes every occurrence of one word type from the current
    text. Two reasons: (1) tractability -- a several-thousand-word
    transcript has too many token positions to try one at a time, but
    typically a few hundred unique word types, further cut down by
    restricting candidates to the model's TF-IDF vocabulary (removing an
    out-of-vocabulary word can't move a linear TF-IDF model's score at
    all); (2) interpretability -- "removing every instance of 'play'" is
    the natural unit for asking whether one trigger word is doing the
    work, not "removing the word at position 812".

    predict_proba_fn takes a text string and returns P(target_class) --
    the caller closes over the fitted vectorizer/model so this function
    stays model-agnostic.

    Diagnostic tool, not a generative-fluency method (MiCE/Polyjuice-
    style) on purpose: the object of interest is *which* words the model
    leans on, not producing readable edited text.
    """
    original_prob = predict_proba_fn(text)

    word_re = re.compile(r"\b\w+\b")
    all_words = word_re.findall(text.lower())
    unique_types = set(all_words)
    if candidate_vocab is not None:
        unique_types &= candidate_vocab

    removed_words: List[str] = []
    trajectory: List[float] = []
    current_text = text
    current_prob = original_prob
    remaining = set(unique_types)

    for _ in range(max_removals):
        if current_prob < flip_threshold or not remaining:
            break

        best_word = None
        best_prob = current_prob
        best_text = None
        for w in remaining:
            pat = re.compile(r"\b" + re.escape(w) + r"\b", re.IGNORECASE)
            candidate_text = pat.sub("", current_text)
            p = predict_proba_fn(candidate_text)
            if p < best_prob:
                best_prob = p
                best_word = w
                best_text = candidate_text

        if best_word is None:
            break  # no single remaining word-type removal helps further

        removed_words.append(best_word)
        remaining.discard(best_word)
        current_text = best_text
        current_prob = best_prob
        trajectory.append(current_prob)

    return CounterfactualResult(
        original_prob=original_prob,
        final_prob=current_prob,
        flipped=current_prob < flip_threshold,
        removed_words=removed_words,
        trajectory=trajectory,
    )


EXPLAINERS.register("counterfactual_word_removal")(ExplainerSpec(
    name="counterfactual_word_removal",
    build=lambda: word_removal_counterfactual,
    describe="Greedy word-removal counterfactual search -- see "
              "Scripts/phase5_counterfactual_explanations.py for the "
              "Play-alias-collision cross-check this was built for",
))
