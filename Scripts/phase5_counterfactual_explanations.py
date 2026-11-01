"""
RansKnow Phase 5.2 -- counterfactual explanations, Play-collision cross-check.

The plan's original framing: does a trained model's Family=Play
prediction bottom out on removing the word "play" alone? That's a
question about a model that predicts Play. This script asks a sharper
version: take the 275 videos where the OLD (pre-fix) regex said Play but
Phase 0.2's context-gating fix removed it -- these are NOT labeled Play in
the data the model trains on today. Does a TF-IDF+LogReg model trained on
the *fixed* labels, and scored only on videos it did not train on, still
rank these videos above other not-Play videos? If so, that's independent
evidence (from a completely different model class than the regex) that the
same word-collision pattern is discoverable from the raw text, not an
artifact of the old regex's specific bug -- and the counterfactual search
then asks whether "play" alone is what's driving it.

Usage:
    python3 Scripts/phase5_counterfactual_explanations.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rk_pipeline.data import load_features, load_transcript  # noqa: E402
from rk_pipeline.explainers import word_removal_counterfactual  # noqa: E402
from rk_pipeline.models import _logreg_pipeline  # noqa: E402
from rk_pipeline.tasks import register_tasks, TASKS  # noqa: E402

OUT = ROOT / "outputs" / "phase5_counterfactual_play.json"
N_CASES_TO_EXPLAIN = 8


def main():
    df = load_features()
    register_tasks(df)
    task = TASKS.get("family")
    if "Play" not in task.classes:
        print("Play not in current class vocabulary -- nothing to check")
        return
    play_idx = task.classes.index("Play")
    y = task.build_y(df)
    y_play = y[:, play_idx]

    # Every score below comes from a model that did not train on that video.
    # The removed videos are labeled not-Play in the training data, so a
    # model that had seen them would score them low regardless of whether it
    # learned the word collision.
    print("Fitting 5-fold TF-IDF + LogReg for 'Play' on current (fixed) labels...")
    texts = [load_transcript(p) for p in df["Transcript_Path"]]
    text_by_id = dict(zip(df["Video_ID"], texts))
    fold_of = {}
    fold_models = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, y_play)):
        vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)
        X = vec.fit_transform([texts[i] for i in train_idx])
        pipe = _logreg_pipeline()
        pipe.fit(X, y_play[train_idx])
        vocab = set(vec.vocabulary_.keys())
        # also include unigram components of bigram vocab entries
        for term in list(vocab):
            if " " in term:
                vocab.update(term.split(" "))
        fold_models.append((vec, pipe, vocab))
        for i in test_idx:
            fold_of[df["Video_ID"].iloc[i]] = fold

    def proba_fn(vid):
        vec, pipe, _ = fold_models[fold_of[vid]]
        return lambda text: float(pipe.predict_proba(vec.transform([text]))[0, 1])

    oof = {vid: proba_fn(vid)(text_by_id[vid]) for vid in df["Video_ID"]}

    # The videos the Phase 0.2 context fix removed from Play. "Removed" is
    # what this set is: the fix's own judgment, not a per-video human review.
    audit = pd.read_csv(ROOT / "outputs" / "Family_Alias_Audit.csv")
    lost_play_ids = set(audit[audit["lost"].fillna("").str.contains("Play")]["Video_ID"])

    print(f"\nHeld-out scores for the {len(lost_play_ids)} videos the fix removed from Play...")
    probs = sorted(((vid, oof[vid]) for vid in lost_play_ids if vid in oof), key=lambda x: -x[1])

    high_conf = [(v, p) for v, p in probs if p >= 0.5]
    print(f"Of those, the held-out model predicts Play (P>=0.5) for: "
          f"{len(high_conf)} / {len(probs)} ({len(high_conf)/len(probs):.0%})")
    print(f"Probability distribution: min={min(p for _,p in probs):.3f} "
          f"median={np.median([p for _,p in probs]):.3f} max={max(p for _,p in probs):.3f}")

    # The inheritance question: among videos the fixed labels call not-Play,
    # does the model rate the ones containing the colliding use of "play"
    # above the rest? 0.5 would mean it does not tell them apart.
    other_neg = [oof[vid] for vid, y_i in zip(df["Video_ID"], y_play)
                 if y_i == 0 and vid not in lost_play_ids]
    lost_scores = [p for _, p in probs]
    removed_vs_other = float(roc_auc_score(
        [1] * len(lost_scores) + [0] * len(other_neg), lost_scores + other_neg))
    positive_scores = [oof[vid] for vid, y_i in zip(df["Video_ID"], y_play) if y_i == 1]
    print(f"AUROC removed vs. other not-Play videos: {removed_vs_other:.3f}")
    print(f"Median P(Play): removed={np.median(lost_scores):.3f} "
          f"other_not_play={np.median(other_neg):.3f} labeled_play={np.median(positive_scores):.3f}")

    # The complementary question: for videos the fixed labels call Play and
    # the held-out model ranks highest, what vocabulary is it relying on?
    # If removal sets are dominated by the bare word "play" with nothing
    # else, that's a concern. If removal sets pull in ransomware-specific
    # vocabulary, that's evidence of a genuine learned signal.
    print(f"\n=== Counterfactual search: what drives held-out positive Play predictions? ===")
    tp_ids = df.loc[y_play == 1, "Video_ID"].tolist()
    tp_probs = sorted(((vid, oof[vid]) for vid in tp_ids), key=lambda x: -x[1])

    cases = []
    for vid, p0 in tp_probs[:N_CASES_TO_EXPLAIN]:
        text = text_by_id[vid]
        predict_proba = proba_fn(vid)
        vocab = fold_models[fold_of[vid]][2]
        result = word_removal_counterfactual(
            text, "Play", predict_proba, max_removals=15, candidate_vocab=vocab
        )
        title = df.loc[df["Video_ID"] == vid, "Video_Title"].values[0]
        print(f"\n{vid} | {title[:60]}")
        print(f"  P(Play) start={result.original_prob:.3f} -> end={result.final_prob:.3f} "
              f"({'FLIPPED' if result.flipped else 'did not flip'} in {len(result.removed_words)} removals)")
        print(f"  removal order: {result.removed_words}")
        cases.append({
            "video_id": vid, "title": title,
            "original_prob": result.original_prob, "final_prob": result.final_prob,
            "flipped": result.flipped, "removed_words": result.removed_words,
            "trajectory": result.trajectory,
        })

    bare_play_only = sum(
        1 for c in cases if c["flipped"] and c["removed_words"] == ["play"]
    )
    print(f"\nOf {len(cases)} explained true-positive cases, {bare_play_only} flipped on "
          f"removing 'play' alone with nothing else -- {'a real concern' if bare_play_only else 'none did'}.")

    import json
    OUT.write_text(json.dumps({
        "false_positive_check": {
            "scoring": "out-of-fold, 5-fold stratified",
            "n_removed_by_fix": len(probs),
            "n_predicted_play": len(high_conf),
            "share_predicted_play": len(high_conf) / len(probs),
            "max_prob": max(p for _, p in probs),
            "median_prob_removed": float(np.median(lost_scores)),
            "median_prob_other_not_play": float(np.median(other_neg)),
            "median_prob_labeled_play": float(np.median(positive_scores)),
            "auroc_removed_vs_other_not_play": removed_vs_other,
            "n_labeled_play": len(positive_scores),
            "share_labeled_play_predicted_play": float(np.mean(np.array(positive_scores) >= 0.5)),
        },
        "true_positive_explanations": cases,
        "n_flipped_on_bare_play_alone": bare_play_only,
    }, indent=2))
    print(f"\nWrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
