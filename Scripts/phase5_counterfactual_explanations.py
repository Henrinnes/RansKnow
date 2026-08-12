"""
RansKnow Phase 5.2 -- counterfactual explanations, Play-collision cross-check.

The plan's original framing: does a trained model's Family=Play
prediction bottom out on removing the word "play" alone? That's a
question about a model that predicts Play. This script asks a sharper
version: take the 275 videos where the OLD (pre-fix) regex said Play but
Phase 0.2's context-gating fix correctly removed it (real false
positives, confirmed by manual review at the time) -- these are NOT
labeled Play in the data the model trains on today. Does a TF-IDF+LogReg
model trained on the *fixed* labels still assign these videos a high
P(Play) anyway? If so, that's independent evidence (from a completely
different model class than the regex) that the same word-collision
pattern is discoverable from the raw text, not an artifact of the old
regex's specific bug -- and the counterfactual search then asks whether
"play" alone is what's driving it.

Usage:
    python3 Scripts/phase5_counterfactual_explanations.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

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

    print("Fitting TF-IDF + LogReg for 'Play' on full current (fixed) labels...")
    texts = [load_transcript(p) for p in df["Transcript_Path"]]
    vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform(texts)
    pipe = _logreg_pipeline()
    pipe.fit(X, y_play)

    def predict_proba(text: str) -> float:
        Xt = vec.transform([text])
        return float(pipe.predict_proba(Xt)[0, 1])

    # The 275 videos Phase 0.2 confirmed were false positives under the
    # old regex -- load directly rather than re-deriving, so this stays
    # anchored to the actual audited set.
    audit = pd.read_csv(ROOT / "outputs" / "Family_Alias_Audit.csv")
    lost_play_ids = set(audit[audit["lost"].fillna("").str.contains("Play")]["Video_ID"])

    path_by_id = dict(zip(df["Video_ID"], df["Transcript_Path"]))
    text_by_id = dict(zip(df["Video_ID"], texts))

    print(f"\nScoring {len(lost_play_ids)} confirmed-false-positive videos with the TRAINED model...")
    probs = []
    for vid in lost_play_ids:
        if vid not in text_by_id:
            continue
        p = predict_proba(text_by_id[vid])
        probs.append((vid, p))
    probs.sort(key=lambda x: -x[1])

    high_conf = [(v, p) for v, p in probs if p >= 0.5]
    print(f"Of those, the trained model STILL predicts Play (P>=0.5) for: "
          f"{len(high_conf)} / {len(probs)} ({len(high_conf)/len(probs):.0%})")
    print(f"Probability distribution: min={min(p for _,p in probs):.3f} "
          f"median={np.median([p for _,p in probs]):.3f} max={max(p for _,p in probs):.3f}")

    vocab = set(vec.vocabulary_.keys())
    # also include unigram components of bigram vocab entries
    for term in list(vocab):
        if " " in term:
            vocab.update(term.split(" "))

    # The false-positive set turned out entirely negative under the
    # trained model (max P=0.013) -- nothing there for a counterfactual
    # search to explain away. Pivot to the complementary question: for
    # videos the model DOES confidently predict Play on, what vocabulary
    # is it actually relying on? If removal sets are dominated by the
    # bare word "play" with nothing else, that's still a concern even
    # though the false-positive set came back clean. If removal sets
    # pull in ransomware-specific vocabulary, that's evidence of a
    # genuine learned signal.
    print(f"\n=== Counterfactual search: what drives the model's actual positive Play predictions? ===")
    true_positive_mask = y_play == 1
    tp_ids = df.loc[true_positive_mask, "Video_ID"].tolist()
    tp_probs = [(vid, predict_proba(text_by_id[vid])) for vid in tp_ids if vid in text_by_id]
    tp_probs.sort(key=lambda x: -x[1])

    cases = []
    for vid, p0 in tp_probs[:N_CASES_TO_EXPLAIN]:
        text = text_by_id[vid]
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
            "n_confirmed_false_positives": len(probs),
            "n_still_predicted_play": len(high_conf),
            "share_still_predicted": len(high_conf) / len(probs),
            "max_prob": max(p for _, p in probs),
        },
        "true_positive_explanations": cases,
        "n_flipped_on_bare_play_alone": bare_play_only,
    }, indent=2))
    print(f"\nWrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
