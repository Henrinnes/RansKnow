"""
RansKnow Phase 5.3 -- perturbation-robustness sweep.

Untargeted counterpart to Phase 5.2's counterfactual search: perturb
inputs within a semantic-preserving budget and measure how much trained
predictions move, without aiming at a specific flip. Two perturbation
types:

1. Simulated ASR noise -- restricted to the 74 videos with official
   YouTube captions (Transcript_Provider == youtube_api), the "clean"
   subset. Injects the kind of errors a smaller ASR model (this corpus
   is 92.7% Whisper-transcribed, mostly whisper_base -- see Finding B)
   actually makes on unfamiliar technical vocabulary: compound proper
   nouns split apart ("WannaCry" -> "wanna cry"), phonetic respelling of
   unusual terms ("Mimikatz" -> "mimi cats"), plus random low-confidence
   token deletion. This operationalizes Finding B as a controlled
   before/after comparison instead of an aggregate-rate observation.

2. Synonym substitution -- swaps common, non-jargon words for synonyms
   across a broader sample. If predictions move a lot under this, the
   model is leaning on surface wording rather than the underlying
   content the wording expresses.

Reports flip rate (does the top prediction change) and mean absolute
probability drift, per task, per perturbation type.

Usage:
    python3 Scripts/phase5_perturbation_robustness.py
"""

import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rk_pipeline.data import load_features, load_transcript  # noqa: E402
from rk_pipeline.models import MODELS  # noqa: E402
from rk_pipeline.tasks import register_tasks, TASKS  # noqa: E402

OUT = ROOT / "outputs" / "phase5_perturbation_robustness.json"
SEED = 0

# Curated, not exhaustive -- targets the exact failure mode Whisper-base
# is known for on unfamiliar technical vocabulary: splitting compound
# proper nouns and phonetically respelling unusual terms. Grounded in
# the same family/tool names knowledge_agent.py's alias map/TOOLS dict
# already track, not invented in isolation.
JARGON_ASR_ERRORS = {
    "wannacry": "wanna cry",
    "blackcat": "black cat",
    "mimikatz": "mimi cats",
    "psexec": "p s exec",
    "rclone": "are clone",
    "anydesk": "any desk",
    "teamviewer": "team viewer",
    "bloodhound": "blood hound",
    "ransomware": "ransom where",
    "malware": "mal where",
    "phishing": "fishing",
    "exfiltrate": "ex filtrate",
    "exfiltration": "ex filtration",
    "qilin": "key lin",
    "conti": "con tea",
}

SYNONYM_MAP = {
    "attackers": "adversaries", "attacker": "adversary",
    "used": "utilized", "using": "utilizing",
    "network": "infrastructure",
    "files": "documents", "file": "document",
    "victim": "target", "victims": "targets",
    "attack": "intrusion", "attacks": "intrusions",
    "detected": "identified", "detect": "identify",
    "found": "discovered", "showed": "demonstrated",
    "important": "significant", "big": "large",
    "helps": "assists", "shows": "illustrates",
    "good": "effective", "said": "stated",
}


def inject_asr_noise(text: str, deletion_rate: float, rng: random.Random) -> str:
    lowered = text
    for term, mangled in JARGON_ASR_ERRORS.items():
        lowered = re.sub(r"\b" + re.escape(term) + r"\b", mangled, lowered, flags=re.IGNORECASE)

    words = lowered.split()
    kept = [w for w in words if rng.random() > deletion_rate]
    return " ".join(kept) if kept else lowered


def inject_synonyms(text: str) -> str:
    def repl(m):
        w = m.group(0)
        lw = w.lower()
        if lw in SYNONYM_MAP:
            syn = SYNONYM_MAP[lw]
            return syn.capitalize() if w[0].isupper() else syn
        return w
    return re.sub(r"\b\w+\b", repl, text)


def fit_task_model(df, task, texts, vec):
    X = vec.transform(texts) if hasattr(vec, "vocabulary_") else vec.fit_transform(texts)
    y = task.build_y(df)
    spec = MODELS.get("logreg")
    model = spec.build(task.label_type, task)
    model.fit(X, y)
    return model


def top_prediction(model, vec, text, task):
    Xt = vec.transform([text])
    if task.label_type == "multilabel":
        # MultiOutputClassifier of per-class binary pipelines
        probs = np.array([est.predict_proba(Xt)[0, 1] for est in model.estimators_])
        top_i = int(np.argmax(probs))
        return task.classes[top_i], float(probs[top_i])
    else:
        probs = model.predict_proba(Xt)[0]
        classes = model.steps[-1][1].classes_ if hasattr(model, "steps") else model.classes_
        top_i = int(np.argmax(probs))
        return classes[top_i], float(probs[top_i])


def sweep(df, task_name, texts_by_id, video_ids, perturb_fn, vec, model, task, label):
    flips = 0
    drifts = []
    for vid in video_ids:
        orig_text = texts_by_id[vid]
        pert_text = perturb_fn(orig_text)

        orig_class, orig_p = top_prediction(model, vec, orig_text, task)
        pert_class, _ = top_prediction(model, vec, pert_text, task)

        # probability drift = change in probability assigned to the
        # ORIGINAL top class after perturbation (not the new top class)
        Xt = vec.transform([pert_text])
        if task.label_type == "multilabel":
            idx = task.classes.index(orig_class)
            pert_p_of_orig_class = float(model.estimators_[idx].predict_proba(Xt)[0, 1])
        else:
            classes = model.steps[-1][1].classes_
            idx = list(classes).index(orig_class)
            pert_p_of_orig_class = float(model.predict_proba(Xt)[0][idx])

        drifts.append(abs(orig_p - pert_p_of_orig_class))
        if orig_class != pert_class:
            flips += 1

    return {
        "label": label, "n": len(video_ids),
        "flip_rate": flips / len(video_ids) if video_ids else None,
        "mean_abs_drift": float(np.mean(drifts)) if drifts else None,
    }


def main():
    df = load_features()
    register_tasks(df)
    all_texts = {vid: load_transcript(p) for vid, p in zip(df["Video_ID"], df["Transcript_Path"])}

    asr_clean_ids = df.loc[df["Transcript_Provider"] == "youtube_api", "Video_ID"].tolist()
    print(f"ASR-noise test set (official captions): {len(asr_clean_ids)} videos")

    rng_seed_base = SEED
    synonym_sample_ids = df["Video_ID"].sample(150, random_state=SEED).tolist()
    print(f"Synonym-substitution test set: {len(synonym_sample_ids)} videos")

    results = {}
    for task_name in ["dominant_tactic", "platform", "relevance", "family"]:
        task = TASKS.get(task_name)
        print(f"\n=== {task_name} ===")

        texts_all = [all_texts[vid] for vid in df["Video_ID"]]
        vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)
        model = fit_task_model(df, task, texts_all, vec)

        def asr_perturb(text, _rng=random.Random(rng_seed_base)):
            return inject_asr_noise(text, deletion_rate=0.08, rng=_rng)

        asr_result = sweep(df, task_name, all_texts, asr_clean_ids, asr_perturb, vec, model, task,
                            "simulated_asr_noise")
        print(f"  ASR-noise:  flip_rate={asr_result['flip_rate']:.1%}  "
              f"mean_abs_drift={asr_result['mean_abs_drift']:.3f}  (n={asr_result['n']})")

        syn_result = sweep(df, task_name, all_texts, synonym_sample_ids, inject_synonyms, vec, model, task,
                            "synonym_substitution")
        print(f"  Synonyms:   flip_rate={syn_result['flip_rate']:.1%}  "
              f"mean_abs_drift={syn_result['mean_abs_drift']:.3f}  (n={syn_result['n']})")

        results[task_name] = {"asr_noise": asr_result, "synonym_substitution": syn_result}

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
