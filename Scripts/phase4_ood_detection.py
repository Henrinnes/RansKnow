"""
RansKnow Phase 4 -- out-of-distribution (OOD) detection.

Unlike Phases 1-3 (paper Sections 5-6), this phase's results are NOT
subject to the weak-label scope note (Section sec:gold-scope): OOD
detection asks whether a model's own confidence or embedding-space
behavior differs between in-distribution (ID) and out-of-distribution
(OOD) input, which is answerable without knowing whether either input's
*label* is correct. Ground truth here is "which population did this
input come from," which is known by construction in every setup below.

Four setups, each with a clean, known ID/OOD split:

1. temporal_emergence -- ID: pre-2024 videos. OOD: post-2024 videos
   mentioning a ransomware family that never appears before 2024 in this
   corpus (Maze, Medusa, Black Basta, INC, Cactus -- verified against
   the actual per-family first-occurrence year, not assumed).
2. channel_type_shift -- ID: Vendor+DFIR channels. OOD: Conference+
   Independent channels.
3. leave_rare_family_out -- ID: videos never mentioning "Cl0p" (11
   occurrences total, a genuinely long-tailed family, not a major one
   like Conti/LockBit/Qilin/Akira which all have 19+). OOD: the 11
   Cl0p-mentioning videos.
4. external_corpus -- ID: RansKnow transcripts. OOD: an unrelated public
   text corpus (20 Newsgroups, rec.sport.baseball + sci.space), included
   as a sanity check that the methods below detect an obviously-OOD
   input at all before trusting them on the harder setups 1-3.

Four detection methods, all built from a single shared classifier
(dominant_tactic, structured features, LightGBM -- the paper's
strongest non-circular result and a natural multiclass fit), refit
fresh on each setup's ID-only training pool:

- max_softmax_probability (MSP): score = 1 - max_c p(c|x). Higher = more OOD.
- energy: score = -logsumexp(raw pre-softmax margins). Higher = more OOD.
  (Computed from real decision-function/raw-score margins, not from
  predict_proba -- probabilities already sum to 1, which makes the
  energy score degenerate to a constant if computed from them.)
- mahalanobis: class-conditional Gaussian Mahalanobis distance on real
  384-dim mean-pooled sentence embeddings (all-MiniLM-L6-v2, precomputed
  via Scripts/precompute_embeddings.py under .venv-embeddings and cached
  to outputs/embedding_cache_1034.npz -- loaded here as a plain array,
  no torch dependency at OOD-detection time). Score = min over classes
  of Mahalanobis distance to that class's ID-train mean, using the
  pooled ID-train covariance (Lee et al. 2018).
- conformal_nonconformity: Section 6's LAC score, s(x) = 1 - p(predicted
  class | x), calibrated to a 90%-target-coverage threshold q_hat on a
  held-out ID calibration split. Reported two ways: AUROC (for direct
  comparability with the other three methods -- note this is
  mathematically the same quantity as MSP evaluated at the top class,
  which is an expected, not a bug, equivalence) and the operationally
  distinct number: exceedance rate above q_hat (ID should be ~10% by
  construction; OOD populations exceeding well above that rate is the
  conformal-specific signal).

For setups 1-3, "external_corpus" for that setup's classifier is the
withheld RansKnow test split, not 20 Newsgroups -- see external_corpus
setup for the cross-corpus check specifically.

Usage:
    python3 Scripts/phase4_ood_detection.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rk_pipeline.data import load_features, load_transcript  # noqa: E402
from rk_pipeline.features import FEATURES  # noqa: E402
from rk_pipeline.models import MODELS  # noqa: E402
from rk_pipeline.tasks import register_tasks, TASKS  # noqa: E402

OUT = ROOT / "outputs" / "phase4_ood.json"
EMBED_CACHE = ROOT / "outputs" / "embedding_cache_1034.npz"
ALPHA = 0.10
SEED = 0

TASK_NAME, FEATURE_NAME = "dominant_tactic", "structured"

POST2023_EMERGING_FAMILIES = ["Maze", "Medusa", "Black Basta", "INC", "Cactus"]
RARE_FAMILY = "Cl0p"


def _load_embeddings():
    d = np.load(EMBED_CACHE, allow_pickle=True)
    return pd.Series(list(d["embedding"]), index=d["video_id"])


def _has_family(df, name):
    return df["Family_List"].fillna("").apply(lambda s: name in [f for f in s.split(", ") if f])


def _fit_classifier(train_df):
    """Fresh dominant_tactic/structured/GBM fit on an ID-only pool.
    Returns (estimator, feature_builder, classes)."""
    task = TASKS.get(TASK_NAME)
    builder = FEATURES.get(FEATURE_NAME)()
    X_train = builder.fit_transform(train_df)
    y_train = task.build_y(train_df)
    est = MODELS.get("gbm").build(task.label_type, task)
    est.fit(X_train, y_train)
    classes = list(est.classes_)
    return est, builder, classes


def _raw_margins(est, X):
    """Pre-softmax margins for LightGBM multiclass -- required for a
    non-degenerate energy score (see module docstring)."""
    booster = est.booster_
    raw = booster.predict(X, raw_score=True)  # (n, n_classes)
    return raw


def _msp_and_energy(est, builder, df):
    X = builder.transform(df)
    proba = est.predict_proba(X)
    msp_score = 1 - proba.max(axis=1)
    raw = _raw_margins(est, X)
    energy_score = -logsumexp(raw, axis=1)
    return msp_score, energy_score, proba


def _mahalanobis_setup(train_df, id_test_df, ood_df, est, builder, embeddings):
    """Class-conditional Gaussian Mahalanobis distance on real sentence
    embeddings, using the shared classifier's predicted class on the
    ID-train pool to define class-conditional groups (Lee et al. 2018),
    since these tasks don't have independently-verified labels either
    (see scope note) -- using predicted rather than weak-label classes
    keeps this method's assumptions consistent with the rest of the paper."""
    def emb_matrix(d):
        return np.stack([embeddings[vid] for vid in d["Video_ID"]])

    train_pred = est.predict(builder.transform(train_df))
    E_train = emb_matrix(train_df)

    classes = np.unique(train_pred)
    means = {c: E_train[train_pred == c].mean(axis=0) for c in classes if (train_pred == c).sum() > 1}
    centered = np.vstack([E_train[train_pred == c] - means[c] for c in means])
    cov = np.cov(centered, rowvar=False) + 1e-6 * np.eye(centered.shape[1])
    cov_inv = np.linalg.pinv(cov)

    def score(d):
        E = emb_matrix(d)
        dists = np.stack([
            np.einsum("ij,jk,ik->i", E - mu, cov_inv, E - mu) for mu in means.values()
        ], axis=1)
        return dists.min(axis=1)

    return score(id_test_df), score(ood_df)


def _conformal_flags(train_df, calib_df, id_test_df, ood_df, est, builder):
    task = TASKS.get(TASK_NAME)
    classes = list(est.classes_)

    def scores_for(d):
        X = builder.transform(d)
        proba = est.predict_proba(X)
        return 1 - proba.max(axis=1)  # score at PREDICTED class -- the OOD-flagging use, see docstring

    y_calib = task.build_y(calib_df)
    X_calib = builder.transform(calib_df)
    proba_calib = est.predict_proba(X_calib)
    true_score = np.array([
        1 - proba_calib[i, classes.index(y_calib[i])] if y_calib[i] in classes else 1.0
        for i in range(len(calib_df))
    ])
    n = len(true_score)
    level = min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)
    qhat = float(np.quantile(true_score, level, method="higher"))

    id_scores = scores_for(id_test_df)
    ood_scores = scores_for(ood_df)
    return id_scores, ood_scores, qhat


def _auroc(id_scores, ood_scores):
    y = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    s = np.concatenate([id_scores, ood_scores])
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return None


def run_setup(name, train_df, id_test_df, ood_df, embeddings):
    print(f"\n=== {name} === (ID train n={len(train_df)}, ID test n={len(id_test_df)}, OOD n={len(ood_df)})")
    est, builder, classes = _fit_classifier(train_df)

    id_msp, id_energy, _ = _msp_and_energy(est, builder, id_test_df)
    ood_msp, ood_energy, _ = _msp_and_energy(est, builder, ood_df)

    auroc_msp = _auroc(id_msp, ood_msp)
    auroc_energy = _auroc(id_energy, ood_energy)
    print(f"  MSP    AUROC={auroc_msp}")
    print(f"  Energy AUROC={auroc_energy}")

    id_maha, ood_maha = _mahalanobis_setup(train_df, id_test_df, ood_df, est, builder, embeddings)
    auroc_maha = _auroc(id_maha, ood_maha)
    print(f"  Mahalanobis AUROC={auroc_maha}")

    train_idx, calib_idx = train_test_split(np.arange(len(train_df)), test_size=0.25, random_state=SEED)
    calib_df = train_df.iloc[calib_idx].reset_index(drop=True)
    fit_df = train_df.iloc[train_idx].reset_index(drop=True)
    est2, builder2, _ = _fit_classifier(fit_df)
    id_conf, ood_conf, qhat = _conformal_flags(fit_df, calib_df, id_test_df, ood_df, est2, builder2)
    auroc_conf = _auroc(id_conf, ood_conf)
    id_exceed = float((id_conf > qhat).mean())
    ood_exceed = float((ood_conf > qhat).mean())
    print(f"  Conformal AUROC={auroc_conf}  qhat={qhat:.3f}  "
          f"ID_exceed_rate={id_exceed:.3f}  OOD_exceed_rate={ood_exceed:.3f}")

    return {
        "n_id_train": len(train_df), "n_id_test": len(id_test_df), "n_ood": len(ood_df),
        "auroc": {"max_softmax_probability": auroc_msp, "energy": auroc_energy,
                  "mahalanobis": auroc_maha, "conformal_nonconformity": auroc_conf},
        "conformal_qhat": qhat, "conformal_id_exceed_rate": id_exceed, "conformal_ood_exceed_rate": ood_exceed,
    }


def setup_temporal_emergence(df, embeddings):
    pre2024 = df[df["Year"] < 2024].reset_index(drop=True)
    post2024 = df[df["Year"] >= 2024].reset_index(drop=True)
    ood_mask = np.zeros(len(post2024), dtype=bool)
    for f in POST2023_EMERGING_FAMILIES:
        ood_mask |= _has_family(post2024, f).to_numpy()
    ood_df = post2024[ood_mask].reset_index(drop=True)

    train_idx, test_idx = train_test_split(np.arange(len(pre2024)), test_size=0.25, random_state=SEED)
    train_df = pre2024.iloc[train_idx].reset_index(drop=True)
    id_test_df = pre2024.iloc[test_idx].reset_index(drop=True)
    return run_setup("temporal_emergence", train_df, id_test_df, ood_df, embeddings)


def setup_channel_type_shift(df, embeddings):
    id_pool = df[df["Channel_Type"].isin(["Vendor", "DFIR"])].reset_index(drop=True)
    ood_df = df[df["Channel_Type"].isin(["Conference", "Independent"])].reset_index(drop=True)

    train_idx, test_idx = train_test_split(np.arange(len(id_pool)), test_size=0.25, random_state=SEED)
    train_df = id_pool.iloc[train_idx].reset_index(drop=True)
    id_test_df = id_pool.iloc[test_idx].reset_index(drop=True)
    return run_setup("channel_type_shift", train_df, id_test_df, ood_df, embeddings)


def setup_leave_rare_family_out(df, embeddings):
    is_rare = _has_family(df, RARE_FAMILY)
    id_pool = df[~is_rare].reset_index(drop=True)
    ood_df = df[is_rare].reset_index(drop=True)

    train_idx, test_idx = train_test_split(np.arange(len(id_pool)), test_size=0.25, random_state=SEED)
    train_df = id_pool.iloc[train_idx].reset_index(drop=True)
    id_test_df = id_pool.iloc[test_idx].reset_index(drop=True)
    return run_setup("leave_rare_family_out", train_df, id_test_df, ood_df, embeddings)


def setup_external_corpus(df, embeddings):
    """Sanity check only -- uses a dedicated TF-IDF classifier (not the
    shared structured-feature one) since 20 Newsgroups text has no
    Knowledge Agent structured columns. Mahalanobis/conformal need the
    same fitted classifier's embedding/probability space, so this setup
    reimplements MSP/energy/conformal on TF-IDF and skips Mahalanobis
    (20 Newsgroups has no precomputed sentence embedding in the cache)."""
    print(f"\n=== external_corpus === (sanity check: RansKnow vs 20 Newsgroups)")
    task = TASKS.get(TASK_NAME)
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.25, random_state=SEED)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    id_test_df = df.iloc[test_idx].reset_index(drop=True)

    vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)
    train_texts = [load_transcript(p) for p in train_df["Transcript_Path"]]
    X_train = vec.fit_transform(train_texts)
    y_train = task.build_y(train_df)
    est = MODELS.get("gbm").build(task.label_type, task)
    est.fit(X_train, y_train)

    id_texts = [load_transcript(p) for p in id_test_df["Transcript_Path"]]
    X_id = vec.transform(id_texts)

    news = fetch_20newsgroups(subset="train", categories=["rec.sport.baseball", "sci.space"],
                               remove=("headers", "footers", "quotes"))
    ood_texts = list(news.data)
    X_ood = vec.transform(ood_texts)

    proba_id = est.predict_proba(X_id)
    proba_ood = est.predict_proba(X_ood)
    id_msp, ood_msp = 1 - proba_id.max(axis=1), 1 - proba_ood.max(axis=1)
    raw_id = _raw_margins(est, X_id)
    raw_ood = _raw_margins(est, X_ood)
    id_energy, ood_energy = -logsumexp(raw_id, axis=1), -logsumexp(raw_ood, axis=1)

    auroc_msp = _auroc(id_msp, ood_msp)
    auroc_energy = _auroc(id_energy, ood_energy)
    print(f"  MSP    AUROC={auroc_msp}")
    print(f"  Energy AUROC={auroc_energy}")

    fit_idx, calib_idx = train_test_split(np.arange(len(train_df)), test_size=0.25, random_state=SEED)
    fit_texts = [train_texts[i] for i in fit_idx]
    calib_texts = [train_texts[i] for i in calib_idx]
    vec2 = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)
    X_fit = vec2.fit_transform(fit_texts)
    y_fit = task.build_y(train_df.iloc[fit_idx].reset_index(drop=True))
    est2 = MODELS.get("gbm").build(task.label_type, task)
    est2.fit(X_fit, y_fit)
    classes2 = list(est2.classes_)

    X_calib = vec2.transform(calib_texts)
    y_calib = task.build_y(train_df.iloc[calib_idx].reset_index(drop=True))
    proba_calib = est2.predict_proba(X_calib)
    true_score = np.array([
        1 - proba_calib[i, classes2.index(y_calib[i])] if y_calib[i] in classes2 else 1.0
        for i in range(len(calib_idx))
    ])
    n = len(true_score)
    level = min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)
    qhat = float(np.quantile(true_score, level, method="higher"))

    X_id2 = vec2.transform(id_texts)
    X_ood2 = vec2.transform(ood_texts)
    id_conf = 1 - est2.predict_proba(X_id2).max(axis=1)
    ood_conf = 1 - est2.predict_proba(X_ood2).max(axis=1)
    auroc_conf = _auroc(id_conf, ood_conf)
    id_exceed = float((id_conf > qhat).mean())
    ood_exceed = float((ood_conf > qhat).mean())
    print(f"  Conformal AUROC={auroc_conf}  qhat={qhat:.3f}  "
          f"ID_exceed_rate={id_exceed:.3f}  OOD_exceed_rate={ood_exceed:.3f}")

    return {
        "n_id_train": len(train_df), "n_id_test": len(id_test_df), "n_ood": len(ood_texts),
        "auroc": {"max_softmax_probability": auroc_msp, "energy": auroc_energy,
                  "mahalanobis": None, "conformal_nonconformity": auroc_conf},
        "conformal_qhat": qhat, "conformal_id_exceed_rate": id_exceed, "conformal_ood_exceed_rate": ood_exceed,
        "note": "Mahalanobis skipped -- 20 Newsgroups has no precomputed sentence embedding "
                "in outputs/embedding_cache_1034.npz (RansKnow-only cache).",
    }


def main():
    df = load_features()
    register_tasks(df)
    embeddings = _load_embeddings()

    print("Verifying family first-occurrence years (must match module docstring claims)...")
    fams = df[["Year", "Family_List"]].dropna(subset=["Family_List"])
    exploded = fams.assign(Family=fams["Family_List"].str.split(", ")).explode("Family")
    exploded = exploded[exploded["Family"] != ""]
    first_year = exploded.groupby("Family")["Year"].min()
    for f in POST2023_EMERGING_FAMILIES:
        assert first_year.get(f, 9999) >= 2024, f"{f} first appears {first_year.get(f)}, not >=2024 as assumed"
    print("  OK:", {f: int(first_year[f]) for f in POST2023_EMERGING_FAMILIES})

    results = {}
    results["temporal_emergence"] = setup_temporal_emergence(df, embeddings)
    results["channel_type_shift"] = setup_channel_type_shift(df, embeddings)
    results["leave_rare_family_out"] = setup_leave_rare_family_out(df, embeddings)
    results["external_corpus"] = setup_external_corpus(df, embeddings)

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
