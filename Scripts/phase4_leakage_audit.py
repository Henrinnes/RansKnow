"""
RansKnow Phase 4 -- leakage and confound audit for the OOD results.

Checks whether the AUROCs in outputs/phase4_ood.json depend on information
shared between training and evaluation populations:

1. Direct overlap: Video_IDs and YouTube URLs shared between train, ID test,
   and OOD in each setup.
2. Channel overlap: Phase 4 splits ID videos at random, so ID-test videos
   can share a channel with training videos. Reports how often, then reruns
   each setup with ID channels held out, over several seeds, next to the
   same seeds under the random split.
3. Near-duplicates: transcript embedding pairs above a cosine threshold
   (the same talk re-uploaded to another channel), then a rerun with ID-test
   and OOD videos dropped when they have a near-duplicate in training.
4. Family-name contamination: ID videos whose raw text contains an OOD
   family name even though the context-gated regex did not tag them.
5. External-corpus length: 20 Newsgroups posts are far shorter than
   transcripts, and mean-pooling few chunks leaves a vector farther from the
   class means. Re-embeds ID-test videos truncated to one chunk.
6. Timestamp formatting: every stored transcript contains timestamp lines
   ("0:05"), which reach both TF-IDF and the embeddings, while 20 Newsgroups
   has none. Re-embeds the corpus with them stripped and reruns every setup
   on stripped text.

Usage:
    python3 Scripts/phase4_leakage_audit.py [--stripped-cache path.npz]

--stripped-cache reuses or stores the timestamp-free embeddings, which take
a while to compute on a CPU.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import phase4_ood_detection as p4  # noqa: E402
from rk_pipeline import features as rk_features  # noqa: E402
from rk_pipeline.data import load_features, load_transcript  # noqa: E402
from rk_pipeline.features import _chunk_words, _get_embedding_model, _EMBEDDING_CHUNK_WORDS  # noqa: E402
from rk_pipeline.tasks import register_tasks  # noqa: E402

OUT = ROOT / "outputs" / "phase4_leakage_audit.json"
SEEDS = [0, 1, 2, 3, 4]
NEAR_DUP_COSINE = 0.98
MIN_UNSEEN_OOD = 5
TIMESTAMP_LINE = re.compile(r"^\s*\d+:\d\d(:\d\d)?\s*$", re.M)
FAMILY_TERMS = {
    "temporal_emergence": r"\b(maze|medusa|black\s?basta|inc\s+ransom|cactus)\b",
    "leave_rare_family_out": r"\b(cl0p|clop)\b",
}


def populations(df):
    """(ID pool, OOD) per within-corpus setup, as in phase4_ood_detection.py."""
    pre = df[df["Year"] < 2024].reset_index(drop=True)
    post = df[df["Year"] >= 2024].reset_index(drop=True)
    mask = np.zeros(len(post), dtype=bool)
    for fam in p4.POST2023_EMERGING_FAMILIES:
        mask |= p4._has_family(post, fam).to_numpy()
    rare = p4._has_family(df, p4.RARE_FAMILY)
    return {
        "temporal_emergence": (pre, post[mask].reset_index(drop=True)),
        "channel_type_shift": (
            df[df["Channel_Type"].isin(["Vendor", "DFIR"])].reset_index(drop=True),
            df[df["Channel_Type"].isin(["Conference", "Independent"])].reset_index(drop=True),
        ),
        "leave_rare_family_out": (df[~rare].reset_index(drop=True), df[rare].reset_index(drop=True)),
    }


def split(pool, seed, by_channel):
    if by_channel:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        train_idx, test_idx = next(splitter.split(pool, groups=pool["Channel_ID"]))
    else:
        train_idx, test_idx = train_test_split(np.arange(len(pool)), test_size=0.25, random_state=seed)
    return pool.iloc[train_idx].reset_index(drop=True), pool.iloc[test_idx].reset_index(drop=True)


def score(train_df, id_test_df, ood_df, embeddings):
    est, builder, _ = p4._fit_classifier(train_df)
    id_msp, id_energy, _ = p4._msp_and_energy(est, builder, id_test_df)
    ood_msp, ood_energy, _ = p4._msp_and_energy(est, builder, ood_df)
    id_maha, ood_maha = p4._mahalanobis_setup(train_df, id_test_df, ood_df, est, builder, embeddings)
    return {
        "msp": p4._auroc(id_msp, ood_msp),
        "energy": p4._auroc(id_energy, ood_energy),
        "mahalanobis": p4._auroc(id_maha, ood_maha),
        "n_id_test": len(id_test_df), "n_ood": len(ood_df),
    }


def summarize(runs):
    return {
        m: {"mean": float(np.mean([r[m] for r in runs])), "min": float(np.min([r[m] for r in runs])),
            "max": float(np.max([r[m] for r in runs]))}
        for m in ["msp", "energy", "mahalanobis"]
    }


def near_duplicates(df, embeddings):
    ids = df["Video_ID"].tolist()
    E = np.stack([embeddings[v] for v in ids])
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    sims = E @ E.T
    np.fill_diagonal(sims, 0)
    partners = {ids[i]: {ids[j] for j in np.where(sims[i] >= NEAR_DUP_COSINE)[0]} for i in range(len(ids))}
    channel = dict(zip(df["Video_ID"], df["Channel_ID"]))
    pairs = {tuple(sorted((a, b))) for a, bs in partners.items() for b in bs}
    return partners, {
        "cosine_threshold": NEAR_DUP_COSINE,
        "n_pairs": len(pairs),
        "n_pairs_across_channels": sum(channel[a] != channel[b] for a, b in pairs),
        "n_videos_involved": sum(bool(v) for v in partners.values()),
    }


def stripped_transcript(rel_path):
    return TIMESTAMP_LINE.sub("", load_transcript(rel_path))


def stripped_embeddings(df, model, cache_path):
    if cache_path and cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        return pd.Series(list(d["embedding"]), index=d["video_id"])
    vecs = []
    for path in df["Transcript_Path"]:
        chunks = _chunk_words(stripped_transcript(path), _EMBEDDING_CHUNK_WORDS)
        vecs.append(np.asarray(model.encode(chunks, show_progress_bar=False)).mean(axis=0))
    if cache_path:
        np.savez_compressed(cache_path, video_id=df["Video_ID"].to_numpy(), embedding=np.stack(vecs))
    return pd.Series(vecs, index=df["Video_ID"].to_numpy())


def main(cache_path=None):
    df = load_features()
    register_tasks(df)
    embeddings = p4._load_embeddings()
    results = {}

    urls = df["YouTube_URL"].value_counts()
    partners, dup_summary = near_duplicates(df, embeddings)
    results["near_duplicates"] = dup_summary
    results["duplicate_youtube_urls"] = int((urls > 1).sum())
    print("Near-duplicates:", dup_summary, "duplicate URLs:", results["duplicate_youtube_urls"])

    for name, (pool, ood) in populations(df).items():
        print(f"\n=== {name} ===")
        train_df, id_test_df = split(pool, p4.SEED, by_channel=False)
        train_ids, test_ids, ood_ids = set(train_df["Video_ID"]), set(id_test_df["Video_ID"]), set(ood["Video_ID"])
        train_channels = set(train_df["Channel_ID"])
        entry = {
            "video_overlap": {"train_test": len(train_ids & test_ids), "train_ood": len(train_ids & ood_ids),
                              "test_ood": len(test_ids & ood_ids)},
            "url_overlap_train_ood": len(set(train_df["YouTube_URL"]) & set(ood["YouTube_URL"])),
            "share_id_test_in_train_channels": float(id_test_df["Channel_ID"].isin(train_channels).mean()),
            "share_ood_in_train_channels": float(ood["Channel_ID"].isin(train_channels).mean()),
        }

        random_runs, channel_runs, unseen_runs = [], [], []
        for seed in SEEDS:
            random_runs.append(score(*split(pool, seed, False), ood, embeddings))
            channel_train, channel_test = split(pool, seed, True)
            channel_runs.append(score(channel_train, channel_test, ood, embeddings))
            # Same held-out ID channels, with OOD also limited to channels the
            # model never trained on, so neither side shares a channel with it.
            unseen = ood[~ood["Channel_ID"].isin(set(channel_train["Channel_ID"]))].reset_index(drop=True)
            if len(unseen) >= MIN_UNSEEN_OOD:
                unseen_runs.append(score(channel_train, channel_test, unseen, embeddings))
        entry["random_split_5_seeds"] = summarize(random_runs)
        entry["channel_held_out_id_5_seeds"] = summarize(channel_runs)
        entry["n_seeds_with_enough_unseen_channel_ood"] = len(unseen_runs)
        if unseen_runs:
            entry["channel_held_out_both_sides"] = summarize(unseen_runs)
            entry["channel_held_out_both_sides_n_ood"] = [r["n_ood"] for r in unseen_runs]

        keep = lambda d: d[~d["Video_ID"].apply(lambda v: bool(partners[v] & train_ids))].reset_index(drop=True)
        deduped_test, deduped_ood = keep(id_test_df), keep(ood)
        entry["near_duplicates_removed"] = score(train_df, deduped_test, deduped_ood, embeddings)

        if name in FAMILY_TERMS:
            pattern = re.compile(FAMILY_TERMS[name], re.I)
            hits = pool["Transcript_Path"].apply(lambda p: bool(pattern.search(load_transcript(p))))
            entry["id_videos_with_raw_family_term"] = {"n": int(hits.sum()), "of": len(pool)}
            if hits.sum():
                clean_train, clean_test = split(pool[~hits].reset_index(drop=True), p4.SEED, False)
                entry["contaminated_id_removed"] = score(clean_train, clean_test, ood, embeddings)

        print(json.dumps(entry, indent=1))
        results[name] = entry

    print("\n=== external_corpus length control ===")
    model = _get_embedding_model()
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.25, random_state=p4.SEED)
    train_df, id_test_df = df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
    ext = np.load(p4.EXTERNAL_EMBED_CACHE)
    est, builder, _ = p4._fit_classifier(train_df)
    train_pred = est.predict(builder.transform(train_df))
    E_train = np.stack([embeddings[v] for v in train_df["Video_ID"]])
    E_full = np.stack([embeddings[v] for v in id_test_df["Video_ID"]])
    first_chunks = [_chunk_words(load_transcript(p), _EMBEDDING_CHUNK_WORDS)[0] for p in id_test_df["Transcript_Path"]]
    E_short = np.asarray(model.encode(first_chunks, show_progress_bar=False))
    id_full, ood = p4._mahalanobis_scores(E_train, train_pred, E_full, ext["embedding"])
    id_short, ood = p4._mahalanobis_scores(E_train, train_pred, E_short, ext["embedding"])
    from sklearn.datasets import fetch_20newsgroups
    news = fetch_20newsgroups(subset="train", categories=["rec.sport.baseball", "sci.space"],
                              remove=("headers", "footers", "quotes"))
    results["external_corpus"] = {
        "median_words_ood": float(np.median([len(news.data[i].split()) for i in ext["doc_index"]])),
        "median_words_id": float(np.median([len(load_transcript(p).split()) for p in id_test_df["Transcript_Path"]])),
        "auroc_full_transcripts": p4._auroc(id_full, ood),
        "auroc_id_truncated_to_one_chunk": p4._auroc(id_short, ood),
        "auroc_id_truncated_to_one_chunk_ci95": p4._auroc_ci(id_short, ood),
        "auroc_truncated_id_vs_full_id": p4._auroc(id_full, id_short),
    }
    print(json.dumps(results["external_corpus"], indent=1))
    OUT.write_text(json.dumps(results, indent=2))

    print("\n=== timestamps stripped ===")
    stripped = stripped_embeddings(df, model, cache_path)
    rk_features.load_transcript = stripped_transcript
    p4.load_transcript = stripped_transcript
    timestamp_free = {}
    for name, (pool, ood) in populations(df).items():
        timestamp_free[name] = score(*split(pool, p4.SEED, False), ood, stripped)
    E_short = np.asarray(model.encode(
        [_chunk_words(stripped_transcript(p), _EMBEDDING_CHUNK_WORDS)[0] for p in id_test_df["Transcript_Path"]],
        show_progress_bar=False))
    est, builder, _ = p4._fit_classifier(train_df)
    train_pred = est.predict(builder.transform(train_df))
    E_train = np.stack([stripped[v] for v in train_df["Video_ID"]])
    E_full = np.stack([stripped[v] for v in id_test_df["Video_ID"]])
    id_full, ood_full = p4._mahalanobis_scores(E_train, train_pred, E_full, ext["embedding"])
    id_short, ood_short = p4._mahalanobis_scores(E_train, train_pred, E_short, ext["embedding"])
    timestamp_free["external_corpus"] = {
        "auroc_full_transcripts": p4._auroc(id_full, ood_full),
        "auroc_id_truncated_to_one_chunk": p4._auroc(id_short, ood_short),
        "auroc_id_truncated_to_one_chunk_ci95": p4._auroc_ci(id_short, ood_short),
    }
    results["timestamps_stripped"] = timestamp_free
    print(json.dumps(timestamp_free, indent=1))

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stripped-cache", type=Path)
    main(parser.parse_args().stripped_cache)
