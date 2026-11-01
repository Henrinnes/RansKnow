"""
One-off precompute: mean-pooled sentence embeddings (all-MiniLM-L6-v2,
384-dim) for the 20 Newsgroups documents Phase 4 uses as its external
corpus, so Mahalanobis distance can run on the sanity-check setup too.
Same chunking as Scripts/precompute_embeddings.py. Cached to disk so
Phase 4 needs no torch at detection time.

Documents left empty after header/footer/quote removal are dropped, since
they are not text a detector could be tested on.

Also re-embeds a few RansKnow transcripts and compares them against
outputs/embedding_cache_1034.npz, to catch a library upgrade changing the
embedding space.

Usage:
    python3 Scripts/precompute_external_embeddings.py
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_20newsgroups

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rk_pipeline.data import load_features, load_transcript  # noqa: E402
from rk_pipeline.features import _chunk_words, _get_embedding_model, _EMBEDDING_CHUNK_WORDS  # noqa: E402

OUT = ROOT / "outputs" / "embedding_cache_20newsgroups.npz"
RANSKNOW_CACHE = ROOT / "outputs" / "embedding_cache_1034.npz"
N_CONSISTENCY_CHECK = 5


def embed(text, model):
    chunks = _chunk_words(text, _EMBEDDING_CHUNK_WORDS)
    return np.asarray(model.encode(chunks, show_progress_bar=False)).mean(axis=0)


def main():
    model = _get_embedding_model()

    cached = np.load(RANSKNOW_CACHE, allow_pickle=True)
    by_id = dict(zip(cached["video_id"], cached["embedding"]))
    df = load_features().head(N_CONSISTENCY_CHECK)
    sims = []
    for vid, path in zip(df["Video_ID"], df["Transcript_Path"]):
        a, b = embed(load_transcript(path), model), by_id[vid]
        sims.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
    print(f"Cosine similarity to cached RansKnow embeddings: min={min(sims):.6f}")
    assert min(sims) > 0.999, "Embedding space differs from outputs/embedding_cache_1034.npz"

    news = fetch_20newsgroups(subset="train", categories=["rec.sport.baseball", "sci.space"],
                              remove=("headers", "footers", "quotes"))
    keep = [i for i, t in enumerate(news.data) if t.strip()]
    print(f"Embedding {len(keep)} documents ({len(news.data) - len(keep)} empty ones dropped)...")
    vecs = np.stack([embed(news.data[i], model) for i in keep])
    np.savez_compressed(OUT, doc_index=np.array(keep), embedding=vecs.astype(np.float32))
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
