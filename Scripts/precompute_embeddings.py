"""
One-off precompute: mean-pooled sentence embeddings (all-MiniLM-L6-v2,
384-dim) for every transcript in the corpus, cached to disk so downstream
scripts (Phase 4's Mahalanobis-distance OOD method) can load a plain
.npz file from the default interpreter instead of needing torch.

Must be run with the dedicated .venv-embeddings interpreter:
    .venv-embeddings/bin/python3 Scripts/precompute_embeddings.py

Reuses rk_pipeline/features.py's EmbeddingFeatureBuilder chunking logic
exactly, so vectors here are identical to what the embedding feature
representation produces elsewhere in the pipeline.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rk_pipeline.data import load_features  # noqa: E402
from rk_pipeline.features import _chunk_words, _get_embedding_model, _EMBEDDING_CHUNK_WORDS  # noqa: E402
from rk_pipeline.data import load_transcript  # noqa: E402

OUT = ROOT / "outputs" / "embedding_cache_1034.npz"


def main():
    df = load_features()
    model = _get_embedding_model()
    print(f"Encoding {len(df)} transcripts with all-MiniLM-L6-v2...")

    vecs = []
    for i, p in enumerate(df["Transcript_Path"]):
        text = load_transcript(p)
        chunks = _chunk_words(text, _EMBEDDING_CHUNK_WORDS)
        chunk_vecs = model.encode(chunks, show_progress_bar=False)
        vecs.append(np.asarray(chunk_vecs).mean(axis=0))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(df)}")

    emb = np.vstack(vecs).astype(np.float32)
    np.savez(OUT, video_id=df["Video_ID"].to_numpy(), embedding=emb)
    print(f"Wrote {OUT.relative_to(ROOT)}  shape={emb.shape}")


if __name__ == "__main__":
    main()
