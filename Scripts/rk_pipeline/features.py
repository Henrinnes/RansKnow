"""
Feature representation builders.

Each registered factory is a zero-arg callable returning a *fresh,
unfit* builder with .fit_transform(train_df) / .transform(df) -- so
every CV fold gets its own fit (critical for TF-IDF: fitting
vocabulary/IDF on the full dataset before splitting would leak test-set
vocabulary into training).

Embedding representation is a registered stub: sentence_transformers is
broken in this environment (torch 2.0.1 / transformers 4.57 mismatch --
`from .training_args import TrainingArguments` fails on import). Fix the
environment or swap to a raw `transformers.AutoModel` mean-pooling
implementation before using it.
"""

from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .data import load_transcript
from .registry import FEATURES

# Columns that are identifiers, free text, or label-adjacent -- excluding
# these from the structured representation avoids trivial leakage (e.g.
# using Family_List to predict Family, or Dominant_Tactic to predict itself).
STRUCTURED_EXCLUDE = {
    "Video_ID", "Channel_ID", "Channel_Name", "Video_Title", "YouTube_URL",
    "Transcript_Path", "Transcript_Provider", "Channel_Type",
    "Family_Count", "Family_List", "Tool_List",
    "Dominant_Tactic", "Platform_Signal",
}


class StructuredFeatureBuilder:
    """Existing Knowledge Agent numeric columns. Stateless (no leakage
    risk from column selection), but implements fit_transform/transform
    for interface uniformity with TfidfFeatureBuilder."""

    def __init__(self):
        self.columns: List[str] = []

    def _select(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.columns].fillna(0).to_numpy(dtype=float)

    def fit_transform(self, train_df: pd.DataFrame) -> np.ndarray:
        self.columns = [
            c for c in train_df.columns
            if c not in STRUCTURED_EXCLUDE and pd.api.types.is_numeric_dtype(train_df[c])
        ]
        return self._select(train_df)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self._select(df)


class TfidfFeatureBuilder:
    """1-2 gram TF-IDF over transcript text, fit on train fold only.

    max_features is capped at 3000, not the more typical 20000+ -- this
    representation is shared with RandomForest/LightGBM, and tree
    ensembles wrapped 21x (one per family, multilabel) over a
    high-dimensional sparse matrix is what made the first smoke-test run
    hang for 2+ hours of CPU time before being killed. 3000 keeps this
    usable for tree models; LogReg would tolerate a much wider matrix
    fine on its own if that's ever worth splitting into two variants.
    """

    def __init__(self, max_features=3000, ngram_range=(1, 2), min_df=2):
        self.vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df)

    def _texts(self, df: pd.DataFrame) -> List[str]:
        return [load_transcript(p) for p in df["Transcript_Path"]]

    def fit_transform(self, train_df: pd.DataFrame):
        return self.vec.fit_transform(self._texts(train_df))

    def transform(self, df: pd.DataFrame):
        return self.vec.transform(self._texts(df))


_EMBEDDING_MODEL = None  # lazy singleton -- loading is the expensive part

_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_EMBEDDING_CHUNK_WORDS = 200  # rough proxy for staying under the model's
                              # 256-token max sequence length; transcripts
                              # run well past that, so naive .encode() on
                              # the raw text would silently truncate to
                              # the opening ~200 words and throw away the
                              # rest of the video.


def _get_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError as e:
            raise ImportError(
                "sentence_transformers/torch not importable in the *current* "
                "interpreter. This representation requires the dedicated venv: "
                "run via .venv-embeddings/bin/python3, not the default "
                "environment (torch 2.0.1 there can't satisfy transformers' "
                "torch>=2.1 requirement -- see Scripts/rk_pipeline/features.py)."
            ) from e
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _EMBEDDING_MODEL = SentenceTransformer(_EMBEDDING_MODEL_NAME, device=device)
    return _EMBEDDING_MODEL


def _chunk_words(text: str, chunk_words: int):
    words = text.split()
    if not words:
        return [""]
    return [" ".join(words[i:i + chunk_words]) for i in range(0, len(words), chunk_words)]


_EMBEDDING_CACHE: dict = {}  # Transcript_Path -> np.ndarray(384,)


def _embed_document_cached(rel_path: str, model) -> np.ndarray:
    # The representation is stateless (no fitting), so the same document
    # produces the same vector regardless of which fold or model is
    # currently using it. Without this cache, a 5-fold split evaluated
    # across 2 models re-encodes every document up to 10x over -- this is
    # the embedding-representation analogue of the rule-based-baseline
    # caching fix in models.py, and without it a single task/split
    # smoke test didn't finish inside a 100s budget.
    if rel_path not in _EMBEDDING_CACHE:
        text = load_transcript(rel_path)
        chunks = _chunk_words(text, _EMBEDDING_CHUNK_WORDS)
        chunk_vecs = model.encode(chunks, show_progress_bar=False)
        _EMBEDDING_CACHE[rel_path] = np.asarray(chunk_vecs).mean(axis=0)
    return _EMBEDDING_CACHE[rel_path]


class EmbeddingFeatureBuilder:
    """Mean-pooled sentence embeddings over chunked transcript text
    (all-MiniLM-L6-v2, 384-dim). Chunked rather than encoded whole
    because the model's ~256-token limit would otherwise silently
    truncate every transcript to its opening ~200 words.

    Requires the dedicated .venv-embeddings/ virtualenv -- the default
    environment's torch 2.0.1 can't satisfy transformers' torch>=2.1
    requirement. Stateless (no fitting), same representation for train
    and test, so fit_transform/transform do the same thing.
    """

    def fit_transform(self, train_df: pd.DataFrame) -> np.ndarray:
        return self.transform(train_df)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        model = _get_embedding_model()
        return np.vstack([
            _embed_document_cached(p, model) for p in df["Transcript_Path"]
        ])


FEATURES.register("structured")(StructuredFeatureBuilder)
FEATURES.register("tfidf")(TfidfFeatureBuilder)
FEATURES.register("embedding")(EmbeddingFeatureBuilder)
