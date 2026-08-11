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


class EmbeddingFeatureBuilder:
    """Stub -- sentence_transformers import is currently broken in this
    environment. Raises on use rather than silently no-op."""

    def fit_transform(self, train_df: pd.DataFrame):
        raise NotImplementedError(
            "Embedding representation not available: sentence_transformers "
            "fails to import in this environment (torch 2.0.1 / transformers "
            "4.57 mismatch). Either pin compatible versions or implement "
            "mean-pooling directly over transformers.AutoModel."
        )

    def transform(self, df: pd.DataFrame):
        raise NotImplementedError("see fit_transform")


FEATURES.register("structured")(StructuredFeatureBuilder)
FEATURES.register("tfidf")(TfidfFeatureBuilder)
FEATURES.register("embedding")(EmbeddingFeatureBuilder)
