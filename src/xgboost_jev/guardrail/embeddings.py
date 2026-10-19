"""Text embeddings reduced to a few dense columns XGBoost can split on.

The default ``HashingSVDEmbedder`` uses char n-gram hashing plus TruncatedSVD
(PCA for sparse input): no model download, well under 1 ms per text because
the projection only gathers the rows of the component matrix a text actually hits. Any object with
``fit``/``transform`` can be swapped in, e.g. a sentence-transformer + PCA.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer


class HashingSVDEmbedder:
    def __init__(self, n_components: int = 32, n_features: int = 2**18, seed: int = 0) -> None:
        self.n_components = n_components
        self.vectorizer = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=seed)
        self._fitted = False

    @property
    def columns(self) -> list[str]:
        return [f"emb_{i}" for i in range(self.n_components)]

    def fit(self, texts: Sequence[str]) -> HashingSVDEmbedder:
        self.svd.fit(self.vectorizer.transform(texts))
        # (n_features, k) row-major so a sparse row only touches its non-zero columns.
        self._proj = np.ascontiguousarray(self.svd.components_.T, dtype=np.float32)
        self._fitted = True
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("embedder must be fit before transform")
        x = self.vectorizer.transform(texts).tocsr()
        out = np.empty((x.shape[0], self.n_components), dtype=np.float32)
        for i in range(x.shape[0]):
            lo, hi = x.indptr[i], x.indptr[i + 1]
            out[i] = x.data[lo:hi].astype(np.float32) @ self._proj[x.indices[lo:hi]]
        return out
