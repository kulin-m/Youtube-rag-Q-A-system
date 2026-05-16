"""
app/retriever.py
Hybrid FAISS + BM25 Retriever with Iterative Two-Pass Retrieval.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    text: str
    source_url: str
    source_title: str
    chunk_id: int
    timestamp: str = "" # NEW: Preserves timestamp data
    keywords: list = field(default_factory=list)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

@dataclass
class RetrievalResult:
    chunk: Chunk
    semantic_score: float
    keyword_score: float
    fusion_score: float

class HybridRetriever:
    def __init__(self, embedding_model: SentenceTransformer, semantic_weight: float = 0.7):
        self.model = embedding_model
        self.semantic_weight = semantic_weight
        self.keyword_weight = 1.0 - semantic_weight
        self.chunks: list = []
        self._index: Optional[faiss.IndexFlatIP] = None
        self._bm25: Optional[BM25Okapi] = None
        self._dim: int = 0

    def add_chunks(self, chunks: list):
        self.chunks.extend(chunks)
        self._build_index()

    def _build_index(self):
        if not self.chunks:
            return
        texts = [c.text for c in self.chunks]
        logger.info(f"[Retriever] Encoding {len(texts)} chunks …")
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )
        self._dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(embeddings.astype(np.float32))
        for i, chunk in enumerate(self.chunks):
            chunk.embedding = embeddings[i]
        tokenised = [text.lower().split() for text in texts]
        self._bm25 = BM25Okapi(tokenised)
        logger.info("[Retriever] Index built.")

    def retrieve(self, query: str, top_k: int = 8, source_filter: Optional[str] = None) -> list:
        if not self.chunks or self._index is None:
            return []

        n = len(self.chunks)
        k_search = min(n, max(top_k * 3, 20))

        q_embed = self.model.encode(
            query, convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32).reshape(1, -1)

        dense_scores, dense_indices = self._index.search(q_embed, k_search)
        dense_scores, dense_indices = dense_scores[0], dense_indices[0]

        dense_rank = {
            int(idx): (rank + 1, float(score))
            for rank, (idx, score) in enumerate(zip(dense_indices, dense_scores))
            if idx < n
        }

        bm25_scores = self._bm25.get_scores(query.lower().split())
        bm25_order = np.argsort(bm25_scores)[::-1][:k_search]
        bm25_rank = {
            int(idx): (rank + 1, float(bm25_scores[idx]))
            for rank, idx in enumerate(bm25_order)
        }

        k_rrf = 60
        results = []
        for idx in set(dense_rank) | set(bm25_rank):
            chunk = self.chunks[idx]
            if source_filter and chunk.source_url != source_filter:
                continue
            dense_r, d_score = dense_rank.get(idx, (k_search + 1, 0.0))
            bm25_r, b_score  = bm25_rank.get(idx, (k_search + 1, 0.0))
            fusion = (self.semantic_weight  / (k_rrf + dense_r) +
                      self.keyword_weight   / (k_rrf + bm25_r))
            results.append(RetrievalResult(
                chunk=chunk, semantic_score=d_score,
                keyword_score=b_score, fusion_score=fusion,
            ))

        results.sort(key=lambda r: r.fusion_score, reverse=True)
        return results[:top_k]

    def retrieve_diverse(self, query: str, top_k: int = 8, diversity_penalty: float = 0.2) -> list:
        candidates = self.retrieve(query, top_k=top_k * 4)
        seen: dict = {}
        diverse = []
        for res in candidates:
            src = res.chunk.source_url
            penalty = seen.get(src, 0) * diversity_penalty
            diverse.append(RetrievalResult(
                chunk=res.chunk,
                semantic_score=res.semantic_score,
                keyword_score=res.keyword_score,
                fusion_score=res.fusion_score - penalty,
            ))
            seen[src] = seen.get(src, 0) + 1
        diverse.sort(key=lambda r: r.fusion_score, reverse=True)
        return diverse[:top_k]

    @staticmethod
    def _second_pass_query(original_query: str, top_chunk_text: str) -> str:
        try:
            vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=100)
            vec.fit_transform([top_chunk_text])
            features = vec.get_feature_names_out()
            scores = vec.transform([top_chunk_text]).toarray()[0]
            ranked = sorted(zip(features, scores), key=lambda x: x[1], reverse=True)
            keywords = [kw for kw, sc in ranked[:4] if sc > 0.0]
            if keywords:
                return f"{original_query} {' '.join(keywords)}"
        except Exception as e:
            logger.warning(f"[Retriever] Second-pass query build failed: {e}")
        return original_query

    def iterative_retrieve(self, query: str, top_k: int = 8, multi_video: bool = False, diversity_penalty: float = 0.2) -> list:
        if multi_video:
            round1 = self.retrieve_diverse(query, top_k=top_k, diversity_penalty=diversity_penalty)
        else:
            round1 = self.retrieve(query, top_k=top_k)

        if not round1:
            return []

        top_text = round1[0].chunk.text
        followup_query = self._second_pass_query(query, top_text)

        round2_k = max(top_k // 2, 3)
        if multi_video:
            round2 = self.retrieve_diverse(followup_query, top_k=round2_k, diversity_penalty=diversity_penalty)
        else:
            round2 = self.retrieve(followup_query, top_k=round2_k)

        seen_ids: set = set()
        merged: list = []
        for result in round1 + round2:
            cid = result.chunk.chunk_id
            if cid not in seen_ids:
                seen_ids.add(cid)
                merged.append(result)

        merged.sort(key=lambda r: r.fusion_score, reverse=True)
        return merged[:top_k]