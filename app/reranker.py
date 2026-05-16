"""
app/reranker.py
Cross-Encoder based reranker.
Takes (query, chunk) pairs and scores them with a fine-tuned cross-encoder,
producing a more accurate relevance signal than bi-encoder similarity alone.
"""

import logging
from typing import Optional

from sentence_transformers import CrossEncoder

from app.retriever import RetrievalResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Wraps a HuggingFace CrossEncoder model (ms-marco-MiniLM-L-6-v2) to
    rerank a set of retrieved candidates.

    The cross-encoder sees (query, passage) jointly, giving it access to
    fine-grained interaction signals not available to a bi-encoder.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info(f"[Reranker] Loading cross-encoder: {model_name} …")
        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_n: Optional[int] = None,
    ) -> list[RetrievalResult]:
        """
        Rerank a list of RetrievalResult objects using the cross-encoder.

        Args:
            query:   The user's query (or normalised/expanded version).
            results: Candidate results from the retriever.
            top_n:   Return only the top_n after reranking. None = return all.

        Returns:
            Reranked (and optionally truncated) list of RetrievalResult objects.
        """
        if not results:
            return []

        pairs = [(query, r.chunk.text) for r in results]

        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            logger.error(f"[Reranker] Cross-encoder inference failed: {e}. Falling back to original order.")
            return results[:top_n] if top_n else results

        # Attach cross-encoder score to each result (overwrite fusion_score)
        scored = list(zip(results, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked = []
        for result, ce_score in scored:
            reranked.append(
                RetrievalResult(
                    chunk=result.chunk,
                    semantic_score=result.semantic_score,
                    keyword_score=result.keyword_score,
                    fusion_score=float(ce_score),  # reuse fusion_score field for CE score
                )
            )

        if top_n:
            reranked = reranked[:top_n]

        logger.debug(
            f"[Reranker] Reranked {len(results)} → {len(reranked)} results. "
            f"Top score: {reranked[0].fusion_score:.4f}"
        )
        return reranked
