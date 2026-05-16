"""
app/rag_pipeline.py
Orchestrator for the RAG Pipeline.
"""

import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np

from sentence_transformers import SentenceTransformer

from config.config import cfg
from app.transcript_extractor import extract_multiple_transcripts
from app.utils import (
    clean_text,
    sentence_chunk,
    timestamped_chunk,
    extract_keywords,
    compute_confidence,
)
from app.query_understanding import QueryUnderstanding
from app.retriever import HybridRetriever, Chunk
from app.reranker import CrossEncoderReranker
from app.generator import GroqGenerator

logger = logging.getLogger(__name__)

@dataclass
class RAGResult:
    question: str
    intent: str
    intent_confidence: float
    contextual_answer: str
    generalized_answer: str
    retrieved_chunks: list
    source_videos: list
    confidence_score: float
    keywords: list
    error: Optional[str] = None

class RAGPipeline:

    def __init__(self):
        logger.info("[Pipeline] Initialising components …")
        self.embed_model = SentenceTransformer(cfg.EMBEDDING_MODEL)
        self.query_engine = QueryUnderstanding(cfg.EMBEDDING_MODEL)
        self.reranker     = CrossEncoderReranker(cfg.RERANKER_MODEL)
        self.generator    = GroqGenerator()
        logger.info("[Pipeline] All components ready.")

    def _fresh_retriever(self) -> HybridRetriever:
        return HybridRetriever(
            embedding_model=self.embed_model,
            semantic_weight=cfg.SEMANTIC_WEIGHT,
        )

    def _ingest(self, transcripts: list, retriever: HybridRetriever) -> list:
        sources = []
        chunk_counter = 0

        for t in transcripts:
            url, title = t["url"], t["title"]
            sources.append({"url": url, "title": title})

            if t.get("transcript_data"):
                # NEW: Use Timestamped Chunking if data is available
                chunk_data = timestamped_chunk(t["transcript_data"], max_words=cfg.CHUNK_SIZE)
                chunks = []
                for item in chunk_data:
                    kws = extract_keywords(item["text"], top_n=cfg.KEYBERT_TOP_N, ngram_range=cfg.KEYBERT_NGRAM_RANGE)
                    chunks.append(Chunk(
                        text=item["text"], source_url=url, source_title=title,
                        chunk_id=chunk_counter, timestamp=item["timestamp"], keywords=kws
                    ))
                    chunk_counter += 1
                retriever.add_chunks(chunks)
            else:
                # Fallback to pure text chunking
                cleaned = clean_text(t["transcript"])
                text_chunks = sentence_chunk(cleaned, max_words=cfg.CHUNK_SIZE, overlap_sentences=2)
                chunks = []
                for text in text_chunks:
                    kws = extract_keywords(text, top_n=cfg.KEYBERT_TOP_N, ngram_range=cfg.KEYBERT_NGRAM_RANGE)
                    chunks.append(Chunk(
                        text=text, source_url=url, source_title=title,
                        chunk_id=chunk_counter, timestamp="", keywords=kws
                    ))
                    chunk_counter += 1
                retriever.add_chunks(chunks)

        return sources

    def _retrieve_and_rerank(self, query_analysis, multi_video: bool, retriever: HybridRetriever) -> tuple:
        query = query_analysis.expanded

        raw = retriever.iterative_retrieve(
            query=query,
            top_k=cfg.TOP_K,
            multi_video=multi_video,
            diversity_penalty=cfg.DIVERSITY_PENALTY,
        )

        if not raw:
            return [], [], [], []

        reranked = self.reranker.rerank(
            query=query_analysis.normalised,
            results=raw,
            top_n=cfg.RERANK_TOP_N,
        )

        chunk_texts       = [r.chunk.text         for r in reranked]
        scores            = [r.fusion_score       for r in reranked]
        source_titles     = [r.chunk.source_title for r in reranked]
        source_timestamps = [r.chunk.timestamp    for r in reranked] # NEW

        return chunk_texts, scores, source_titles, source_timestamps

    def run(self, urls: list, question: str) -> RAGResult:
        multi_video = len(urls) > 1

        try:
            qa = self.query_engine.analyse(question)
        except Exception as e:
            return RAGResult(question=question, intent="unknown", intent_confidence=0.0, contextual_answer="", generalized_answer="", retrieved_chunks=[], source_videos=[], confidence_score=0.0, keywords=[], error=str(e))

        try:
            transcripts = extract_multiple_transcripts(urls)
        except (ValueError, RuntimeError) as e:
            return RAGResult(question=question, intent=qa.intent, intent_confidence=qa.intent_confidence, contextual_answer="", generalized_answer="", retrieved_chunks=[], source_videos=[], confidence_score=0.0, keywords=qa.keywords, error=str(e))

        retriever = self._fresh_retriever()
        sources   = self._ingest(transcripts, retriever)

        chunk_texts, scores, source_titles, source_timestamps = self._retrieve_and_rerank(
            qa, multi_video, retriever
        )

        if not chunk_texts:
            msg = "No relevant content found in the video transcript(s)."
            return RAGResult(question=question, intent=qa.intent, intent_confidence=qa.intent_confidence, contextual_answer=msg, generalized_answer="", retrieved_chunks=[], source_videos=sources, confidence_score=0.0, keywords=qa.keywords)

        selected_chunks, selected_titles, selected_timestamps = self._mmr_with_titles_and_timestamps(
            query=qa.normalised,
            chunks=chunk_texts,
            titles=source_titles,
            timestamps=source_timestamps,
            top_n=cfg.RERANK_TOP_N,
        )

        confidence = compute_confidence(scores)

        # Contextual generator receives timestamps for precise citations
        contextual = self.generator.generate_contextual_answer(
            question, selected_chunks, selected_titles, selected_timestamps
        )
        
        # Generalized answer creates the API logic documentation
        generalized = self.generator.generate_generalized_answer(
            question, selected_chunks, selected_titles
        )

        return RAGResult(
            question=question,
            intent=qa.intent,
            intent_confidence=qa.intent_confidence,
            contextual_answer=contextual,
            generalized_answer=generalized,
            retrieved_chunks=selected_chunks,
            source_videos=sources,
            confidence_score=confidence,
            keywords=qa.keywords,
        )

    def _mmr_with_titles_and_timestamps(self, query: str, chunks: list, titles: list, timestamps: list, top_n: int) -> tuple:
        if not chunks:
            return [], [], []
        if len(chunks) <= top_n:
            return chunks, titles, timestamps

        all_texts   = [query] + chunks
        all_embeds  = self.embed_model.encode(all_texts, convert_to_numpy=True, normalize_embeddings=True)
        query_embed  = all_embeds[0]
        chunk_embeds = all_embeds[1:]
        query_sims   = np.dot(chunk_embeds, query_embed)

        selected_indices = []
        remaining = list(range(len(chunks)))
        lambda_ = 0.6

        for _ in range(min(top_n, len(chunks))):
            if not selected_indices:
                best = remaining[int(np.argmax(query_sims[remaining]))]
            else:
                sel_embeds = chunk_embeds[selected_indices]
                mmr_scores = []
                for i in remaining:
                    rel = float(query_sims[i])
                    red = float(np.max(np.dot(sel_embeds, chunk_embeds[i])))
                    mmr_scores.append((i, lambda_ * rel - (1 - lambda_) * red))
                best = max(mmr_scores, key=lambda x: x[1])[0]
            selected_indices.append(best)
            remaining.remove(best)

        sel_chunks = [chunks[i] for i in selected_indices]
        sel_titles = [titles[i] for i in selected_indices]
        sel_timestamps = [timestamps[i] for i in selected_indices]

        return sel_chunks, sel_titles, sel_timestamps