"""
config/config.py
Central configuration for the YouTube RAG QA System.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ─── API Keys ─────────────────────────────────────────────────────────────
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # ─── Embedding Model ──────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # ─── Cross-Encoder Reranker ───────────────────────────────────────────────
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ─── Gemini Generation Model ──────────────────────────────────────────────
    # Use models/gemini-1.5-flash (with prefix) for v1beta API compatibility
    # Fallback chain: flash → pro → gemini-pro (older alias)
    GEMINI_MODEL: str = "models/gemini-1.5-flash"
    GEMINI_MODEL_FALLBACKS: list = [
        "models/gemini-1.5-flash",
        "models/gemini-1.5-pro",
        "models/gemini-pro",
        "gemini-1.0-pro",
    ]

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # ─── Chunking ─────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 512))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 64))

    # ─── Retrieval ────────────────────────────────────────────────────────────
    TOP_K: int = int(os.getenv("TOP_K", 5))
    RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N", 3))

    # ─── Hybrid Search Weights ────────────────────────────────────────────────
    SEMANTIC_WEIGHT: float = 0.7
    KEYWORD_WEIGHT: float = 0.3

    # ─── Confidence Threshold ─────────────────────────────────────────────────
    CONFIDENCE_THRESHOLD: float = 0.3

    # ─── Keyword Extraction ───────────────────────────────────────────────────
    KEYBERT_TOP_N: int = 5
    KEYBERT_NGRAM_RANGE: tuple = (1, 2)

    # ─── Query Expansion ──────────────────────────────────────────────────────
    QUERY_EXPANSION_ENABLED: bool = True

    # ─── Multi-video ──────────────────────────────────────────────────────────
    MAX_VIDEOS: int = 3
    DIVERSITY_PENALTY: float = 0.2


cfg = Config()