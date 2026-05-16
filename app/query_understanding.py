"""
app/query_understanding.py
Intelligent query understanding module using Sentence-BERT for semantic
intent detection and normalisation. Uses semantic similarity — NOT keyword
matching — to classify intent.
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Intent taxonomy
# ──────────────────────────────────────────────────────────────────────────────

# Each intent has a label and representative canonical phrases.
# The module embeds these phrases and compares against the user query embedding.
INTENT_DEFINITIONS = {
    "definition": [
        "What is X?",
        "Define X",
        "Explain the concept of X",
        "What does X mean?",
        "Describe X",
        "Tell me about X",
    ],
    "explanation": [
        "How does X work?",
        "Explain how X functions",
        "Walk me through X",
        "How is X done?",
        "Describe the process of X",
    ],
    "comparison": [
        "What is the difference between X and Y?",
        "Compare X and Y",
        "X vs Y",
        "How does X differ from Y?",
        "Which is better, X or Y?",
    ],
    "summarization": [
        "Summarize the video",
        "Give me a summary",
        "What are the key points?",
        "What is the video about?",
        "Main takeaways from the video",
    ],
    "factual": [
        "When did X happen?",
        "Who is X?",
        "Where is X located?",
        "What year was X?",
        "How many X are there?",
    ],
    "opinion": [
        "What do you think about X?",
        "Is X good or bad?",
        "What is the speaker's view on X?",
        "Opinion on X",
        "What does the author believe?",
    ],
    "procedural": [
        "How do I do X?",
        "Steps to achieve X",
        "Guide me through X",
        "Tutorial on X",
        "How can I implement X?",
    ],
}


@dataclass
class QueryAnalysis:
    original: str
    normalised: str
    expanded: str
    intent: str
    intent_confidence: float
    keywords: list[str]


# ──────────────────────────────────────────────────────────────────────────────
# QueryUnderstanding class
# ──────────────────────────────────────────────────────────────────────────────

class QueryUnderstanding:
    """
    Analyses and enriches user queries using semantic similarity.

    Pipeline:
        1. Normalise query (lowercase, strip filler)
        2. Detect intent via S-BERT cosine similarity against intent clusters
        3. Expand query with synonyms
        4. Extract keywords
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info("[QueryUnderstanding] Loading S-BERT model …")
        self.model = SentenceTransformer(model_name)
        self._intent_embeddings: dict[str, np.ndarray] = {}
        self._build_intent_embeddings()

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_intent_embeddings(self):
        """Pre-compute centroid embeddings for each intent cluster."""
        for intent, phrases in INTENT_DEFINITIONS.items():
            phrase_embeds = self.model.encode(phrases, convert_to_numpy=True, normalize_embeddings=True)
            centroid = phrase_embeds.mean(axis=0)
            # Renormalise centroid
            norm = np.linalg.norm(centroid)
            self._intent_embeddings[intent] = centroid / (norm + 1e-9)
        logger.debug("[QueryUnderstanding] Intent embeddings built.")

    @staticmethod
    def _normalise(query: str) -> str:
        """
        Normalise query:
        - Lowercase
        - Strip leading filler phrases ('can you', 'please', etc.)
        - Remove extra whitespace
        """
        q = query.lower().strip()
        # Strip common filler prefixes
        fillers = [
            r"^can you\s+",
            r"^could you\s+",
            r"^please\s+",
            r"^i want to know\s+",
            r"^tell me\s+",
            r"^i'd like to know\s+",
            r"^i wonder\s+",
        ]
        for pattern in fillers:
            q = re.sub(pattern, "", q, flags=re.IGNORECASE)
        q = re.sub(r"\s+", " ", q).strip()
        return q

    def _detect_intent(self, normalised_query: str) -> tuple[str, float]:
        """
        Classify intent by comparing query embedding to pre-built intent centroids.

        Returns:
            (intent_label, confidence_score)
        """
        q_embed = self.model.encode(normalised_query, convert_to_numpy=True, normalize_embeddings=True)
        best_intent = "factual"
        best_score = -1.0

        for intent, centroid in self._intent_embeddings.items():
            sim = float(np.dot(q_embed, centroid))
            if sim > best_score:
                best_score = sim
                best_intent = intent

        return best_intent, round(best_score, 4)

    @staticmethod
    def _extract_keywords_simple(query: str) -> list[str]:
        """Lightweight keyword extraction — removes stopwords, returns content words."""
        stopwords = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "need", "dare", "ought",
            "what", "how", "why", "when", "where", "who", "which", "that", "this",
            "these", "those", "it", "its", "in", "on", "at", "by", "for", "with",
            "about", "against", "between", "into", "through", "of", "to", "from",
            "up", "down", "and", "but", "or", "if", "then", "so", "me", "my",
            "please", "tell", "explain", "describe", "give", "show",
        }
        words = re.findall(r"\b[a-z]+\b", query.lower())
        return [w for w in words if w not in stopwords and len(w) > 2]

    # ── Public API ────────────────────────────────────────────────────────────

    def analyse(self, query: str) -> QueryAnalysis:
        """
        Full analysis of a user query.

        Returns a QueryAnalysis dataclass with all enriched fields.
        """
        from app.utils import expand_query

        normalised = self._normalise(query)
        intent, confidence = self._detect_intent(normalised)
        expanded = expand_query(normalised)
        keywords = self._extract_keywords_simple(normalised)

        logger.info(
            f"[QueryUnderstanding] intent='{intent}' (conf={confidence:.3f}) | "
            f"keywords={keywords} | query='{normalised}'"
        )

        return QueryAnalysis(
            original=query,
            normalised=normalised,
            expanded=expanded,
            intent=intent,
            intent_confidence=confidence,
            keywords=keywords,
        )

    def are_semantically_similar(self, q1: str, q2: str, threshold: float = 0.80) -> bool:
        """
        Check if two queries express the same intent / meaning.
        Useful for deduplication or caching.
        """
        embeds = self.model.encode([q1, q2], convert_to_numpy=True, normalize_embeddings=True)
        sim = float(np.dot(embeds[0], embeds[1]))
        return sim >= threshold
