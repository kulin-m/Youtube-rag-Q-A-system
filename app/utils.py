"""
app/utils.py
Core utility functions for text cleaning, chunking, keyword extraction, and prompt building.
"""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Basic text cleaning for transcripts."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def sentence_chunk(text: str, max_words: int = 400, overlap_sentences: int = 2) -> list:
    """Split plain text into chunks based on sentence boundaries."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        words = sentence.split()
        if current_word_count + len(words) > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
            current_word_count = sum(len(s.split()) for s in current_chunk)
        
        current_chunk.append(sentence)
        current_word_count += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def format_timestamp(seconds: float) -> str:
    """Converts seconds float to [MM:SS] format."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"[{mins:02d}:{secs:02d}]"

def timestamped_chunk(transcript_data: list, max_words: int = 400) -> list:
    """
    Split transcript dictionaries into chunks while preserving the starting timestamp.
    Expects transcript_data as [{'text': '...', 'start': 12.5}, ...]
    """
    chunks = []
    current_text = []
    current_word_count = 0
    start_time = 0.0

    for entry in transcript_data:
        if not current_text:
            start_time = entry.get('start', 0.0)
            
        text_segment = entry.get('text', '')
        words = text_segment.split()
        current_text.append(text_segment)
        current_word_count += len(words)

        if current_word_count >= max_words:
            chunks.append({
                "text": " ".join(current_text),
                "timestamp": format_timestamp(start_time)
            })
            current_text = []
            current_word_count = 0
            
    if current_text:
        chunks.append({
            "text": " ".join(current_text),
            "timestamp": format_timestamp(start_time)
        })
    return chunks

def extract_keywords(text: str, top_n: int = 5, ngram_range=(1, 2)) -> list:
    """Extract keywords using TF-IDF as a fallback for KeyBERT."""
    try:
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=ngram_range, max_features=100)
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        keyword_scores = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
        return [kw for kw, score in keyword_scores[:top_n] if score > 0]
    except Exception:
        return []

def compute_confidence(scores: list) -> float:
    """Compute average retrieval confidence score."""
    if not scores: return 0.0
    return float(np.mean(scores))

def expand_query(query: str) -> str:
    """Expand query with synonyms (placeholder for future expansion logic)."""
    return query

def build_fid_style_contextual_prompt(question: str, chunks: list, titles: list, timestamps: list) -> str:
    """
    Strict extractor prompt: Outputs ONLY timestamps and their exact captions.
    """
    context = ""
    for i, (text, title, ts) in enumerate(zip(chunks, titles, timestamps)):
        ts_str = f" | Starts at: {ts}" if ts else ""
        context += f"--- DOCUMENT {i+1} (Source: {title}{ts_str}) ---\n{text}\n\n"
    
    return f"""You are a direct transcript extraction tool. Your ONLY job is to output the exact captions from the provided documents that are relevant to the user's query.

CRITICAL INSTRUCTIONS:
1. Do NOT attempt to explain the concept.
2. Do NOT complain that the document lacks a detailed explanation or definition.
3. Straight up give the exact caption/quote from the text and its corresponding timestamp.
4. Format your output as a simple list of quotes with their timestamps and sources.

Example Format:
* **[01:23]** (Source: Video Title): "Here is the exact transcript text..."

Question: {question}

Relevant Video Segments:
{context}

Answer:"""

def build_fid_style_generalized_prompt(question: str, chunks: list, titles: list) -> str:
    """Educational prompt designed to output an API Documentation / Technical Manual style response."""
    context = "\n".join([f"Source {i+1}: {t}" for i, t in enumerate(set(titles))])
    
    return f"""You are an expert technical writer and engineer. Using the following video sources as a starting reference point, explain the requested concept as if you are writing API Documentation or a Technical Manual.

CRITICAL INSTRUCTIONS:
Even if the video only mentions the topic briefly, use your extensive internal knowledge to provide a fully fleshed-out, highly technical explanation. Do NOT refuse to answer or state that information is missing.

STRUCTURE REQUIRED:
1. Endpoint / Concept Definition: A concise, technical summary.
2. Parameters & Logic: How the concept functions under the hood.
3. Pseudo-Code / Architecture Workflow: A code-like representation or step-by-step logic flow.
4. Production Use Cases: Real-world applications.

Question: {question}

Reference Context (For topic anchoring only):
{context}

Technical API-Style Explanation:"""