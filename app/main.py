"""
app/main.py
FastAPI backend for the YouTube RAG QA System.

Endpoints:
    GET  /                  – health check
    POST /ask/single        – single YouTube URL + question
    POST /ask/multi         – multiple YouTube URLs (2-3) + question
    POST /evaluate          – offline evaluation metrics
"""

import sys
import os
import logging
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional

from app.rag_pipeline import RAGPipeline
from evaluation.metrics import evaluate_answer

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="YouTube RAG QA System",
    description=(
        "Answer questions from YouTube videos using a full RAG pipeline.\n\n"
        "**Single video:** `POST /ask/single`\n\n"
        "**Multiple videos:** `POST /ask/multi`"
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singleton pipeline ────────────────────────────────────────────────────────
_pipeline: Optional[RAGPipeline] = None

def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        logger.info("[Startup] Initialising RAG pipeline …")
        _pipeline = RAGPipeline()
        logger.info("[Startup] Pipeline ready.")
    return _pipeline


# ── Validators ────────────────────────────────────────────────────────────────
import re
YT_PATTERN = r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w\-]+"

def _validate_yt_url(url: str) -> str:
    url = url.strip()
    if not re.match(YT_PATTERN, url):
        raise ValueError(f"Not a valid YouTube URL: {url}")
    return url


# ── Request schemas ───────────────────────────────────────────────────────────

class SingleAskRequest(BaseModel):
    url: str = Field(
        ...,
        description="A single YouTube video URL.",
        examples=["https://www.youtube.com/watch?v=aircAruvnKk"],
    )
    question: str = Field(
        ...,
        min_length=3,
        description="The question to answer from the video.",
        examples=["What is a neural network?"],
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, v):
        return _validate_yt_url(v)

    @field_validator("question")
    @classmethod
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be blank.")
        return v.strip()


class MultiAskRequest(BaseModel):
    urls: list[str] = Field(
        ...,
        min_length=2,
        max_length=3,
        description="Two or three YouTube video URLs.",
        examples=[["https://youtu.be/url1", "https://youtu.be/url2"]],
    )
    question: str = Field(
        ...,
        min_length=3,
        description="The question to answer across all videos.",
        examples=["Compare the concepts discussed in each video."],
    )

    @field_validator("urls")
    @classmethod
    def validate_urls(cls, urls):
        return [_validate_yt_url(u) for u in urls]

    @field_validator("question")
    @classmethod
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be blank.")
        return v.strip()


class EvaluateRequest(BaseModel):
    generated_answer: str = Field(..., description="Answer produced by the system.")
    reference_answer: str = Field(..., description="Ground-truth reference answer.")
    include_bertscore: bool = Field(False, description="Include BERTScore (slower).")


# ── Response schemas ──────────────────────────────────────────────────────────

class VideoSource(BaseModel):
    url: str
    title: str


class AnswerSection(BaseModel):
    heading: str
    content: str


class RAGResponse(BaseModel):
    # ── Query info ────────────────────────────────────────────────────────────
    question: str
    detected_intent: str
    intent_confidence: str          # e.g. "87.3%"
    extracted_keywords: list[str]

    # ── Source info ───────────────────────────────────────────────────────────
    source_videos: list[VideoSource]
    retrieval_confidence: str       # e.g. "High (0.82)"

    # ── Answers ───────────────────────────────────────────────────────────────
    contextual_answer: AnswerSection
    generalized_answer: AnswerSection

    # ── Supporting context ────────────────────────────────────────────────────
    retrieved_context: list[dict]   # [{index, preview, length}]

    # ── Meta ──────────────────────────────────────────────────────────────────
    processing_time_seconds: float
    error: Optional[str] = None


class EvaluateResponse(BaseModel):
    semantic_similarity: float
    rouge1: float
    rouge2: float
    rougeL: float
    bleu: float
    bertscore_f1: Optional[float] = None
    interpretation: dict            # human-readable quality labels


# ── Helpers ───────────────────────────────────────────────────────────────────

def _confidence_label(score: float) -> str:
    if score >= 0.75:   return f"High ({score:.2f})"
    if score >= 0.45:   return f"Medium ({score:.2f})"
    return               f"Low ({score:.2f}) — answer may not fully reflect video content"


def _build_response(result, elapsed: float) -> RAGResponse:
    """Convert a RAGResult into a clean, human-readable RAGResponse."""

    # Format retrieved chunks with index + character preview
    context_items = []
    for i, chunk in enumerate(result.retrieved_chunks, 1):
        words = chunk.split()
        preview = " ".join(words[:40]) + ("…" if len(words) > 40 else "")
        context_items.append({
            "index": i,
            "preview": preview,
            "total_words": len(words),
        })

    return RAGResponse(
        # Query
        question=result.question,
        detected_intent=result.intent.capitalize(),
        intent_confidence=f"{result.intent_confidence * 100:.1f}%",
        extracted_keywords=result.keywords if result.keywords else ["(none detected)"],

        
        # Sources
        source_videos=[VideoSource(**v) for v in result.source_videos],
        retrieval_confidence=_confidence_label(result.confidence_score),

        # Context
        retrieved_context=context_items,
        
        # Answers
        contextual_answer=AnswerSection(
            heading="Answer from Video Content",
            content=result.contextual_answer or "No contextual answer could be generated.",
        ),
        generalized_answer=AnswerSection(
            heading="Expanded Answer (LLM-Enhanced)",
            content=result.generalized_answer or "No generalized answer could be generated.",
        ),


        # Meta
        processing_time_seconds=round(elapsed, 2),
        error=result.error,
    )


def _run_pipeline(urls: list[str], question: str) -> tuple:
    """Run pipeline and return (result, elapsed_seconds)."""
    pipeline = get_pipeline()
    start = time.time()
    try:
        result = pipeline.run(urls=urls, question=question)
    except Exception as e:
        logger.error(f"[Pipeline] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    elapsed = time.time() - start

    if result.error:
        raise HTTPException(status_code=422, detail=result.error)

    return result, elapsed


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check", tags=["System"])
def root():
    return {
        "status": "running",
        "version": "3.0.0",
        "endpoints": {
            "single_video_qa": "POST /ask/single",
            "multi_video_qa":  "POST /ask/multi",
            "evaluate":        "POST /evaluate",
            "docs":            "GET  /docs",
        }
    }


@app.post(
    "/ask/single",
    response_model=RAGResponse,
    summary="Ask a question from a single YouTube video",
    tags=["QA"],
)
def ask_single(request: SingleAskRequest):
    """
    Provide **one** YouTube URL and a question.
    Returns a contextual answer (grounded in the video) and a
    generalized answer (LLM-enhanced with broader knowledge).
    """
    result, elapsed = _run_pipeline(urls=[request.url], question=request.question)
    return _build_response(result, elapsed)


@app.post(
    "/ask/multi",
    response_model=RAGResponse,
    summary="Ask a question across multiple YouTube videos (2–3)",
    tags=["QA"],
)
def ask_multi(request: MultiAskRequest):
    """
    Provide **two or three** YouTube URLs and a question.
    The pipeline retrieves diverse chunks across all videos and
    synthesizes a unified answer.
    """
    result, elapsed = _run_pipeline(urls=request.urls, question=request.question)
    return _build_response(result, elapsed)


@app.post(
    "/evaluate",
    response_model=EvaluateResponse,
    summary="Evaluate a generated answer against a reference",
    tags=["Evaluation"],
)
def evaluate(request: EvaluateRequest):
    """
    Compute ROUGE, BLEU, and Semantic Similarity scores between a
    generated answer and a ground-truth reference answer.
    """
    try:
        metrics = evaluate_answer(
            generated=request.generated_answer,
            reference=request.reference_answer,
            include_bertscore=request.include_bertscore,
        )
    except Exception as e:
        logger.error(f"[/evaluate] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    def _label(score: float, thresholds=(0.4, 0.7)) -> str:
        if score >= thresholds[1]: return "Good"
        if score >= thresholds[0]: return "Fair"
        return "Poor"

    return EvaluateResponse(
        semantic_similarity=metrics.get("semantic_similarity", 0.0),
        rouge1=metrics.get("rouge1", 0.0),
        rouge2=metrics.get("rouge2", 0.0),
        rougeL=metrics.get("rougeL", 0.0),
        bleu=metrics.get("bleu", 0.0),
        bertscore_f1=metrics.get("bertscore_f1"),
        interpretation={
            "semantic_similarity": _label(metrics.get("semantic_similarity", 0)),
            "rouge1":              _label(metrics.get("rouge1", 0)),
            "rouge2":              _label(metrics.get("rouge2", 0)),
            "rougeL":              _label(metrics.get("rougeL", 0)),
            "bleu":                _label(metrics.get("bleu", 0), (0.1, 0.4)),
        },
    )