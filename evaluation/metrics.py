"""
evaluation/metrics.py

COMPLETE RELIABLE EVALUATION SUITE FOR A YOUTUBE RAG QA SYSTEM
===============================================================

Organised into 3 tiers based on what they measure:

TIER 1 — RETRIEVAL QUALITY  (did we fetch the right chunks?)
  1. Context Precision@K    — are relevant chunks ranked at the top?
  2. Context Recall         — do retrieved chunks cover all needed information?
  3. MRR (Mean Reciprocal Rank) — how high is the first relevant chunk?
  4. Hit@K                  — did ANY relevant chunk appear in top K?

TIER 2 — GENERATION QUALITY  (is the answer good?)
  5. Faithfulness           — are all answer claims supported by context?
  6. Answer Relevancy       — does the answer address the question?
  7. Answer Correctness     — does the answer match a reference answer?
  8. Hallucination Rate     — what fraction of claims are NOT in context?

TIER 3 — SURFACE OVERLAP  (traditional NLP metrics — weakest for RAG)
  9. Semantic Similarity    — cosine sim between answer and reference embeddings
  10. ROUGE-1/2/L           — n-gram overlap with reference
  11. BLEU                  — precision-based n-gram overlap
  12. METEOR                — recall-weighted unigram overlap

WHY ROUGE/BLEU ARE WEAK FOR RAG
================================
ROUGE and BLEU compare surface n-gram overlap against a reference string.
They do NOT check:
  - Whether the answer is factually grounded in the retrieved context
  - Whether the retriever found relevant chunks
  - Whether the LLM hallucinated
A model can score 0.8 ROUGE by copying irrelevant text, or score 0.2 ROUGE
while giving a perfectly correct paraphrase. Use them as supplementary signals
only, never as primary metrics for a RAG system.

RECOMMENDED PRIMARY METRICS FOR THIS PROJECT
=============================================
  Primary:    Faithfulness, Context Precision@K, Answer Relevancy
  Secondary:  Context Recall, Semantic Similarity, MRR
  Tertiary:   ROUGE-L, BLEU, METEOR (surface sanity checks only)

WHAT INPUTS EACH METRIC NEEDS
==============================
  question       — the user's query
  answer         — the generated answer (contextual or generalized)
  contexts       — list of retrieved chunks passed to the generator
  reference      — ground-truth answer (needed for Recall, Correctness, ROUGE, BLEU)
  relevant_flags — list of bool per chunk (needed for Precision@K, MRR, Hit@K)
"""

import re
import logging
import math
from typing import Optional

import numpy as np
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── Shared embedding model ────────────────────────────────────────────────────
_embed_model: Optional[SentenceTransformer] = None

def _get_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        from config.config import cfg
        _embed_model = SentenceTransformer(cfg.EMBEDDING_MODEL)
    return _embed_model

_rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


# ══════════════════════════════════════════════════════════════════════════════
# TIER 1 — RETRIEVAL QUALITY METRICS
# ══════════════════════════════════════════════════════════════════════════════

def context_precision_at_k(relevant_flags: list) -> float:
    """
    Context Precision@K  (RAGAS-style, no LLM needed)

    Measures whether relevant chunks are ranked higher than irrelevant ones.
    Rewards systems that put the best evidence at the top of the context list.

    Formula:
        CP@K = (1 / |relevant|) * sum_k [ Precision@k * v_k ]
        where v_k = 1 if chunk k is relevant, 0 otherwise
        Precision@k = (relevant chunks in top k) / k

    Args:
        relevant_flags: List of bool — True if chunk[i] is relevant to the query.
                        Order matters: position 0 = first chunk passed to LLM.

    Returns:
        Float in [0, 1]. 1.0 = all relevant chunks ranked before all irrelevant.

    Example:
        [True, False, True, False] → 0.583
        [True, True, False, False] → 1.0   (best case)
        [False, False, True, True] → 0.0   (worst case)
    """
    if not relevant_flags or sum(relevant_flags) == 0:
        return 0.0

    num_relevant = sum(relevant_flags)
    running_relevant = 0
    score = 0.0

    for k, is_rel in enumerate(relevant_flags, start=1):
        if is_rel:
            running_relevant += 1
            precision_at_k = running_relevant / k
            score += precision_at_k

    return round(score / num_relevant, 4)


def context_recall(answer: str, contexts: list, reference: str) -> float:
    """
    Context Recall  (semantic approximation, no LLM needed)

    Measures how much of the reference answer's information is present
    in the retrieved context. A low recall means the retriever missed
    evidence needed to produce the correct answer.

    Approach: Split reference into sentences. For each sentence, check
    whether it has a semantically similar match in the context pool.
    Recall = matched_sentences / total_reference_sentences.

    Args:
        answer:    Generated answer (unused here, kept for API consistency).
        contexts:  List of retrieved chunk strings.
        reference: Ground-truth reference answer.

    Returns:
        Float in [0, 1].
    """
    if not reference or not contexts:
        return 0.0

    model = _get_model()
    ref_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', reference) if s.strip()]
    if not ref_sentences:
        return 0.0

    context_blob = " ".join(contexts)
    ctx_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', context_blob) if s.strip()]
    if not ctx_sentences:
        return 0.0

    ref_embeds = model.encode(ref_sentences, normalize_embeddings=True)
    ctx_embeds = model.encode(ctx_sentences, normalize_embeddings=True)

    matched = 0
    threshold = 0.65   # sentence is "covered" if cosine sim ≥ 0.65

    for ref_e in ref_embeds:
        sims = np.dot(ctx_embeds, ref_e)
        if float(np.max(sims)) >= threshold:
            matched += 1

    return round(matched / len(ref_sentences), 4)


def mean_reciprocal_rank(relevant_flags: list) -> float:
    """
    Mean Reciprocal Rank (MRR)

    How early does the first relevant chunk appear?
    MRR = 1 / rank_of_first_relevant_chunk

    Returns 0 if no chunk is relevant. Returns 1.0 if the first chunk is relevant.

    Args:
        relevant_flags: Ordered list of bool — True if chunk is relevant.
    """
    for rank, is_rel in enumerate(relevant_flags, start=1):
        if is_rel:
            return round(1.0 / rank, 4)
    return 0.0


def hit_at_k(relevant_flags: list, k: int = None) -> float:
    """
    Hit@K — binary: did ANY relevant chunk appear in the top K?

    Returns 1.0 if at least one of the top-K chunks is relevant, else 0.0.
    K defaults to len(relevant_flags) (i.e., the full retrieved set).
    """
    if k is None:
        k = len(relevant_flags)
    return 1.0 if any(relevant_flags[:k]) else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# TIER 2 — GENERATION QUALITY METRICS
# ══════════════════════════════════════════════════════════════════════════════

def faithfulness(answer: str, contexts: list) -> float:
    """
    Faithfulness  (semantic approximation of RAGAS Faithfulness)

    Measures what fraction of sentences in the generated answer are
    semantically supported by the retrieved context.

    Official RAGAS formula:
        Faithfulness = supported_claims / total_claims

    Our approach (no LLM needed):
        Split answer into sentences. For each sentence, check if a
        semantically similar sentence exists in the context pool.
        Score = matched_sentences / total_answer_sentences.

    Args:
        answer:   The generated answer string.
        contexts: List of retrieved chunk strings.

    Returns:
        Float in [0, 1]. 1.0 = every answer sentence is grounded in context.
    """
    if not answer or not contexts:
        return 0.0

    model = _get_model()
    ans_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if s.strip()]
    if not ans_sentences:
        return 0.0

    context_blob = " ".join(contexts)
    ctx_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', context_blob) if s.strip()]
    if not ctx_sentences:
        return 0.0

    ans_embeds = model.encode(ans_sentences, normalize_embeddings=True)
    ctx_embeds = model.encode(ctx_sentences, normalize_embeddings=True)

    threshold = 0.65
    supported = 0
    for ans_e in ans_embeds:
        sims = np.dot(ctx_embeds, ans_e)
        if float(np.max(sims)) >= threshold:
            supported += 1

    return round(supported / len(ans_sentences), 4)


def hallucination_rate(answer: str, contexts: list) -> float:
    """
    Hallucination Rate = 1 - Faithfulness

    What fraction of answer sentences are NOT grounded in the context?
    High hallucination = the model is making things up.
    """
    return round(1.0 - faithfulness(answer, contexts), 4)


def answer_relevancy(question: str, answer: str) -> float:
    """
    Answer Relevancy  (semantic approximation of RAGAS Answer Relevancy)

    Measures whether the generated answer actually addresses the question.
    Uses cosine similarity between question and answer embeddings.

    RAGAS uses an LLM to generate synthetic questions from the answer and
    averages their similarity to the original question. We approximate this
    with direct cosine similarity, which correlates well and needs no LLM.

    Args:
        question: The user's query string.
        answer:   The generated answer string.

    Returns:
        Float in [0, 1]. Higher = more relevant to the question.
    """
    if not question or not answer:
        return 0.0
    model = _get_model()
    embeds = model.encode([question, answer], normalize_embeddings=True)
    return round(float(np.dot(embeds[0], embeds[1])), 4)


def answer_correctness(answer: str, reference: str) -> float:
    """
    Answer Correctness  (semantic similarity to ground truth)

    Combines ROUGE-L F1 (surface overlap) and semantic cosine similarity
    for a balanced correctness score that rewards both factual content
    and phrasing flexibility.

    Formula:
        correctness = 0.4 * rouge_L + 0.6 * semantic_sim

    Returns:
        Float in [0, 1].
    """
    if not answer or not reference:
        return 0.0
    r_scores = _rouge.score(reference, answer)
    rouge_l = r_scores["rougeL"].fmeasure
    sem_sim  = semantic_similarity(answer, reference)
    return round(0.4 * rouge_l + 0.6 * sem_sim, 4)


# ══════════════════════════════════════════════════════════════════════════════
# TIER 3 — SURFACE OVERLAP METRICS  (supplementary only)
# ══════════════════════════════════════════════════════════════════════════════

def semantic_similarity(generated: str, reference: str) -> float:
    """
    Cosine similarity between S-BERT embeddings.
    Good for detecting paraphrases; insensitive to factual correctness.
    """
    if not generated or not reference:
        return 0.0
    model = _get_model()
    embeds = model.encode([generated, reference], normalize_embeddings=True)
    return round(float(np.dot(embeds[0], embeds[1])), 4)


def rouge_scores(generated: str, reference: str) -> dict:
    """ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    if not generated or not reference:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    scores = _rouge.score(reference, generated)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def bleu_score(generated: str, reference: str) -> float:
    """Sentence-level BLEU with smoothing."""
    try:
        import nltk
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        smooth = SmoothingFunction().method1
        return round(float(sentence_bleu(
            [reference.lower().split()],
            generated.lower().split(),
            smoothing_function=smooth,
        )), 4)
    except Exception as e:
        logger.warning(f"[BLEU] Failed: {e}")
        return 0.0


def meteor_score(generated: str, reference: str) -> float:
    """
    METEOR — recall-weighted unigram F-score with stemming and synonym matching.
    Better than BLEU for short answers. Requires nltk.
    """
    try:
        import nltk
        from nltk.translate.meteor_score import single_meteor_score
        for res in ["wordnet", "omw-1.4", "averaged_perceptron_tagger"]:
            try:
                nltk.data.find(f"corpora/{res}")
            except LookupError:
                nltk.download(res, quiet=True)
        score = single_meteor_score(
            reference.lower().split(),
            generated.lower().split(),
        )
        return round(float(score), 4)
    except Exception as e:
        logger.warning(f"[METEOR] Failed: {e}")
        return 0.0


def bertscore(generated: str, reference: str) -> Optional[dict]:
    """BERTScore P/R/F1 (optional — requires bert-score package)."""
    try:
        from bert_score import score as _bs
        P, R, F1 = _bs([generated], [reference], lang="en", verbose=False)
        return {
            "precision": round(float(P.mean()), 4),
            "recall":    round(float(R.mean()), 4),
            "f1":        round(float(F1.mean()), 4),
        }
    except ImportError:
        logger.warning("[BERTScore] bert-score not installed.")
        return None
    except Exception as e:
        logger.warning(f"[BERTScore] Failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# COMPOSITE EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_answer(
    generated: str,
    reference: str,
    include_bertscore: bool = False,
) -> dict:
    """
    Legacy composite evaluator (Tier 3 only).
    Kept for backward compatibility with evaluation.py.
    """
    metrics: dict = {
        "semantic_similarity": semantic_similarity(generated, reference),
    }
    metrics.update(rouge_scores(generated, reference))
    metrics["bleu"] = bleu_score(generated, reference)
    if include_bertscore:
        bs = bertscore(generated, reference)
        if bs:
            metrics["bertscore_f1"] = bs["f1"]
    return metrics


def evaluate_rag_response(
    question:        str,
    answer:          str,
    contexts:        list,
    reference:       str  = "",
    relevant_flags:  list = None,
    include_bertscore: bool = False,
) -> dict:
    """
    FULL RAG evaluation across all three tiers.

    Args:
        question:       User query string.
        answer:         Generated answer (contextual or generalized).
        contexts:       List of retrieved chunk strings fed to the generator.
        reference:      Ground-truth answer (required for Tier 1 recall,
                        Tier 2 correctness, and all Tier 3 metrics).
        relevant_flags: List of bool per chunk. True = chunk is relevant.
                        If not provided, Tier 1 ranking metrics are skipped.
        include_bertscore: Compute BERTScore (slower).

    Returns:
        Dict with all available metric scores and a human-readable summary.
    """
    result = {}

    # ── Tier 1: Retrieval ─────────────────────────────────────────────────────
    if relevant_flags:
        result["context_precision_at_k"] = context_precision_at_k(relevant_flags)
        result["mrr"]                    = mean_reciprocal_rank(relevant_flags)
        result["hit_at_k"]               = hit_at_k(relevant_flags)
    if reference:
        result["context_recall"]         = context_recall(answer, contexts, reference)

    # ── Tier 2: Generation ────────────────────────────────────────────────────
    result["faithfulness"]       = faithfulness(answer, contexts)
    result["hallucination_rate"] = hallucination_rate(answer, contexts)
    result["answer_relevancy"]   = answer_relevancy(question, answer)
    if reference:
        result["answer_correctness"] = answer_correctness(answer, reference)

    # ── Tier 3: Surface overlap ───────────────────────────────────────────────
    result["semantic_similarity"] = semantic_similarity(answer, reference) if reference else None
    if reference:
        result.update(rouge_scores(answer, reference))
        result["bleu"]   = bleu_score(answer, reference)
        result["meteor"] = meteor_score(answer, reference)
        if include_bertscore:
            bs = bertscore(answer, reference)
            if bs:
                result["bertscore_f1"] = bs["f1"]

    # ── Human-readable quality labels ─────────────────────────────────────────
    def _label(v, lo=0.4, hi=0.7):
        if v is None: return "N/A"
        if v >= hi:   return "Good"
        if v >= lo:   return "Fair"
        return "Poor"

    result["quality_summary"] = {
        "faithfulness":         _label(result.get("faithfulness"), 0.6, 0.8),
        "hallucination_risk":   "Low" if result.get("hallucination_rate", 1) < 0.2
                                else "Medium" if result.get("hallucination_rate", 1) < 0.5
                                else "High",
        "answer_relevancy":     _label(result.get("answer_relevancy"), 0.5, 0.75),
        "context_precision":    _label(result.get("context_precision_at_k"), 0.4, 0.7),
        "retrieval_coverage":   _label(result.get("context_recall"), 0.4, 0.7),
        "answer_correctness":   _label(result.get("answer_correctness"), 0.4, 0.7),
    }

    return result


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def plot_single_sample(metrics: dict, title: str = "RAG Evaluation — Single Sample",
                       save_path: str = None) -> None:
    """
    Renders a radar chart + horizontal bar chart for a single sample's metrics.

    Args:
        metrics:   Output dict from evaluate_rag_response (before quality_summary pop).
        title:     Figure suptitle.
        save_path: If given, saves figure to this path; otherwise plt.show().
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    import math

    METRIC_GROUPS = {
        "Retrieval": [
            ("context_precision_at_k", "CP@K"),
            ("context_recall",         "C-Recall"),
            ("mrr",                    "MRR"),
            ("hit_at_k",               "Hit@K"),
        ],
        "Generation": [
            ("faithfulness",       "Faithfulness"),
            ("answer_relevancy",   "Ans. Relevancy"),
            ("answer_correctness", "Ans. Correctness"),
            ("hallucination_rate", "Halluc. Rate"),
        ],
        "Surface": [
            ("semantic_similarity", "Sem. Sim."),
            ("rouge1",              "ROUGE-1"),
            ("rouge2",              "ROUGE-2"),
            ("rougeL",              "ROUGE-L"),
            ("bleu",                "BLEU"),
            ("meteor",              "METEOR"),
        ],
    }

    # Palette
    GROUP_COLORS = {
        "Retrieval":  "#4C9BE8",
        "Generation": "#E87C4C",
        "Surface":    "#6DBE6D",
    }

    # Flatten all available metrics for the bar chart
    bar_labels, bar_values, bar_colors = [], [], []
    for grp, pairs in METRIC_GROUPS.items():
        for key, short in pairs:
            val = metrics.get(key)
            if val is not None:
                bar_labels.append(short)
                bar_values.append(float(val))
                bar_colors.append(GROUP_COLORS[grp])

    # ── Radar data: use only Generation + Retrieval metrics that are present ──
    radar_pairs = [
        ("faithfulness",           "Faithfulness"),
        ("answer_relevancy",       "Ans. Relevancy"),
        ("answer_correctness",     "Ans. Correctness"),
        ("context_precision_at_k", "CP@K"),
        ("context_recall",         "C-Recall"),
        ("mrr",                    "MRR"),
        ("semantic_similarity",    "Sem. Sim."),
    ]
    radar_data = [(lbl, float(metrics[k])) for k, lbl in radar_pairs if metrics.get(k) is not None]

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    gs = GridSpec(1, 2, figure=fig, wspace=0.35)

    # ── LEFT: Radar chart ────────────────────────────────────────────────────
    if len(radar_data) >= 3:
        ax_radar = fig.add_subplot(gs[0], polar=True)
        labels_r = [d[0] for d in radar_data]
        values_r = [d[1] for d in radar_data]
        N = len(labels_r)
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]
        values_r += values_r[:1]

        ax_radar.set_theta_offset(math.pi / 2)
        ax_radar.set_theta_direction(-1)
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(labels_r, fontsize=9)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax_radar.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7, color="grey")
        ax_radar.plot(angles, values_r, color="#4C9BE8", linewidth=2)
        ax_radar.fill(angles, values_r, color="#4C9BE8", alpha=0.25)
        ax_radar.set_title("Key Metrics Radar", pad=18, fontsize=11)

    # ── RIGHT: Horizontal bar chart ──────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[1])
    y_pos = range(len(bar_labels) - 1, -1, -1)   # top-to-bottom
    bars = ax_bar.barh(list(y_pos), bar_values, color=bar_colors, edgecolor="white",
                       height=0.6, linewidth=0.5)

    # Threshold lines
    ax_bar.axvline(0.4, color="orange", linestyle="--", linewidth=1, alpha=0.7, label="Fair (0.4)")
    ax_bar.axvline(0.7, color="green",  linestyle="--", linewidth=1, alpha=0.7, label="Good (0.7)")

    # Value annotations
    for bar, val in zip(bars, bar_values):
        ax_bar.text(min(val + 0.02, 0.97), bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha="left", fontsize=8)

    ax_bar.set_yticks(list(y_pos))
    ax_bar.set_yticklabels(bar_labels, fontsize=9)
    ax_bar.set_xlim(0, 1.1)
    ax_bar.set_xlabel("Score", fontsize=10)
    ax_bar.set_title("All Metrics", fontsize=11)
    ax_bar.legend(fontsize=8, loc="lower right")

    # Legend for groups
    patches = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    ax_bar.legend(handles=patches + [
        mpatches.Patch(color="none", label=""),
        mpatches.Patch(facecolor="none", edgecolor="orange", linestyle="--", label="Fair (0.4)"),
        mpatches.Patch(facecolor="none", edgecolor="green",  linestyle="--", label="Good (0.7)"),
    ], fontsize=8, loc="lower right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"[Viz] Saved single-sample plot → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_metrics_dashboard(df, save_path: str = None) -> None:
    """
    Full dashboard with 6 subplots for a batch evaluation DataFrame:

      1. Bar chart  — mean scores per tier
      2. Box plots  — score distributions (Tier 2 generation metrics)
      3. Heatmap    — per-sample metric matrix
      4. Scatter    — Faithfulness vs Answer Relevancy (coloured by correctness)
      5. Histogram  — Hallucination Rate distribution
      6. Radar      — aggregate mean across key metrics

    Args:
        df:        DataFrame returned by evaluate_from_file / evaluate_live.
        save_path: If given, saves figure; otherwise plt.show().
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    import math

    TIER1 = [c for c in ["context_precision_at_k", "context_recall", "mrr", "hit_at_k"] if c in df.columns]
    TIER2 = [c for c in ["faithfulness", "hallucination_rate", "answer_relevancy", "answer_correctness"] if c in df.columns]
    TIER3 = [c for c in ["semantic_similarity", "rouge1", "rouge2", "rougeL", "bleu", "meteor"] if c in df.columns]

    SHORT = {
        "context_precision_at_k": "CP@K",
        "context_recall":         "C-Recall",
        "mrr":                    "MRR",
        "hit_at_k":               "Hit@K",
        "faithfulness":           "Faithfulness",
        "hallucination_rate":     "Halluc. Rate",
        "answer_relevancy":       "Ans. Relevancy",
        "answer_correctness":     "Ans. Correctness",
        "semantic_similarity":    "Sem. Sim.",
        "rouge1": "ROUGE-1", "rouge2": "ROUGE-2",
        "rougeL": "ROUGE-L", "bleu": "BLEU", "meteor": "METEOR",
    }

    PALETTE = {"Tier 1": "#4C9BE8", "Tier 2": "#E87C4C", "Tier 3": "#6DBE6D"}

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("RAG Evaluation Dashboard", fontsize=16, fontweight="bold", y=1.01)

    axes = fig.subplot_mosaic(
        [["bar",   "box",   "scatter"],
         ["heat",  "hist",  "radar"]],
        gridspec_kw={"hspace": 0.45, "wspace": 0.35},
    )

    num_df = df.select_dtypes(include="number")

    # ── 1. Mean scores by tier ───────────────────────────────────────────────
    ax1 = axes["bar"]
    tier_data = []
    tier_labels_all = []
    tier_colors_all = []
    for tier_name, cols, color in [("Tier 1", TIER1, PALETTE["Tier 1"]),
                                    ("Tier 2", TIER2, PALETTE["Tier 2"]),
                                    ("Tier 3", TIER3, PALETTE["Tier 3"])]:
        for c in cols:
            tier_data.append(df[c].mean())
            tier_labels_all.append(SHORT.get(c, c))
            tier_colors_all.append(color)

    bars = ax1.bar(range(len(tier_data)), tier_data, color=tier_colors_all,
                   edgecolor="white", linewidth=0.5)
    ax1.set_xticks(range(len(tier_data)))
    ax1.set_xticklabels(tier_labels_all, rotation=35, ha="right", fontsize=8)
    ax1.set_ylim(0, 1.1)
    ax1.axhline(0.4, color="orange", linestyle="--", linewidth=1, alpha=0.7)
    ax1.axhline(0.7, color="green",  linestyle="--", linewidth=1, alpha=0.7)
    ax1.set_title("Mean Scores by Metric", fontsize=11)
    ax1.set_ylabel("Mean Score")
    for bar, val in zip(bars, tier_data):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    patches = [mpatches.Patch(color=c, label=t) for t, c in PALETTE.items()]
    ax1.legend(handles=patches, fontsize=8)

    # ── 2. Box plots — generation quality ────────────────────────────────────
    ax2 = axes["box"]
    if TIER2:
        box_data = [df[c].dropna().values for c in TIER2]
        bp = ax2.boxplot(box_data, patch_artist=True, notch=False,
                         medianprops=dict(color="black", linewidth=2))
        for patch in bp["boxes"]:
            patch.set_facecolor(PALETTE["Tier 2"])
            patch.set_alpha(0.7)
        ax2.set_xticklabels([SHORT.get(c, c) for c in TIER2], rotation=20, ha="right", fontsize=8)
        ax2.set_ylim(0, 1.1)
        ax2.axhline(0.7, color="green", linestyle="--", linewidth=1, alpha=0.6)
        ax2.set_title("Generation Quality — Distributions", fontsize=11)
        ax2.set_ylabel("Score")

    # ── 3. Scatter — Faithfulness vs Answer Relevancy ────────────────────────
    ax3 = axes["scatter"]
    if "faithfulness" in df.columns and "answer_relevancy" in df.columns:
        color_col = df["answer_correctness"] if "answer_correctness" in df.columns \
                    else pd.Series([0.5] * len(df))
        sc = ax3.scatter(df["faithfulness"], df["answer_relevancy"],
                         c=color_col, cmap="RdYlGn", vmin=0, vmax=1,
                         s=60, edgecolors="grey", linewidths=0.4, alpha=0.85)
        cb = plt.colorbar(sc, ax=ax3)
        cb.set_label("Answer Correctness", fontsize=8)
        ax3.set_xlabel("Faithfulness", fontsize=9)
        ax3.set_ylabel("Answer Relevancy", fontsize=9)
        ax3.set_xlim(0, 1.05)
        ax3.set_ylim(0, 1.05)
        ax3.axhline(0.5, color="grey", linestyle=":", linewidth=1)
        ax3.axvline(0.6, color="grey", linestyle=":", linewidth=1)
        ax3.set_title("Faithfulness vs Answer Relevancy", fontsize=11)

    # ── 4. Heatmap — per-sample metric matrix ────────────────────────────────
    ax4 = axes["heat"]
    heat_cols = [c for c in (TIER1 + TIER2 + TIER3) if c in num_df.columns]
    if heat_cols:
        heat_data = df[heat_cols].fillna(0).values
        im = ax4.imshow(heat_data.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax4.set_yticks(range(len(heat_cols)))
        ax4.set_yticklabels([SHORT.get(c, c) for c in heat_cols], fontsize=8)
        ax4.set_xlabel("Sample Index", fontsize=9)
        ax4.set_title("Per-Sample Metric Heatmap", fontsize=11)
        plt.colorbar(im, ax=ax4, fraction=0.03, pad=0.04)
        ax4.set_xticks(range(len(df)))
        ax4.set_xticklabels([str(i + 1) for i in range(len(df))], fontsize=7)

    # ── 5. Histogram — Hallucination Rate ────────────────────────────────────
    ax5 = axes["hist"]
    if "hallucination_rate" in df.columns:
        vals = df["hallucination_rate"].dropna().values
        n, bins, patches_h = ax5.hist(vals, bins=min(15, max(5, len(vals) // 2)),
                                       edgecolor="white", linewidth=0.5)
        # Colour by risk zone
        for patch, left in zip(patches_h, bins[:-1]):
            if left < 0.2:
                patch.set_facecolor("#6DBE6D")
            elif left < 0.5:
                patch.set_facecolor("#F0C040")
            else:
                patch.set_facecolor("#E05050")
        ax5.axvline(0.2, color="green",  linestyle="--", linewidth=1.2, label="Low/Med (0.2)")
        ax5.axvline(0.5, color="orange", linestyle="--", linewidth=1.2, label="Med/High (0.5)")
        ax5.set_xlabel("Hallucination Rate", fontsize=9)
        ax5.set_ylabel("Count", fontsize=9)
        ax5.set_title("Hallucination Rate Distribution", fontsize=11)
        ax5.legend(fontsize=8)

    # ── 6. Radar — aggregate means ───────────────────────────────────────────
    ax6 = axes["radar"]
    ax6.remove()
    ax6 = fig.add_subplot(2, 3, 6, polar=True)

    radar_cols = [c for c in [
        "faithfulness", "answer_relevancy", "answer_correctness",
        "context_precision_at_k", "context_recall", "mrr", "semantic_similarity",
    ] if c in df.columns]

    if len(radar_cols) >= 3:
        r_vals = [df[c].mean() for c in radar_cols]
        r_labels = [SHORT.get(c, c) for c in radar_cols]
        N = len(r_labels)
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]
        r_vals  += r_vals[:1]

        ax6.set_theta_offset(math.pi / 2)
        ax6.set_theta_direction(-1)
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(r_labels, fontsize=8)
        ax6.set_ylim(0, 1)
        ax6.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax6.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=6, color="grey")
        ax6.plot(angles, r_vals, color="#4C9BE8", linewidth=2)
        ax6.fill(angles, r_vals, color="#4C9BE8", alpha=0.25)
        ax6.set_title("Aggregate Radar", pad=16, fontsize=11)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"[Viz] Saved dashboard → {save_path}")
    else:
        plt.show()
    plt.close(fig)