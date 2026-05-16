"""
evaluation/evaluation.py
Batch evaluation runner using the full metrics suite.

Usage:
    # Offline (pre-generated answers in JSON):
    python evaluation/evaluation.py --data data/sample_questions.json

    # Live (runs pipeline end-to-end):
    python evaluation/evaluation.py --data data/sample_questions.json --live

    # Save plots to a directory:
    python evaluation/evaluation.py --data data/sample_questions.json --plots plots/

Expected JSON schema:
    [
      {
        "question":        "What is a neural network?",
        "url":             "https://youtu.be/...",        (needed for --live)
        "reference_answer":"...",
        "generated_answer":"...",                         (needed for offline)
        "retrieved_chunks":["chunk1", "chunk2", ...],    (optional but recommended)
        "relevant_flags":  [true, false, true]           (optional)
      }
    ]
"""

import argparse
import json
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import evaluate_rag_response, plot_metrics_dashboard, plot_single_sample

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)

SEP = "─" * 72


# ══════════════════════════════════════════════════════════════════════════════
# CORE EVALUATION RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_from_file(data_path: str, include_bertscore: bool = False) -> pd.DataFrame:
    """Offline evaluation using pre-generated answers in the JSON file."""
    with open(data_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    records = []
    for i, s in enumerate(samples):
        question  = s.get("question", "")
        reference = s.get("reference_answer", "")
        generated = s.get("generated_answer", "")
        contexts  = s.get("retrieved_chunks", [])
        flags     = s.get("relevant_flags", None)

        if not generated:
            logger.warning(f"[Eval] Sample {i+1} missing generated_answer — skipping.")
            continue

        metrics = evaluate_rag_response(
            question=question, answer=generated, contexts=contexts,
            reference=reference, relevant_flags=flags,
            include_bertscore=include_bertscore,
        )
        summary = metrics.pop("quality_summary", {})
        row = {"question": question[:60], **metrics,
               **{f"label_{k}": v for k, v in summary.items()}}
        records.append(row)

        logger.info(
            f"[Eval] {i+1}/{len(samples)} | "
            f"Faith={metrics.get('faithfulness','?'):.3f} | "
            f"Rel={metrics.get('answer_relevancy','?'):.3f} | "
            f"Halluc={metrics.get('hallucination_rate','?'):.3f}"
        )

    return pd.DataFrame(records)


def evaluate_live(data_path: str, include_bertscore: bool = False) -> pd.DataFrame:
    """Live evaluation — runs the RAG pipeline for each sample."""
    from app.rag_pipeline import RAGPipeline

    with open(data_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    pipeline = RAGPipeline()
    records = []

    for i, s in enumerate(samples):
        question  = s.get("question", "")
        reference = s.get("reference_answer", "")
        url       = s.get("url", "")

        if not url:
            logger.warning(f"[Eval-Live] Sample {i+1} missing URL — skipping.")
            continue

        logger.info(f"[Eval-Live] Running sample {i+1}: {question[:60]}")
        try:
            result   = pipeline.run(urls=[url], question=question)
            generated = result.contextual_answer
            contexts  = result.retrieved_chunks
        except Exception as e:
            logger.error(f"[Eval-Live] Pipeline error: {e}")
            continue

        metrics = evaluate_rag_response(
            question=question, answer=generated, contexts=contexts,
            reference=reference, include_bertscore=include_bertscore,
        )
        summary = metrics.pop("quality_summary", {})
        row = {"question": question[:60], **metrics,
               **{f"label_{k}": v for k, v in summary.items()}}
        records.append(row)

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_report(df: pd.DataFrame) -> None:
    print(f"\n{SEP}")
    print("  EVALUATION REPORT")
    print(SEP)

    num_cols = [c for c in df.select_dtypes(include="number").columns]
    tier1 = [c for c in num_cols if c in
              ["context_precision_at_k","context_recall","mrr","hit_at_k"]]
    tier2 = [c for c in num_cols if c in
              ["faithfulness","hallucination_rate","answer_relevancy","answer_correctness"]]
    tier3 = [c for c in num_cols if c in
              ["semantic_similarity","rouge1","rouge2","rougeL","bleu","meteor","bertscore_f1"]]

    for tier_name, cols in [
        ("TIER 1 — Retrieval Quality",    tier1),
        ("TIER 2 — Generation Quality",   tier2),
        ("TIER 3 — Surface Overlap",      tier3),
    ]:
        if not cols:
            continue
        print(f"\n  {tier_name}")
        print(f"  {'Metric':<30} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for col in cols:
            m = df[col].mean(); s = df[col].std()
            lo = df[col].min(); hi = df[col].max()
            print(f"  {col:<30} {m:8.4f} {s:8.4f} {lo:8.4f} {hi:8.4f}")

    print(f"\n  Total samples: {len(df)}")
    print(SEP)


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION ENTRY-POINT
# ══════════════════════════════════════════════════════════════════════════════

def generate_plots(df: pd.DataFrame, plots_dir: str = None) -> None:
    """
    Produces all visual reports for a completed evaluation run.

    Plots generated
    ───────────────
      1. eval_dashboard.png   — 6-panel batch dashboard
          • Mean scores bar chart (all tiers, colour-coded)
          • Box-plot distributions for generation metrics
          • Scatter: Faithfulness vs Answer Relevancy (coloured by correctness)
          • Per-sample heatmap
          • Hallucination Rate histogram (green / amber / red bins)
          • Aggregate radar chart

      2. sample_<N>.png (one per row)  — single-sample radar + bar chart

    Args:
        df:        Evaluation DataFrame from evaluate_from_file / evaluate_live.
        plots_dir: Directory to save PNGs. If None, figures are shown interactively.
    """
    import matplotlib
    if plots_dir:
        matplotlib.use("Agg")   # non-interactive backend when saving to disk

    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)

    # ── Dashboard (full batch) ───────────────────────────────────────────────
    dash_path = os.path.join(plots_dir, "eval_dashboard.png") if plots_dir else None
    logger.info("[Viz] Generating batch dashboard …")
    plot_metrics_dashboard(df, save_path=dash_path)
    if dash_path:
        logger.info(f"[Viz]   → {dash_path}")

    # ── Per-sample plots ─────────────────────────────────────────────────────
    num_metric_cols = df.select_dtypes(include="number").columns.tolist()
    for idx, row in df.iterrows():
        sample_metrics = {c: row[c] for c in num_metric_cols if pd.notna(row.get(c))}
        question_snippet = str(row.get("question", f"Sample {idx+1}"))[:50]
        title = f"Sample {idx+1}: {question_snippet}"
        sample_path = (
            os.path.join(plots_dir, f"sample_{idx+1:03d}.png") if plots_dir else None
        )
        logger.info(f"[Viz] Generating per-sample plot for sample {idx+1} …")
        plot_single_sample(sample_metrics, title=title, save_path=sample_path)
        if sample_path:
            logger.info(f"[Viz]   → {sample_path}")

    if plots_dir:
        print(f"\n  All plots saved to: {os.path.abspath(plots_dir)}/")
    else:
        print("\n  (Plots displayed interactively — pass --plots <dir> to save instead)")


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY-POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluate YouTube RAG QA System")
    parser.add_argument("--data",       default="data/sample_questions.json",
                        help="Path to the evaluation JSON file.")
    parser.add_argument("--live",       action="store_true",
                        help="Run the RAG pipeline live instead of using pre-generated answers.")
    parser.add_argument("--bertscore",  action="store_true",
                        help="Include BERTScore (slower; requires bert-score package).")
    parser.add_argument("--output",     default=None,
                        help="Save evaluation results DataFrame to this CSV path.")
    parser.add_argument("--plots",      default=None,
                        help="Directory to save PNG plots. Omit to display interactively.")
    parser.add_argument("--no-plots",   action="store_true",
                        help="Skip all plot generation entirely.")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Dataset not found: {args.data}")
        sys.exit(1)

    # ── Run evaluation ───────────────────────────────────────────────────────
    df = evaluate_live(args.data, args.bertscore) if args.live \
         else evaluate_from_file(args.data, args.bertscore)

    if df.empty:
        print("No results.")
        sys.exit(1)

    # ── Console report ───────────────────────────────────────────────────────
    print_report(df)

    # ── CSV export ───────────────────────────────────────────────────────────
    if args.output:
        df.to_csv(args.output, index=False)
        logger.info(f"Saved CSV → {args.output}")

    # ── Visual reports ───────────────────────────────────────────────────────
    if not args.no_plots:
        generate_plots(df, plots_dir=args.plots)


if __name__ == "__main__":
    main()