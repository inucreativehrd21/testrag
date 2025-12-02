#!/usr/bin/env python3
"""
RAGAS Evaluation Script
- Supports both Optimized (Enhanced) RAG and LangGraph RAG
- Evaluates Python and Git questions (40 each = 80 total)
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from answerer_v2_optimized import EnhancedRAGPipeline, setup_logging

# RAGAS imports
try:
    from ragas import evaluate
    from ragas.metrics import (
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
        answer_correctness,
    )
    from datasets import Dataset
except ImportError:
    print("Error: RAGAS not installed. Install with: pip install ragas datasets")
    sys.exit(1)


def convert_to_serializable(obj):
    """Convert numpy/pandas types to native Python types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    return obj


def load_evaluation_questions(file_path: str = "ragas_evaluation_questions.json") -> List[Dict]:
    """Load evaluation questions from JSON"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


class OptimizedRAGRunner:
    """Wrapper for the existing EnhancedRAGPipeline (optimized version)"""

    def __init__(self, config_path: str):
        self.pipeline = EnhancedRAGPipeline(config_path)

    def answer_with_contexts(self, question: str) -> Tuple[str, List[str], Dict[str, Any]]:
        """Return answer, contexts, and optional metadata."""
        answer, contexts = self.pipeline.answer_with_contexts(question)
        return answer, contexts, {"pipeline_type": "optimized"}


class LangGraphRAGRunner:
    """Wrapper for the LangGraph RAG workflow"""

    def __init__(self, config_path: str):
        # Lazy imports to avoid loading LangGraph when not needed
        from langgraph_rag.config import get_config
        from langgraph_rag.graph import create_rag_graph
        from langgraph_rag.state import create_initial_state

        # Initialize config/global resources once
        self.config = get_config(config_path)
        self.create_state = create_initial_state
        self.graph = create_rag_graph()

    def answer_with_contexts(self, question: str) -> Tuple[str, List[str], Dict[str, Any]]:
        """Execute LangGraph and return answer + contexts + metadata."""
        final_state: Optional[Dict[str, Any]] = None

        initial_state = self.create_state(question)
        for state in self.graph.stream(initial_state):
            # state is a mapping of node name -> RAGState
            for _, node_state in state.items():
                final_state = node_state

        if not final_state:
            raise RuntimeError("LangGraph pipeline returned no state")

        contexts = final_state.get("final_documents") or final_state.get("documents") or []
        metadatas = final_state.get("final_metadatas") or final_state.get("metadatas") or []

        metadata = {
            "pipeline_type": "langgraph",
            "workflow_history": final_state.get("workflow_history", []),
            "route": final_state.get("route"),
            "retry_count": final_state.get("retry_count"),
            "hallucination_grade": final_state.get("hallucination_grade"),
            "answer_usefulness": final_state.get("answer_usefulness"),
            "intent": final_state.get("intent"),
            "context_metadata": metadatas,
        }

        return final_state.get("generation", ""), contexts, metadata


def build_pipeline_runner(pipeline_type: str, config_path: str):
    """Factory that returns the appropriate RAG runner."""
    pipeline_type = pipeline_type.lower()
    if pipeline_type == "optimized":
        return OptimizedRAGRunner(config_path)
    if pipeline_type == "langgraph":
        return LangGraphRAGRunner(config_path)
    raise ValueError(f"Unsupported pipeline type: {pipeline_type}")


def run_rag_pipeline(pipeline_runner: Any, questions: List[Dict], pipeline_label: str) -> List[Dict]:
    """Run the selected RAG pipeline on all questions and collect results"""
    logger = logging.getLogger(__name__)

    results = []
    total_time = 0

    logger.info(f"Running {pipeline_label} pipeline on {len(questions)} questions...")

    for i, q in enumerate(questions, 1):
        question_text = q["question"]
        question_id = q["id"]
        domain = q["domain"]
        difficulty = q["difficulty"]

        logger.info(f"\n[{i}/{len(questions)}] {question_id} ({domain}, {difficulty})")
        logger.info(f"Question: {question_text[:100]}...")

        try:
            start_time = time.time()

            answer, contexts, metadata = pipeline_runner.answer_with_contexts(question_text)
            context_metadata = metadata.pop("context_metadata", [])

            elapsed = time.time() - start_time
            total_time += elapsed

            result = {
                "question_id": question_id,
                "question": question_text,
                "ground_truth": q["ground_truth"],
                "answer": answer,
                "contexts": contexts,
                "context_metadata": context_metadata,
                "pipeline_metadata": metadata,
                "domain": domain,
                "difficulty": difficulty,
                "type": q["type"],
                "time": elapsed,
                "success": True
            }

            logger.info(f"✓ Completed in {elapsed:.2f}s")
            logger.info(f"  Answer length: {len(answer)} chars")
            logger.info(f"  Contexts: {len(contexts)}")

        except Exception as e:
            logger.error(f"✗ Failed: {e}")
            result = {
                "question_id": question_id,
                "question": question_text,
                "ground_truth": q["ground_truth"],
                "answer": "",
                "contexts": [],
                "context_metadata": [],
                "pipeline_metadata": {"error": str(e)},
                "domain": domain,
                "difficulty": difficulty,
                "type": q["type"],
                "time": 0,
                "error": str(e),
                "success": False
            }

        results.append(result)

    avg_time = total_time / len(questions) if questions else 0
    logger.info(f"\n{'='*80}")
    logger.info(f"Pipeline execution complete:")
    logger.info(f"  Total questions: {len(questions)}")
    logger.info(f"  Successful: {sum(1 for r in results if r['success'])}")
    logger.info(f"  Failed: {sum(1 for r in results if not r['success'])}")
    logger.info(f"  Avg time: {avg_time:.2f}s")
    logger.info(f"  Total time: {total_time:.2f}s ({total_time/60:.1f} min)")
    logger.info(f"{'='*80}")

    return results


def prepare_ragas_dataset(results: List[Dict]) -> Dataset:
    """Prepare dataset in RAGAS format"""
    # Filter successful results only
    successful = [r for r in results if r["success"]]

    # RAGAS expects these fields
    dataset_dict = {
        "question": [r["question"] for r in successful],
        "answer": [r["answer"] for r in successful],
        "contexts": [r["contexts"] for r in successful],
        "ground_truth": [r["ground_truth"] for r in successful],
    }

    return Dataset.from_dict(dataset_dict)


def run_ragas_evaluation(dataset: Dataset) -> Dict:
    """Run RAGAS evaluation"""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*80)
    logger.info("Starting RAGAS Evaluation...")
    logger.info("="*80)

    # Run evaluation with all metrics
    eval_start = time.time()

    evaluation_result = evaluate(
        dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
            answer_correctness,
        ],
    )

    eval_time = time.time() - eval_start

    logger.info(f"\n✓ RAGAS evaluation completed in {eval_time:.2f}s ({eval_time/60:.1f} min)")

    # Convert EvaluationResult to dict for JSON serialization
    # RAGAS returns lists of scores per question, need to compute mean
    try:
        # Try to convert to pandas and get mean (handles NaN automatically)
        result_df = evaluation_result.to_pandas()
        ragas_scores = {
            'context_precision': float(result_df['context_precision'].mean()),
            'context_recall': float(result_df['context_recall'].mean()),
            'faithfulness': float(result_df['faithfulness'].mean()),
            'answer_relevancy': float(result_df['answer_relevancy'].mean()),
            'answer_correctness': float(result_df['answer_correctness'].mean()),
        }
    except AttributeError:
        # If to_pandas() doesn't exist, manually compute mean with NaN handling
        ragas_scores = {}
        for metric_name in ['context_precision', 'context_recall', 'faithfulness',
                            'answer_relevancy', 'answer_correctness']:
            values = evaluation_result[metric_name]
            # Filter out NaN values
            clean_values = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
            if clean_values:
                ragas_scores[metric_name] = float(np.mean(clean_values))
            else:
                ragas_scores[metric_name] = 0.0
                logger.warning(f"{metric_name} has no valid values, defaulting to 0.0")

    # Log scores
    logger.info("\nRAGAS Scores:")
    for metric, score in ragas_scores.items():
        logger.info(f"  {metric}: {score:.4f} ({score*100:.2f}%)")

    return ragas_scores


def save_results(
    results: List[Dict],
    ragas_scores: Dict,
    output_dir: str = "artifacts/ragas_evals",
    pipeline_label: str = "Enhanced RAG",
    config_path: str = "config/enhanced.yaml",
    questions_path: str = "ragas_evaluation_questions.json",
):
    """Save evaluation results to JSON and generate report"""
    logger = logging.getLogger(__name__)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed results (convert to serializable first)
    results_file = output_path / f"ragas_eval_{timestamp}_detailed.json"
    output_data = {
        "timestamp": timestamp,
        "pipeline": pipeline_label,
        "config_path": config_path,
        "questions_path": questions_path,
        "total_questions": len(results),
        "successful": sum(1 for r in results if r["success"]),
        "ragas_scores": ragas_scores,
        "results": results
    }
    # Convert numpy types to native Python types
    output_data = convert_to_serializable(output_data)

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"✓ Detailed results saved to {results_file}")

    # Generate text report
    report_file = output_path / f"ragas_eval_{timestamp}_report.txt"
    with open(report_file, "w", encoding="utf-8-sig") as f:
        f.write("="*80 + "\n")
        f.write("RAGAS EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Pipeline: {pipeline_label}\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"Questions: {questions_path}\n\n")

        f.write(f"Total Questions: {len(results)}\n")
        f.write(f"Successful: {sum(1 for r in results if r['success'])}\n")
        f.write(f"Failed: {sum(1 for r in results if not r['success'])}\n\n")

        # Domain breakdown
        python_count = sum(1 for r in results if r['domain'] == 'python')
        git_count = sum(1 for r in results if r['domain'] == 'git')
        f.write(f"By Domain:\n")
        f.write(f"  Python: {python_count}\n")
        f.write(f"  Git: {git_count}\n\n")

        # Difficulty breakdown
        difficulties = {}
        for r in results:
            diff = r.get('difficulty', 'unknown')
            difficulties[diff] = difficulties.get(diff, 0) + 1

        f.write(f"By Difficulty:\n")
        for diff, count in sorted(difficulties.items()):
            f.write(f"  {diff.capitalize()}: {count}\n")
        f.write("\n")

        # RAGAS scores
        f.write("="*80 + "\n")
        f.write("RAGAS SCORES\n")
        f.write("="*80 + "\n\n")

        for metric, score in ragas_scores.items():
            percentage = score * 100
            f.write(f"{metric:25s}: {percentage:6.2f}%\n")

        f.write("\n")

        # Performance comparison (if baseline exists)
        f.write("="*80 + "\n")
        f.write("PERFORMANCE ANALYSIS\n")
        f.write("="*80 + "\n\n")

        # Target vs Actual
        targets = {
            "context_precision": 80.0,
            "context_recall": 85.0,
            "faithfulness": 93.0,
            "answer_relevancy": 80.0,
            "answer_correctness": 75.0
        }

        f.write(f"{'Metric':<25} {'Target':<10} {'Actual':<10} {'Gap':<10} {'Status'}\n")
        f.write("-"*70 + "\n")

        for metric, target in targets.items():
            actual = ragas_scores.get(metric, 0) * 100
            gap = actual - target
            status = "✓ PASS" if gap >= 0 else "✗ BELOW"
            f.write(f"{metric:<25} {target:<10.1f} {actual:<10.2f} {gap:>+9.2f} {status}\n")

        f.write("\n")

        # Sample answers
        f.write("="*80 + "\n")
        f.write("SAMPLE ANSWERS (First 3)\n")
        f.write("="*80 + "\n\n")

        for i, r in enumerate(results[:3], 1):
            if r['success']:
                f.write(f"[Sample {i}] {r['question_id']}\n")
                f.write(f"Question: {r['question']}\n\n")
                f.write(f"Answer:\n{r['answer']}\n\n")
                f.write(f"Ground Truth:\n{r['ground_truth']}\n\n")
                f.write(f"Contexts: {len(r['contexts'])} retrieved\n")
                f.write("-"*80 + "\n\n")

    logger.info(f"✓ Report saved to {report_file}")

    return results_file, report_file


def print_summary(ragas_scores: Dict):
    """Print evaluation summary to console"""
    print("\n" + "="*80)
    print("RAGAS EVALUATION RESULTS")
    print("="*80 + "\n")

    for metric, score in ragas_scores.items():
        percentage = score * 100

        # Color coding (if terminal supports)
        if percentage >= 80:
            status = "✓ EXCELLENT"
        elif percentage >= 70:
            status = "✓ GOOD"
        elif percentage >= 60:
            status = "⚠ ACCEPTABLE"
        else:
            status = "✗ NEEDS IMPROVEMENT"

        print(f"{metric:25s}: {percentage:6.2f}%  {status}")

    print("\n" + "="*80)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments"""
    parser = argparse.ArgumentParser(description="Run RAGAS benchmark against RAG pipelines")
    parser.add_argument(
        "--pipeline",
        choices=["langgraph", "optimized"],
        default="langgraph",
        help="Which pipeline to evaluate (default: langgraph)",
    )
    parser.add_argument(
        "--config",
        default="config/enhanced.yaml",
        help="Path to the pipeline config file",
    )
    parser.add_argument(
        "--questions",
        default="ragas_evaluation_questions.json",
        help="Path to evaluation question set",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/ragas_evals",
        help="Directory to store evaluation artifacts",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def main():
    """Main evaluation function"""
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    pipeline_label = "LangGraph RAG" if args.pipeline == "langgraph" else "Optimized RAG"

    logger.info("="*80)
    logger.info(f"RAGAS EVALUATION - {pipeline_label}")
    logger.info("="*80)

    # Load evaluation questions
    logger.info("\nLoading evaluation questions...")
    questions = load_evaluation_questions(args.questions)
    logger.info(f"Loaded {len(questions)} questions")

    # Filter by domain (optional)
    python_questions = [q for q in questions if q["domain"] == "python"]
    git_questions = [q for q in questions if q["domain"] == "git"]

    logger.info(f"  Python: {len(python_questions)}")
    logger.info(f"  Git: {len(git_questions)}")
    logger.info(f"Questions file: {args.questions}")
    logger.info(f"Config file: {args.config}")

    # Initialize RAG pipeline
    logger.info(f"\nInitializing {pipeline_label}...")
    pipeline_runner = build_pipeline_runner(args.pipeline, args.config)
    logger.info("✓ Pipeline initialized")

    # Run RAG pipeline on all questions
    logger.info("\n" + "="*80)
    logger.info("STEP 1: Running RAG Pipeline")
    logger.info("="*80)

    results = run_rag_pipeline(pipeline_runner, questions, pipeline_label)

    # Prepare RAGAS dataset
    logger.info("\n" + "="*80)
    logger.info("STEP 2: Preparing RAGAS Dataset")
    logger.info("="*80)

    dataset = prepare_ragas_dataset(results)
    logger.info(f"✓ Dataset prepared: {len(dataset)} samples")

    # Run RAGAS evaluation
    logger.info("\n" + "="*80)
    logger.info("STEP 3: Running RAGAS Evaluation")
    logger.info("="*80)

    ragas_scores = run_ragas_evaluation(dataset)

    # Save results
    logger.info("\n" + "="*80)
    logger.info("STEP 4: Saving Results")
    logger.info("="*80)

    results_file, report_file = save_results(
        results,
        ragas_scores,
        output_dir=args.output_dir,
        pipeline_label=pipeline_label,
        config_path=args.config,
        questions_path=args.questions,
    )

    # Print summary
    print_summary(ragas_scores)

    logger.info(f"\n✓ Evaluation complete!")
    logger.info(f"  Detailed results: {results_file}")
    logger.info(f"  Report: {report_file}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.exception(f"Evaluation failed: {e}")
        sys.exit(1)
