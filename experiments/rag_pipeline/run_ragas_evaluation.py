#!/usr/bin/env python3
"""
RAGAS Evaluation Script for Enhanced RAG Pipeline
Evaluates Python and Git questions (40 each = 80 total)
"""
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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


def load_evaluation_questions(file_path: str = "ragas_evaluation_questions.json") -> List[Dict]:
    """Load evaluation questions from JSON"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def run_rag_pipeline(pipeline: EnhancedRAGPipeline, questions: List[Dict]) -> List[Dict]:
    """Run RAG pipeline on all questions and collect results"""
    logger = logging.getLogger(__name__)

    results = []
    total_time = 0

    logger.info(f"Running RAG pipeline on {len(questions)} questions...")

    for i, q in enumerate(questions, 1):
        question_text = q["question"]
        question_id = q["id"]
        domain = q["domain"]
        difficulty = q["difficulty"]

        logger.info(f"\n[{i}/{len(questions)}] {question_id} ({domain}, {difficulty})")
        logger.info(f"Question: {question_text[:100]}...")

        try:
            start_time = time.time()

            # Get contexts and metadatas (retrieve returns tuple)
            contexts, metadatas = pipeline.retrieve(question_text)

            # Get answer
            answer = pipeline.answer(question_text)

            elapsed = time.time() - start_time
            total_time += elapsed

            result = {
                "question_id": question_id,
                "question": question_text,
                "ground_truth": q["ground_truth"],
                "answer": answer,
                "contexts": contexts,
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
                "domain": domain,
                "difficulty": difficulty,
                "type": q["type"],
                "time": 0,
                "error": str(e),
                "success": False
            }

        results.append(result)

    avg_time = total_time / len(questions)
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

    return evaluation_result


def save_results(
    results: List[Dict],
    ragas_scores: Dict,
    output_dir: str = "artifacts/ragas_evals"
):
    """Save evaluation results to JSON and generate report"""
    logger = logging.getLogger(__name__)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed results
    results_file = output_path / f"ragas_eval_{timestamp}_detailed.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "total_questions": len(results),
            "successful": sum(1 for r in results if r["success"]),
            "ragas_scores": ragas_scores,
            "results": results
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"✓ Detailed results saved to {results_file}")

    # Generate text report
    report_file = output_path / f"ragas_eval_{timestamp}_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("RAGAS EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Pipeline: Enhanced RAG (Hybrid Search + Context Quality Filter)\n")
        f.write(f"Config: config/enhanced.yaml\n\n")

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


def main():
    """Main evaluation function"""
    setup_logging("INFO")
    logger = logging.getLogger(__name__)

    logger.info("="*80)
    logger.info("RAGAS EVALUATION - Enhanced RAG Pipeline")
    logger.info("="*80)

    # Load evaluation questions
    logger.info("\nLoading evaluation questions...")
    questions = load_evaluation_questions()
    logger.info(f"Loaded {len(questions)} questions")

    # Filter by domain (optional)
    python_questions = [q for q in questions if q["domain"] == "python"]
    git_questions = [q for q in questions if q["domain"] == "git"]

    logger.info(f"  Python: {len(python_questions)}")
    logger.info(f"  Git: {len(git_questions)}")

    # Initialize RAG pipeline
    logger.info("\nInitializing Enhanced RAG Pipeline...")
    pipeline = EnhancedRAGPipeline("config/enhanced.yaml")
    logger.info("✓ Pipeline initialized")

    # Run RAG pipeline on all questions
    logger.info("\n" + "="*80)
    logger.info("STEP 1: Running RAG Pipeline")
    logger.info("="*80)

    results = run_rag_pipeline(pipeline, questions)

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

    results_file, report_file = save_results(results, ragas_scores)

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
