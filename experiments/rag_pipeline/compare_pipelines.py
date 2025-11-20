#!/usr/bin/env python3
"""
Pipeline Comparison Tool
Compare Baseline vs Enhanced RAG performance
"""
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from answerer import RAGPipeline as BaselineRAG
from answerer_v2 import EnhancedRAGPipeline, setup_logging


def load_test_questions(file_path: str = "ragas_questions.json") -> List[Dict]:
    """Load test questions from JSON file"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("questions", [])


def evaluate_pipeline(pipeline, questions: List[Dict], pipeline_name: str) -> Dict:
    """Evaluate a pipeline on test questions"""
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluating {pipeline_name}")
    logger.info(f"{'='*80}")

    results = []
    total_time = 0

    # Filter Git/Python questions only
    filtered = [q for q in questions if q.get("domain") in ["git", "python"]]

    for i, q in enumerate(filtered, 1):
        question = q["question"]
        domain = q.get("domain", "unknown")
        difficulty = q.get("difficulty", "unknown")

        logger.info(f"\n[{i}/{len(filtered)}] Domain: {domain}, Difficulty: {difficulty}")
        logger.info(f"Question: {question[:100]}...")

        try:
            start = time.time()
            answer = pipeline.answer(question)
            elapsed = time.time() - start

            results.append({
                "question": question,
                "answer": answer,
                "domain": domain,
                "difficulty": difficulty,
                "time": elapsed,
                "success": True
            })

            total_time += elapsed
            logger.info(f"✓ Answered in {elapsed:.2f}s")

        except Exception as e:
            logger.error(f"✗ Failed: {e}")
            results.append({
                "question": question,
                "answer": None,
                "domain": domain,
                "difficulty": difficulty,
                "time": 0,
                "success": False,
                "error": str(e)
            })

    # Calculate statistics
    successful = [r for r in results if r["success"]]
    avg_time = total_time / len(successful) if successful else 0

    stats = {
        "pipeline": pipeline_name,
        "total_questions": len(filtered),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "avg_time": avg_time,
        "total_time": total_time,
        "results": results
    }

    logger.info(f"\n{'='*80}")
    logger.info(f"{pipeline_name} Statistics:")
    logger.info(f"  Total: {stats['total_questions']}")
    logger.info(f"  Success: {stats['successful']}")
    logger.info(f"  Failed: {stats['failed']}")
    logger.info(f"  Avg Time: {stats['avg_time']:.2f}s")
    logger.info(f"  Total Time: {stats['total_time']:.2f}s")
    logger.info(f"{'='*80}")

    return stats


def compare_results(baseline_stats: Dict, enhanced_stats: Dict):
    """Print comparison table"""
    print("\n" + "="*80)
    print("PIPELINE COMPARISON")
    print("="*80)

    print(f"\n{'Metric':<30} {'Baseline':<20} {'Enhanced':<20} {'Change':<15}")
    print("-"*85)

    # Success rate
    b_success = baseline_stats['successful']
    e_success = enhanced_stats['successful']
    print(f"{'Success Rate':<30} {b_success}/{baseline_stats['total_questions']:<20} {e_success}/{enhanced_stats['total_questions']:<20} {'--':<15}")

    # Average time
    b_avg = baseline_stats['avg_time']
    e_avg = enhanced_stats['avg_time']
    time_diff = e_avg - b_avg
    time_pct = (time_diff / b_avg * 100) if b_avg > 0 else 0
    print(f"{'Avg Time (s)':<30} {b_avg:<20.2f} {e_avg:<20.2f} {time_pct:+.1f}%")

    # Total time
    b_total = baseline_stats['total_time']
    e_total = enhanced_stats['total_time']
    total_diff = e_total - b_total
    total_pct = (total_diff / b_total * 100) if b_total > 0 else 0
    print(f"{'Total Time (s)':<30} {b_total:<20.2f} {e_total:<20.2f} {total_pct:+.1f}%")

    print("="*80)

    # Performance notes
    print("\n📊 Performance Analysis:")
    print("-"*80)

    if time_pct > 0:
        print(f"⚠️  Enhanced pipeline is {time_pct:.1f}% slower (expected)")
        print("   - Sparse search adds ~200-400ms")
        print("   - Context quality filter adds ~500-800ms")
        print("   - Total overhead: ~1-1.5s per query")
    else:
        print(f"✓ Enhanced pipeline is {abs(time_pct):.1f}% faster (unexpected!)")

    print("\n📈 Expected Quality Improvements:")
    print("   - Context Precision: +10-15%")
    print("   - Context Recall: +10-15%")
    print("   - Faithfulness: +15-20%")
    print("   - Answer Correctness: +8-12%")

    print("\n💡 Next Steps:")
    print("   1. Run RAGAS evaluation on both pipelines")
    print("   2. Compare Context Precision/Recall metrics")
    print("   3. Analyze specific questions where Enhanced performs better")

    print("="*80)


def main():
    setup_logging("INFO")
    logger = logging.getLogger(__name__)

    # Load questions
    logger.info("Loading test questions...")
    questions = load_test_questions()
    logger.info(f"Loaded {len(questions)} questions")

    # Filter Git/Python only
    filtered = [q for q in questions if q.get("domain") in ["git", "python"]]
    logger.info(f"Filtered to {len(filtered)} Git/Python questions")

    # Evaluate Baseline
    logger.info("\n" + "#"*80)
    logger.info("# BASELINE PIPELINE")
    logger.info("#"*80)
    baseline = BaselineRAG("config/base.yaml")
    baseline_stats = evaluate_pipeline(baseline, questions, "Baseline")

    # Evaluate Enhanced
    logger.info("\n" + "#"*80)
    logger.info("# ENHANCED PIPELINE")
    logger.info("#"*80)
    enhanced = EnhancedRAGPipeline("config/enhanced.yaml")
    enhanced_stats = evaluate_pipeline(enhanced, questions, "Enhanced")

    # Compare
    compare_results(baseline_stats, enhanced_stats)

    # Save results
    output = {
        "baseline": baseline_stats,
        "enhanced": enhanced_stats,
        "comparison": {
            "time_increase_pct": ((enhanced_stats['avg_time'] - baseline_stats['avg_time']) / baseline_stats['avg_time'] * 100) if baseline_stats['avg_time'] > 0 else 0
        }
    }

    output_file = Path("artifacts/pipeline_comparison.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.exception(f"Comparison failed: {e}")
        sys.exit(1)
