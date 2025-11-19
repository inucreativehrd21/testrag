"""
Batch evaluation script for RAG pipeline.

This script runs a batch of questions through the RAG pipeline and saves
detailed results including retrieved chunks, answers, and performance metrics.

Usage:
    python evaluate.py --questions questions.txt
    python evaluate.py --questions-json questions.json
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from dotenv import load_dotenv

from answerer import RAGPipeline

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluates RAG pipeline on a batch of questions."""

    def __init__(self, config_path: str, output_dir: Optional[str] = None):
        """
        Initialize evaluator.

        Args:
            config_path: Path to configuration file
            output_dir: Directory for evaluation results (default: artifacts/evals)
        """
        logger.info(f"Initializing RAGEvaluator with config: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.pipeline = RAGPipeline(config_path)

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            artifacts_dir = Path(config["project"]["artifacts_dir"])
            self.output_dir = artifacts_dir / "evals"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Evaluation results will be saved to: {self.output_dir}")

    def evaluate_batch(self, questions: List[str], metadata: Optional[Dict] = None) -> str:
        """
        Evaluate a batch of questions.

        Args:
            questions: List of question strings
            metadata: Optional metadata to include in output

        Returns:
            Path to output JSONL file
        """
        logger.info(f"Starting batch evaluation of {len(questions)} questions")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"eval_{timestamp}.jsonl"

        results = []
        for idx, question in enumerate(questions, 1):
            logger.info(f"Evaluating question {idx}/{len(questions)}: {question[:80]}...")
            result = self._evaluate_single(question, idx)
            results.append(result)

            # Write incrementally to JSONL
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        # Generate summary
        summary = self._generate_summary(results, metadata)
        summary_path = self.output_dir / f"eval_{timestamp}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"Evaluation complete. Results saved to {output_path}")
        logger.info(f"Summary saved to {summary_path}")

        return str(output_path)

    def _evaluate_single(self, question: str, question_id: int) -> Dict:
        """
        Evaluate a single question.

        Args:
            question: Question string
            question_id: Numeric ID for the question

        Returns:
            Dictionary with evaluation results
        """
        start_time = time.time()

        # Route the query
        decision = self.pipeline.router.classify(question)

        # Retrieve contexts
        retrieval_start = time.time()
        contexts = self.pipeline.retrieve(question)
        retrieval_time = time.time() - retrieval_start

        # Generate answer
        answer_start = time.time()
        if contexts:
            context_block = "\n\n".join(f"근거 {i+1}: {chunk}" for i, chunk in enumerate(contexts))
            messages = [
                {"role": "system", "content": self.pipeline.system_prompt},
                {"role": "user", "content": f"질문 유형: {decision.reason}\n질문: {question}\n\n컨텍스트:\n{context_block}\n\n지침: 근거를 인용하며 한국어로 답변하고, 추가 확인이 필요하면 명시하세요."}
            ]

            response = self.pipeline.llm_client.chat.completions.create(
                model=self.pipeline.llm_cfg["model_name"],
                messages=messages,
                temperature=self.pipeline.llm_cfg.get("temperature", 0.2),
                top_p=self.pipeline.llm_cfg.get("top_p", 0.9),
                max_tokens=self.pipeline.llm_cfg.get("max_new_tokens", 300)
            )
            answer = response.choices[0].message.content.strip()
        else:
            answer = "관련 문서를 찾지 못했습니다."

        answer_time = time.time() - answer_start
        total_time = time.time() - start_time

        result = {
            "question_id": question_id,
            "question": question,
            "routing": {
                "difficulty": decision.difficulty,
                "strategy": decision.strategy,
                "reason": decision.reason,
            },
            "retrieved_chunks": contexts,
            "num_chunks_retrieved": len(contexts),
            "answer": answer,
            "metadata": {
                "retrieval_time_sec": round(retrieval_time, 3),
                "answer_time_sec": round(answer_time, 3),
                "total_time_sec": round(total_time, 3),
                "timestamp": datetime.now().isoformat(),
            }
        }

        return result

    def _generate_summary(self, results: List[Dict], metadata: Optional[Dict] = None) -> Dict:
        """Generate summary statistics from evaluation results."""
        total_questions = len(results)
        total_retrieval_time = sum(r["metadata"]["retrieval_time_sec"] for r in results)
        total_answer_time = sum(r["metadata"]["answer_time_sec"] for r in results)
        total_time = sum(r["metadata"]["total_time_sec"] for r in results)

        avg_chunks = sum(r["num_chunks_retrieved"] for r in results) / total_questions if total_questions > 0 else 0

        summary = {
            "evaluation_metadata": metadata or {},
            "total_questions": total_questions,
            "performance": {
                "avg_retrieval_time_sec": round(total_retrieval_time / total_questions, 3) if total_questions > 0 else 0,
                "avg_answer_time_sec": round(total_answer_time / total_questions, 3) if total_questions > 0 else 0,
                "avg_total_time_sec": round(total_time / total_questions, 3) if total_questions > 0 else 0,
                "total_time_sec": round(total_time, 3),
            },
            "retrieval_stats": {
                "avg_chunks_retrieved": round(avg_chunks, 2),
                "questions_with_no_chunks": sum(1 for r in results if r["num_chunks_retrieved"] == 0),
            },
            "routing_distribution": self._count_routing(results),
            "timestamp": datetime.now().isoformat(),
        }

        return summary

    def _count_routing(self, results: List[Dict]) -> Dict:
        """Count routing decisions."""
        difficulty_counts = {}
        strategy_counts = {}

        for result in results:
            diff = result["routing"]["difficulty"]
            strat = result["routing"]["strategy"]

            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
            strategy_counts[strat] = strategy_counts.get(strat, 0) + 1

        return {
            "by_difficulty": difficulty_counts,
            "by_strategy": strategy_counts,
        }


def load_questions_from_txt(file_path: str) -> List[str]:
    """Load questions from a text file (one per line)."""
    with open(file_path, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]
    return questions


def load_questions_from_json(file_path: str) -> List[str]:
    """
    Load questions from a JSON file.

    Expected format:
        {"questions": ["question1", "question2", ...]}
    or
        [{"question": "..."}, {"question": "..."}, ...]
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "questions" in data:
        return data["questions"]
    elif isinstance(data, list):
        if all(isinstance(item, str) for item in data):
            return data
        elif all(isinstance(item, dict) and "question" in item for item in data):
            return [item["question"] for item in data]

    raise ValueError(f"Unsupported JSON format in {file_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch evaluation for RAG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate from text file (one question per line)
    python evaluate.py --questions questions.txt

    # Evaluate from JSON file
    python evaluate.py --questions-json questions.json

    # Custom output directory
    python evaluate.py --questions questions.txt --output-dir results/

    # With custom config
    python evaluate.py --questions questions.txt --config config/custom.yaml
        """
    )

    parser.add_argument("--questions", help="Path to questions text file (one per line)")
    parser.add_argument("--questions-json", help="Path to questions JSON file")
    parser.add_argument("--config", default="config/base.yaml", help="Path to config file")
    parser.add_argument("--output-dir", help="Custom output directory for evaluation results")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return parser.parse_args()


def setup_logging(level: str = "INFO"):
    """Configure structured logging."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_level)

    # Load questions
    if args.questions:
        logger.info(f"Loading questions from text file: {args.questions}")
        questions = load_questions_from_txt(args.questions)
    elif args.questions_json:
        logger.info(f"Loading questions from JSON file: {args.questions_json}")
        questions = load_questions_from_json(args.questions_json)
    else:
        logger.error("No questions file provided. Use --questions or --questions-json")
        exit(1)

    logger.info(f"Loaded {len(questions)} questions")

    try:
        evaluator = RAGEvaluator(args.config, args.output_dir)
        output_path = evaluator.evaluate_batch(questions)
        print(f"\nEvaluation complete! Results saved to: {output_path}")
    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        raise
