"""
Local evaluation script for the RAG pipeline.

Runs a set of questions with ground-truth answers through the pipeline
and computes text-overlap metrics (precision, recall, F1, exact match).

Usage:
    python local_eval.py --dataset data/test_queries.json
    python local_eval.py --dataset custom.json --config config/base.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import yaml
from dotenv import load_dotenv

from answerer import RAGPipeline

load_dotenv()
logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^0-9a-z가-힣\s]", " ", text)
    return " ".join(text.split())


def compute_metrics(prediction: str, reference: str) -> Dict[str, float]:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())

    precision = overlap / len(pred_tokens) if pred_tokens else 0.0
    recall = overlap / len(ref_tokens) if ref_tokens else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    exact = 1.0 if normalize_text(prediction) == normalize_text(reference) and reference.strip() else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact,
    }


class LocalEvaluator:
    def __init__(self, config_path: str, output_dir: Path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.pipeline = RAGPipeline(config_path)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, dataset: List[Dict[str, str]]) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        detail_path = self.output_dir / f"local_eval_{timestamp}.jsonl"
        summary_path = self.output_dir / f"local_eval_{timestamp}_summary.json"

        results = []
        for idx, item in enumerate(dataset, 1):
            question = item["query"]
            answer = item["ground_truth"]
            domain = item.get("domain", "unknown")
            logger.info("(%d/%d) Evaluating domain=%s question=%s", idx, len(dataset), domain, question)
            start = time.time()
            response = self.pipeline.answer(question)
            latency = time.time() - start
            metrics = compute_metrics(response, answer)

            result = {
                "id": idx,
                "domain": domain,
                "question": question,
                "prediction": response,
                "reference": answer,
                "latency_sec": round(latency, 3),
                "metrics": metrics,
            }
            results.append(result)

            with open(detail_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        summary = self._summarize(results)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info("Evaluation complete. Details: %s", detail_path)
        logger.info("Summary: %s", summary_path)

    def _summarize(self, results: List[Dict]) -> Dict:
        by_domain = defaultdict(lambda: defaultdict(float))
        counts = defaultdict(int)

        for result in results:
            domain = result["domain"]
            counts[domain] += 1
            for metric, value in result["metrics"].items():
                by_domain[domain][metric] += value
            by_domain[domain]["latency_sec"] += result["latency_sec"]

        domain_summary = {}
        for domain, metrics in by_domain.items():
            n = counts[domain]
            domain_summary[domain] = {
                "questions": n,
                "avg_precision": round(metrics["precision"] / n, 3),
                "avg_recall": round(metrics["recall"] / n, 3),
                "avg_f1": round(metrics["f1"] / n, 3),
                "exact_match": round(metrics["exact_match"] / n, 3),
                "avg_latency_sec": round(metrics["latency_sec"] / n, 3),
            }

        if results:
            overall = {
                "questions": len(results),
                "avg_precision": round(sum(r["metrics"]["precision"] for r in results) / len(results), 3),
                "avg_recall": round(sum(r["metrics"]["recall"] for r in results) / len(results), 3),
                "avg_f1": round(sum(r["metrics"]["f1"] for r in results) / len(results), 3),
                "exact_match": round(sum(r["metrics"]["exact_match"] for r in results) / len(results), 3),
                "avg_latency_sec": round(sum(r["latency_sec"] for r in results) / len(results), 3),
            }
        else:
            overall = {}

        return {
            "overall": overall,
            "by_domain": domain_summary,
            "timestamp": datetime.now().isoformat(),
        }


def load_dataset(path: str) -> List[Dict[str, str]]:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        items = []
        for domain, entries in data.items():
            for entry in entries:
                items.append(
                    {
                        "domain": domain,
                        "query": entry["query"],
                        "ground_truth": entry["ground_truth"],
                    }
                )
        return items

    if isinstance(data, list):
        return data

    raise ValueError("Dataset format must be list or dict of lists.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local evaluation benchmark for RAG pipeline")
    parser.add_argument("--dataset", required=True, help="Path to JSON dataset with ground_truth fields")
    parser.add_argument("--config", default="config/base.yaml", help="Pipeline config file")
    parser.add_argument("--output-dir", default="artifacts/evals_local", help="Directory to store evaluation artifacts")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    dataset = load_dataset(args.dataset)
    logger.info("Loaded %d evaluation samples from %s", len(dataset), args.dataset)

    evaluator = LocalEvaluator(args.config, Path(args.output_dir))
    evaluator.evaluate(dataset)


if __name__ == "__main__":
    main()
