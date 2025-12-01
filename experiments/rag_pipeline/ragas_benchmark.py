#!/usr/bin/env python3
"""
Ragas Benchmark for RAG Pipeline
Evaluates RAG system using Ragas metrics: context precision, context recall,
faithfulness, and answer relevancy.

Usage:
    python ragas_benchmark.py
    python ragas_benchmark.py --questions custom_questions.json
    python ragas_benchmark.py --config config/base.yaml --output results.json
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
from dotenv import load_dotenv

# Ragas imports
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
    print("Error: ragas not installed. Install with: pip install ragas")
    sys.exit(1)

# Import RAG pipeline
from answerer import RAGPipeline

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class RagasBenchmark:
    """Ragas benchmark runner for RAG pipeline"""

    def __init__(
        self,
        config_path: str = "config/enhanced.yaml",
        questions_path: str = "ragas_evaluation_questions.json",
        output_dir: str = "artifacts/ragas_evals"
    ):
        """
        Initialize benchmark runner

        Args:
            config_path: Path to RAG pipeline config
            questions_path: Path to questions JSON file
            output_dir: Directory to save evaluation results
        """
        self.config_path = config_path
        self.questions_path = questions_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing Ragas Benchmark")
        logger.info(f"Config: {config_path}")
        logger.info(f"Questions: {questions_path}")
        logger.info(f"Output: {output_dir}")

        # Load RAG pipeline
        logger.info("Loading RAG pipeline...")
        self.pipeline = RAGPipeline(config_path)
        logger.info("RAG pipeline loaded")

        # Load questions
        logger.info("Loading questions...")
        self.questions_data = self._load_questions()
        logger.info(f"Loaded {len(self.questions_data['questions'])} questions")

    def _load_questions(self) -> Dict[str, Any]:
        """Load questions from JSON file"""
        with open(self.questions_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _get_rag_response(self, question: str) -> Dict[str, Any]:
        """
        Get RAG pipeline response for a question

        Args:
            question: Question text

        Returns:
            Dict with answer and contexts
        """
        # Classify question
        decision = self.pipeline.router.classify(question)

        # Retrieve contexts
        contexts_with_scores = self.pipeline.retrieve(question)

        # Extract context texts (handle both dict and str formats)
        contexts: List[str] = []
        for ctx in contexts_with_scores:
            if isinstance(ctx, dict):
                contexts.append(ctx.get("text", ""))
            else:
                contexts.append(str(ctx))

        # Generate answer using the retrieved contexts
        answer = self._generate_answer(question, contexts, decision)
        decision_data = asdict(decision)

        return {
            'answer': answer,
            'contexts': contexts,
            'decision': decision_data,
            'num_contexts': len(contexts)
        }

    def _generate_answer(self, question: str, contexts: List[str], decision) -> str:
        """Generate answer using existing contexts to avoid duplicate retrieval."""
        if not contexts:
            logger.warning("No contexts retrieved, returning fallback message")
            return "관련 문서를 찾지 못했습니다."

        context_block = "\n\n".join(f"근거 {i + 1}: {chunk}" for i, chunk in enumerate(contexts))
        messages = [
            {"role": "system", "content": self.pipeline.system_prompt},
            {
                "role": "user",
                "content": (
                    f"질문 유형: {decision.reason}\n질문: {question}\n\n"
                    f"컨텍스트:\n{context_block}\n\n"
                    "지침: 근거를 인용하며 한국어로 답변하고, 추가 확인이 필요하면 명시하세요."
                ),
            },
        ]

        llm_cfg = self.pipeline.llm_cfg
        response = self.pipeline.llm_client.chat.completions.create(
            model=llm_cfg["model_name"],
            messages=messages,
            temperature=llm_cfg.get("temperature", 0.2),
            top_p=llm_cfg.get("top_p", 0.9),
            max_tokens=llm_cfg.get("max_new_tokens", 300),
        )
        return response.choices[0].message.content.strip()

    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run Ragas evaluation on all questions

        Returns:
            Dictionary with evaluation results
        """
        logger.info("="*60)
        logger.info("Starting Ragas Evaluation")
        logger.info("="*60)

        # Prepare data for Ragas
        questions = []
        ground_truths = []
        answers = []
        contexts = []

        # Additional metadata
        metadata = []

        for idx, q_data in enumerate(self.questions_data['questions'], 1):
            question = q_data['question']
            ground_truth = q_data['ground_truth']
            difficulty = q_data['difficulty']
            domain = q_data['domain']

            logger.info(f"\n[{idx}/{len(self.questions_data['questions'])}] Processing question")
            logger.info(f"Domain: {domain} | Difficulty: {difficulty}")
            logger.info(f"Question: {question}")

            # Get RAG response
            response = self._get_rag_response(question)

            logger.info(f"Answer: {response['answer'][:100]}...")
            logger.info(f"Retrieved {response['num_contexts']} contexts")

            # Collect data
            questions.append(question)
            ground_truths.append(ground_truth)
            answers.append(response['answer'])
            contexts.append(response['contexts'])

            metadata.append({
                'id': q_data['id'],
                'difficulty': difficulty,
                'domain': domain,
                'decision_difficulty': response['decision'].get('difficulty'),
                'decision_strategy': response['decision'].get('strategy'),
                'decision_reason': response['decision'].get('reason'),
                'num_contexts': response['num_contexts']
            })

        logger.info("\n" + "="*60)
        logger.info("Running Ragas Metrics Evaluation...")
        logger.info("="*60)

        # Create Ragas dataset
        data = {
            'question': questions,
            'answer': answers,
            'contexts': contexts,
            'ground_truth': ground_truths
        }

        dataset = Dataset.from_dict(data)

        # Run evaluation
        logger.info("Computing metrics (this may take a few minutes)...")

        result = evaluate(
            dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
                answer_correctness,
            ],
        )

        logger.info("Metrics computation complete!")

        # Convert to DataFrame and add original data
        results_df = result.to_pandas()

        # Add original question data (Ragas may not include these in output)
        if 'question' not in results_df.columns:
            results_df['question'] = questions
        if 'answer' not in results_df.columns:
            results_df['answer'] = answers
        if 'contexts' not in results_df.columns:
            results_df['contexts'] = contexts
        if 'ground_truth' not in results_df.columns:
            results_df['ground_truth'] = ground_truths

        # Add metadata
        for key in metadata[0].keys():
            results_df[key] = [m[key] for m in metadata]

        # Calculate summary statistics (Ragas returns lists, so we compute means)
        def safe_mean(values):
            """Calculate mean, handling both lists and scalars"""
            if isinstance(values, (list, tuple)):
                valid_values = [v for v in values if v is not None and not pd.isna(v)]
                return sum(valid_values) / len(valid_values) if valid_values else 0.0
            return float(values) if values is not None else 0.0

        summary = {
            'context_precision': safe_mean(result['context_precision']),
            'context_recall': safe_mean(result['context_recall']),
            'faithfulness': safe_mean(result['faithfulness']),
            'answer_relevancy': safe_mean(result['answer_relevancy']),
            'answer_correctness': safe_mean(result['answer_correctness']),
        }

        logger.info(f"\nSummary Metrics:")
        logger.info(f"  Context Precision:   {summary['context_precision']:.4f}")
        logger.info(f"  Context Recall:      {summary['context_recall']:.4f}")
        logger.info(f"  Faithfulness:        {summary['faithfulness']:.4f}")
        logger.info(f"  Answer Relevancy:    {summary['answer_relevancy']:.4f}")
        logger.info(f"  Answer Correctness:  {summary['answer_correctness']:.4f}")

        return {
            'dataset': result,
            'dataframe': results_df,
            'metadata': metadata,
            'summary': summary
        }

    def save_results(self, results: Dict[str, Any], output_name: str = None) -> str:
        """
        Save evaluation results to files

        Args:
            results: Results dictionary from run_evaluation
            output_name: Optional custom output name

        Returns:
            Path to saved results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if output_name:
            base_name = output_name
        else:
            base_name = f"ragas_eval_{timestamp}"

        # Save detailed results as JSON
        json_path = self.output_dir / f"{base_name}.json"

        # Prepare JSON-serializable results
        # Convert DataFrame to dict, handling complex types
        df_dict = results['dataframe'].copy()

        # Convert contexts (list of lists) to JSON-serializable format
        if 'contexts' in df_dict.columns:
            df_dict['contexts'] = df_dict['contexts'].apply(
                lambda x: x if isinstance(x, list) else []
            )

        json_results = {
            'timestamp': timestamp,
            'config_path': str(self.config_path),
            'questions_path': str(self.questions_path),
            'num_questions': len(self.questions_data['questions']),
            'summary': {k: float(v) if v is not None else 0.0
                       for k, v in results['summary'].items()},
            'detailed_results': df_dict.to_dict(orient='records')
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Saved JSON results to: {json_path}")

        # Save as CSV (convert complex types to strings)
        csv_path = self.output_dir / f"{base_name}.csv"
        df_csv = results['dataframe'].copy()

        # Convert contexts list to string for CSV
        if 'contexts' in df_csv.columns:
            df_csv['contexts'] = df_csv['contexts'].apply(
                lambda x: str(x) if not isinstance(x, str) else x
            )

        df_csv.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"Saved CSV results to: {csv_path}")

        # Save summary report
        report_path = self.output_dir / f"{base_name}_report.txt"
        self._save_report(results, report_path)
        logger.info(f"Saved report to: {report_path}")

        return str(json_path)

    def _save_report(self, results: Dict[str, Any], report_path: Path):
        """Generate and save a human-readable report"""
        df = results['dataframe']
        summary = results['summary']

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("RAGAS EVALUATION REPORT\n")
            f.write("="*70 + "\n\n")

            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Config: {self.config_path}\n")
            f.write(f"Questions: {self.questions_path}\n")
            f.write(f"Total Questions: {len(df)}\n\n")

            # Overall metrics
            f.write("-"*70 + "\n")
            f.write("OVERALL METRICS\n")
            f.write("-"*70 + "\n")
            f.write(f"Context Precision:   {summary['context_precision']:.4f}\n")
            f.write(f"Context Recall:      {summary['context_recall']:.4f}\n")
            f.write(f"Faithfulness:        {summary['faithfulness']:.4f}\n")
            f.write(f"Answer Relevancy:    {summary['answer_relevancy']:.4f}\n")
            f.write(f"Answer Correctness:  {summary['answer_correctness']:.4f}\n\n")

            # Metrics by difficulty
            f.write("-"*70 + "\n")
            f.write("METRICS BY DIFFICULTY\n")
            f.write("-"*70 + "\n")

            if 'difficulty' in df.columns:
                for difficulty in ['easy', 'medium', 'hard']:
                    subset = df[df['difficulty'] == difficulty]
                    if len(subset) > 0:
                        f.write(f"\n{difficulty.upper()} ({len(subset)} questions):\n")
                        if 'context_precision' in df.columns:
                            f.write(f"  Context Precision:   {subset['context_precision'].mean():.4f}\n")
                        if 'context_recall' in df.columns:
                            f.write(f"  Context Recall:      {subset['context_recall'].mean():.4f}\n")
                        if 'faithfulness' in df.columns:
                            f.write(f"  Faithfulness:        {subset['faithfulness'].mean():.4f}\n")
                        if 'answer_relevancy' in df.columns:
                            f.write(f"  Answer Relevancy:    {subset['answer_relevancy'].mean():.4f}\n")
                        if 'answer_correctness' in df.columns:
                            f.write(f"  Answer Correctness:  {subset['answer_correctness'].mean():.4f}\n")

            # Metrics by domain
            f.write("\n" + "-"*70 + "\n")
            f.write("METRICS BY DOMAIN\n")
            f.write("-"*70 + "\n")

            if 'domain' in df.columns:
                for domain in df['domain'].unique():
                    subset = df[df['domain'] == domain]
                    f.write(f"\n{domain.upper()} ({len(subset)} questions):\n")
                    if 'context_precision' in df.columns:
                        f.write(f"  Context Precision:   {subset['context_precision'].mean():.4f}\n")
                    if 'context_recall' in df.columns:
                        f.write(f"  Context Recall:      {subset['context_recall'].mean():.4f}\n")
                    if 'faithfulness' in df.columns:
                        f.write(f"  Faithfulness:        {subset['faithfulness'].mean():.4f}\n")
                    if 'answer_relevancy' in df.columns:
                        f.write(f"  Answer Relevancy:    {subset['answer_relevancy'].mean():.4f}\n")
                    if 'answer_correctness' in df.columns:
                        f.write(f"  Answer Correctness:  {subset['answer_correctness'].mean():.4f}\n")

            # Top and bottom performers
            f.write("\n" + "-"*70 + "\n")
            f.write("TOP 3 QUESTIONS (by answer_relevancy)\n")
            f.write("-"*70 + "\n")

            if 'answer_relevancy' in df.columns and len(df) > 0:
                top_3 = df.nlargest(min(3, len(df)), 'answer_relevancy')
                for idx, row in top_3.iterrows():
                    q_id = row.get('id', idx)
                    question = str(row.get('question', ''))[:60]
                    relevancy = row.get('answer_relevancy', 0.0)
                    faith = row.get('faithfulness', 0.0)

                    f.write(f"\nQ{q_id}: {question}...\n")
                    f.write(f"  Relevancy: {relevancy:.4f} | ")
                    f.write(f"Faithfulness: {faith:.4f}\n")

            f.write("\n" + "-"*70 + "\n")
            f.write("BOTTOM 3 QUESTIONS (by answer_relevancy)\n")
            f.write("-"*70 + "\n")

            if 'answer_relevancy' in df.columns and len(df) > 0:
                bottom_3 = df.nsmallest(min(3, len(df)), 'answer_relevancy')
                for idx, row in bottom_3.iterrows():
                    q_id = row.get('id', idx)
                    question = str(row.get('question', ''))[:60]
                    relevancy = row.get('answer_relevancy', 0.0)
                    faith = row.get('faithfulness', 0.0)

                    f.write(f"\nQ{q_id}: {question}...\n")
                    f.write(f"  Relevancy: {relevancy:.4f} | ")
                    f.write(f"Faithfulness: {faith:.4f}\n")

            f.write("\n" + "="*70 + "\n")

    def print_summary(self, results: Dict[str, Any]):
        """Print summary to console"""
        summary = results['summary']
        df = results['dataframe']

        print("\n" + "="*70)
        print("RAGAS EVALUATION SUMMARY")
        print("="*70)
        print(f"\nOverall Metrics:")
        print(f"  Context Precision:   {summary['context_precision']:.4f}")
        print(f"  Context Recall:      {summary['context_recall']:.4f}")
        print(f"  Faithfulness:        {summary['faithfulness']:.4f}")
        print(f"  Answer Relevancy:    {summary['answer_relevancy']:.4f}")
        print(f"  Answer Correctness:  {summary['answer_correctness']:.4f}")

        print(f"\nBy Difficulty:")
        for difficulty in ['easy', 'medium', 'hard']:
            subset = df[df['difficulty'] == difficulty]
            if len(subset) > 0:
                avg_relevancy = subset['answer_relevancy'].mean()
                print(f"  {difficulty.capitalize():8} ({len(subset):2} questions): {avg_relevancy:.4f}")

        print("="*70 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run Ragas benchmark evaluation on RAG pipeline"
    )
    parser.add_argument(
        '--config',
        default='config/base.yaml',
        help='Path to RAG pipeline config'
    )
    parser.add_argument(
        '--questions',
        default='ragas_questions.json',
        help='Path to questions JSON file'
    )
    parser.add_argument(
        '--output-dir',
        default='artifacts/ragas_evals',
        help='Output directory for results'
    )
    parser.add_argument(
        '--output-name',
        help='Custom output file name (without extension)'
    )

    args = parser.parse_args()

    # Check OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        logger.error("OPENAI_API_KEY not set in environment")
        sys.exit(1)

    # Run benchmark
    benchmark = RagasBenchmark(
        config_path=args.config,
        questions_path=args.questions,
        output_dir=args.output_dir
    )

    try:
        results = benchmark.run_evaluation()
        output_path = benchmark.save_results(results, args.output_name)
        benchmark.print_summary(results)

        logger.info(f"\n✓ Evaluation complete!")
        logger.info(f"Results saved to: {output_path}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
