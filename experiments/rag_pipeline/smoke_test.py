"""
End-to-end smoke test for RAG pipeline.

This script validates the entire pipeline by:
1. Running data_prep.py to create chunks
2. Running index_builder.py to build the vector index
3. Running answerer.py with sample questions
4. Verifying all outputs and artifacts are created correctly

Usage:
    python smoke_test.py
    python smoke_test.py --config config/base.yaml
    python smoke_test.py --skip-prep  # Skip data prep if already done
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Sample questions for testing
SAMPLE_QUESTIONS = [
    "Git의 브랜치란 무엇인가요?",
    "Python의 리스트와 튜플의 차이는 무엇인가요?",
    "Docker 컨테이너의 개념을 설명해주세요.",
]


class SmokeTest:
    """End-to-end smoke test for RAG pipeline."""

    def __init__(self, config_path: str, skip_prep: bool = False, skip_index: bool = False):
        """
        Initialize smoke test.

        Args:
            config_path: Path to configuration file
            skip_prep: Skip data preparation step
            skip_index: Skip index building step
        """
        self.config_path = config_path
        self.skip_prep = skip_prep
        self.skip_index = skip_index

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        artifacts_dir = Path(self.config["project"]["artifacts_dir"])
        self.chunks_path = artifacts_dir / "chunks.parquet"
        self.chroma_path = artifacts_dir / "chroma_db"

    def run(self):
        """Run the complete smoke test."""
        logger.info("=" * 80)
        logger.info("Starting RAG Pipeline Smoke Test")
        logger.info("=" * 80)

        total_start = time.time()
        failed = False

        try:
            # Check prerequisites
            logger.info("\n[1/6] Checking prerequisites...")
            self._check_prerequisites()
            logger.info("✓ Prerequisites check passed")

            # Step 1: Data preparation
            if not self.skip_prep:
                logger.info("\n[2/6] Running data preparation...")
                self._run_data_prep()
                logger.info("✓ Data preparation completed")
            else:
                logger.info("\n[2/6] Skipping data preparation (--skip-prep flag)")

            # Verify chunks
            logger.info("\n[3/6] Verifying chunks...")
            self._verify_chunks()
            logger.info("✓ Chunks verification passed")

            # Step 2: Index building
            if not self.skip_index:
                logger.info("\n[4/6] Running index builder...")
                self._run_index_builder()
                logger.info("✓ Index building completed")
            else:
                logger.info("\n[4/6] Skipping index building (--skip-index flag)")

            # Verify index
            logger.info("\n[5/6] Verifying ChromaDB index...")
            self._verify_index()
            logger.info("✓ Index verification passed")

            # Step 3: Answer questions
            logger.info("\n[6/6] Testing question answering...")
            self._test_answering()
            logger.info("✓ Question answering test passed")

        except Exception as e:
            logger.exception(f"Smoke test failed: {e}")
            failed = True

        total_time = time.time() - total_start

        # Print summary
        logger.info("\n" + "=" * 80)
        if failed:
            logger.error(f"SMOKE TEST FAILED after {total_time:.2f}s")
            logger.info("=" * 80)
            return False
        else:
            logger.info(f"SMOKE TEST PASSED in {total_time:.2f}s")
            logger.info("=" * 80)
            logger.info("\nAll pipeline components are working correctly!")
            logger.info("You can now:")
            logger.info("  - Run individual questions: python answerer.py \"your question\"")
            logger.info("  - Start API server: python serve.py")
            logger.info("  - Run evaluations: python evaluate.py --questions questions.txt")
            return True

    def _check_prerequisites(self):
        """Check that all required environment variables and files exist."""
        # Check OPENAI_API_KEY
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY not set - LLM calls will fail")
            logger.warning("Set it with: export OPENAI_API_KEY=your-key-here")

        # Check raw data directory
        raw_dir = Path(self.config["data"]["raw_dir"])
        if not raw_dir.exists():
            raise FileNotFoundError(
                f"Raw data directory not found: {raw_dir}\n"
                f"Please create the directory and add your data files."
            )

        domains = self.config["data"].get("domains", [])
        if domains:
            for domain in domains:
                domain_dir = raw_dir / domain
                if not domain_dir.exists():
                    logger.warning(f"Domain directory not found: {domain_dir}")

    def _run_data_prep(self):
        """Run data preparation script."""
        from data_prep import DataPreparer

        preparer = DataPreparer(self.config_path)
        preparer.run()

    def _verify_chunks(self):
        """Verify that chunks were created correctly."""
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {self.chunks_path}")

        df = pd.read_parquet(self.chunks_path)
        logger.info(f"  Found {len(df)} chunks")

        # Check required columns
        required_cols = ["domain", "chunk_id", "text", "length"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Check data quality
        if df["text"].isna().any():
            raise ValueError("Found null text values in chunks")

        if len(df) == 0:
            raise ValueError("No chunks created - check your raw data")

        logger.info(f"  Domains: {df['domain'].unique().tolist()}")
        logger.info(f"  Avg chunk length: {df['length'].mean():.0f} chars")

    def _run_index_builder(self):
        """Run index builder script."""
        from index_builder import IndexBuilder

        builder = IndexBuilder(self.config_path)
        builder.run()

    def _verify_index(self):
        """Verify that ChromaDB index was created correctly."""
        if not self.chroma_path.exists():
            raise FileNotFoundError(f"ChromaDB path not found: {self.chroma_path}")

        import chromadb

        client = chromadb.PersistentClient(path=str(self.chroma_path))
        try:
            collection = client.get_collection(name="rag_chunks")
            count = collection.count()
            logger.info(f"  ChromaDB collection has {count} documents")

            if count == 0:
                raise ValueError("ChromaDB collection is empty")

        except Exception as e:
            raise RuntimeError(f"Failed to access ChromaDB collection: {e}")

    def _test_answering(self):
        """Test question answering with sample questions."""
        from answerer import RAGPipeline

        pipeline = RAGPipeline(self.config_path)

        for idx, question in enumerate(SAMPLE_QUESTIONS, 1):
            logger.info(f"\n  Question {idx}: {question}")
            start = time.time()

            try:
                answer = pipeline.answer(question)
                elapsed = time.time() - start

                logger.info(f"  Answer: {answer[:150]}...")
                logger.info(f"  Response time: {elapsed:.2f}s")

                if not answer or answer == "관련 문서를 찾지 못했습니다.":
                    logger.warning(f"  Warning: No relevant documents found for question {idx}")

            except Exception as e:
                raise RuntimeError(f"Failed to answer question {idx}: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end smoke test for RAG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full smoke test
    python smoke_test.py

    # Skip data prep (if already done)
    python smoke_test.py --skip-prep

    # Skip both prep and indexing (test only answering)
    python smoke_test.py --skip-prep --skip-index

    # Use custom config
    python smoke_test.py --config config/custom.yaml
        """
    )

    parser.add_argument("--config", default="config/base.yaml", help="Path to config file")
    parser.add_argument("--skip-prep", action="store_true", help="Skip data preparation step")
    parser.add_argument("--skip-index", action="store_true", help="Skip index building step")
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

    try:
        tester = SmokeTest(args.config, args.skip_prep, args.skip_index)
        success = tester.run()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.exception(f"Smoke test crashed: {e}")
        sys.exit(1)
