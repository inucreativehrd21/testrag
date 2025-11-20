#!/usr/bin/env python3
"""
Quick test script for Enhanced RAG Pipeline
Tests: Hybrid Search + Context Quality Filter
"""
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from answerer_v2 import EnhancedRAGPipeline, setup_logging

def main():
    setup_logging("INFO")
    logger = logging.getLogger(__name__)

    # Test questions
    test_questions = [
        # Git easy
        "Git에서 마지막 커밋 메시지를 수정하려면?",

        # Git medium
        "여러 사람이 사용하는 릴리스 브랜치에서 충돌 없이 긴급 패치를 주입하려면?",

        # Python easy
        "Python에서 리스트 컴프리헨션으로 1부터 5까지 제곱 값을 만드는 코드는?",

        # Python medium
        "Python의 GIL(Global Interpreter Lock)이 무엇인가요?",
    ]

    try:
        logger.info("="*80)
        logger.info("Initializing Enhanced RAG Pipeline")
        logger.info("="*80)

        pipeline = EnhancedRAGPipeline("config/enhanced.yaml")

        for i, question in enumerate(test_questions, 1):
            logger.info("\n" + "="*80)
            logger.info(f"Test {i}/{len(test_questions)}: {question}")
            logger.info("="*80)

            answer = pipeline.answer(question)

            print(f"\n질문 {i}: {question}")
            print("-"*80)
            print("답변:")
            print(answer)
            print("="*80)
            print()

        logger.info("\n" + "="*80)
        logger.info("All tests completed successfully!")
        logger.info("="*80)

    except Exception as e:
        logger.exception(f"Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
