#!/usr/bin/env python3
"""
Git + Python 크롤러 실행 스크립트
URL 태깅 포함
"""
import sys
from pathlib import Path

# Add crawler to path
sys.path.insert(0, str(Path(__file__).parent))

from scrapers.git_scraper import GitDocsScraper
from scrapers.python_scraper import PythonDocsScraper
from utils.logger import get_logger
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = get_logger(__name__)


def main():
    """메인 실행 함수"""
    print("="*80)
    print("RAG 문서 크롤러 (URL 태깅 포함)")
    print("="*80)
    print()

    choice = input("크롤링할 도메인을 선택하세요:\n1. Git\n2. Python\n3. 둘 다\n선택: ")

    if choice == "1" or choice == "3":
        logger.info("Git 문서 크롤링 시작...")
        git_scraper = GitDocsScraper()
        git_docs = git_scraper.scrape_all()
        logger.info(f"✓ Git 크롤링 완료: {len(git_docs)}개 문서")

    if choice == "2" or choice == "3":
        logger.info("\nPython 문서 크롤링 시작...")
        python_scraper = PythonDocsScraper()
        python_docs = python_scraper.scrape_all()
        logger.info(f"✓ Python 크롤링 완료: {len(python_docs)}개 문서")

    print()
    print("="*80)
    print("✓ 크롤링 완료!")
    print("="*80)
    print()
    print("다음 단계:")
    print("1. data_prep.py 실행 (청킹)")
    print("2. index_builder.py 실행 (벡터 인덱싱)")
    print()


if __name__ == "__main__":
    main()
