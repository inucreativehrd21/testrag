"""
대규모 크롤링 실행 스크립트 - 확장판
Git 501개 + Python 500개 = 총 1001개 페이지
"""
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from scrapers.git_scraper_extended import GitDocsScraperExtended
from scrapers.python_scraper_extended import PythonDocsScraperExtended
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    print("=" * 80)
    print("🚀 대규모 RAG 크롤링 시작 - 확장판")
    print("=" * 80)
    print("\n크롤링 대상:")
    print("  - Git: 501개 페이지 (9개 소스)")
    print("  - Python: 500개 페이지 (12개 소스)")
    print("  - 총 1001개 페이지")
    print("\n특징:")
    print("  ✓ 모든 페이지에 URL 태깅")
    print("  ✓ 신뢰할 수 있는 고품질 소스만 선별")
    print("  ✓ 404 에러 최소화를 위한 검증된 URL")
    print("=" * 80)

    choice = input("\n크롤링할 도메인을 선택하세요:\n1. Git (501개)\n2. Python (500개)\n3. 둘 다 (1001개)\n선택: ")

    git_docs = []
    python_docs = []

    if choice == "1" or choice == "3":
        print("\n" + "=" * 80)
        print("📚 Git 문서 크롤링 시작 (501개 페이지)")
        print("=" * 80)

        git_scraper = GitDocsScraperExtended()
        git_docs = git_scraper.scrape_all()

        print(f"\n✓ Git 크롤링 완료: {len(git_docs)}개 수집")

    if choice == "2" or choice == "3":
        print("\n" + "=" * 80)
        print("🐍 Python 문서 크롤링 시작 (500개 페이지)")
        print("=" * 80)

        python_scraper = PythonDocsScraperExtended()
        python_docs = python_scraper.scrape_all()

        print(f"\n✓ Python 크롤링 완료: {len(python_docs)}개 수집")

    # 최종 요약
    print("\n" + "=" * 80)
    print("🎉 크롤링 완료!")
    print("=" * 80)

    if git_docs:
        print(f"\n📚 Git:")
        print(f"  - 수집: {len(git_docs)}개 문서")
        print(f"  - 저장: data/raw/git/pages.json")
        print(f"  - URL 태깅: ✓")

    if python_docs:
        print(f"\n🐍 Python:")
        print(f"  - 수집: {len(python_docs)}개 문서")
        print(f"  - 저장: data/raw/python/pages.json")
        print(f"  - URL 태깅: ✓")

    total = len(git_docs) + len(python_docs)
    print(f"\n📊 총 {total}개 문서 크롤링 완료!")

    print("\n" + "=" * 80)
    print("다음 단계:")
    print("  1. cd c:\\develop1\\test\\experiments\\rag_pipeline")
    print("  2. python data_prep.py --config config/enhanced.yaml")
    print("  3. python index_builder.py --config config/enhanced.yaml")
    print("  4. python answerer_v2_optimized.py \"테스트 질문\" --config config/enhanced.yaml")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  크롤링이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"크롤링 중 오류 발생: {e}", exc_info=True)
        print(f"\n❌ 오류 발생: {e}")
