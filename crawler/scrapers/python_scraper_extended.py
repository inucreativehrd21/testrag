"""
Python 문서 크롤러 - 확장판 (500 URLs)
settings_extended.py 사용
"""
import json
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings_extended import TARGET_URLS, PYTHON_RAW_DIR
from scrapers.base_scraper import BaseScraper
from scrapers.content_extractor import QualityContentExtractor
from utils.logger import get_logger

logger = get_logger(__name__)


class PythonDocsScraperExtended(BaseScraper):
    """Python 문서 확장 크롤러 - 500개 페이지"""

    def __init__(self):
        super().__init__()
        self.extractor = QualityContentExtractor()
        self.doc_type = "python"

    def scrape_all(self) -> List[Dict]:
        """모든 Python 문서 크롤링 (12개 소스, 500개 페이지)"""
        logger.info("=" * 80)
        logger.info("Python 문서 확장 크롤링 시작 (500개 페이지 목표)")
        logger.info("=" * 80)

        all_docs = []
        python_sources = TARGET_URLS['python']
        total_sources = len(python_sources)

        for idx, (source_name, config) in enumerate(python_sources.items(), 1):
            logger.info(f"\n[{idx}/{total_sources}] {source_name} 크롤링...")
            docs = self._scrape_source(source_name, config)
            all_docs.extend(docs)
            logger.info(f"  수집: {len(docs)}개")

        self._save_all(all_docs)

        logger.info("\n" + "=" * 80)
        logger.info(f"✓ Python 문서 크롤링 완료: {len(all_docs)}개 수집")
        logger.info("=" * 80)

        return all_docs

    def _scrape_source(self, source_name: str, config: Dict) -> List[Dict]:
        """개별 소스 크롤링"""
        base_url = config['base']
        pages = config.get('pages', [])

        docs = []
        for page in tqdm(pages, desc=f"{source_name}"):
            # URL 조합
            if page.startswith('http'):
                url = page
            else:
                # trailing slash 처리
                separator = '' if base_url.endswith('/') or page.startswith('/') else '/'
                url = f"{base_url}{separator}{page}"

            try:
                doc = self._scrape_page(url)
                if doc:
                    docs.append(doc)
            except Exception as e:
                logger.warning(f"  ✗ 크롤링 실패 ({url}): {e}")
                continue

        return docs

    def _scrape_page(self, url: str) -> Dict:
        """개별 페이지 크롤링 (URL 태깅 포함)"""
        html = self.get_html(url)
        if not html:
            return None

        # 콘텐츠 추출 (URL 포함)
        doc = self.extractor.extract(html, url, self.doc_type)
        return doc

    def _save_all(self, docs: List[Dict]):
        """크롤링 결과 저장 (URL 태깅 포함)"""
        if not docs:
            logger.warning("저장할 문서가 없습니다.")
            return

        # pages.json 저장
        output_file = PYTHON_RAW_DIR / "pages.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)

        logger.info(f"\n✓ 저장 완료: {output_file}")
        logger.info(f"  - 총 {len(docs)}개 문서")
        logger.info(f"  - 모든 문서에 URL 태깅 포함")

        # metadata.json 저장
        metadata = {
            "total_documents": len(docs),
            "sources": list(TARGET_URLS['python'].keys()),
            "url_tagged": True,
            "crawl_version": "extended_v1"
        }

        metadata_file = PYTHON_RAW_DIR / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    scraper = PythonDocsScraperExtended()
    docs = scraper.scrape_all()
    print(f"\n✓ 총 {len(docs)}개 Python 문서 크롤링 완료!")
