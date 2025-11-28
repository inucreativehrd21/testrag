"""
Python 문서 크롤러 - URL 태깅 포함
"""
import json
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import TARGET_URLS, PYTHON_RAW_DIR
from scrapers.base_scraper import BaseScraper
from scrapers.content_extractor import QualityContentExtractor
from utils.logger import get_logger

logger = get_logger(__name__)


class PythonDocsScraper(BaseScraper):
    """Python 문서 크롤러"""

    def __init__(self):
        super().__init__()
        self.extractor = QualityContentExtractor()
        self.doc_type = "python"

    def scrape_all(self) -> List[Dict]:
        """모든 Python 문서 크롤링"""
        logger.info("=" * 80)
        logger.info("Python 문서 크롤링 시작 (URL 태깅 포함)")
        logger.info("=" * 80)

        all_docs = []

        # 1. Real Python
        logger.info("\n[1/9] Real Python 크롤링...")
        realpython_docs = self._scrape_realpython()
        all_docs.extend(realpython_docs)
        logger.info(f"  수집: {len(realpython_docs)}개")

        # 2. Python 공식 튜토리얼
        logger.info("\n[2/9] Python Official Tutorial 크롤링...")
        official_tutorial_docs = self._scrape_official_tutorial()
        all_docs.extend(official_tutorial_docs)
        logger.info(f"  수집: {len(official_tutorial_docs)}개")

        # 3. W3Schools
        logger.info("\n[3/9] W3Schools Python 크롤링...")
        w3_docs = self._scrape_w3schools()
        all_docs.extend(w3_docs)
        logger.info(f"  수집: {len(w3_docs)}개")

        # 4. Python Docs - Library Reference
        logger.info("\n[4/9] Python Official Library 크롤링...")
        official_library_docs = self._scrape_official_library()
        all_docs.extend(official_library_docs)
        logger.info(f"  수집: {len(official_library_docs)}개")

        # 5. Python Official HOWTOs
        logger.info("\n[5/9] Python Official HOWTOs 크롤링...")
        official_howto_docs = self._scrape_official_howto()
        all_docs.extend(official_howto_docs)
        logger.info(f"  수집: {len(official_howto_docs)}개")

        # 6. GeeksforGeeks
        logger.info("\n[6/9] GeeksforGeeks Python 크롤링...")
        geeksforgeeks_docs = self._scrape_geeksforgeeks()
        all_docs.extend(geeksforgeeks_docs)
        logger.info(f"  수집: {len(geeksforgeeks_docs)}개")

        # 7. Programiz
        logger.info("\n[7/9] Programiz Python 크롤링...")
        programiz_docs = self._scrape_programiz()
        all_docs.extend(programiz_docs)
        logger.info(f"  수집: {len(programiz_docs)}개")

        # 8. PyMOTW
        logger.info("\n[8/9] PyMOTW 크롤링...")
        pymotw_docs = self._scrape_pymotw()
        all_docs.extend(pymotw_docs)
        logger.info(f"  수집: {len(pymotw_docs)}개")

        # 9. Python Advanced
        logger.info("\n[9/9] Python Official Advanced 크롤링...")
        official_advanced_docs = self._scrape_official_advanced()
        all_docs.extend(official_advanced_docs)
        logger.info(f"  수집: {len(official_advanced_docs)}개")

        logger.info(f"\n✓ Python 문서 수집 완료: 총 {len(all_docs)}개 (URL 포함)")

        # 전체 저장 (URL 포함)
        self._save_all(all_docs)

        return all_docs

    def _scrape_realpython(self) -> List[Dict]:
        config = TARGET_URLS['python']['realpython']
        return self._scrape_source(config, "Real Python")

    def _scrape_official_tutorial(self) -> List[Dict]:
        config = TARGET_URLS['python']['official_tutorial']
        return self._scrape_source(config, "Official Tutorial", use_chapters=True)

    def _scrape_w3schools(self) -> List[Dict]:
        config = TARGET_URLS['python']['w3schools']
        return self._scrape_source(config, "W3Schools")

    def _scrape_official_library(self) -> List[Dict]:
        config = TARGET_URLS['python']['official_library']
        return self._scrape_source(config, "Official Library")

    def _scrape_official_howto(self) -> List[Dict]:
        config = TARGET_URLS['python']['official_howto']
        return self._scrape_source(config, "Official HOWTO")

    def _scrape_geeksforgeeks(self) -> List[Dict]:
        config = TARGET_URLS['python']['geeksforgeeks']
        return self._scrape_source(config, "GeeksforGeeks")

    def _scrape_programiz(self) -> List[Dict]:
        config = TARGET_URLS['python']['programiz']
        return self._scrape_source(config, "Programiz")

    def _scrape_pymotw(self) -> List[Dict]:
        config = TARGET_URLS['python']['pymotw']
        return self._scrape_source(config, "PyMOTW")

    def _scrape_official_advanced(self) -> List[Dict]:
        config = TARGET_URLS['python']['official_advanced']
        return self._scrape_source(config, "Official Advanced")

    def _scrape_source(self, config: Dict, name: str, use_chapters: bool = False) -> List[Dict]:
        """공통 크롤링 로직"""
        base_url = config['base']
        pages = config.get('chapters' if use_chapters else 'pages', [])

        docs = []

        for page in tqdm(pages, desc=name):
            url = f"{base_url}/{page}"
            doc = self._scrape_page(url)
            if doc:
                docs.append(doc)

        return docs

    def _scrape_page(self, url: str) -> Dict:
        """개별 페이지 크롤링 (URL 포함)"""
        html = self.get_html(url)
        if not html:
            return None

        doc = self.extractor.extract(html, url, self.doc_type)
        return doc

    def _save_all(self, docs: List[Dict]):
        """전체 저장 (pages.json - URL 포함)"""
        output_file = PYTHON_RAW_DIR / "pages.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)
        logger.info(f"\n✓ 저장 완료: {output_file}")
        logger.info(f"  총 {len(docs)}개 문서 (URL 태깅 포함)")

        # metadata.json
        from datetime import datetime
        metadata = {
            "domain": "python",
            "total_pages": len(docs),
            "crawled_at": datetime.now().isoformat()
        }
        meta_file = PYTHON_RAW_DIR / "metadata.json"
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
