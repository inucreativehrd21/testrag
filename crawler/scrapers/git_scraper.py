"""
Git 문서 크롤러 - URL 태깅 포함
"""
import json
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import TARGET_URLS, GIT_RAW_DIR
from scrapers.base_scraper import BaseScraper
from scrapers.content_extractor import QualityContentExtractor
from utils.logger import get_logger

logger = get_logger(__name__)


class GitDocsScraper(BaseScraper):
    """Git 문서 크롤러"""

    def __init__(self):
        super().__init__()
        self.extractor = QualityContentExtractor()
        self.doc_type = "git"

    def scrape_all(self) -> List[Dict]:
        """모든 Git 문서 크롤링"""
        logger.info("=" * 80)
        logger.info("Git 문서 크롤링 시작 (URL 태깅 포함)")
        logger.info("=" * 80)

        all_docs = []

        # 1. Atlassian Tutorials
        logger.info("\n[1/5] Atlassian Git Tutorials 크롤링...")
        atlassian_docs = self._scrape_atlassian()
        all_docs.extend(atlassian_docs)
        logger.info(f"  수집: {len(atlassian_docs)}개")

        # 2. Pro Git Korean
        logger.info("\n[2/5] Pro Git Korean (git-scm.com) 크롤링...")
        pro_git_docs = self._scrape_pro_git_ko()
        all_docs.extend(pro_git_docs)
        logger.info(f"  수집: {len(pro_git_docs)}개")

        # 3. GitHub Docs
        logger.info("\n[3/5] GitHub Docs 크롤링...")
        github_docs = self._scrape_github_docs()
        all_docs.extend(github_docs)
        logger.info(f"  수집: {len(github_docs)}개")

        # 4. W3Schools
        logger.info("\n[4/5] W3Schools Git 크롤링...")
        w3_docs = self._scrape_w3schools()
        all_docs.extend(w3_docs)
        logger.info(f"  수집: {len(w3_docs)}개")

        # 5. Git Official Reference Manual
        logger.info("\n[5/5] Git Official Reference 크롤링...")
        reference_docs = self._scrape_official_reference()
        all_docs.extend(reference_docs)
        logger.info(f"  수집: {len(reference_docs)}개")

        logger.info(f"\n✓ Git 문서 수집 완료: 총 {len(all_docs)}개 (URL 포함)")

        # 전체 저장 (URL 포함)
        self._save_all(all_docs)

        return all_docs

    def _scrape_atlassian(self) -> List[Dict]:
        """Atlassian Tutorials 크롤링"""
        config = TARGET_URLS['git']['atlassian']
        base_url = config['base']
        pages = config['pages']

        docs = []

        for page in tqdm(pages, desc="Atlassian"):
            url = f"{base_url}/{page}"
            doc = self._scrape_page(url)
            if doc:
                docs.append(doc)

        return docs

    def _scrape_pro_git_ko(self) -> List[Dict]:
        """Pro Git Korean (git-scm.com) 크롤링"""
        config = TARGET_URLS['git']['pro_git_ko']
        base_url = config['base']
        pages = config['pages']

        docs = []

        for page in tqdm(pages, desc="Pro Git KO"):
            url = f"{base_url}{page}"
            doc = self._scrape_page(url)
            if doc:
                docs.append(doc)

        return docs

    def _scrape_github_docs(self) -> List[Dict]:
        """GitHub Docs 크롤링"""
        config = TARGET_URLS['git']['github_docs']
        base_url = config['base']
        pages = config['pages']

        docs = []

        for page in tqdm(pages, desc="GitHub Docs"):
            url = f"{base_url}/{page}"
            doc = self._scrape_page(url)
            if doc:
                docs.append(doc)

        return docs

    def _scrape_w3schools(self) -> List[Dict]:
        """W3Schools Git 크롤링"""
        config = TARGET_URLS['git']['w3schools']
        base_url = config['base']
        pages = config['pages']

        docs = []

        for page in tqdm(pages, desc="W3Schools"):
            url = f"{base_url}/{page}"
            doc = self._scrape_page(url)
            if doc:
                docs.append(doc)

        return docs

    def _scrape_official_reference(self) -> List[Dict]:
        """Git Official Reference 크롤링"""
        config = TARGET_URLS['git']['official_reference']
        base_url = config['base']
        pages = config['pages']

        docs = []

        for page in tqdm(pages, desc="Git Reference"):
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
        output_file = GIT_RAW_DIR / "pages.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)
        logger.info(f"\n✓ 저장 완료: {output_file}")
        logger.info(f"  총 {len(docs)}개 문서 (URL 태깅 포함)")

        # metadata.json
        from datetime import datetime
        metadata = {
            "domain": "git",
            "total_pages": len(docs),
            "crawled_at": datetime.now().isoformat()
        }
        meta_file = GIT_RAW_DIR / "metadata.json"
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
