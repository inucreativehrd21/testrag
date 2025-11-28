import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DataPreparer:
    def __init__(self, config_path: str):
        logger.info(f"Initializing DataPreparer with config: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        project = self.config["project"]
        self.artifacts_dir = Path(project["artifacts_dir"]).resolve()
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Artifacts directory: {self.artifacts_dir}")
        self.raw_dir = Path(self.config["data"]["raw_dir"])
        self.domains = self.config["data"].get("domains", [])
        logger.info(f"Target domains: {self.domains}")
        chunk_cfg = self.config["chunking"]["primary"]
        self.chunker = RecursiveCharacterTextSplitter(
            chunk_size=chunk_cfg["chunk_size"],
            chunk_overlap=chunk_cfg["chunk_overlap"],
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        logger.info(f"Primary chunker: size={chunk_cfg['chunk_size']}, overlap={chunk_cfg['chunk_overlap']}")
        fallback_cfg = self.config["chunking"].get("fallback")
        self.fallback_chunker = None
        if fallback_cfg:
            self.fallback_chunker = RecursiveCharacterTextSplitter(
                chunk_size=fallback_cfg["chunk_size"],
                chunk_overlap=fallback_cfg["chunk_overlap"],
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            logger.info(f"Fallback chunker: size={fallback_cfg['chunk_size']}, overlap={fallback_cfg['chunk_overlap']}")

    def run(self):
        logger.info("Starting data preparation pipeline (with URL metadata)")
        start_time = time.time()

        # Load documents with URL metadata
        documents_with_metadata = self._load_documents_with_metadata()
        logger.info(f"Loaded {len(documents_with_metadata)} domain documents (with URLs)")

        chunk_rows = []
        for domain_data in documents_with_metadata:
            domain = domain_data["domain"]
            doc_chunks = domain_data["chunks"]  # List of {text, url}

            domain_start = time.time()

            for chunk_info in doc_chunks:
                chunk_text = chunk_info["text"]
                chunk_url = chunk_info["url"]

                # Split chunk if needed
                chunks = self.chunker.split_text(chunk_text)
                used_fallback = False
                if not chunks and self.fallback_chunker is not None:
                    chunks = self.fallback_chunker.split_text(chunk_text)
                    used_fallback = True

                # Add all sub-chunks with URL
                for idx, chunk in enumerate(chunks):
                    chunk_rows.append(
                        {
                            "domain": domain,
                            "chunk_id": f"{domain}_{len(chunk_rows)}",
                            "text": chunk,
                            "length": len(chunk),
                            "url": chunk_url  # ← URL 추가!
                        }
                    )

            domain_time = time.time() - domain_start
            logger.info(
                f"Domain '{domain}': {len([r for r in chunk_rows if r['domain'] == domain])} chunks created in {domain_time:.2f}s (with URLs)"
            )

        df = pd.DataFrame(chunk_rows)
        output_path = self.artifacts_dir / "chunks.parquet"
        df.to_parquet(output_path, index=False)

        elapsed = time.time() - start_time
        logger.info(f"Data preparation complete: {len(df)} total chunks saved to {output_path}")
        logger.info(f"Total time: {elapsed:.2f}s")
        print(f"Saved {len(df)} chunks to {output_path}")

    def _load_documents_with_metadata(self) -> List[Dict]:
        """
        Load documents with URL metadata

        Returns:
            [
                {
                    "domain": "python",
                    "chunks": [
                        {"text": "...", "url": "https://..."},
                        {"text": "...", "url": "https://..."},
                    ]
                },
                ...
            ]
        """
        logger.info(f"Loading documents with URL metadata from {self.raw_dir}")

        if not self.raw_dir.exists():
            logger.error(f"Raw data directory not found: {self.raw_dir}")
            raise FileNotFoundError(f"Raw data directory not found: {self.raw_dir}")

        target_domains = self.domains if self.domains else [
            d.name for d in self.raw_dir.iterdir() if d.is_dir()
        ]

        all_domain_data = []

        for domain in target_domains:
            domain_dir = self.raw_dir / domain
            if not domain_dir.exists():
                logger.warning(f"Domain directory not found: {domain_dir}")
                continue

            # pages.json 파일 찾기
            pages_file = domain_dir / "pages.json"
            if not pages_file.exists():
                logger.warning(f"pages.json not found in {domain_dir}")
                continue

            try:
                with open(pages_file, "r", encoding="utf-8") as f:
                    pages_data = json.load(f)

                chunks = []
                for page in pages_data:
                    url = page.get("url", "unknown")

                    # Extract all text from sections
                    page_texts = []

                    # Add title
                    if "title" in page:
                        page_texts.append(page["title"])

                    # Add summary
                    if "summary" in page:
                        page_texts.append(page["summary"])

                    # Add sections
                    if "sections" in page:
                        for section in page["sections"]:
                            if section.get("header"):
                                page_texts.append(section["header"])
                            if section.get("content"):
                                page_texts.append(section["content"])

                    # Combine and add to chunks
                    if page_texts:
                        combined_text = "\n\n".join(page_texts)
                        chunks.append({
                            "text": combined_text,
                            "url": url
                        })

                all_domain_data.append({
                    "domain": domain,
                    "chunks": chunks
                })

                logger.info(f"Domain '{domain}': loaded {len(chunks)} documents with URLs")

            except Exception as e:
                logger.error(f"Failed to load {pages_file}: {e}")
                continue

        return all_domain_data

    def _load_documents(self) -> Dict[str, str]:
        logger.info(f"Loading documents from {self.raw_dir}")
        documents: Dict[str, str] = {}
        if not self.raw_dir.exists():
            logger.error(f"Raw data directory not found: {self.raw_dir}")
            raise FileNotFoundError(f"Raw data directory not found: {self.raw_dir}")
        target_domains = self.domains if self.domains else [
            d.name for d in self.raw_dir.iterdir() if d.is_dir()
        ]
        for domain in target_domains:
            domain_dir = self.raw_dir / domain
            if not domain_dir.exists():
                logger.warning(f"Domain directory not found: {domain_dir}")
                continue
            combined = []
            file_count = 0
            for path in domain_dir.glob("**/*"):
                if path.is_dir() or path.name.startswith("."):
                    continue
                if path.suffix == ".json":
                    with open(path, "r", encoding="utf-8") as f:
                        try:
                            data = json.load(f)
                            file_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to parse JSON file {path}: {e}")
                            continue
                        combined.extend(self._extract_texts(data))
                else:
                    try:
                        combined.append(path.read_text(encoding="utf-8"))
                        file_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to read file {path}: {e}")
                        continue
            if combined:
                documents[domain] = "\n\n".join(combined)
                logger.info(f"Domain '{domain}': loaded {file_count} files, {len(combined)} text segments")
        return documents

    def _extract_texts(self, data) -> List[str]:
        """
        Recursively extract text content from nested JSON structures.
        Handles various formats including sections arrays, summaries, etc.
        """
        texts: List[str] = []

        if isinstance(data, list):
            for item in data:
                texts.extend(self._extract_texts(item))

        elif isinstance(data, dict):
            # Extract common text fields
            for key in ("summary", "content", "text", "body", "page_content", "description"):
                if key in data and data[key]:
                    text = str(data[key]).strip()
                    if text:
                        texts.append(text)

            # Handle nested sections/paragraphs/items arrays
            for key in ("sections", "paragraphs", "pages", "documents", "data", "items", "chunks"):
                if key in data and isinstance(data[key], list):
                    texts.extend(self._extract_texts(data[key]))

        elif data:
            text = str(data).strip()
            if text:
                texts.append(text)

        return [t for t in texts if t and t.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare chunked data for RAG pipeline")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def setup_logging(level: str = "INFO"):
    """Configure structured logging for the pipeline."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_level)
    try:
        preparer = DataPreparer(args.config)
        preparer.run()
    except Exception as e:
        logger.exception(f"Data preparation failed: {e}")
        raise
