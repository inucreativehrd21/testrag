import argparse
import logging
import time
from pathlib import Path

import chromadb
import pandas as pd
import yaml
from FlagEmbedding import BGEM3FlagModel

logger = logging.getLogger(__name__)


class IndexBuilder:
    def __init__(self, config_path: str):
        logger.info(f"Initializing IndexBuilder with config: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        project = self.config["project"]
        self.artifacts_dir = Path(project["artifacts_dir"]).resolve()
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_path = self.artifacts_dir / "chunks.parquet"
        self.chroma_path = self.artifacts_dir / "chroma_db"
        logger.info(f"Chunks path: {self.chunks_path}")
        logger.info(f"ChromaDB path: {self.chroma_path}")

        embedding_cfg = self.config["embedding"]
        device = embedding_cfg.get("device", "auto")
        if device == "auto":
            device = "cuda" if BGEM3FlagModel.has_cuda() else "cpu"
        logger.info(f"Loading embedding model {embedding_cfg['model_name']} on device: {device}")

        model_start = time.time()
        self.embedding_model = BGEM3FlagModel(
            embedding_cfg["model_name"],
            use_fp16=True,
            device=device
        )
        model_load_time = time.time() - model_start
        logger.info(f"Embedding model loaded in {model_load_time:.2f}s")

        self.batch_size = embedding_cfg.get("batch_size", 32)
        self.max_length = embedding_cfg.get("max_length", 1024)
        logger.info(f"Embedding config: batch_size={self.batch_size}, max_length={self.max_length}")

    def run(self):
        logger.info("Starting index building pipeline")
        start_time = time.time()

        if not self.chunks_path.exists():
            logger.error(f"chunks.parquet not found at {self.chunks_path}")
            raise FileNotFoundError("chunks.parquet not found. Run data_prep.py first.")

        logger.info(f"Loading chunks from {self.chunks_path}")
        df = pd.read_parquet(self.chunks_path)
        logger.info(f"Loaded {len(df)} chunks")

        logger.info("Initializing ChromaDB client")
        client = chromadb.PersistentClient(path=str(self.chroma_path))

        # Delete existing collection if it exists
        try:
            client.delete_collection(name="rag_chunks")
            logger.info("Deleted existing collection")
        except Exception:
            logger.info("No existing collection to delete")

        # Create new collection with embedding_function=None since we provide our own embeddings
        collection = client.create_collection(name="rag_chunks", embedding_function=None)
        logger.info("Created new collection")

        batch_size = 512
        num_batches = (len(df) + batch_size - 1) // batch_size
        logger.info(f"Processing {len(df)} chunks in {num_batches} batches of {batch_size}")

        for batch_idx, start in enumerate(range(0, len(df), batch_size), 1):
            batch_start_time = time.time()
            batch = df.iloc[start:start + batch_size]

            logger.info(f"Batch {batch_idx}/{num_batches}: Encoding {len(batch)} chunks")
            embeddings = self.embedding_model.encode(
                batch["text"].tolist(),
                batch_size=self.batch_size,
                max_length=self.max_length,
                return_dense=True
            )["dense_vecs"]

            ids = batch["chunk_id"].tolist()
            # Include URL in metadata
            metadata_columns = ["domain", "length"]
            if "url" in batch.columns:
                metadata_columns.append("url")
            metadatas = batch[metadata_columns].to_dict(orient="records")
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=batch["text"].tolist(),
                metadatas=metadatas
            )

            batch_time = time.time() - batch_start_time
            logger.info(f"Batch {batch_idx}/{num_batches}: Completed in {batch_time:.2f}s")

        elapsed = time.time() - start_time
        logger.info(f"Index building complete: {len(df)} chunks indexed to {self.chroma_path}")
        logger.info(f"Total time: {elapsed:.2f}s")
        print(f"Indexed {len(df)} chunks into {self.chroma_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build chroma index")
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
        builder = IndexBuilder(args.config)
        builder.run()
    except Exception as e:
        logger.exception(f"Index building failed: {e}")
        raise
