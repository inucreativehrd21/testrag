import argparse
import logging
import os
import time
from pathlib import Path
from typing import List

import chromadb
import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel
from openai import OpenAI
from sentence_transformers import CrossEncoder

from router import QueryRouter

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, config_path: str):
        logger.info(f"Initializing RAGPipeline with config: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        artifacts_dir = Path(self.config["project"]["artifacts_dir"])
        self.chroma_path = artifacts_dir / "chroma_db"
        logger.info(f"ChromaDB path: {self.chroma_path}")

        embedding_cfg = self.config["embedding"]
        device = embedding_cfg.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedding model {embedding_cfg['model_name']} on device: {device}")

        model_start = time.time()
        self.embedding_model = BGEM3FlagModel(
            embedding_cfg["model_name"],
            use_fp16=True,
            device=device
        )
        logger.info(f"Embedding model loaded in {time.time() - model_start:.2f}s")

        logger.info("Connecting to ChromaDB")
        # Use embedding_function=None since we provide our own embeddings
        client = chromadb.PersistentClient(path=str(self.chroma_path))
        self.collection = client.get_or_create_collection("rag_chunks", embedding_function=None)
        collection_count = self.collection.count()
        logger.info(f"ChromaDB collection 'rag_chunks' has {collection_count} documents")

        self.retrieval_cfg = self.config["retrieval"]
        logger.info(f"Loading stage 1 reranker: {self.retrieval_cfg['rerankers']['stage1']['model_name']}")
        self.stage1_reranker = self._load_reranker(self.retrieval_cfg["rerankers"]["stage1"])
        logger.info(f"Loading stage 2 reranker: {self.retrieval_cfg['rerankers']['stage2']['model_name']}")
        self.stage2_reranker = self._load_reranker(self.retrieval_cfg["rerankers"]["stage2"])

        self.router = QueryRouter()
        logger.info("Query router initialized")

        llm_cfg = self.config["llm"]
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY environment variable not set")
            raise RuntimeError("OPENAI_API_KEY not set")
        self.llm_client = OpenAI(api_key=api_key)
        self.llm_cfg = llm_cfg
        logger.info(f"LLM client initialized: model={llm_cfg['model_name']}")

        system_path = Path(llm_cfg["system_prompt_path"])
        self.system_prompt = system_path.read_text(encoding="utf-8") if system_path.exists() else ""
        logger.info(f"System prompt loaded from {system_path}")
        logger.info("RAGPipeline initialization complete")

    def _load_reranker(self, cfg):
        model = CrossEncoder(cfg["model_name"], device=cfg.get("device", "cpu"))
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer and tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                if hasattr(model.model, "resize_token_embeddings"):
                    model.model.resize_token_embeddings(len(tokenizer))
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
            tokenizer.padding_side = "right"
            if hasattr(model.model, "config"):
                model.model.config.pad_token_id = tokenizer.pad_token_id
        return model

    def _rerank(self, query: str, docs: List[str], reranker: CrossEncoder, top_k: int) -> List[str]:
        pairs = [[query, doc] for doc in docs]
        scores = reranker.predict(pairs)
        if torch.is_tensor(scores):
            scores = scores.detach().cpu().numpy()
        idx = np.argsort(scores)[::-1][:top_k]
        return [docs[i] for i in idx]

    def retrieve(self, question: str) -> List[str]:
        logger.info(f"Starting retrieval for question: {question[:100]}...")
        retrieval_start = time.time()

        # Encode query
        encode_start = time.time()
        query_emb = self.embedding_model.encode(
            [question],
            batch_size=1,
            max_length=self.config["embedding"].get("max_length", 1024),
            return_dense=True
        )["dense_vecs"].tolist()
        encode_time = time.time() - encode_start
        logger.info(f"Query encoding took {encode_time:.3f}s")

        # Dense retrieval
        dense_start = time.time()
        dense = self.collection.query(query_embeddings=query_emb, n_results=self.retrieval_cfg["dense_top_k"])
        docs = dense["documents"][0]
        dense_time = time.time() - dense_start
        logger.info(f"Dense retrieval: retrieved {len(docs)} documents in {dense_time:.3f}s")

        if not docs:
            logger.warning("No documents retrieved from dense search")
            return []

        # Stage 1 reranking
        stage1_start = time.time()
        stage1 = self._rerank(question, docs, self.stage1_reranker, self.retrieval_cfg["dense_top_k"])
        stage1_time = time.time() - stage1_start
        logger.info(f"Stage 1 reranking: {len(stage1)} documents in {stage1_time:.3f}s")

        # Stage 2 reranking
        stage2_start = time.time()
        final = self._rerank(question, stage1, self.stage2_reranker, self.retrieval_cfg["rerank_top_k"])
        stage2_time = time.time() - stage2_start
        logger.info(f"Stage 2 reranking: {len(final)} documents in {stage2_time:.3f}s")

        total_retrieval_time = time.time() - retrieval_start
        logger.info(f"Total retrieval time: {total_retrieval_time:.3f}s")
        return final

    def answer(self, question: str) -> str:
        logger.info(f"Answering question: {question[:100]}...")
        total_start = time.time()

        # Route query
        decision = self.router.classify(question)
        logger.info(f"Query routing: difficulty={decision.difficulty}, strategy={decision.strategy}, reason={decision.reason}")

        # Retrieve contexts
        contexts = self.retrieve(question)
        if not contexts:
            logger.warning("No contexts retrieved, returning fallback message")
            return "관련 문서를 찾지 못했습니다."

        # Format context
        context_block = "\n\n".join(f"근거 {i+1}: {chunk}" for i, chunk in enumerate(contexts))
        logger.info(f"Formatted {len(contexts)} contexts for LLM")

        # Call LLM
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"질문 유형: {decision.reason}\n질문: {question}\n\n컨텍스트:\n{context_block}\n\n지침: 근거를 인용하며 한국어로 답변하고, 추가 확인이 필요하면 명시하세요."}
        ]

        llm_start = time.time()
        logger.info(f"Calling LLM: model={self.llm_cfg['model_name']}, temperature={self.llm_cfg.get('temperature', 0.2)}")
        response = self.llm_client.chat.completions.create(
            model=self.llm_cfg["model_name"],
            messages=messages,
            temperature=self.llm_cfg.get("temperature", 0.2),
            top_p=self.llm_cfg.get("top_p", 0.9),
            max_tokens=self.llm_cfg.get("max_new_tokens", 300)
        )
        llm_time = time.time() - llm_start
        logger.info(f"LLM response received in {llm_time:.3f}s")

        answer_text = response.choices[0].message.content.strip()
        total_time = time.time() - total_start
        logger.info(f"Total answer time: {total_time:.3f}s (retrieval + LLM)")
        logger.info(f"Answer preview: {answer_text[:100]}...")

        return answer_text


def parse_args():
    parser = argparse.ArgumentParser(description="Serve optimized RAG answer")
    parser.add_argument("question", help="질문 문장")
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
        pipeline = RAGPipeline(args.config)
        answer = pipeline.answer(args.question)
        print(answer)
    except Exception as e:
        logger.exception(f"Answer generation failed: {e}")
        raise
