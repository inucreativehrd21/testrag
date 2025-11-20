"""
Enhanced RAG Pipeline with Hybrid Search + Context Quality Filter
Implements:
1. Hybrid Search (Dense + Sparse with RRF)
2. Context Quality Evaluation (Self-RAG style)
3. Improved retrieval pipeline

Performance improvements expected:
- Context Precision: +10-15%
- Context Recall: +10-15%
- Faithfulness: +15-20%
"""
import argparse
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

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


class EnhancedRAGPipeline:
    """Enhanced RAG Pipeline with Hybrid Search and Context Quality Filter"""

    def __init__(self, config_path: str):
        logger.info(f"Initializing EnhancedRAGPipeline with config: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        artifacts_dir = Path(self.config["project"]["artifacts_dir"])
        self.chroma_path = artifacts_dir / "chroma_db"
        logger.info(f"ChromaDB path: {self.chroma_path}")

        # Load embedding model
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

        # Connect to ChromaDB
        logger.info("Connecting to ChromaDB")
        client = chromadb.PersistentClient(path=str(self.chroma_path))
        self.collection = client.get_or_create_collection("rag_chunks", embedding_function=None)
        collection_count = self.collection.count()
        logger.info(f"ChromaDB collection 'rag_chunks' has {collection_count} documents")

        # Load rerankers
        self.retrieval_cfg = self.config["retrieval"]
        logger.info(f"Loading stage 1 reranker: {self.retrieval_cfg['rerankers']['stage1']['model_name']}")
        self.stage1_reranker = self._load_reranker(self.retrieval_cfg["rerankers"]["stage1"])
        logger.info(f"Loading stage 2 reranker: {self.retrieval_cfg['rerankers']['stage2']['model_name']}")
        self.stage2_reranker = self._load_reranker(self.retrieval_cfg["rerankers"]["stage2"])

        # Query router
        self.router = QueryRouter()
        logger.info("Query router initialized")

        # LLM client
        llm_cfg = self.config["llm"]
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY environment variable not set")
            raise RuntimeError("OPENAI_API_KEY not set")
        self.llm_client = OpenAI(api_key=api_key)
        self.llm_cfg = llm_cfg
        logger.info(f"LLM client initialized: model={llm_cfg['model_name']}")

        # System prompt
        system_path = Path(llm_cfg["system_prompt_path"])
        self.system_prompt = system_path.read_text(encoding="utf-8") if system_path.exists() else ""
        logger.info(f"System prompt loaded from {system_path}")

        # Context quality filter config
        self.quality_cfg = self.config.get("context_quality", {})
        self.enable_quality_filter = self.quality_cfg.get("enabled", True)
        self.quality_threshold = self.quality_cfg.get("threshold", 0.6)
        logger.info(f"Context quality filter: enabled={self.enable_quality_filter}, threshold={self.quality_threshold}")

        logger.info("EnhancedRAGPipeline initialization complete")

    def _load_reranker(self, cfg):
        """Load cross-encoder reranker model"""
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
        """Rerank documents using cross-encoder"""
        pairs = [[query, doc] for doc in docs]
        scores = reranker.predict(pairs)
        if torch.is_tensor(scores):
            scores = scores.detach().cpu().numpy()
        idx = np.argsort(scores)[::-1][:top_k]
        return [docs[i] for i in idx]

    def _sparse_search(self, query_sparse_vector: Dict[int, float], documents: List[str],
                      doc_ids: List[str], top_k: int = 50) -> List[Tuple[str, str, float]]:
        """
        Perform sparse (BM25-like) search using BGE-M3's lexical weights

        Args:
            query_sparse_vector: {token_id: weight} from BGE-M3
            documents: List of document texts
            doc_ids: List of document IDs
            top_k: Number of top results to return

        Returns:
            List of (doc_id, doc_text, sparse_score) tuples
        """
        logger.info(f"Performing sparse search with {len(query_sparse_vector)} query terms")

        # Get sparse vectors for all documents
        doc_sparse_vectors = self.embedding_model.encode(
            documents,
            batch_size=32,
            max_length=1024,
            return_dense=False,
            return_sparse=True
        )["lexical_weights"]

        # Calculate sparse scores (inner product of sparse vectors)
        scores = []
        for doc_sparse in doc_sparse_vectors:
            score = 0.0
            for token_id, query_weight in query_sparse_vector.items():
                if token_id in doc_sparse:
                    score += query_weight * doc_sparse[token_id]
            scores.append(score)

        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(doc_ids[i], documents[i], scores[i]) for i in top_indices]

        logger.info(f"Sparse search: top score={results[0][2]:.4f}, bottom score={results[-1][2]:.4f}")
        return results

    def _reciprocal_rank_fusion(self,
                                dense_results: List[Tuple[str, str, float]],
                                sparse_results: List[Tuple[str, str, float]],
                                k: int = 60) -> List[str]:
        """
        Reciprocal Rank Fusion (RRF) to combine dense and sparse results

        RRF formula: score(d) = sum(1 / (k + rank_i(d))) for all ranking lists i

        Args:
            dense_results: List of (doc_id, doc_text, score) from dense search
            sparse_results: List of (doc_id, doc_text, score) from sparse search
            k: RRF constant (default 60, from literature)

        Returns:
            List of document texts sorted by RRF score
        """
        logger.info(f"Applying RRF fusion with k={k}")
        rrf_scores = defaultdict(float)
        doc_texts = {}

        # Add dense rankings
        for rank, (doc_id, doc_text, _) in enumerate(dense_results):
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
            doc_texts[doc_id] = doc_text

        # Add sparse rankings
        for rank, (doc_id, doc_text, _) in enumerate(sparse_results):
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
            doc_texts[doc_id] = doc_text

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        fused_docs = [doc_texts[doc_id] for doc_id in sorted_ids]

        logger.info(f"RRF fusion: combined {len(dense_results)} dense + {len(sparse_results)} sparse → {len(fused_docs)} unique docs")
        return fused_docs

    def _evaluate_context_quality(self, question: str, contexts: List[str]) -> List[str]:
        """
        Evaluate context quality using LLM (Self-RAG style)

        Each context is evaluated as:
        - RELEVANT: Directly answers the question
        - PARTIAL: Contains related information
        - IRRELEVANT: Not related to the question

        Returns only RELEVANT and PARTIAL contexts
        """
        if not self.enable_quality_filter:
            logger.info("Context quality filter disabled, returning all contexts")
            return contexts

        logger.info(f"Evaluating quality of {len(contexts)} contexts")
        eval_start = time.time()

        evaluated = []
        relevance_scores = []

        for idx, ctx in enumerate(contexts):
            # Truncate context for evaluation (save tokens)
            ctx_preview = ctx[:800] if len(ctx) > 800 else ctx

            prompt = f"""질문: {question}

문서: {ctx_preview}

이 문서가 질문에 답하는 데 도움이 됩니까?

- RELEVANT: 질문에 직접 답할 수 있는 정보 포함
- PARTIAL: 질문과 관련된 정보 일부 포함
- IRRELEVANT: 질문과 무관

반드시 RELEVANT, PARTIAL, IRRELEVANT 중 하나만 출력하세요."""

            try:
                response = self.llm_client.chat.completions.create(
                    model="gpt-4o-mini",  # Use cheaper model for evaluation
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=20
                )

                label = response.choices[0].message.content.strip().upper()

                if label == "RELEVANT":
                    evaluated.append(ctx)
                    relevance_scores.append(1.0)
                    logger.debug(f"Context {idx+1}: RELEVANT")
                elif label == "PARTIAL":
                    evaluated.append(ctx)
                    relevance_scores.append(0.5)
                    logger.debug(f"Context {idx+1}: PARTIAL")
                else:
                    logger.debug(f"Context {idx+1}: IRRELEVANT (filtered out)")

            except Exception as e:
                logger.warning(f"Failed to evaluate context {idx+1}: {e}, keeping it")
                evaluated.append(ctx)
                relevance_scores.append(0.5)

        eval_time = time.time() - eval_start
        quality_ratio = len(evaluated) / len(contexts) if contexts else 0

        logger.info(f"Context quality evaluation: {len(evaluated)}/{len(contexts)} kept ({quality_ratio:.1%}) in {eval_time:.2f}s")

        # If too few contexts remain, log warning
        if len(evaluated) < 2:
            logger.warning(f"Only {len(evaluated)} relevant contexts found - may need query rewriting")

        return evaluated

    def retrieve(self, question: str) -> List[str]:
        """
        Enhanced retrieval with Hybrid Search and Context Quality Filter

        Pipeline:
        1. Encode query (dense + sparse)
        2. Dense search (top 50)
        3. Sparse search on retrieved docs
        4. RRF fusion
        5. Two-stage reranking
        6. Context quality evaluation

        Returns:
            List of top-k relevant document chunks
        """
        logger.info(f"Starting enhanced retrieval for question: {question[:100]}...")
        retrieval_start = time.time()

        # Step 1: Encode query (dense + sparse)
        encode_start = time.time()
        query_encoding = self.embedding_model.encode(
            [question],
            batch_size=1,
            max_length=self.config["embedding"].get("max_length", 1024),
            return_dense=True,
            return_sparse=True,  # Enable sparse encoding
            return_colbert_vecs=False
        )
        query_dense = query_encoding["dense_vecs"][0].tolist()
        query_sparse = query_encoding["lexical_weights"][0]
        encode_time = time.time() - encode_start
        logger.info(f"Query encoding: dense={len(query_dense)}D, sparse={len(query_sparse)} terms in {encode_time:.3f}s")

        # Step 2: Dense retrieval (get more candidates for sparse re-scoring)
        dense_top_k = self.retrieval_cfg.get("hybrid_dense_top_k", 50)
        dense_start = time.time()
        dense_results = self.collection.query(
            query_embeddings=[query_dense],
            n_results=dense_top_k
        )
        dense_docs = dense_results["documents"][0]
        dense_ids = dense_results["ids"][0]
        dense_time = time.time() - dense_start
        logger.info(f"Dense retrieval: {len(dense_docs)} docs in {dense_time:.3f}s")

        if not dense_docs:
            logger.warning("No documents retrieved from dense search")
            return []

        # Step 3: Sparse search on dense candidates
        sparse_start = time.time()
        sparse_top_k = self.retrieval_cfg.get("hybrid_sparse_top_k", 50)

        # Create (id, text, dense_score) tuples
        dense_scored = [(dense_ids[i], dense_docs[i], 1.0 / (i + 1)) for i in range(len(dense_docs))]

        # Perform sparse search
        sparse_scored = self._sparse_search(query_sparse, dense_docs, dense_ids, top_k=sparse_top_k)
        sparse_time = time.time() - sparse_start
        logger.info(f"Sparse search: {len(sparse_scored)} docs in {sparse_time:.3f}s")

        # Step 4: RRF Fusion
        rrf_start = time.time()
        rrf_k = self.retrieval_cfg.get("rrf_k", 60)
        fused_docs = self._reciprocal_rank_fusion(dense_scored, sparse_scored, k=rrf_k)
        rrf_time = time.time() - rrf_start
        logger.info(f"RRF fusion: {len(fused_docs)} docs in {rrf_time:.3f}s")

        # Take top 25 for reranking
        rerank_input_k = min(25, len(fused_docs))
        rerank_input = fused_docs[:rerank_input_k]

        # Step 5: Two-stage reranking
        stage1_start = time.time()
        stage1 = self._rerank(question, rerank_input, self.stage1_reranker, rerank_input_k)
        stage1_time = time.time() - stage1_start
        logger.info(f"Stage 1 reranking: {len(stage1)} docs in {stage1_time:.3f}s")

        stage2_start = time.time()
        final_k = self.retrieval_cfg.get("rerank_top_k", 5)
        stage2 = self._rerank(question, stage1, self.stage2_reranker, final_k)
        stage2_time = time.time() - stage2_start
        logger.info(f"Stage 2 reranking: {len(stage2)} docs in {stage2_time:.3f}s")

        # Step 6: Context quality evaluation
        quality_start = time.time()
        final_contexts = self._evaluate_context_quality(question, stage2)
        quality_time = time.time() - quality_start
        logger.info(f"Context quality filter: {len(final_contexts)}/{len(stage2)} kept in {quality_time:.3f}s")

        total_retrieval_time = time.time() - retrieval_start
        logger.info(f"Total enhanced retrieval time: {total_retrieval_time:.3f}s")

        return final_contexts

    def answer(self, question: str) -> str:
        """Generate answer with enhanced retrieval pipeline"""
        logger.info(f"Answering question: {question[:100]}...")
        total_start = time.time()

        # Route query
        decision = self.router.classify(question)
        logger.info(f"Query routing: difficulty={decision.difficulty}, strategy={decision.strategy}, reason={decision.reason}")

        # Retrieve contexts
        contexts = self.retrieve(question)
        if not contexts:
            logger.warning("No contexts retrieved, returning fallback message")
            return "관련 문서를 찾지 못했습니다. 질문을 다르게 표현해보시겠어요?"

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
    parser = argparse.ArgumentParser(description="Enhanced RAG with Hybrid Search")
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
        pipeline = EnhancedRAGPipeline(args.config)
        answer = pipeline.answer(args.question)
        print("\n" + "="*80)
        print("답변:")
        print("="*80)
        print(answer)
        print("="*80)
    except Exception as e:
        logger.exception(f"Answer generation failed: {e}")
        raise
