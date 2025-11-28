#!/usr/bin/env python3
"""
Enhanced RAG Pipeline - OPTIMIZED VERSION for speed
Optimizations:
1. Async parallel context quality evaluation (5-10x faster)
2. Optimized metadata matching (O(n) → O(1))
3. Reduced logging overhead
4. Maintains exact same logic and quality
"""
import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import chromadb
import numpy as np
import yaml
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from openai import AsyncOpenAI, OpenAI

logger = logging.getLogger(__name__)


@dataclass
class QueryRoutingDecision:
    """Query routing decision with difficulty and strategy"""
    difficulty: str  # easy, medium, hard
    strategy: str    # standard, enhanced, multi_hop
    reason: str      # explanation


class SimpleQueryRouter:
    """Simple rule-based query router"""

    def __init__(self, llm_client: OpenAI, model: str = "gpt-4o-mini"):
        self.llm_client = llm_client
        self.model = model

    def classify(self, question: str) -> QueryRoutingDecision:
        """Classify query difficulty and determine strategy"""
        # Simple heuristic for now
        word_count = len(question.split())

        if word_count < 10:
            return QueryRoutingDecision("easy", "standard", "Short question")
        elif word_count < 20:
            return QueryRoutingDecision("medium", "standard", "Medium question")
        else:
            return QueryRoutingDecision("hard", "enhanced", "Complex question")


class EnhancedRAGPipeline:
    """Enhanced RAG Pipeline with Speed Optimizations"""

    def __init__(self, config_path: str):
        """Initialize pipeline with config"""
        logger.info(f"Initializing Enhanced RAG Pipeline from {config_path}")

        # Load config
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.retrieval_cfg = self.config.get("retrieval", {})
        self.llm_cfg = self.config.get("llm", {})

        # Context quality filter
        context_quality_cfg = self.config.get("context_quality", {})
        self.enable_quality_filter = context_quality_cfg.get("enabled", True)
        self.quality_evaluator_model = context_quality_cfg.get("evaluator_model", "gpt-4o-mini")

        # Embedding model
        embedding_cfg = self.config["embedding"]
        logger.info(f"Loading embedding model: {embedding_cfg['model_name']}")
        self.embedding_model = BGEM3FlagModel(
            embedding_cfg["model_name"],
            use_fp16=True,
            device=embedding_cfg.get("device", "cpu")
        )
        self.batch_size = embedding_cfg.get("batch_size", 32)

        # Rerankers
        reranker_cfg = self.retrieval_cfg.get("rerankers", {})
        stage1_cfg = reranker_cfg.get("stage1", {})
        stage2_cfg = reranker_cfg.get("stage2", {})

        logger.info(f"Loading stage 1 reranker: {stage1_cfg['model_name']}")
        self.stage1_reranker = FlagReranker(
            stage1_cfg["model_name"],
            use_fp16=True,
            device=stage1_cfg.get("device", "cpu")
        )

        logger.info(f"Loading stage 2 reranker: {stage2_cfg['model_name']}")
        self.stage2_reranker = FlagReranker(
            stage2_cfg["model_name"],
            use_fp16=True,
            device=stage2_cfg.get("device", "cpu")
        )

        # ChromaDB
        artifacts_dir = Path(self.config["project"]["artifacts_dir"])
        chroma_path = artifacts_dir / "chroma_db"

        logger.info(f"Connecting to ChromaDB at {chroma_path}")
        client = chromadb.PersistentClient(path=str(chroma_path))
        self.collection = client.get_collection("rag_chunks")
        logger.info(f"Collection loaded: {self.collection.count()} documents")

        # LLM clients (sync and async)
        self.llm_client = OpenAI()
        self.async_llm_client = AsyncOpenAI()

        # Query router
        self.router = SimpleQueryRouter(self.llm_client)

        # System prompt
        system_path = Path(self.llm_cfg["system_prompt_path"])
        self.system_prompt = system_path.read_text(encoding="utf-8") if system_path.exists() else ""
        logger.info(f"System prompt loaded from {system_path}")

        logger.info("✓ Enhanced RAG Pipeline initialized")

    def _sparse_search(self, query_sparse_vector: Dict, documents: List[str], doc_ids: List[str], top_k: int = 50) -> List[Tuple[str, str, float]]:
        """Perform sparse search using BGE-M3's lexical weights"""
        # Encode documents to get sparse vectors
        doc_encodings = self.embedding_model.encode(
            documents,
            batch_size=self.batch_size,
            max_length=1024,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False
        )

        # Calculate sparse scores (inner product)
        scores = []
        for i, doc_sparse in enumerate(doc_encodings["lexical_weights"]):
            score = 0.0
            for term, query_weight in query_sparse_vector.items():
                if term in doc_sparse:
                    score += query_weight * doc_sparse[term]
            scores.append((doc_ids[i], documents[i], score))

        # Sort by score
        scores.sort(key=lambda x: x[2], reverse=True)
        return scores[:top_k]

    def _reciprocal_rank_fusion(self, dense_results: List[Tuple], sparse_results: List[Tuple], k: int = 60) -> List[str]:
        """Reciprocal Rank Fusion: score(d) = sum(1 / (k + rank_i(d)))"""
        rrf_scores = {}
        doc_texts = {}

        # Add dense rankings
        for rank, (doc_id, doc_text, _) in enumerate(dense_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
            doc_texts[doc_id] = doc_text

        # Add sparse rankings
        for rank, (doc_id, doc_text, _) in enumerate(sparse_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
            doc_texts[doc_id] = doc_text

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        fused_docs = [doc_texts[doc_id] for doc_id in sorted_ids]

        logger.debug(f"RRF fusion: combined {len(dense_results)} dense + {len(sparse_results)} sparse → {len(fused_docs)} unique docs")
        return fused_docs

    @staticmethod
    def _strip_existing_sources(answer_text: str) -> str:
        """
        Remove any existing '📚 참고' section that the LLM may have added.
        This prevents duplicate or outdated source lists when we append URLs.
        """
        marker = "📚 참고"
        if marker in answer_text:
            return answer_text.split(marker)[0].rstrip()
        return answer_text

    def _rerank(self, query: str, documents: List[str], reranker: FlagReranker, top_k: int) -> List[str]:
        """Rerank documents using FlagReranker"""
        if not documents:
            return []

        pairs = [[query, doc] for doc in documents]
        scores = reranker.compute_score(pairs, normalize=True)

        # Handle single document case
        if isinstance(scores, (int, float)):
            scores = [scores]

        # Sort by score
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in scored_docs[:top_k]]

    async def _evaluate_single_context(self, question: str, ctx: str, idx: int) -> Tuple[bool, str, int]:
        """
        Evaluate a single context asynchronously

        Returns:
            tuple: (keep: bool, label: str, idx: int)
        """
        ctx_preview = ctx[:800] if len(ctx) > 800 else ctx

        prompt = f"""질문: {question}

문서: {ctx_preview}

이 문서가 질문에 답하는 데 도움이 됩니까?

- RELEVANT: 질문에 직접 답할 수 있는 정보 포함
- PARTIAL: 질문과 관련된 정보 일부 포함
- IRRELEVANT: 질문과 관련 없음

단어 하나만 답변하세요 (RELEVANT, PARTIAL, IRRELEVANT):"""

        try:
            response = await self.async_llm_client.chat.completions.create(
                model=self.quality_evaluator_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10
            )
            label = response.choices[0].message.content.strip().upper()

            keep = "RELEVANT" in label or "PARTIAL" in label
            logger.debug(f"Context {idx+1}: {label} - {'KEPT' if keep else 'FILTERED'}")
            return keep, label, idx

        except Exception as e:
            logger.warning(f"Context evaluation failed for {idx+1}: {e}, keeping by default")
            return True, "ERROR", idx

    async def _evaluate_context_quality_async(self, question: str, contexts: List[str]) -> List[str]:
        """
        Evaluate context quality using LLM with PARALLEL async calls

        OPTIMIZATION: This is 5-10x faster than sequential evaluation
        """
        if not self.enable_quality_filter:
            logger.debug("Context quality filter disabled, returning all contexts")
            return contexts

        logger.info(f"Evaluating quality of {len(contexts)} contexts (parallel)")
        eval_start = time.time()

        # Create tasks for all contexts
        tasks = [
            self._evaluate_single_context(question, ctx, idx)
            for idx, ctx in enumerate(contexts)
        ]

        # Execute all evaluations in parallel
        results = await asyncio.gather(*tasks)

        # Filter contexts based on results
        evaluated = []
        for keep, label, idx in results:
            if keep:
                evaluated.append(contexts[idx])

        eval_time = time.time() - eval_start
        logger.info(f"Context quality evaluation: {len(evaluated)}/{len(contexts)} kept, {eval_time:.2f}s")

        if len(evaluated) < 2:
            logger.warning(f"Only {len(evaluated)} relevant contexts found - may need query rewriting")

        return evaluated

    def _evaluate_context_quality(self, question: str, contexts: List[str]) -> List[str]:
        """
        Synchronous wrapper for async context quality evaluation

        OPTIMIZATION: Uses asyncio.run() to execute parallel evaluations
        """
        return asyncio.run(self._evaluate_context_quality_async(question, contexts))

    def retrieve(self, question: str) -> Tuple[List[str], List[Dict]]:
        """
        Enhanced retrieval with Hybrid Search and Context Quality Filter

        Returns:
            tuple: (contexts, metadatas)
        """
        logger.info(f"Starting enhanced retrieval for question: {question[:100]}...")
        retrieval_start = time.time()

        # Step 1: Encode query
        encode_start = time.time()
        query_encoding = self.embedding_model.encode(
            [question],
            batch_size=1,
            max_length=1024,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False
        )
        query_dense = query_encoding["dense_vecs"][0].tolist()
        query_sparse = query_encoding["lexical_weights"][0]
        encode_time = time.time() - encode_start
        logger.debug(f"Query encoding: {encode_time:.3f}s")

        # Step 2: Dense retrieval with metadatas
        dense_top_k = self.retrieval_cfg.get("hybrid_dense_top_k", 50)
        dense_start = time.time()
        dense_results = self.collection.query(
            query_embeddings=[query_dense],
            n_results=dense_top_k,
            include=["documents", "metadatas"]  # ← IMPORTANT: Include metadatas!
        )
        dense_docs = dense_results["documents"][0]
        dense_ids = dense_results["ids"][0]
        dense_metas = dense_results["metadatas"][0]  # ← Get metadatas
        dense_time = time.time() - dense_start
        logger.debug(f"Dense retrieval: {len(dense_docs)} docs in {dense_time:.3f}s")

        if not dense_docs:
            logger.warning("No documents retrieved")
            return [], []

        # Step 3: Sparse search
        sparse_start = time.time()
        sparse_top_k = self.retrieval_cfg.get("hybrid_sparse_top_k", 50)
        dense_scored = [(dense_ids[i], dense_docs[i], 1.0 / (i + 1)) for i in range(len(dense_docs))]
        sparse_scored = self._sparse_search(query_sparse, dense_docs, dense_ids, top_k=sparse_top_k)
        sparse_time = time.time() - sparse_start
        logger.debug(f"Sparse search: {sparse_time:.3f}s")

        # Step 4: RRF Fusion
        rrf_k = self.retrieval_cfg.get("rrf_k", 60)
        fused_docs = self._reciprocal_rank_fusion(dense_scored, sparse_scored, k=rrf_k)

        # Step 5: Two-stage reranking
        rerank_input_k = min(25, len(fused_docs))
        stage1 = self._rerank(question, fused_docs[:rerank_input_k], self.stage1_reranker, rerank_input_k)

        final_k = self.retrieval_cfg.get("rerank_top_k", 10)
        stage2 = self._rerank(question, stage1, self.stage2_reranker, final_k)

        # Step 6: Context quality evaluation (OPTIMIZED: Parallel async)
        final_contexts = self._evaluate_context_quality(question, stage2)

        # Step 7: Get metadatas for final contexts (OPTIMIZED: O(1) lookup)
        # Build doc_to_meta mapping for O(1) lookup
        doc_to_meta = {}
        for i, doc in enumerate(dense_docs):
            meta = dense_metas[i].copy()
            meta["chunk_id"] = dense_ids[i]
            doc_to_meta[doc] = meta

        context_metadatas = []
        for ctx in final_contexts:
            meta = doc_to_meta.get(ctx, {"domain": "unknown", "chunk_id": "unknown"})
            context_metadatas.append(meta)

        total_time = time.time() - retrieval_start
        logger.info(f"Total retrieval time: {total_time:.3f}s, contexts: {len(final_contexts)}")

        return final_contexts, context_metadatas

    def answer(self, question: str) -> str:
        """Generate answer with enhanced retrieval pipeline"""
        logger.info(f"Answering question: {question[:100]}...")
        total_start = time.time()

        # Route query
        decision = self.router.classify(question)
        logger.debug(f"Query routing: {decision.difficulty}, {decision.reason}")

        # Retrieve contexts with metadatas
        contexts, metadatas = self.retrieve(question)
        if not contexts:
            logger.warning("No contexts retrieved")
            return "관련 문서를 찾지 못했습니다. 질문을 다르게 표현해보시겠어요?"

        # Format context with metadata (including URL for reference)
        context_block = "\n\n".join(
            f"[문서 {i+1}] {meta.get('domain', 'unknown')}\n{ctx}"
            for i, (ctx, meta) in enumerate(zip(contexts, metadatas))
        )
        logger.debug(f"Formatted {len(contexts)} contexts with metadata for LLM")

        # Call LLM
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"질문: {question}\n\n컨텍스트:\n{context_block}"}
        ]

        llm_start = time.time()
        response = self.llm_client.chat.completions.create(
            model=self.llm_cfg["model_name"],
            messages=messages,
            temperature=self.llm_cfg.get("temperature", 0.2),
            max_tokens=self.llm_cfg.get("max_new_tokens", 300),
            top_p=self.llm_cfg.get("top_p", 0.9)
        )
        answer_text = response.choices[0].message.content
        # Remove any pre-existing "📚 참고" section before appending URLs
        answer_text = self._strip_existing_sources(answer_text)
        llm_time = time.time() - llm_start

        # Add source URLs at the end
        source_urls = []
        for meta in metadatas:
            url = meta.get('url', 'unknown')
            if url != 'unknown' and url not in source_urls:
                source_urls.append(url)

        if source_urls:
            sources_section = "\n\n📚 참고:\n" + "\n".join(f"- {url}" for url in source_urls)
            answer = answer_text + sources_section
        else:
            answer = answer_text

        total_time = time.time() - total_start
        logger.info(f"Answer generated in {llm_time:.2f}s, total: {total_time:.2f}s")

        return answer

    def answer_with_contexts(self, question: str) -> Tuple[str, List[str]]:
        """
        Generate answer and return contexts (for RAGAS evaluation)

        OPTIMIZATION: Prevents double retrieve() calls in evaluation

        Returns:
            tuple: (answer: str, contexts: List[str])
        """
        logger.info(f"Answering question with contexts: {question[:100]}...")
        total_start = time.time()

        # Route query
        decision = self.router.classify(question)
        logger.debug(f"Query routing: {decision.difficulty}, {decision.reason}")

        # Retrieve contexts with metadatas
        contexts, metadatas = self.retrieve(question)
        if not contexts:
            logger.warning("No contexts retrieved")
            return "관련 문서를 찾지 못했습니다. 질문을 다르게 표현해보시겠어요?", []

        # Format context with metadata (including URL for reference)
        context_block = "\n\n".join(
            f"[문서 {i+1}] {meta.get('domain', 'unknown')}\n{ctx}"
            for i, (ctx, meta) in enumerate(zip(contexts, metadatas))
        )
        logger.debug(f"Formatted {len(contexts)} contexts with metadata for LLM")

        # Call LLM
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"질문: {question}\n\n컨텍스트:\n{context_block}"}
        ]

        llm_start = time.time()
        response = self.llm_client.chat.completions.create(
            model=self.llm_cfg["model_name"],
            messages=messages,
            temperature=self.llm_cfg.get("temperature", 0.2),
            max_tokens=self.llm_cfg.get("max_new_tokens", 300),
            top_p=self.llm_cfg.get("top_p", 0.9)
        )
        answer_text = response.choices[0].message.content
        # Remove any pre-existing "📚 참고" section before appending URLs
        answer_text = self._strip_existing_sources(answer_text)
        llm_time = time.time() - llm_start

        # Add source URLs at the end
        source_urls = []
        for meta in metadatas:
            url = meta.get('url', 'unknown')
            if url != 'unknown' and url not in source_urls:
                source_urls.append(url)

        if source_urls:
            sources_section = "\n\n📚 참고:\n" + "\n".join(f"- {url}" for url in source_urls)
            answer = answer_text + sources_section
        else:
            answer = answer_text

        total_time = time.time() - total_start
        logger.info(f"Answer generated in {llm_time:.2f}s, total: {total_time:.2f}s")

        # Return answer and contexts (for RAGAS)
        return answer, contexts


def setup_logging(level: str = "INFO"):
    """Configure logging"""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="Enhanced RAG Pipeline (Optimized)")
    parser.add_argument("question", type=str, help="Question to answer")
    parser.add_argument("--config", default="config/enhanced.yaml")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    pipeline = EnhancedRAGPipeline(args.config)
    answer = pipeline.answer(args.question)

    print("\n" + "="*80)
    print("답변:")
    print("="*80)
    print(answer)
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
