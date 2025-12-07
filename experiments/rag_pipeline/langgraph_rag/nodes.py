"""
LangGraph RAG 노드 함수

이 모듈은 LangGraph 워크플로우의 각 노드를 정의합니다.
각 노드는 RAGState를 입력받아 수정하고 반환합니다.

주요 노드:
- query_router: 질문 분석 및 라우팅
- hybrid_retrieve: Hybrid Search (Dense + Sparse + RRF)
- rerank_stage1: 1차 Reranking (BGE-reranker-v2-m3)
- rerank_stage2: 2차 Reranking (BGE-reranker-large)
- grade_documents: 문서 관련성 평가
- transform_query: 쿼리 재작성
- generate: 답변 생성
- hallucination_check: 환각 검증
- answer_grading: 답변 품질 평가
- web_search: 웹 검색 fallback
- load_user_context: 사용자 컨텍스트 로드 (개인화)
- personalize_response: 답변 개인화
- suggest_related_questions: 관련 질문 추천
"""

import asyncio
import logging
import re
import time
from typing import Dict, List, Tuple

import chromadb
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI, OpenAI

from .config import get_config
from .state import (
    RAGState,
    add_to_history,
    IntentClassification,
    IntentType,
    DocumentRelevance,
    RelevanceType,
    RewrittenQuery,
    QueryRewriteAction,
    HallucinationGrade,
    HallucinationType,
    UsefulnessGrade,
    UsefulnessType,
)
from .tools import get_web_search_tool

logger = logging.getLogger(__name__)


# ========== 전역 리소스 (싱글톤 패턴) ==========

class RAGResources:
    """
    RAG 시스템 리소스 관리 (싱글톤)

    LangGraph 노드들이 공유하는 리소스:
    - 임베딩 모델
    - Reranker 모델들
    - ChromaDB 컬렉션
    - LLM 클라이언트
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """리소스 초기화 (최초 1회만 실행)"""
        if self._initialized:
            return

        logger.info("Initializing RAG resources...")
        config = get_config()

        # 임베딩 모델
        logger.info(f"Loading embedding model: {config.embedding_model}")
        self.embedding_model = BGEM3FlagModel(
            config.embedding_model,
            use_fp16=True,
            device=config.embedding_device,
        )
        self.embedding_batch_size = config.embedding_batch_size

        # Reranker Stage 1
        logger.info(f"Loading reranker stage 1: {config.reranker_stage1_model}")
        self.reranker_stage1 = FlagReranker(
            config.reranker_stage1_model,
            use_fp16=True,
            device=config.reranker_stage1_device,
        )

        # Reranker Stage 2
        logger.info(f"Loading reranker stage 2: {config.reranker_stage2_model}")
        self.reranker_stage2 = FlagReranker(
            config.reranker_stage2_model,
            use_fp16=True,
            device=config.reranker_stage2_device,
        )

        # ChromaDB
        logger.info(f"Connecting to ChromaDB at {config.chroma_db_path}")
        client = chromadb.PersistentClient(path=str(config.chroma_db_path))
        self.collection = client.get_collection("rag_chunks")
        logger.info(f"Collection loaded: {self.collection.count()} documents")

        # LLM 클라이언트 (동기/비동기)
        self.llm_client = OpenAI()
        self.async_llm_client = AsyncOpenAI()

        # LangChain LLM 클라이언트 (structured output용)
        self.langchain_llm = ChatOpenAI(
            model=config.llm_model,
            temperature=config.llm_temperature,
        )
        self.langchain_llm_fast = ChatOpenAI(
            model=config.context_quality_model,
            temperature=0,
        )

        # 시스템 프롬프트
        system_prompt_path = config.artifacts_dir.parent / config.config["llm"]["system_prompt_path"]
        self.system_prompt = (
            system_prompt_path.read_text(encoding="utf-8")
            if system_prompt_path.exists()
            else ""
        )

        # 웹 검색 도구
        self.web_search_tool = get_web_search_tool()

        self._initialized = True
        logger.info("✓ RAG resources initialized")


def get_resources() -> RAGResources:
    """전역 RAG 리소스 반환"""
    return RAGResources()

# ========== 노드 0: Intent Classifier ==========


def intent_classifier_node(state: RAGState) -> RAGState:
    """
    질문 의도를 분류해 in_scope가 아니면 초기에 종료시킨다.

    Categories:
    - IN_SCOPE: 개발/프로그래밍/학습 관련
    - GREETING: 인사/감사 등
    - CHITCHAT: 잡담/요청(아이스크림 사줘 등)
    - NONSENSICAL: 무의미/스팸
    """
    logger.info("[Intent] 질문 의도 분류 시작")
    resources = get_resources()
    config = get_config()

    question = state["question"]
    intent = "unknown"

    prompt = f"""다음 사용자의 질문이 개발/프로그래밍/학습 관련인지 분류하세요.
반드시 아래 중 하나의 라벨만 답변:
- IN_SCOPE: 개발, 프로그래밍, 소프트웨어 학습/디버깅/도구 사용
- GREETING: 인사, 감사, 안부
- CHITCHAT: 잡담/사적요청 (예: 아이스크림 사줘, 노래 추천)
- NONSENSICAL: 무의미/스팸/의미 없는 입력

질문: {question}

정답 라벨 한 단어만:"""

    try:
        response = resources.llm_client.chat.completions.create(
            model=config.context_quality_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=4,
        )
        label = response.choices[0].message.content.strip().upper()
        if "IN_SCOPE" in label:
            intent = "in_scope"
        elif "GREETING" in label:
            intent = "greeting"
        elif "CHITCHAT" in label:
            intent = "chitchat"
        elif "NON" in label:
            intent = "nonsensical"
        else:
            intent = "unknown"
    except Exception as e:
        logger.warning(f"[Intent] 분류 실패: {e}, 기본 in_scope로 처리")
        intent = "in_scope"

    state["intent"] = intent

    # in_scope가 아니면 바로 짧은 메시지 후 종료
    if intent != "in_scope":
        reply_map = {
            "greeting": "안녕하세요! 저는 개발·학습 도우미예요. 궁금한 개발/프로그래밍 질문을 알려주시면 도와드릴게요.",
            "chitchat": "저는 개발·학습 관련 질문에 집중하고 있어요. 코드나 에러, 학습 주제를 말씀해 주세요!",
            "nonsensical": "지금 입력으로는 도움을 드리기 어려워요. 개발/프로그래밍 관련 질문을 구체적으로 알려주시면 도와드릴게요.",
        }
        state["generation"] = reply_map.get(
            intent,
            "개발·학습 관련 질문을 알려주시면 도움을 드릴게요.",
        )

    return add_to_history(state, "intent_classifier")


# ========== 노드 1: Query Router ==========

def query_router_node(state: RAGState) -> RAGState:
    """
    질문 분석 및 라우팅 결정

    Args:
        state (RAGState): 현재 상태

    Returns:
        RAGState: 라우팅 결정이 추가된 상태

    라우팅 전략:
    - "vectorstore": 벡터 검색 (기본)
    - "websearch": 웹 검색 (최신 정보 필요)
    - "direct": LLM만 사용 (검색 불필요)

    현재 구현: 간단한 키워드 기반 라우팅
    향후 개선: LLM 기반 분류
    """
    logger.info(f"[QueryRouter] 질문 분석: {state['question'][:100]}")

    question = state["question"].lower()

    # 간단한 키워드 기반 라우팅
    # TODO: LLM 기반 분류로 개선
    if any(
        keyword in question
        for keyword in ["최근", "현재", "2024", "2025", "뉴스", "트렌드"]
    ):
        route = "websearch"
        logger.info("[QueryRouter] → 웹 검색 (최신 정보)")
    elif any(
        keyword in question
        for keyword in ["안녕", "hello", "hi", "감사", "고마워"]
    ):
        route = "direct"
        logger.info("[QueryRouter] → 직접 답변 (검색 불필요)")
    else:
        route = "vectorstore"
        logger.info("[QueryRouter] → 벡터 검색 (기본)")

    state["route"] = route
    return add_to_history(state, "query_router")


# ========== 노드 2: Hybrid Retrieve ==========

def hybrid_retrieve_node(state: RAGState) -> RAGState:
    """
    Hybrid Search (Dense + Sparse + RRF Fusion)

    Args:
        state (RAGState): 현재 상태

    Returns:
        RAGState: 검색 결과가 추가된 상태

    검색 단계:
    1. 쿼리 인코딩 (Dense + Sparse)
    2. Dense 검색 (의미 기반)
    3. Sparse 검색 (키워드 기반)
    4. RRF Fusion (두 결과 결합)
    """
    logger.info("[HybridRetrieve] 검색 시작")
    start_time = time.time()

    resources = get_resources()
    config = get_config()

    question = state["question"]

    # Step 1: 쿼리 인코딩
    query_encoding = resources.embedding_model.encode(
        [question],
        batch_size=1,
        max_length=1024,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    query_dense = query_encoding["dense_vecs"][0].tolist()
    query_sparse = query_encoding["lexical_weights"][0]

    # Step 2: Dense 검색
    dense_top_k = config.hybrid_dense_top_k
    dense_results = resources.collection.query(
        query_embeddings=[query_dense],
        n_results=dense_top_k,
        include=["documents", "metadatas"],
    )
    dense_docs = dense_results["documents"][0]
    dense_ids = dense_results["ids"][0]
    dense_metas = dense_results["metadatas"][0]

    if not dense_docs:
        logger.warning("[HybridRetrieve] 검색 결과 없음")
        state["documents"] = []
        state["metadatas"] = []
        return add_to_history(state, "hybrid_retrieve")

    # Step 3: Sparse 검색
    sparse_top_k = config.hybrid_sparse_top_k
    dense_scored = [
        (dense_ids[i], dense_docs[i], 1.0 / (i + 1)) for i in range(len(dense_docs))
    ]
    sparse_scored = _sparse_search(
        resources, query_sparse, dense_docs, dense_ids, top_k=sparse_top_k
    )

    # Step 4: RRF Fusion
    rrf_k = config.config["retrieval"]["rrf_k"]
    fused_docs = _reciprocal_rank_fusion(dense_scored, sparse_scored, k=rrf_k)

    # 메타데이터 매핑 (O(1) 조회)
    doc_to_meta = {}
    for i, doc in enumerate(dense_docs):
        meta = dense_metas[i].copy()
        meta["chunk_id"] = dense_ids[i]
        doc_to_meta[doc] = meta

    fused_metadatas = [
        doc_to_meta.get(doc, {"domain": "unknown", "chunk_id": "unknown"})
        for doc in fused_docs
    ]

    elapsed = time.time() - start_time
    logger.info(
        f"[HybridRetrieve] {len(fused_docs)}개 문서 검색 완료 ({elapsed:.2f}s)"
    )

    state["documents"] = fused_docs
    state["metadatas"] = fused_metadatas
    return add_to_history(state, "hybrid_retrieve")


def _sparse_search(
    resources: RAGResources,
    query_sparse_vector: Dict,
    documents: List[str],
    doc_ids: List[str],
    top_k: int = 50,
) -> List[Tuple[str, str, float]]:
    """
    Sparse 검색 (BGE-M3 lexical weights 사용)

    Args:
        resources: RAG 리소스
        query_sparse_vector: 쿼리 sparse vector
        documents: 후보 문서들
        doc_ids: 문서 ID들
        top_k: 상위 k개 반환

    Returns:
        List[Tuple[str, str, float]]: (doc_id, doc_text, score)
    """
    # 문서 인코딩 (sparse만)
    doc_encodings = resources.embedding_model.encode(
        documents,
        batch_size=resources.embedding_batch_size,
        max_length=1024,
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    # Sparse score 계산 (inner product)
    scores = []
    for i, doc_sparse in enumerate(doc_encodings["lexical_weights"]):
        score = 0.0
        for term, query_weight in query_sparse_vector.items():
            if term in doc_sparse:
                score += query_weight * doc_sparse[term]
        scores.append((doc_ids[i], documents[i], score))

    # 점수 기준 정렬
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:top_k]


def _reciprocal_rank_fusion(
    dense_results: List[Tuple],
    sparse_results: List[Tuple],
    k: int = 60,
) -> List[str]:
    """
    Reciprocal Rank Fusion

    Args:
        dense_results: Dense 검색 결과 [(doc_id, doc_text, score), ...]
        sparse_results: Sparse 검색 결과 [(doc_id, doc_text, score), ...]
        k: RRF 상수 (기본: 60)

    Returns:
        List[str]: Fusion된 문서 텍스트 리스트

    RRF 공식:
        score(d) = Σ 1 / (k + rank_i(d))
    """
    rrf_scores = {}
    doc_texts = {}

    # Dense 순위 추가
    for rank, (doc_id, doc_text, _) in enumerate(dense_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        doc_texts[doc_id] = doc_text

    # Sparse 순위 추가
    for rank, (doc_id, doc_text, _) in enumerate(sparse_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        doc_texts[doc_id] = doc_text

    # RRF 점수 기준 정렬
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    fused_docs = [doc_texts[doc_id] for doc_id in sorted_ids]

    logger.debug(
        f"[RRF] Dense {len(dense_results)} + Sparse {len(sparse_results)} "
        f"→ {len(fused_docs)} unique docs"
    )
    return fused_docs


# ========== 노드 3: Rerank Stage 1 ==========

def rerank_stage1_node(state: RAGState) -> RAGState:
    """
    1차 Reranking (BGE-reranker-v2-m3)

    Args:
        state (RAGState): 현재 상태

    Returns:
        RAGState: 1차 reranking 결과가 추가된 상태

    전략:
    - Hybrid search 결과 중 상위 25개를 reranking
    - 빠른 모델로 초기 필터링
    """
    logger.info("[Rerank Stage 1] 시작")
    start_time = time.time()

    resources = get_resources()
    config = get_config()

    question = state["question"]
    documents = state["documents"]
    metadatas = state["metadatas"]

    if not documents:
        logger.warning("[Rerank Stage 1] 문서 없음")
        state["reranked_documents"] = []
        state["reranked_metadatas"] = []
        return add_to_history(state, "rerank_stage1")

    # 상위 25개만 reranking (성능 최적화)
    rerank_input_k = min(25, len(documents))
    docs_to_rerank = documents[:rerank_input_k]
    metas_to_rerank = metadatas[:rerank_input_k]

    # Reranking
    reranked_docs = _rerank(
        question, docs_to_rerank, resources.reranker_stage1, rerank_input_k
    )

    # 메타데이터 매핑
    doc_to_meta = {doc: meta for doc, meta in zip(docs_to_rerank, metas_to_rerank)}
    reranked_metas = [
        doc_to_meta.get(doc, {"domain": "unknown"}) for doc in reranked_docs
    ]

    elapsed = time.time() - start_time
    logger.info(
        f"[Rerank Stage 1] {len(reranked_docs)}개 문서 reranking 완료 ({elapsed:.2f}s)"
    )

    state["reranked_documents"] = reranked_docs
    state["reranked_metadatas"] = reranked_metas
    return add_to_history(state, "rerank_stage1")


# ========== 노드 4: Rerank Stage 2 ==========

def rerank_stage2_node(state: RAGState) -> RAGState:
    """
    2차 Reranking (BGE-reranker-large)

    Args:
        state (RAGState): 현재 상태

    Returns:
        RAGState: 2차 reranking 결과가 추가된 상태

    전략:
    - 1차 reranking 결과를 더 강력한 모델로 재평가
    - 최종 top_k개 선택 (기본: 10개)
    """
    logger.info("[Rerank Stage 2] 시작")
    start_time = time.time()

    resources = get_resources()
    config = get_config()

    question = state["question"]
    documents = state["reranked_documents"]
    metadatas = state["reranked_metadatas"]

    if not documents:
        logger.warning("[Rerank Stage 2] 문서 없음")
        state["final_documents"] = []
        state["final_metadatas"] = []
        return add_to_history(state, "rerank_stage2")

    # 최종 top_k 선택
    final_k = config.rerank_top_k
    reranked_docs = _rerank(question, documents, resources.reranker_stage2, final_k)

    # 메타데이터 매핑
    doc_to_meta = {doc: meta for doc, meta in zip(documents, metadatas)}
    reranked_metas = [
        doc_to_meta.get(doc, {"domain": "unknown"}) for doc in reranked_docs
    ]

    elapsed = time.time() - start_time
    logger.info(
        f"[Rerank Stage 2] {len(reranked_docs)}개 최종 문서 선택 ({elapsed:.2f}s)"
    )

    state["final_documents"] = reranked_docs
    state["final_metadatas"] = reranked_metas
    return add_to_history(state, "rerank_stage2")


def _rerank(
    query: str, documents: List[str], reranker: FlagReranker, top_k: int
) -> List[str]:
    """
    문서 Reranking

    Args:
        query: 쿼리
        documents: 문서 리스트
        reranker: Reranker 모델
        top_k: 상위 k개 반환

    Returns:
        List[str]: Reranking된 문서들
    """
    if not documents:
        return []

    pairs = [[query, doc] for doc in documents]
    scores = reranker.compute_score(pairs, normalize=True)

    # 단일 문서 처리
    if isinstance(scores, (int, float)):
        scores = [scores]

    # 점수 기준 정렬
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in scored_docs[:top_k]]


# ========== 계속 (nodes_part2.py로 분할) ==========
# 다음 노드들:
# - grade_documents_node
# - transform_query_node
# - generate_node
# - hallucination_check_node
# - answer_grading_node
# - web_search_node
# nodes.py 계속 - 노드 5~10

# ========== 노드 5: Grade Documents ==========

def grade_documents_node(state):
    """문서 관련성 평가 (Corrective RAG)"""
    logger.info("[GradeDocuments] 문서 관련성 평가 시작")
    start_time = time.time()

    resources = get_resources()
    question = state["question"]
    documents = state["final_documents"]

    if not documents:
        logger.warning("[GradeDocuments] 문서 없음")
        state["document_relevance"] = "not_relevant"
        return add_to_history(state, "grade_documents")

    # 병렬 평가 (비동기)
    results = asyncio.run(
        _evaluate_documents_async(resources.async_llm_client, question, documents)
    )

    # 결과 집계
    relevant_count = sum(
        1 for label in results if "RELEVANT" in label or "PARTIAL" in label
    )
    relevance_ratio = relevant_count / len(results)

    if relevance_ratio >= 0.5:
        state["document_relevance"] = "relevant"
        logger.info(
            f"[GradeDocuments] 문서 관련성: RELEVANT ({relevant_count}/{len(results)})"
        )
    else:
        state["document_relevance"] = "not_relevant"
        logger.info(
            f"[GradeDocuments] 문서 관련성: NOT RELEVANT ({relevant_count}/{len(results)})"
        )
        state["web_search_needed"] = True

    elapsed = time.time() - start_time
    logger.info(f"[GradeDocuments] 평가 완료 ({elapsed:.2f}s)")

    return add_to_history(state, "grade_documents")


async def _evaluate_documents_async(async_client, question, documents):
    """문서 관련성 병렬 평가 (비동기)"""
    config = get_config()

    tasks = [
        _evaluate_single_document(
            async_client, question, doc, config.context_quality_model
        )
        for doc in documents
    ]

    results = await asyncio.gather(*tasks)
    return results


async def _evaluate_single_document(async_client, question, document, model):
    """단일 문서 관련성 평가"""
    doc_preview = document[:800] if len(document) > 800 else document

    prompt = f"""질문: {question}

문서: {doc_preview}

이 문서가 질문에 답하는 데 도움이 됩니까?

- RELEVANT: 질문에 직접 답할 수 있는 정보 포함
- PARTIAL: 질문과 관련된 정보 일부 포함
- IRRELEVANT: 질문과 관련 없음

단어 하나만 답변하세요 (RELEVANT, PARTIAL, IRRELEVANT):"""

    try:
        response = await async_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        label = response.choices[0].message.content.strip().upper()
        return label
    except Exception as e:
        logger.warning(f"문서 평가 실패: {e}, 기본값 PARTIAL 사용")
        return "PARTIAL"


# ========== 노드 6: Transform Query ==========

def transform_query_node(state):
    """쿼리 재작성 (Query Transformation)"""
    logger.info("[TransformQuery] 쿼리 재작성 시작")
    start_time = time.time()

    resources = get_resources()
    config = get_config()

    question = state["question"]

    prompt = f"""다음 질문을 더 구체적이고 검색하기 좋은 형태로 재작성하세요.

원본 질문: {question}

재작성 지침:
- 핵심 키워드 강조
- 구체적인 용어 사용
- 검색에 도움이 되는 컨텍스트 추가

재작성된 질문:"""

    try:
        response = resources.llm_client.chat.completions.create(
            model=config.context_quality_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100,
        )
        transformed = response.choices[0].message.content.strip()
        logger.info(f"[TransformQuery] 원본: {question}")
        logger.info(f"[TransformQuery] 재작성: {transformed}")

        state["transformed_query"] = transformed
        state["question"] = transformed  # 재작성된 쿼리로 교체

    except Exception as e:
        logger.error(f"[TransformQuery] 실패: {e}, 원본 쿼리 유지")
        state["transformed_query"] = question

    elapsed = time.time() - start_time
    logger.info(f"[TransformQuery] 완료 ({elapsed:.2f}s)")

    return add_to_history(state, "transform_query")


# ========== 노드 7: Generate ==========

def generate_node(state):
    """Generate answer with LLM (final step)."""
    logger.info("[Generate] Start generation")
    start_time = time.time()

    resources = get_resources()
    config = get_config()

    question = state["question"]
    documents = state["final_documents"]
    metadatas = state["final_metadatas"]

    if not documents:
        logger.warning("[Generate] No documents; returning fallback message")
        state["generation"] = "?? ??? ?? ?????. ??? ?? ? ????? ?????."
        return add_to_history(state, "generate")

    # Context block
    context_block = "\n\n".join(
        f"[?? {i+1}] {meta.get('domain', 'unknown')}\n{doc}"
        for i, (doc, meta) in enumerate(zip(documents, metadatas))
    )

    # Recent chat history (best-effort, short)
    history = state.get("chat_history", [])
    history_text = ""
    if history:
        recent = history[-6:]
        history_lines = [
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in recent
        ]
        history_text = "[Recent chat]\n" + "\n".join(history_lines) + "\n\n"

    messages = [
        {"role": "system", "content": resources.system_prompt},
        {
            "role": "user",
            "content": (
                f"??: {question}\n\n"
                f"{history_text}????:\n{context_block}\n\n"
                "??: ???? ??? ??? ????, ?? ??? '?? Reference' ???? ?? ???."
            ),
        },
    ]

    try:
        response = resources.llm_client.chat.completions.create(
            model=config.llm_model,
            messages=messages,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            top_p=config.config["llm"].get("top_p", 0.9),
        )
        answer_text = response.choices[0].message.content

        # Strip prior reference sections and tool mentions
        answer_text = _clean_tool_mentions(_strip_existing_sources(answer_text))

        # URL extraction for references
        source_urls = []
        for meta in metadatas:
            url = meta.get("url", "unknown")
            if url != "unknown" and url not in source_urls:
                source_urls.append(url)

        if source_urls:
            sources_section = "\n\n?? Reference:\n" + "\n".join(
                f"- {url}" for url in source_urls
            )
            answer = answer_text + sources_section
        else:
            answer = answer_text

        state["generation"] = answer
        logger.info("[Generate] Generation done")

    except Exception as e:
        logger.error(f"[Generate] Error: {e}")
        state["generation"] = "?? ??? ??? ??????."

    elapsed = time.time() - start_time
    logger.info(f"[Generate] Done ({elapsed:.2f}s)")

    return add_to_history(state, "generate")

def _strip_existing_sources(answer_text: str) -> str:
    """기존 출처 섹션 제거"""
    marker = "📚 참고"
    if marker in answer_text:
        return answer_text.split(marker)[0].rstrip()
    return answer_text


def _clean_tool_mentions(answer_text: str) -> str:
    """
    본문에서 tavily/websearch 등 툴 이름을 제거해 답변을 자연스럽게 만든다.
    """
    cleaned = answer_text
    for token in ["tavily", "websearch", "web search", "Tavily", "WebSearch"]:
        cleaned = re.sub(rf"\(?\b{re.escape(token)}\b\)?", "", cleaned, flags=re.IGNORECASE)
    # Collapse double spaces left by removals
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


# ========== 노드 8: Hallucination Check (개선 버전) ==========

def hallucination_check_node(state):
    """
    환각 검증 (Self-RAG with confidence scoring)

    개선사항:
    1. Structured output (Pydantic) - 파싱 오류 제거
    2. Confidence score - 신뢰도 측정
    3. Keyword overlap validation - LLM 과신 방지
    4. 토큰 기반 truncation - 근거 누락 방지
    """
    logger.info("[HallucinationCheck] 환각 검증 시작 (개선 버전)")
    start_time = time.time()

    resources = get_resources()

    generation = state["generation"]
    documents = state["final_documents"]

    if not documents:
        logger.warning("[HallucinationCheck] 문서 없음, 검증 스킵")
        state["hallucination_grade"] = "not_sure"
        return add_to_history(state, "hallucination_check")

    # 출처 제거한 답변만 검증
    answer_only = _clean_tool_mentions(_strip_existing_sources(generation))

    # Step 1: 간단한 keyword overlap 사전 검증
    keyword_overlap = _calculate_keyword_overlap(answer_only, documents)

    # 명백한 환각 (keyword overlap이 너무 낮음)
    if keyword_overlap < 0.15:
        logger.warning(f"[HallucinationCheck] Keyword overlap 매우 낮음 ({keyword_overlap:.2%}) → NOT_SUPPORTED")
        state["hallucination_grade"] = "not_supported"
        state["web_search_needed"] = True
        return add_to_history(state, "hallucination_check")

    # Step 2: 토큰 기반 문서 truncation (더 많은 문서 포함)
    context_text = _smart_truncate_documents(documents, max_tokens=3000)

    # Step 3: LLM structured output으로 검증
    system_prompt = """You are a fact-checker. Evaluate if the answer is fully grounded in the provided documents.

Provide:
1. reasoning: Your judgment reasoning (2-3 sentences)
2. grade: SUPPORTED / NOT_SUPPORTED / NOT_SURE
3. confidence: Your confidence score (0.0 to 1.0)

Be strict: if ANY claim lacks evidence, mark as NOT_SUPPORTED."""

    user_prompt = f"""Answer:
{answer_only}

Documents:
{context_text}

Is the answer fully supported by the documents?"""

    try:
        structured_llm = resources.langchain_llm_fast.with_structured_output(HallucinationGrade)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        result: HallucinationGrade = structured_llm.invoke(messages)

        # Step 4: Confidence calibration (LLM confidence + keyword overlap)
        # LLM이 과신하는 경향을 keyword overlap로 보정
        calibrated_confidence = result.confidence * 0.7 + keyword_overlap * 0.3

        # Low confidence warning
        if calibrated_confidence < 0.65 and result.grade.value == "supported":
            logger.warning(
                f"[HallucinationCheck] Low confidence ({calibrated_confidence:.2f}), "
                f"keyword_overlap={keyword_overlap:.2%}"
            )
            # Confidence가 낮으면 NOT_SURE로 변경
            if calibrated_confidence < 0.50:
                result.grade = HallucinationType.NOT_SURE

        # State update
        state["hallucination_grade"] = result.grade.value

        if result.grade.value == "not_supported":
            state["web_search_needed"] = True

        logger.info(
            f"[HallucinationCheck] Grade={result.grade.value}, "
            f"LLM_Conf={result.confidence:.2f}, "
            f"Calibrated={calibrated_confidence:.2f}, "
            f"Keyword_Overlap={keyword_overlap:.2%}"
        )
        logger.info(f"[HallucinationCheck] Reasoning: {result.reasoning}")

    except Exception as e:
        logger.error(f"[HallucinationCheck] 실패: {e}", exc_info=True)
        state["hallucination_grade"] = "not_sure"

    elapsed = time.time() - start_time
    logger.info(f"[HallucinationCheck] 완료 ({elapsed:.2f}s)")

    return add_to_history(state, "hallucination_check")


def _calculate_keyword_overlap(answer: str, documents: List[str]) -> float:
    """
    간단한 keyword overlap 계산 (답변과 문서 간의 어휘 중복도)

    Returns:
        float: Overlap ratio (0.0 ~ 1.0)
    """
    import re

    # 답변에서 키워드 추출 (2글자 이상, 알파벳/한글)
    answer_words = set(re.findall(r'[a-zA-Z가-힣]{2,}', answer.lower()))

    if not answer_words:
        return 0.0

    # 문서에서 키워드 추출
    doc_text = " ".join(documents).lower()
    doc_words = set(re.findall(r'[a-zA-Z가-힣]{2,}', doc_text))

    # Overlap 계산
    matching_words = answer_words.intersection(doc_words)
    overlap_ratio = len(matching_words) / len(answer_words)

    return overlap_ratio


def _smart_truncate_documents(documents: List[str], max_tokens: int = 3000) -> str:
    """
    토큰 수 기반으로 문서를 스마트하게 truncate

    Args:
        documents: 문서 리스트
        max_tokens: 최대 토큰 수

    Returns:
        str: Truncated된 문서 텍스트
    """
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o-mini")
    except:
        # tiktoken 없으면 문자 수 기반 fallback (대략 1 token = 4 chars)
        max_chars = max_tokens * 4
        combined = "\n\n".join(f"[Doc {i+1}]\n{doc}" for i, doc in enumerate(documents))
        if len(combined) > max_chars:
            return combined[:max_chars] + "\n... [truncated]"
        return combined

    result = []
    total_tokens = 0

    for i, doc in enumerate(documents):
        doc_with_header = f"[Doc {i+1}]\n{doc}"
        doc_tokens = len(enc.encode(doc_with_header))

        if total_tokens + doc_tokens <= max_tokens:
            result.append(doc_with_header)
            total_tokens += doc_tokens
        else:
            # 남은 토큰으로 일부만 포함
            remaining_tokens = max_tokens - total_tokens
            if remaining_tokens > 50:  # 최소 50 토큰은 있어야 의미 있음
                truncated_doc = enc.decode(enc.encode(doc)[:remaining_tokens])
                result.append(f"[Doc {i+1}]\n{truncated_doc}... [truncated]")
            break

    return "\n\n".join(result)


# ========== 노드 9: Answer Grading ==========

def answer_grading_node(state):
    """답변 품질 평가 (Self-RAG)"""
    logger.info("[AnswerGrading] 답변 품질 평가 시작")
    start_time = time.time()

    resources = get_resources()
    config = get_config()

    question = state["question"]
    generation = state["generation"]

    # 출처 제거한 답변만 평가
    answer_only = _clean_tool_mentions(_strip_existing_sources(generation))

    prompt = f"""다음 답변이 질문에 유용한지 판단하세요.

질문: {question}

답변: {answer_only}

이 답변이 질문에 충분히 답변합니까?

- USEFUL: 질문에 충분히 답변함
- NOT_USEFUL: 질문에 답변하지 못함

단어 하나만 답변하세요 (USEFUL, NOT_USEFUL):"""

    try:
        response = resources.llm_client.chat.completions.create(
            model=config.context_quality_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=20,
        )
        label = response.choices[0].message.content.strip().upper()

        if "USEFUL" in label and "NOT" not in label:
            state["answer_usefulness"] = "useful"
            logger.info("[AnswerGrading] 결과: USEFUL")
        else:
            state["answer_usefulness"] = "not_useful"
            logger.warning("[AnswerGrading] 결과: NOT USEFUL")
            state["web_search_needed"] = True

    except Exception as e:
        logger.error(f"[AnswerGrading] 실패: {e}")
        state["answer_usefulness"] = "useful"  # 실패 시 긍정으로 가정

    elapsed = time.time() - start_time
    logger.info(f"[AnswerGrading] 완료 ({elapsed:.2f}s)")

    return add_to_history(state, "answer_grading")


# ========== 노드 10: Web Search ==========

def web_search_node(state):
    """웹 검색 fallback (Corrective RAG)"""
    logger.info("[WebSearch] 웹 검색 시작")
    start_time = time.time()

    web_search_tool = get_web_search_tool()
    question = state["question"]

    if not web_search_tool.enabled:
        logger.warning("[WebSearch] 웹 검색 비활성화됨")
        state["final_documents"] = []
        state["final_metadatas"] = []
        return add_to_history(state, "web_search")

    # 웹 검색 실행
    documents, metadatas = web_search_tool.search_with_metadata(question)

    state["final_documents"] = documents
    state["final_metadatas"] = metadatas

    elapsed = time.time() - start_time
    logger.info(f"[WebSearch] {len(documents)}개 결과 검색 완료 ({elapsed:.2f}s)")

    return add_to_history(state, "web_search")


# ========== 개인화 및 질문 추천 노드 ==========

# ========== 노드 11: Load User Context (개인화) ==========

def load_user_context_node(state: RAGState) -> RAGState:
    """
    사용자 컨텍스트 로드 (개인화 - 간소화 버전)

    Django에서 전달받은 user_context를 기반으로 현재 질문과 관련된
    사용자 학습 목표 및 관심사를 분석합니다.

    Note: 멘토님의 원본과 다르게 DB 쿼리를 하지 않고,
          Django에서 이미 전달받은 user_context를 사용합니다.

    Args:
        state (RAGState): 현재 상태

    Returns:
        RAGState: 개인화 컨텍스트가 추가된 상태
    """
    logger.info("[LoadUserContext] 사용자 컨텍스트 로드 시작")
    start_time = time.time()

    user_id = state.get("user_id", "")
    user_context = state.get("user_context", {})
    question = state["question"]

    if not user_id or not user_context:
        logger.info("[LoadUserContext] user_id 또는 user_context 없음, 개인화 스킵")
        return add_to_history(state, "load_user_context")

    try:
        # Django에서 전달받은 user_context 사용
        # user_context 구조: {
        #     "learning_goals": "Python 마스터하기, Django 학습",
        #     "interested_topics": "웹 개발, API 설계, 데이터베이스",
        # }
        learning_goals = user_context.get("learning_goals", "")
        interested_topics = user_context.get("interested_topics", "")

        if not learning_goals and not interested_topics:
            logger.info("[LoadUserContext] 사용자 학습 데이터 없음")
            return add_to_history(state, "load_user_context")

        # 질문에서 키워드 추출
        question_keywords = _extract_keywords(question)

        # 관련 항목 찾기
        related_items = []
        forgotten_items = []

        # learning_goals 분석
        if learning_goals:
            goals_list = [g.strip() for g in learning_goals.split(",")]
            for goal in goals_list:
                if _is_related_to_question(goal, question_keywords, question):
                    related_items.append({
                        "type": "learning_goal",
                        "content": goal,
                    })
                    # 질문에 직접 언급되지 않은 경우 상기 후보
                    if not _is_mentioned_in_question(goal, question):
                        forgotten_items.append({
                            "type": "learning_goal",
                            "content": goal,
                        })

        # interested_topics 분석
        if interested_topics:
            topics_list = [t.strip() for t in interested_topics.split(",")]
            for topic in topics_list:
                if _is_related_to_question(topic, question_keywords, question):
                    related_items.append({
                        "type": "interested_topic",
                        "content": topic,
                    })
                    # 질문에 직접 언급되지 않은 경우 상기 후보
                    if not _is_mentioned_in_question(topic, question):
                        forgotten_items.append({
                            "type": "interested_topic",
                            "content": topic,
                        })

        state["related_selections"] = related_items
        state["forgotten_candidates"] = forgotten_items

        logger.info(
            f"[LoadUserContext] 로드 완료 - "
            f"관련: {len(related_items)}, "
            f"상기 후보: {len(forgotten_items)}"
        )

    except Exception as e:
        logger.error(f"[LoadUserContext] 실패: {e}")
        state["related_selections"] = []
        state["forgotten_candidates"] = []

    elapsed = time.time() - start_time
    logger.info(f"[LoadUserContext] 완료 ({elapsed:.2f}s)")

    return add_to_history(state, "load_user_context")


def _extract_keywords(text: str) -> List[str]:
    """
    텍스트에서 키워드 추출

    Args:
        text: 입력 텍스트

    Returns:
        List[str]: 추출된 키워드 목록
    """
    # 간단한 키워드 추출 (공백 기준 분리 + 불용어 제거)
    stopwords = {"은", "는", "이", "가", "을", "를", "의", "에", "에서", "으로", "로", "와", "과", "하고", "있", "없", "수", "더", "등"}

    words = text.lower().replace("?", "").replace(".", "").split()
    keywords = [w for w in words if len(w) > 1 and w not in stopwords]

    return keywords


def _is_related_to_question(item: str, keywords: List[str], question: str) -> bool:
    """
    항목이 질문과 관련 있는지 판단

    Args:
        item: 사용자 학습 목표 또는 관심사
        keywords: 질문 키워드 목록
        question: 원본 질문

    Returns:
        bool: 관련 여부
    """
    item_lower = item.lower()
    question_lower = question.lower()

    # 직접 언급된 경우
    if item_lower in question_lower:
        return True

    # 키워드 중 하나라도 포함되면 관련 있음
    for keyword in keywords:
        if keyword in item_lower:
            return True

    return False


def _is_mentioned_in_question(item: str, question: str) -> bool:
    """
    항목이 질문에 직접 언급되었는지 확인

    Args:
        item: 사용자 학습 목표 또는 관심사
        question: 원본 질문

    Returns:
        bool: 언급 여부
    """
    item_lower = item.lower()
    question_lower = question.lower()

    return item_lower in question_lower


# ========== 노드 12: Personalize Response (개인화) ==========

def personalize_response_node(state: RAGState) -> RAGState:
    """
    답변 개인화 (상기 메시지 주입)

    생성된 답변에 사용자가 잊었을 수 있는 과거 학습 목표나 관심사를
    상기시키는 메시지를 자연스럽게 추가합니다.

    Args:
        state (RAGState): 현재 상태

    Returns:
        RAGState: 개인화된 답변이 포함된 상태
    """
    logger.info("[PersonalizeResponse] 답변 개인화 시작")
    start_time = time.time()

    generation = state["generation"]
    forgotten_candidates = state.get("forgotten_candidates", [])

    if not forgotten_candidates:
        logger.info("[PersonalizeResponse] 상기할 내용 없음, 스킵")
        state["reminder_added"] = False
        return add_to_history(state, "personalize_response")

    try:
        # 상기 메시지 생성 (최대 2개 항목만)
        reminder_items = forgotten_candidates[:2]
        reminder_parts = []

        for item in reminder_items:
            item_type = item.get("type", "")
            content = item.get("content", "")

            if content:
                if item_type == "learning_goal":
                    reminder_parts.append(f"'{content}' 학습 목표")
                else:
                    reminder_parts.append(f"'{content}'")

        if reminder_parts:
            # 자연스러운 상기 메시지 구성
            if len(reminder_parts) == 1:
                items_text = reminder_parts[0]
            else:
                items_text = f"{reminder_parts[0]}과(와) {reminder_parts[1]}"

            reminder_message = (
                f"\n\n💡 **참고**: 이전에 관심을 보이셨던 {items_text}도 "
                f"함께 확인해보시면 도움이 될 수 있습니다."
            )

            # 출처 섹션 앞에 삽입
            if "📚 참고:" in generation:
                parts = generation.split("📚 참고:")
                personalized_generation = parts[0].rstrip() + reminder_message + "\n\n📚 참고:" + parts[1]
            else:
                personalized_generation = generation + reminder_message

            state["generation"] = personalized_generation
            state["reminder_added"] = True

            logger.info(f"[PersonalizeResponse] 상기 메시지 추가: {items_text}")
        else:
            state["reminder_added"] = False

    except Exception as e:
        logger.error(f"[PersonalizeResponse] 실패: {e}")
        state["reminder_added"] = False

    elapsed = time.time() - start_time
    logger.info(f"[PersonalizeResponse] 완료 ({elapsed:.2f}s)")

    return add_to_history(state, "personalize_response")


# ========== 노드 13: Suggest Related Questions ==========

def suggest_related_questions_node(state: RAGState) -> RAGState:
    """
    관련 질문 추천

    현재 질문과 답변, 그리고 사용자 컨텍스트를 바탕으로
    사용자가 다음에 할 수 있는 관련 질문 3개를 추천합니다.

    Args:
        state (RAGState): 현재 상태

    Returns:
        RAGState: 관련 질문 목록이 포함된 상태
    """
    logger.info("[SuggestQuestions] 관련 질문 추천 시작")
    start_time = time.time()

    resources = get_resources()

    question = state["question"]
    generation = state["generation"]
    user_context = state.get("user_context", {})

    # 사용자 컨텍스트 요약
    context_summary = ""
    if user_context:
        learning_goals = user_context.get("learning_goals", "")
        interested_topics = user_context.get("interested_topics", "")
        if learning_goals:
            context_summary += f"학습 목표: {learning_goals}\n"
        if interested_topics:
            context_summary += f"관심 주제: {interested_topics}\n"

    # 답변에서 출처 제거
    answer_only = _strip_existing_sources(generation)
    if len(answer_only) > 500:
        answer_only = answer_only[:500] + "..."

    system_prompt = """당신은 학습 도우미입니다. 사용자의 현재 질문과 답변을 보고,
자연스럽게 이어질 수 있는 관련 질문 3개를 추천하세요.

추천 질문은:
- 현재 질문에서 자연스럽게 파생되는 내용
- 더 깊이 있는 이해를 돕는 내용
- 실용적이고 구체적인 내용
- 사용자의 학습 목표/관심사와 관련된 내용

각 질문은 한 줄로 작성하고, 번호나 불릿 없이 줄바꿈으로만 구분하세요."""

    user_prompt = f"""현재 질문: {question}

답변 요약: {answer_only}"""

    if context_summary:
        user_prompt += f"\n\n사용자 정보:\n{context_summary}"

    user_prompt += "\n\n관련 질문 3개를 추천해주세요 (각 질문을 줄바꿈으로 구분):"

    try:
        response = resources.llm_client.chat.completions.create(
            model=resources.langchain_llm_fast.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=200,
        )

        suggestions_text = response.choices[0].message.content.strip()

        # 줄바꿈으로 분리하고 정리
        suggestions = [
            line.strip().lstrip("0123456789.-) ").strip()
            for line in suggestions_text.split("\n")
            if line.strip() and len(line.strip()) > 10
        ]

        # 최대 3개만
        suggestions = suggestions[:3]

        state["related_questions"] = suggestions
        logger.info(f"[SuggestQuestions] {len(suggestions)}개 질문 추천 완료")

    except Exception as e:
        logger.error(f"[SuggestQuestions] 실패: {e}")
        state["related_questions"] = []

    elapsed = time.time() - start_time
    logger.info(f"[SuggestQuestions] 완료 ({elapsed:.2f}s)")

    return add_to_history(state, "suggest_related_questions")
