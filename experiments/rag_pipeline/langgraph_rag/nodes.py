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
"""

import logging
import re
import time
from typing import Dict, List, Tuple

import chromadb
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from openai import AsyncOpenAI, OpenAI

# ### 수정 시작 ###
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
# ### 수정 완료 ###

from .config import get_config
from .state import (
    RAGState,
    add_to_history,
    # ### 수정 시작 ###
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
    # ### 수정 완료 ###
)
from .tools import get_web_search_tool, get_rag_tools  # ### 수정: get_rag_tools 추가 ###

logger = logging.getLogger(__name__)


def _increment_retry_count(state: RAGState) -> None:
    """retry_count는 라우팅 함수 대신 실제 재시도 노드에서만 증가시킨다."""
    config = get_config()
    state["retry_count"] = min(state["retry_count"] + 1, config.max_retries)


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

        # ### 수정 시작 ###
        # LangChain LLM 클라이언트 (structured output / bind_tools 용)
        self.langchain_llm = ChatOpenAI(
            model=config.llm_model,
            temperature=config.llm_temperature,
        )
        self.langchain_llm_fast = ChatOpenAI(
            model=config.context_quality_model,
            temperature=0,
        )
        # ### 수정 완료 ###

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


# ### 수정 시작 ###
def intent_classifier_node(state: RAGState) -> RAGState:
    """
    질문 의도를 분류해 in_scope가 아니면 초기에 종료시킨다.
    with_structured_output을 사용하여 tool calling 방식으로 분류.

    Categories:
    - IN_SCOPE: 개발/프로그래밍/학습 관련
    - GREETING: 인사/감사 등
    - CHITCHAT: 잡담/요청(아이스크림 사줘 등)
    - NONSENSICAL: 무의미/스팸
    """
    logger.info("[Intent] 질문 의도 분류 시작")
    resources = get_resources()

    question = state["question"]
    intent = "unknown"

    system_prompt = """당신은 질문 의도 분류기입니다. 사용자의 질문을 분석하여 의도를 분류하세요.

분류 기준:
- IN_SCOPE: 개발, 프로그래밍, 소프트웨어 학습/디버깅/도구 사용 관련
- GREETING: 인사, 감사, 안부
- CHITCHAT: 잡담/사적요청 (예: 아이스크림 사줘, 노래 추천)
- NONSENSICAL: 무의미/스팸/의미 없는 입력"""

    user_prompt = f"질문: {question}"

    try:
        # with_structured_output으로 IntentClassification Pydantic 모델 강제
        structured_llm = resources.langchain_llm_fast.with_structured_output(
            IntentClassification,
            method="function_calling",
        )
        result: IntentClassification = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])

        intent = result.intent.value
        logger.info(f"[Intent] 분류 결과: {intent}, 근거: {result.reasoning}")

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
# ### 수정 완료 ###


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

# ### 수정 시작 ###
def grade_documents_node(state):
    """
    문서 관련성 평가 (Corrective RAG)
    with_structured_output을 사용하여 tool calling 방식으로 평가.
    """
    logger.info("[GradeDocuments] 문서 관련성 평가 시작")
    start_time = time.time()

    resources = get_resources()
    question = state["question"]
    documents = state["final_documents"]

    if not documents:
        logger.warning("[GradeDocuments] 문서 없음")
        state["document_relevance"] = "not_relevant"
        return add_to_history(state, "grade_documents")

    # with_structured_output으로 DocumentRelevance 모델 사용
    structured_llm = resources.langchain_llm_fast.with_structured_output(
        DocumentRelevance,
        method="function_calling",
    )

    system_prompt = """당신은 문서 관련성 평가기입니다. 주어진 문서가 질문에 답하는 데 도움이 되는지 평가하세요.

평가 기준:
- RELEVANT: 질문에 직접 답할 수 있는 정보 포함
- PARTIAL: 질문과 관련된 정보 일부 포함
- IRRELEVANT: 질문과 관련 없음"""

    # 병렬 처리를 위한 메시지 리스트 준비
    message_batches = []
    for doc in documents:
        doc_preview = doc[:800] if len(doc) > 800 else doc
        user_prompt = f"질문: {question}\n\n문서: {doc_preview}"
        message_batches.append([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])

    # 병렬 처리 (LangChain batch 사용 - 10개 동시 요청)
    try:
        batch_results = structured_llm.batch(message_batches)
        results = []
        for result in batch_results:
            if isinstance(result, DocumentRelevance):
                results.append(result.relevance)
                logger.debug(f"[GradeDocuments] 문서 평가: {result.relevance.value}, 근거: {result.reasoning}")
            else:
                logger.warning(f"예상치 못한 결과 타입, 기본값 PARTIAL 사용")
                results.append(RelevanceType.PARTIAL)
    except Exception as e:
        logger.warning(f"배치 평가 실패: {e}, 모든 문서 PARTIAL 처리")
        results = [RelevanceType.PARTIAL] * len(documents)


    # 결과 집계
    relevant_count = sum(
        1 for r in results if r in (RelevanceType.RELEVANT, RelevanceType.PARTIAL)
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
# ### 수정 완료 ###


# ========== 노드 6: Transform Query ==========

# ### 수정 시작 ###
def transform_query_node(state):
    """
    쿼리 재작성 (Query Transformation)
    with_structured_output을 사용하여 tool calling 방식으로 재작성.
    """
    logger.info("[TransformQuery] 쿼리 재작성 시작")
    start_time = time.time()

    _increment_retry_count(state)

    resources = get_resources()
    question = state["question"]

    system_prompt = """당신은 검색 쿼리 최적화 전문가입니다.
사용자의 질문을 분석하여 검색에 더 적합한 형태로 재작성해야 하는지 판단하세요.

판단 기준:
- PRESERVE: 질문이 이미 충분히 구체적이고 검색에 적합함
- REWRITE: 질문을 더 구체적이고 검색하기 좋은 형태로 재작성 필요

재작성 지침 (REWRITE인 경우):
- 핵심 키워드 강조
- 구체적인 용어 사용
- 검색에 도움이 되는 컨텍스트 추가"""

    user_prompt = f"원본 질문: {question}"

    try:
        structured_llm = resources.langchain_llm_fast.with_structured_output(
            RewrittenQuery,
            method="function_calling",
        )
        result: RewrittenQuery = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])

        if result.action == QueryRewriteAction.REWRITE and result.rewritten_query.strip():
            transformed = result.rewritten_query.strip()
            logger.info(f"[TransformQuery] 원본: {question}")
            logger.info(f"[TransformQuery] 재작성: {transformed}")
            logger.info(f"[TransformQuery] 근거: {result.reasoning}")
            state["transformed_query"] = transformed
            state["question"] = transformed
        else:
            logger.info(f"[TransformQuery] 원본 유지: {question}")
            logger.info(f"[TransformQuery] 근거: {result.reasoning}")
            state["transformed_query"] = question

    except Exception as e:
        logger.error(f"[TransformQuery] 실패: {e}, 원본 쿼리 유지")
        state["transformed_query"] = question

    elapsed = time.time() - start_time
    logger.info(f"[TransformQuery] 완료 ({elapsed:.2f}s)")

    return add_to_history(state, "transform_query")
# ### 수정 완료 ###


# ========== 노드 7: Generate ==========

# ### 수정 시작 ###
def generate_node(state):
    """
    답변 생성 (LLM)
    bind_tools를 사용하여 tool calling 방식으로 답변 생성.
    LLM이 컨텍스트가 불충분하다고 판단하면 web_search 도구를 호출할 수 있음.
    """
    logger.info("[Generate] 답변 생성 시작")
    start_time = time.time()

    resources = get_resources()

    question = state["question"]
    documents = state["final_documents"]
    metadatas = state["final_metadatas"]

    if not documents:
        logger.warning("[Generate] 문서 없음")
        state["generation"] = "관련 문서를 찾지 못했습니다. 질문을 다르게 표현해보시겠어요?"
        return add_to_history(state, "generate")

    # 컨텍스트 포맷팅
    context_block = "\n\n".join(
        f"[문서 {i+1}] {meta.get('domain', 'unknown')}\n{doc}"
        for i, (doc, meta) in enumerate(zip(documents, metadatas))
    )

    system_content = resources.system_prompt + "\n\n규칙: 본문에 툴/출처명(tavily, websearch 등)을 넣지 말고, 출처는 마지막 '📚 참고' 섹션에만 표기하세요."
    user_content = f"질문: {question}\n\n컨텍스트:\n{context_block}"

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_content),
    ]

    try:
        # bind_tools로 도구 바인딩 (LLM이 필요시 web_search 호출 가능)
        rag_tools = get_rag_tools()
        llm_with_tools = resources.langchain_llm.bind_tools(
            rag_tools,
            tool_choice="auto",  # LLM이 자동으로 도구 호출 여부 결정
        )

        response = llm_with_tools.invoke(messages)

        # tool_calls가 있으면 처리
        if response.tool_calls:
            logger.info(f"[Generate] Tool calls 감지: {[tc['name'] for tc in response.tool_calls]}")

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                if tool_name == "web_search":
                    # 웹 검색 수행
                    from .tools import web_search as web_search_tool_func
                    search_result = web_search_tool_func.invoke(tool_args)
                    logger.info(f"[Generate] 웹 검색 수행: {tool_args.get('query', '')}")

                    # 웹 검색 결과로 다시 답변 생성
                    enhanced_context = f"{context_block}\n\n[웹 검색 결과]\n{search_result}"
                    enhanced_messages = [
                        SystemMessage(content=system_content),
                        HumanMessage(content=f"질문: {question}\n\n컨텍스트:\n{enhanced_context}"),
                    ]

                    # 도구 없이 최종 답변 생성
                    final_response = resources.langchain_llm.invoke(enhanced_messages)
                    answer_text = final_response.content

                elif tool_name == "answer_directly":
                    # 바로 답변 (tool 없이 재호출)
                    final_response = resources.langchain_llm.invoke(messages)
                    answer_text = final_response.content
                else:
                    answer_text = response.content or "답변을 생성할 수 없습니다."
        else:
            # tool_calls가 없으면 직접 답변
            answer_text = response.content

        # 기존 출처 제거 및 툴명 정리
        answer_text = _clean_tool_mentions(_strip_existing_sources(answer_text))

        # URL 출처 추가
        source_urls = []
        for meta in metadatas:
            url = meta.get("url", "unknown")
            if url != "unknown" and url not in source_urls:
                source_urls.append(url)

        if source_urls:
            sources_section = "\n\n📚 참고:\n" + "\n".join(
                f"- {url}" for url in source_urls
            )
            answer = answer_text + sources_section
        else:
            answer = answer_text

        state["generation"] = answer
        logger.info("[Generate] 답변 생성 완료")

    except Exception as e:
        logger.error(f"[Generate] 실패: {e}")
        state["generation"] = "답변 생성 중 오류가 발생했습니다."

    elapsed = time.time() - start_time
    logger.info(f"[Generate] 완료 ({elapsed:.2f}s)")

    return add_to_history(state, "generate")
# ### 수정 완료 ###


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


# ========== 노드 8: Hallucination Check ==========

# ### 수정 시작 ###
def hallucination_check_node(state):
    """
    환각 검증 (Self-RAG)
    with_structured_output을 사용하여 tool calling 방식으로 검증.
    """
    logger.info("[HallucinationCheck] 환각 검증 시작")
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

    # 컨텍스트 요약 (너무 길면 truncate)
    context_preview = "\n\n".join(documents[:3])
    if len(context_preview) > 2000:
        context_preview = context_preview[:2000] + "..."

    system_prompt = """당신은 환각(hallucination) 검증 전문가입니다.
답변이 제공된 문서에 근거하는지 판단하세요.

판단 기준:
- SUPPORTED: 답변의 모든 내용이 문서에 근거함
- NOT_SUPPORTED: 문서에 없는 내용이 포함됨 (환각)
- NOT_SURE: 판단하기 어려움"""

    user_prompt = f"""답변:
{answer_only}

제공된 문서:
{context_preview}

답변의 모든 주장이 문서에서 확인됩니까?"""

    try:
        structured_llm = resources.langchain_llm_fast.with_structured_output(
            HallucinationGrade,
            method="function_calling",
        )
        result: HallucinationGrade = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])

        state["hallucination_grade"] = result.grade.value
        logger.info(f"[HallucinationCheck] 결과: {result.grade.value}, 근거: {result.reasoning}")

        if result.grade == HallucinationType.NOT_SUPPORTED:
            state["web_search_needed"] = True

    except Exception as e:
        logger.error(f"[HallucinationCheck] 실패: {e}")
        state["hallucination_grade"] = "not_sure"

    elapsed = time.time() - start_time
    logger.info(f"[HallucinationCheck] 완료 ({elapsed:.2f}s)")

    return add_to_history(state, "hallucination_check")
# ### 수정 완료 ###


# ========== 노드 9: Answer Grading ==========

# ### 수정 시작 ###
def answer_grading_node(state):
    """
    답변 품질 평가 (Self-RAG)
    with_structured_output을 사용하여 tool calling 방식으로 평가.
    """
    logger.info("[AnswerGrading] 답변 품질 평가 시작")
    start_time = time.time()

    resources = get_resources()
    question = state["question"]
    generation = state["generation"]

    # 출처 제거한 답변만 평가
    answer_only = _clean_tool_mentions(_strip_existing_sources(generation))

    system_prompt = """당신은 답변 품질 평가 전문가입니다.
답변이 질문에 유용한지 판단하세요.

평가 기준:
- USEFUL: 질문에 충분히 답변함
- NOT_USEFUL: 질문에 답변하지 못함"""

    user_prompt = f"""질문: {question}

답변: {answer_only}

이 답변이 질문에 충분히 답변합니까?"""

    try:
        structured_llm = resources.langchain_llm_fast.with_structured_output(
            UsefulnessGrade,
            method="function_calling",
        )
        result: UsefulnessGrade = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])

        state["answer_usefulness"] = result.grade.value
        logger.info(f"[AnswerGrading] 결과: {result.grade.value}, 근거: {result.reasoning}")

        if result.grade == UsefulnessType.NOT_USEFUL:
            state["web_search_needed"] = True

    except Exception as e:
        logger.error(f"[AnswerGrading] 실패: {e}")
        state["answer_usefulness"] = "useful"  # 실패 시 긍정으로 가정

    elapsed = time.time() - start_time
    logger.info(f"[AnswerGrading] 완료 ({elapsed:.2f}s)")

    return add_to_history(state, "answer_grading")
# ### 수정 완료 ###


# ========== 노드 10: Web Search ==========

def web_search_node(state):
    """웹 검색 fallback (Corrective RAG)"""
    logger.info("[WebSearch] 웹 검색 시작")
    start_time = time.time()

    _increment_retry_count(state)

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


# NEW START - 개인화 노드

# ========== 노드 11: Load User Context (개인화) ==========

def load_user_context_node(state: RAGState) -> RAGState:
    """
    사용자 컨텍스트 로드 (개인화)

    DB에서 사용자의 과거 서비스 선택 이력을 조회하고,
    현재 질문과 관련된 선택 항목 및 "잊었을 가능성이 있는" 항목을 식별합니다.

    Args:
        state (RAGState): 현재 상태

    Returns:
        RAGState: 개인화 컨텍스트가 추가된 상태
    """
    logger.info("[LoadUserContext] 사용자 컨텍스트 로드 시작")
    start_time = time.time()

    user_id = state.get("user_id", "")
    question = state["question"]

    if not user_id:
        logger.info("[LoadUserContext] user_id 없음, 개인화 스킵")
        return add_to_history(state, "load_user_context")

    try:
        # Step 1: DB에서 사용자 선택 이력 조회
        # TODO: 실제 DB 연동 구현 필요 (현재는 mock)
        user_selections = _fetch_user_selections_from_db(user_id)

        if not user_selections:
            logger.info(f"[LoadUserContext] 사용자 {user_id}의 선택 이력 없음")
            return add_to_history(state, "load_user_context")

        # Step 2: 질문에서 키워드 추출
        question_keywords = _extract_keywords(question)

        # Step 3: 질문과 관련된 선택 항목 필터링
        related_selections = []
        for selection in user_selections:
            if _has_relevance(selection, question_keywords):
                related_selections.append(selection)

        # Step 4: "잊었을 가능성" 판단
        # 선택했지만 질문에서 직접 언급하지 않은 항목 = 상기 후보
        question_lower = question.lower()
        forgotten_candidates = []
        for selection in related_selections:
            service_name = selection.get("service_name", "").lower()
            selected_option = selection.get("selected_option", "").lower()

            # 서비스명이나 선택 옵션이 질문에 없으면 잊었을 가능성
            if service_name not in question_lower and selected_option not in question_lower:
                forgotten_candidates.append(selection)

        state["user_selections"] = user_selections
        state["related_selections"] = related_selections
        state["forgotten_candidates"] = forgotten_candidates

        logger.info(
            f"[LoadUserContext] 로드 완료 - "
            f"전체: {len(user_selections)}, "
            f"관련: {len(related_selections)}, "
            f"상기 후보: {len(forgotten_candidates)}"
        )

    except Exception as e:
        logger.error(f"[LoadUserContext] 실패: {e}")
        state["user_selections"] = []
        state["related_selections"] = []
        state["forgotten_candidates"] = []

    elapsed = time.time() - start_time
    logger.info(f"[LoadUserContext] 완료 ({elapsed:.2f}s)")

    return add_to_history(state, "load_user_context")


def _fetch_user_selections_from_db(user_id: str) -> List[Dict]:
    """
    DB에서 사용자 선택 이력 조회 (Mock 구현)

    TODO: 실제 DB 연동으로 교체 필요

    Returns:
        List[Dict]: 사용자 선택 이력
            - service_name: 서비스명
            - selected_option: 선택한 옵션
            - category: 카테고리
            - selected_at: 선택 일시
    """
    # Mock 데이터 - 실제 구현 시 DB 쿼리로 교체
    # 예: SELECT * FROM user_selections WHERE user_id = ?
    logger.debug(f"[DB Mock] 사용자 {user_id} 선택 이력 조회")

    # 실제 구현 예시:
    # from .database import get_db_connection
    # db = get_db_connection()
    # return db.query("SELECT * FROM user_selections WHERE user_id = ?", [user_id])

    return []  # 실제 DB 연동 전까지 빈 리스트 반환


def _extract_keywords(text: str) -> List[str]:
    """
    텍스트에서 키워드 추출

    Args:
        text: 입력 텍스트

    Returns:
        List[str]: 추출된 키워드 목록
    """
    # 간단한 키워드 추출 (공백 기준 분리 + 불용어 제거)
    # TODO: 더 정교한 키워드 추출 (형태소 분석 등)
    stopwords = {"은", "는", "이", "가", "을", "를", "의", "에", "에서", "으로", "로", "와", "과", "하고", "있", "없", "수"}

    words = text.lower().replace("?", "").replace(".", "").split()
    keywords = [w for w in words if len(w) > 1 and w not in stopwords]

    return keywords


def _has_relevance(selection: Dict, keywords: List[str]) -> bool:
    """
    선택 항목이 키워드와 관련 있는지 판단

    Args:
        selection: 사용자 선택 항목
        keywords: 질문 키워드 목록

    Returns:
        bool: 관련 여부
    """
    service_name = selection.get("service_name", "").lower()
    category = selection.get("category", "").lower()
    selected_option = selection.get("selected_option", "").lower()

    selection_text = f"{service_name} {category} {selected_option}"

    # 키워드 중 하나라도 포함되면 관련 있음
    for keyword in keywords:
        if keyword in selection_text:
            return True

    return False


# ========== 노드 12: Personalize Response (개인화) ==========

def personalize_response_node(state: RAGState) -> RAGState:
    """
    답변 개인화 (상기 메시지 주입)

    생성된 답변에 사용자가 잊었을 수 있는 과거 선택 사항을 상기시키는
    메시지를 자연스럽게 추가합니다.

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
            service_name = item.get("service_name", "")
            selected_option = item.get("selected_option", "")

            if service_name and selected_option:
                reminder_parts.append(f"'{service_name}'에서 '{selected_option}'")
            elif service_name:
                reminder_parts.append(f"'{service_name}'")

        if reminder_parts:
            # 자연스러운 상기 메시지 구성
            if len(reminder_parts) == 1:
                items_text = reminder_parts[0]
            else:
                items_text = f"{reminder_parts[0]}과(와) {reminder_parts[1]}"

            reminder_message = (
                f"\n\n💡 **참고**: 고객님께서는 이전에 {items_text}을(를) "
                f"선택하셨는데요, 이 부분도 함께 확인해보시면 도움이 될 수 있습니다."
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

# NEW END - 개인화 노드


# ========== 노드 13: Suggest Related Questions (비동기 개인화) ==========

def suggest_related_questions_node(state: RAGState) -> RAGState:
    """
    관련 질문 추천 (비동기 개인화)

    현재 질문과 생성된 답변을 바탕으로
    사용자가 이어서 물어볼 만한 관련 질문 3개를 추천합니다.

    Args:
        state (RAGState): 현재 상태

    Returns:
        RAGState: 관련 질문이 추가된 상태
    """
    logger.info("[SuggestQuestions] 관련 질문 추천 시작")
    start_time = time.time()

    resources = get_resources()

    question = state["question"]
    answer = state["generation"]
    user_context = state.get("user_context", {})

    # user_context에서 학습 목표와 관심 주제 추출
    learning_goals = user_context.get("learning_goals", "")
    interested_topics = user_context.get("interested_topics", "")

    # 출처 제거한 답변만 사용
    answer_only = _clean_tool_mentions(_strip_existing_sources(answer))

    # 개인화 컨텍스트 구성
    context_text = ""
    if learning_goals or interested_topics:
        context_text = "\n\n사용자 프로필:"
        if learning_goals:
            context_text += f"\n- 학습 목표: {learning_goals}"
        if interested_topics:
            context_text += f"\n- 관심 주제: {interested_topics}"

    system_prompt = """당신은 학습 도우미입니다. 사용자의 현재 질문과 답변을 보고,
자연스럽게 이어질 수 있는 관련 질문 3개를 추천하세요.

추천 기준:
- 현재 주제와 직접 연관된 심화 질문
- 학습 단계를 고려한 적절한 난이도
- 실무에서 자주 마주치는 상황
- 사용자의 학습 목표와 관심사 반영

형식: 각 질문은 한 줄로, 구체적이고 명확하게 작성
예시:
- Python에서 리스트 컴프리헨션은 어떻게 사용하나요?
- git merge와 git rebase의 차이는 무엇인가요?
- 딕셔너리에서 특정 키가 존재하는지 확인하는 방법은?"""

    user_prompt = f"""현재 질문: {question}

답변 요약: {answer_only[:500]}...{context_text}

위 질문과 답변을 바탕으로 사용자가 이어서 물어볼 만한 관련 질문 3개를 추천하세요."""

    try:
        response = resources.llm_client.chat.completions.create(
            model=get_config().context_quality_model,  # 빠른 모델 사용
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,  # 다양성을 위해 약간 높게
            max_tokens=200,
        )

        suggestions_text = response.choices[0].message.content.strip()

        # 파싱: 각 줄을 질문으로 추출 (- 나 숫자로 시작하는 줄)
        import re
        questions = []
        for line in suggestions_text.split("\n"):
            line = line.strip()
            # - 나 1. 2. 등으로 시작하는 줄 추출
            match = re.match(r'^[-•*\d.)\]]+\s*(.+)$', line)
            if match:
                question_text = match.group(1).strip()
                if question_text and len(question_text) > 10:  # 최소 길이 필터
                    questions.append(question_text)

        # 최대 3개만
        related_questions = questions[:3]

        state["related_questions"] = related_questions

        logger.info(f"[SuggestQuestions] {len(related_questions)}개 질문 추천 완료")
        for i, q in enumerate(related_questions, 1):
            logger.info(f"  {i}. {q[:50]}...")

    except Exception as e:
        logger.error(f"[SuggestQuestions] 실패: {e}")
        state["related_questions"] = []

    elapsed = time.time() - start_time
    logger.info(f"[SuggestQuestions] 완료 ({elapsed:.2f}s)")

    return add_to_history(state, "suggest_related_questions")
