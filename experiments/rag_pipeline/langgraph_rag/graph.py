"""
LangGraph RAG Workflow 구성

이 모듈은 LangGraph StateGraph를 구성하고 조건부 라우팅 로직을 정의합니다.

워크플로우 구조:
1. query_router → 질문 분석
2. hybrid_retrieve → 벡터 검색
3. rerank_stage1 → 1차 reranking
4. rerank_stage2 → 2차 reranking
5. grade_documents → 문서 관련성 평가
   ├─ relevant → generate
   └─ not_relevant → transform_query (또는 web_search)
6. generate → 답변 생성
7. hallucination_check → 환각 검증
8. answer_grading → 답변 품질 평가
   ├─ useful → END
   └─ not_useful → web_search (재시도)
"""

import logging
from typing import Literal

from langgraph.graph import StateGraph, END

from .config import get_config
from .nodes import (
    query_router_node,
    hybrid_retrieve_node,
    rerank_stage1_node,
    rerank_stage2_node,
    grade_documents_node,
    transform_query_node,
    generate_node,
    hallucination_check_node,
    answer_grading_node,
    web_search_node,
)
from .state import RAGState

logger = logging.getLogger(__name__)


# ========== 조건부 라우팅 함수들 ==========


def route_question(state: RAGState) -> Literal["vectorstore", "websearch", "direct"]:
    """
    질문 라우팅 결정

    Args:
        state (RAGState): 현재 상태

    Returns:
        str: "vectorstore" | "websearch" | "direct"

    라우팅 로직:
    - query_router_node에서 결정된 route 값 사용
    - vectorstore: 기본 벡터 검색 파이프라인
    - websearch: 바로 웹 검색으로 이동
    - direct: 검색 없이 바로 답변 생성 (간단한 인사 등)
    """
    route = state["route"]
    logger.info(f"[Router] 질문 라우팅: {route}")
    return route


def decide_to_generate_or_transform(
    state: RAGState,
) -> Literal["transform_query", "generate", "websearch"]:
    """
    문서 관련성 평가 후 다음 단계 결정

    Args:
        state (RAGState): 현재 상태

    Returns:
        str: "transform_query" | "generate" | "websearch"

    결정 로직:
    - relevant: 문서 관련성 높음 → generate (답변 생성)
    - not_relevant + retry < max: 쿼리 재작성 → transform_query
    - not_relevant + retry >= max: 웹 검색 → websearch
    """
    config = get_config()

    document_relevance = state["document_relevance"]
    retry_count = state["retry_count"]

    if document_relevance == "relevant":
        logger.info("[Decision] 문서 관련성 높음 → 답변 생성")
        return "generate"
    elif retry_count < config.max_retries:
        logger.info(f"[Decision] 문서 관련성 낮음 → 쿼리 재작성 (시도 {retry_count + 1}/{config.max_retries})")
        state["retry_count"] += 1
        return "transform_query"
    else:
        logger.warning(f"[Decision] 최대 재시도 횟수 초과 → 웹 검색")
        return "websearch"


def check_hallucination_and_usefulness(
    state: RAGState,
) -> Literal["answer_grading", "websearch", "retry_generate"]:
    """
    환각 검증 후 다음 단계 결정

    Args:
        state (RAGState): 현재 상태

    Returns:
        str: "answer_grading" | "websearch" | "retry_generate"

    결정 로직:
    - supported: 문서에 근거함 → answer_grading (답변 품질 평가)
    - not_supported: 환각 발견 → websearch (웹 검색으로 보완)
    - not_sure: 판단 불가 → answer_grading (일단 진행)
    """
    hallucination_grade = state["hallucination_grade"]
    retry_count = state["retry_count"]
    config = get_config()

    if hallucination_grade == "supported":
        logger.info("[Decision] 환각 없음 → 답변 품질 평가")
        return "answer_grading"
    elif hallucination_grade == "not_supported":
        if retry_count < config.max_retries:
            logger.warning("[Decision] 환각 발견 → 웹 검색으로 보완")
            state["retry_count"] += 1
            return "websearch"
        else:
            logger.warning("[Decision] 최대 재시도 초과, 환각 있지만 답변 반환")
            return "answer_grading"
    else:  # not_sure
        logger.info("[Decision] 환각 판단 불가 → 답변 품질 평가")
        return "answer_grading"


def grade_generation_usefulness(state: RAGState) -> Literal["end", "websearch"]:
    """
    답변 품질 평가 후 최종 결정

    Args:
        state (RAGState): 현재 상태

    Returns:
        str: "end" | "websearch"

    결정 로직:
    - useful: 답변 유용함 → end (완료)
    - not_useful + retry < max: 웹 검색으로 재시도 → websearch
    - not_useful + retry >= max: 강제 종료 → end
    """
    config = get_config()

    answer_usefulness = state["answer_usefulness"]
    retry_count = state["retry_count"]

    if answer_usefulness == "useful":
        logger.info("[Decision] 답변 유용함 → 완료")
        return "end"
    elif retry_count < config.max_retries:
        logger.warning(f"[Decision] 답변 품질 낮음 → 웹 검색 재시도 (시도 {retry_count + 1}/{config.max_retries})")
        state["retry_count"] += 1
        return "websearch"
    else:
        logger.warning("[Decision] 최대 재시도 초과 → 현재 답변으로 완료")
        return "end"


# ========== LangGraph 구성 함수 ==========


def create_rag_graph() -> StateGraph:
    """
    Adaptive RAG StateGraph 생성

    Returns:
        StateGraph: 컴파일된 LangGraph

    그래프 구조:
        START
          ↓
        query_router ──────────────┐
          ↓                        ↓
        vectorstore            websearch
          ↓                        ↓
        hybrid_retrieve          generate
          ↓                        ↓
        rerank_stage1            END
          ↓
        rerank_stage2
          ↓
        grade_documents
          ├─ relevant → generate
          ├─ not_relevant (retry) → transform_query → hybrid_retrieve
          └─ not_relevant (max retry) → websearch → generate
               ↓
        hallucination_check
          ├─ supported → answer_grading
          └─ not_supported → websearch → generate
               ↓
        answer_grading
          ├─ useful → END
          └─ not_useful → websearch → generate
    """
    logger.info("Creating Adaptive RAG graph...")

    # StateGraph 생성
    workflow = StateGraph(RAGState)

    # ========== 노드 추가 ==========

    # 1. Query Router
    workflow.add_node("query_router", query_router_node)

    # 2. Vectorstore Path (기본)
    workflow.add_node("hybrid_retrieve", hybrid_retrieve_node)
    workflow.add_node("rerank_stage1", rerank_stage1_node)
    workflow.add_node("rerank_stage2", rerank_stage2_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("transform_query", transform_query_node)

    # 3. Generation
    workflow.add_node("generate", generate_node)

    # 4. Self-Correction
    workflow.add_node("hallucination_check", hallucination_check_node)
    workflow.add_node("answer_grading", answer_grading_node)

    # 5. Web Search Fallback
    workflow.add_node("web_search", web_search_node)

    # ========== 엣지 추가 ==========

    # 시작점: query_router
    workflow.set_entry_point("query_router")

    # Query Router → Vectorstore / WebSearch / Direct
    workflow.add_conditional_edges(
        "query_router",
        route_question,
        {
            "vectorstore": "hybrid_retrieve",
            "websearch": "web_search",
            "direct": "generate",  # 검색 없이 바로 생성
        },
    )

    # Vectorstore Path: Hybrid Retrieve → Rerank Stage 1 → Rerank Stage 2 → Grade Documents
    workflow.add_edge("hybrid_retrieve", "rerank_stage1")
    workflow.add_edge("rerank_stage1", "rerank_stage2")
    workflow.add_edge("rerank_stage2", "grade_documents")

    # Grade Documents → Generate / Transform Query / WebSearch
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate_or_transform,
        {
            "generate": "generate",
            "transform_query": "transform_query",
            "websearch": "web_search",
        },
    )

    # Transform Query → Hybrid Retrieve (재검색)
    workflow.add_edge("transform_query", "hybrid_retrieve")

    # Web Search → Generate
    workflow.add_edge("web_search", "generate")

    # Generate → Hallucination Check
    workflow.add_edge("generate", "hallucination_check")

    # Hallucination Check → Answer Grading / WebSearch
    workflow.add_conditional_edges(
        "hallucination_check",
        check_hallucination_and_usefulness,
        {
            "answer_grading": "answer_grading",
            "websearch": "web_search",
            "retry_generate": "generate",  # 재생성 (현재 미사용)
        },
    )

    # Answer Grading → END / WebSearch
    workflow.add_conditional_edges(
        "answer_grading",
        grade_generation_usefulness,
        {
            "end": END,
            "websearch": "web_search",
        },
    )

    # 그래프 컴파일
    app = workflow.compile()

    logger.info("✓ Adaptive RAG graph created successfully")
    return app


# ========== 그래프 실행 함수 ==========


def run_rag_graph(question: str, config_path: str = None) -> dict:
    """
    RAG 그래프 실행

    Args:
        question (str): 질문
        config_path (str, optional): 설정 파일 경로

    Returns:
        dict: 최종 상태

    사용 예시:
        >>> result = run_rag_graph("git rebase란 무엇인가요?")
        >>> print(result["generation"])
        >>> print(result["workflow_history"])
    """
    from .state import create_initial_state
    from .config import get_config

    # 설정 로드
    if config_path:
        config = get_config(config_path)
    else:
        config = get_config()

    # 초기 상태 생성
    initial_state = create_initial_state(question)

    # 그래프 생성
    app = create_rag_graph()

    # 그래프 실행
    logger.info(f"\n{'='*80}")
    logger.info(f"질문: {question}")
    logger.info(f"{'='*80}\n")

    try:
        # LangGraph 실행 (stream 모드로 각 노드 출력 확인 가능)
        final_state = None
        for state in app.stream(initial_state):
            # state는 {node_name: updated_state} 형태
            for node_name, node_state in state.items():
                logger.debug(f"[Stream] {node_name} 완료")
                final_state = node_state

        logger.info(f"\n{'='*80}")
        logger.info(f"워크플로우 완료")
        logger.info(f"실행된 노드: {' → '.join(final_state['workflow_history'])}")
        logger.info(f"{'='*80}\n")

        return final_state

    except Exception as e:
        logger.error(f"그래프 실행 실패: {e}", exc_info=True)
        raise


def visualize_graph(output_path: str = "rag_graph.png"):
    """
    그래프 시각화 (선택사항)

    Args:
        output_path (str): 출력 파일 경로

    Note:
        - graphviz 설치 필요: pip install graphviz
        - 그래프 구조를 PNG로 저장
    """
    try:
        from langgraph.graph import Graph

        app = create_rag_graph()

        # Mermaid 다이어그램 출력
        print("\n=== LangGraph Mermaid Diagram ===\n")
        print(app.get_graph().draw_mermaid())
        print("\n" + "=" * 80 + "\n")

        logger.info(f"그래프 다이어그램 출력 완료")

    except ImportError:
        logger.warning("graphviz가 설치되지 않았습니다. 시각화를 건너뜁니다.")
    except Exception as e:
        logger.warning(f"그래프 시각화 실패: {e}")
