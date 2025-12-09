"""
LangGraph RAG Workflow 구성

Adaptive RAG StateGraph를 정의하고 조건부 분기를 설정한다.
"""

import logging
from typing import Literal

from langgraph.graph import StateGraph, END

from .nodes import (
    intent_classifier_node,
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
    # NEW START - 개인화 노드 import
    load_user_context_node,
    suggest_related_questions_node,
    # NEW END - 개인화 노드 import
)
from .state import RAGState

logger = logging.getLogger(__name__)


# ========== 조건부 경로 함수 ==========


def decide_intent_path(state: RAGState) -> Literal["proceed", "end"]:
    """Intent 분류 결과에 따라 진행 여부 결정."""
    intent = state.get("intent", "unknown")
    if intent == "in_scope":
        return "proceed"
    return "end"


def route_question(state: RAGState) -> Literal["vectorstore", "websearch", "direct"]:
    """
    질문 라우팅 결정

    Returns:
        str: "vectorstore" | "websearch" | "direct"
    """
    route = state["route"]
    logger.info(f"[Router] 질문 라우팅: {route}")
    return route


def decide_to_generate_or_transform(
    state: RAGState,
) -> Literal["transform_query", "generate", "websearch"]:
    """
    문서 관련성 평가 이후 다음 경로 결정

    수정된 로직:
    - 문서 관련성이 높으면 generate
    - 문서 관련성이 낮으면:
      - retry_count가 0이면 쿼리 재작성 (1번만 시도)
      - retry_count가 1 이상이면 바로 웹서치
    """
    document_relevance = state["document_relevance"]
    retry_count = state["retry_count"]

    if document_relevance == "relevant":
        logger.info("[Decision] 문서 관련성 높음 → generate")
        return "generate"
    elif retry_count == 0:
        logger.info("[Decision] 문서 관련성 부족 → 쿼리 재작성 (1회)")
        return "transform_query"
    else:
        logger.warning("[Decision] 쿼리 재작성 후에도 관련성 부족 → 웹 검색")
        return "websearch"


def check_hallucination_and_usefulness(
    state: RAGState,
) -> Literal["answer_grading", "websearch", "retry_generate"]:
    """
    환각 여부에 따라 다음 단계 결정

    수정된 로직:
    - 웹서치를 이미 수행했으면 환각이 있어도 answer_grading으로 이동
    - 웹서치를 수행하지 않았고 환각이 있으면 웹서치로 이동
    """
    hallucination_grade = state["hallucination_grade"]
    workflow_history = state.get("workflow_history", [])

    # 웹서치를 이미 수행했는지 확인
    web_search_done = "web_search" in workflow_history

    if hallucination_grade == "supported":
        logger.info("[Decision] 환각 없음 → answer_grading")
        return "answer_grading"
    elif hallucination_grade == "not_supported":
        if web_search_done:
            logger.warning("[Decision] 환각 발견했지만 웹서치 이미 수행됨 → answer_grading")
            return "answer_grading"
        else:
            logger.warning("[Decision] 환각 발견 → 웹 검색으로 보완")
            return "websearch"
    else:
        logger.info("[Decision] 환각 불확실 → answer_grading")
        return "answer_grading"


def grade_generation_usefulness(state: RAGState) -> Literal["end", "websearch"]:
    """
    답변 유용성 평가 이후 종료 또는 웹 검색 결정

    수정된 로직:
    - 답변이 유용하면 종료
    - 답변이 유용하지 않으면:
      - 웹서치를 이미 수행했으면 종료 (더 이상 재시도 안 함)
      - 웹서치를 수행하지 않았으면 웹서치 진행
    """
    answer_usefulness = state["answer_usefulness"]
    workflow_history = state.get("workflow_history", [])

    # 웹서치를 이미 수행했는지 확인
    web_search_done = "web_search" in workflow_history

    if answer_usefulness == "useful":
        logger.info("[Decision] 답변 유용 → 종료")
        return "end"
    else:
        if web_search_done:
            logger.warning("[Decision] 답변 품질 부족하지만 웹서치 이미 수행됨 → 종료")
            return "end"
        else:
            logger.warning("[Decision] 답변 품질 부족 → 웹 검색으로 재시도")
            return "websearch"


# ========== LangGraph 구성 함수 ==========


# NEW START - enable_personalization 파라미터 추가
def create_rag_graph(enable_personalization: bool = True) -> StateGraph:
    """
    Adaptive RAG StateGraph 생성

    Args:
        enable_personalization: True면 개인화 노드 포함, False면 제외
    """
    logger.info(f"Creating Adaptive RAG graph (personalization={enable_personalization})...")

    workflow = StateGraph(RAGState)

    # 기본 노드 등록
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("query_router", query_router_node)
    workflow.add_node("hybrid_retrieve", hybrid_retrieve_node)
    workflow.add_node("rerank_stage1", rerank_stage1_node)
    workflow.add_node("rerank_stage2", rerank_stage2_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("transform_query", transform_query_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("hallucination_check", hallucination_check_node)
    workflow.add_node("answer_grading", answer_grading_node)
    workflow.add_node("web_search", web_search_node)

    # 개인화 노드 등록 (enable_personalization=True일 때만)
    if enable_personalization:
        workflow.add_node("load_user_context", load_user_context_node)
        workflow.add_node("suggest_related_questions", suggest_related_questions_node)

    # 시작점: intent classifier
    workflow.set_entry_point("intent_classifier")

    # Intent → 다음 노드 (개인화 여부에 따라 분기)
    if enable_personalization:
        workflow.add_conditional_edges(
            "intent_classifier",
            decide_intent_path,
            {
                "proceed": "load_user_context",
                "end": END,
            },
        )
        workflow.add_edge("load_user_context", "query_router")
    else:
        workflow.add_conditional_edges(
            "intent_classifier",
            decide_intent_path,
            {
                "proceed": "query_router",
                "end": END,
            },
        )

    # Query Router → Vectorstore / WebSearch / Direct
    workflow.add_conditional_edges(
        "query_router",
        route_question,
        {
            "vectorstore": "hybrid_retrieve",
            "websearch": "web_search",
            "direct": "generate",
        },
    )

    # Vectorstore path
    workflow.add_edge("hybrid_retrieve", "rerank_stage1")
    workflow.add_edge("rerank_stage1", "rerank_stage2")
    workflow.add_edge("rerank_stage2", "grade_documents")

    # Grade Documents → Generate / Transform / WebSearch
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate_or_transform,
        {
            "generate": "generate",
            "transform_query": "transform_query",
            "websearch": "web_search",
        },
    )

    # Transform Query → Hybrid Retrieve
    workflow.add_edge("transform_query", "hybrid_retrieve")

    # Web Search → Generate
    workflow.add_edge("web_search", "generate")

    # Generate → Hallucination Check (개인화 노드 제거됨)
    workflow.add_edge("generate", "hallucination_check")

    # Hallucination Check → Answer Grading / WebSearch / Retry Generate
    workflow.add_conditional_edges(
        "hallucination_check",
        check_hallucination_and_usefulness,
        {
            "answer_grading": "answer_grading",
            "websearch": "web_search",
            "retry_generate": "generate",
        },
    )

    # Answer Grading → END / WebSearch / Suggest Questions (개인화 여부에 따라)
    if enable_personalization:
        workflow.add_conditional_edges(
            "answer_grading",
            grade_generation_usefulness,
            {
                "end": "suggest_related_questions",  # 개인화: 관련 질문 생성 후 종료
                "websearch": "web_search",
            },
        )
        workflow.add_edge("suggest_related_questions", END)
    else:
        workflow.add_conditional_edges(
            "answer_grading",
            grade_generation_usefulness,
            {
                "end": END,
                "websearch": "web_search",
            },
        )

    app = workflow.compile()
    logger.info(f"✓ Adaptive RAG graph created successfully (personalization={enable_personalization})")
    return app
# NEW END - enable_personalization 파라미터 추가


# ========== 그래프 실행/시각화 ==========


# NEW START - enable_personalization 파라미터 추가
def run_rag_graph(
    question: str,
    config_path: str = None,
    user_id: str = "",
    user_context: dict = None,
    enable_personalization: bool = True,
) -> dict:
    """
    RAG 그래프 실행

    Args:
        question: 사용자 질문
        config_path: 설정 파일 경로 (선택)
        user_id: 사용자 식별자 (개인화에 사용, 선택)
        user_context: 사용자 컨텍스트 (Django에서 전달, 선택)
        enable_personalization: 개인화 기능 활성화 여부
    """
    from .state import create_initial_state
    from .config import get_config

    if config_path:
        _ = get_config(config_path)
    else:
        _ = get_config()

    initial_state = create_initial_state(question, user_id=user_id, user_context=user_context)
    app = create_rag_graph(enable_personalization=enable_personalization)

    logger.info(f"\n{'=' * 80}")
    logger.info(f"질문: {question}")
    logger.info(f"개인화: {enable_personalization}")
    logger.info(f"{'=' * 80}\n")

    try:
        final_state = None
        for state in app.stream(initial_state):
            for _, node_state in state.items():
                final_state = node_state

        logger.info(f"\n{'=' * 80}")
        logger.info(f"워크플로우 완료")
        logger.info(f"실행된 노드: {' → '.join(final_state['workflow_history'])}")
        logger.info(f"{'=' * 80}\n")

        return final_state

    except Exception as e:
        logger.error(f"그래프 실행 실패: {e}", exc_info=True)
        raise


async def run_parallel_rag(
    question: str,
    user_id: str = "",
    config_path: str = None,
) -> dict:
    """
    개인화 ON/OFF 두 버전을 병렬 실행하여 A/B 비교

    Args:
        question: 사용자 질문
        user_id: 사용자 식별자 (개인화 버전에서 사용)
        config_path: 설정 파일 경로 (선택)

    Returns:
        dict: {
            "answer_without_personalization": str,
            "answer_with_personalization": str,
            "state_without_personalization": dict,
            "state_with_personalization": dict,
        }
    """
    import asyncio
    from .state import create_initial_state
    from .config import get_config

    if config_path:
        _ = get_config(config_path)
    else:
        _ = get_config()

    logger.info(f"\n{'=' * 80}")
    logger.info(f"[Parallel RAG] 병렬 실행 시작")
    logger.info(f"질문: {question}")
    logger.info(f"{'=' * 80}\n")

    # 같은 함수를 파라미터만 다르게 호출
    app_no_p = create_rag_graph(enable_personalization=False)
    app_with_p = create_rag_graph(enable_personalization=True)

    state_no_p = create_initial_state(question, user_id="")
    state_with_p = create_initial_state(question, user_id=user_id)

    def run_graph(app, initial_state, label: str):
        """그래프 실행 헬퍼"""
        logger.info(f"[{label}] 실행 시작")
        final_state = None
        for state in app.stream(initial_state):
            for _, node_state in state.items():
                final_state = node_state
        logger.info(f"[{label}] 실행 완료")
        return final_state

    # 병렬 실행
    result_no_p, result_with_p = await asyncio.gather(
        asyncio.to_thread(run_graph, app_no_p, state_no_p, "Without Personalization"),
        asyncio.to_thread(run_graph, app_with_p, state_with_p, "With Personalization"),
    )

    logger.info(f"\n{'=' * 80}")
    logger.info(f"[Parallel RAG] 병렬 실행 완료")
    logger.info(f"{'=' * 80}\n")

    return {
        "answer_without_personalization": result_no_p.get("generation", ""),
        "answer_with_personalization": result_with_p.get("generation", ""),
        "state_without_personalization": result_no_p,
        "state_with_personalization": result_with_p,
    }


def run_parallel_rag_sync(
    question: str,
    user_id: str = "",
    config_path: str = None,
) -> dict:
    """
    run_parallel_rag의 동기 버전 (asyncio.run 래퍼)
    """
    import asyncio

    return asyncio.run(run_parallel_rag(question, user_id, config_path))
# NEW END - enable_personalization 파라미터 추가


def visualize_graph(output_path: str = "rag_graph.png"):
    """
    그래프 시각화 (Mermaid + PNG)

    Note:
        - graphviz / pydot 설치 필요
    """
    try:
        app = create_rag_graph()

        print("\n=== LangGraph Mermaid Diagram ===\n")
        print(app.get_graph().draw_mermaid())
        print("\n" + "=" * 80 + "\n")

        # PNG 저장 (graphviz 설치 필요)
        try:
            app.get_graph().draw_png(output_path)
            logger.info(f"그래프 PNG 저장 완료: {output_path}")
        except Exception as e:  # graphviz가 없으면 무시
            logger.warning(f"PNG 저장 실패(그래프 도구 미설치 가능): {e}")

        logger.info("그래프 시각화 완료")
    except ImportError:
        logger.warning("graphviz가 설치되지 않아 시각화를 건너뜁니다.")
    except Exception as e:
        logger.warning(f"그래프 시각화 실패: {e}")
