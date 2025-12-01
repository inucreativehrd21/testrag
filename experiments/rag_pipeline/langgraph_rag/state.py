"""
LangGraph RAG State 정의

이 모듈은 RAG 파이프라인의 상태를 관리합니다.
LangGraph는 상태 기반 워크플로우를 사용하며, 각 노드는 이 상태를 읽고 수정합니다.
"""

from typing import List, Dict, Any, TypedDict, Literal


class RAGState(TypedDict):
    """
    RAG 파이프라인의 전체 상태를 나타내는 TypedDict

    LangGraph는 이 상태 객체를 노드 간에 전달하며,
    각 노드는 상태의 일부를 읽거나 수정할 수 있습니다.

    Attributes:
        question (str): 사용자의 원본 질문
        route (str): 질문 라우팅 결정 ("vectorstore" | "websearch" | "direct")
        documents (List[str]): 검색된 문서 텍스트 리스트
        metadatas (List[Dict[str, Any]]): 검색된 문서의 메타데이터 (url, domain, length 등)
        reranked_documents (List[str]): 1차 reranking 후 문서
        reranked_metadatas (List[Dict[str, Any]]): 1차 reranking 후 메타데이터
        final_documents (List[str]): 2차 reranking 후 최종 문서
        final_metadatas (List[Dict[str, Any]]): 2차 reranking 후 최종 메타데이터
        generation (str): LLM이 생성한 답변
        web_search_needed (bool): 웹 검색 필요 여부 플래그
        retry_count (int): 재시도 횟수 (무한 루프 방지)
        document_relevance (str): 문서 관련성 평가 결과 ("relevant" | "not_relevant")
        hallucination_grade (str): 환각 검증 결과 ("supported" | "not_supported" | "not_sure")
        answer_usefulness (str): 답변 유용성 평가 결과 ("useful" | "not_useful")
        transformed_query (str): 재작성된 검색 쿼리
        workflow_history (List[str]): 실행된 노드 이름 기록 (디버깅용)
    """

    # ========== 입력 ==========
    question: str

    # ========== 라우팅 ==========
    route: Literal["vectorstore", "websearch", "direct"]

    # ========== 검색 결과 ==========
    documents: List[str]
    metadatas: List[Dict[str, Any]]

    # ========== Reranking 결과 ==========
    reranked_documents: List[str]
    reranked_metadatas: List[Dict[str, Any]]
    final_documents: List[str]
    final_metadatas: List[Dict[str, Any]]

    # ========== 생성 결과 ==========
    generation: str

    # ========== 제어 플래그 ==========
    web_search_needed: bool
    retry_count: int

    # ========== 평가 결과 ==========
    document_relevance: Literal["relevant", "not_relevant", "unknown"]
    hallucination_grade: Literal["supported", "not_supported", "not_sure"]
    answer_usefulness: Literal["useful", "not_useful", "unknown"]

    # ========== 쿼리 변환 ==========
    transformed_query: str

    # ========== 디버깅 ==========
    workflow_history: List[str]


def create_initial_state(question: str) -> RAGState:
    """
    초기 RAG 상태 생성

    Args:
        question (str): 사용자 질문

    Returns:
        RAGState: 초기화된 상태 객체

    Note:
        - 모든 리스트는 빈 리스트로 초기화
        - 플래그는 False로 초기화
        - retry_count는 0으로 초기화
    """
    return {
        # 입력
        "question": question,

        # 라우팅 (초기값: 알 수 없음)
        "route": "vectorstore",  # Default to vectorstore

        # 검색 결과 (빈 리스트)
        "documents": [],
        "metadatas": [],

        # Reranking 결과 (빈 리스트)
        "reranked_documents": [],
        "reranked_metadatas": [],
        "final_documents": [],
        "final_metadatas": [],

        # 생성 결과 (빈 문자열)
        "generation": "",

        # 제어 플래그
        "web_search_needed": False,
        "retry_count": 0,

        # 평가 결과
        "document_relevance": "unknown",
        "hallucination_grade": "not_sure",
        "answer_usefulness": "unknown",

        # 쿼리 변환
        "transformed_query": "",

        # 디버깅
        "workflow_history": [],
    }


def add_to_history(state: RAGState, node_name: str) -> RAGState:
    """
    워크플로우 히스토리에 노드 이름 추가

    LangSmith에서 추적할 때 유용합니다.

    Args:
        state (RAGState): 현재 상태
        node_name (str): 실행된 노드 이름

    Returns:
        RAGState: 업데이트된 상태
    """
    state["workflow_history"].append(node_name)
    return state
