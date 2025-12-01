"""
LangGraph RAG State definition.

This module defines the shared state passed between LangGraph nodes.
"""

from typing import List, Dict, Any, TypedDict, Literal


class RAGState(TypedDict):
    """
    RAG workflow state carried across LangGraph nodes.

    Attributes:
        question (str): 사용자 원 질문
        route (str): 질문 라우팅 결정 ("vectorstore" | "websearch" | "direct")
        intent (str): 질문 의도 분류 ("in_scope" | "greeting" | "chitchat" | "nonsensical" | "unknown")
        documents (List[str]): 검색된 문서 리스트
        metadatas (List[Dict[str, Any]]): 문서 메타데이터 리스트
        reranked_documents (List[str]): 1차 reranking 결과 문서
        reranked_metadatas (List[Dict[str, Any]]): 1차 reranking 메타데이터
        final_documents (List[str]): 2차 reranking 최종 문서
        final_metadatas (List[Dict[str, Any]]): 2차 reranking 최종 메타데이터
        generation (str): 최종 생성 답변
        web_search_needed (bool): 웹 검색 필요 여부
        retry_count (int): 재시도 횟수
        document_relevance (str): 문서 관련성 평가 ("relevant" | "not_relevant" | "unknown")
        hallucination_grade (str): 환각 여부 ("supported" | "not_supported" | "not_sure")
        answer_usefulness (str): 답변 유용성 평가 ("useful" | "not_useful" | "unknown")
        transformed_query (str): 변환된 쿼리
        workflow_history (List[str]): 실행된 노드 기록
    """

    # 입력
    question: str

    # 라우팅/의도
    route: Literal["vectorstore", "websearch", "direct"]
    intent: Literal["in_scope", "greeting", "chitchat", "nonsensical", "unknown"]

    # 검색 결과
    documents: List[str]
    metadatas: List[Dict[str, Any]]

    # Reranking 결과
    reranked_documents: List[str]
    reranked_metadatas: List[Dict[str, Any]]
    final_documents: List[str]
    final_metadatas: List[Dict[str, Any]]

    # 생성 결과
    generation: str

    # 흐름 제어
    web_search_needed: bool
    retry_count: int

    # 평가 결과
    document_relevance: Literal["relevant", "not_relevant", "unknown"]
    hallucination_grade: Literal["supported", "not_supported", "not_sure"]
    answer_usefulness: Literal["useful", "not_useful", "unknown"]

    # 변환/로그
    transformed_query: str
    workflow_history: List[str]


def create_initial_state(question: str) -> RAGState:
    """
    Initialize the RAG state.
    """
    return {
        "question": question,
        "route": "vectorstore",
        "intent": "unknown",
        "documents": [],
        "metadatas": [],
        "reranked_documents": [],
        "reranked_metadatas": [],
        "final_documents": [],
        "final_metadatas": [],
        "generation": "",
        "web_search_needed": False,
        "retry_count": 0,
        "document_relevance": "unknown",
        "hallucination_grade": "not_sure",
        "answer_usefulness": "unknown",
        "transformed_query": "",
        "workflow_history": [],
    }


def add_to_history(state: RAGState, node_name: str) -> RAGState:
    """
    Append executed node name to workflow history.
    """
    state["workflow_history"].append(node_name)
    return state
