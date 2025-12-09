"""
LangGraph RAG State definition.

This module defines the shared state passed between LangGraph nodes.
"""

from enum import Enum
from typing import List, Dict, Any, TypedDict, Literal

from pydantic import BaseModel, Field


# ### 수정 시작 ###
# ========== Pydantic 모델 (Structured Output용) ==========

class IntentType(str, Enum):
    """질문 의도 분류"""
    IN_SCOPE = "in_scope"
    GREETING = "greeting"
    CHITCHAT = "chitchat"
    NONSENSICAL = "nonsensical"


class IntentClassification(BaseModel):
    """Intent 분류 결과"""
    reasoning: str = Field(description="분류 근거 (1-2문장)")
    intent: IntentType = Field(description="분류된 의도")


class RelevanceType(str, Enum):
    """문서 관련성 평가"""
    RELEVANT = "relevant"
    PARTIAL = "partial"
    IRRELEVANT = "irrelevant"


class DocumentRelevance(BaseModel):
    """단일 문서 관련성 평가 결과"""
    reasoning: str = Field(description="평가 근거")
    relevance: RelevanceType = Field(description="관련성 평가")


class QueryRewriteAction(str, Enum):
    """쿼리 재작성 액션"""
    PRESERVE = "preserve"
    REWRITE = "rewrite"


class RewrittenQuery(BaseModel):
    """쿼리 재작성 결과"""
    reasoning: str = Field(description="재작성 근거")
    action: QueryRewriteAction = Field(description="재작성 여부")
    rewritten_query: str = Field(description="재작성된 쿼리 (action이 rewrite일 때만 사용)")


class HallucinationType(str, Enum):
    """환각 여부 평가"""
    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    NOT_SURE = "not_sure"


class HallucinationGrade(BaseModel):
    """환각 검증 결과"""
    reasoning: str = Field(description="판단 근거")
    grade: HallucinationType = Field(description="환각 여부")


class UsefulnessType(str, Enum):
    """답변 유용성 평가"""
    USEFUL = "useful"
    NOT_USEFUL = "not_useful"


class UsefulnessGrade(BaseModel):
    """답변 유용성 평가 결과"""
    reasoning: str = Field(description="평가 근거")
    grade: UsefulnessType = Field(description="유용성 평가")


# ### 수정 완료 ###


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

    # NEW START - 개인화 필드
    user_id: str  # 사용자 식별자
    user_context: Dict[str, Any]  # Django에서 전달받은 사용자 컨텍스트
    user_selections: List[Dict[str, Any]]  # 사용자 전체 선택 이력 (DB에서 로드)
    related_selections: List[Dict[str, Any]]  # 현재 질문과 관련된 선택 항목
    forgotten_candidates: List[Dict[str, Any]]  # 사용자가 잊었을 가능성 있는 항목 (상기 후보)
    reminder_added: bool  # 상기 메시지 추가 여부
    reminder_message: str  # 생성된 상기 메시지 (suggest_related_questions_node에서 생성)
    related_questions: List[str]  # 추천 관련 질문 (비동기로 생성)
    # NEW END - 개인화 필드


def create_initial_state(question: str, user_id: str = "", user_context: Dict[str, Any] = None) -> RAGState:
    """
    Initialize the RAG state.

    Args:
        question: 사용자 질문
        user_id: 사용자 식별자 (개인화에 사용)
        user_context: Django에서 전달받은 사용자 컨텍스트 (learning_goals, interested_topics 등)
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
        # NEW START - 개인화 초기값
        "user_id": user_id,
        "user_context": user_context or {},
        "user_selections": [],
        "related_selections": [],
        "forgotten_candidates": [],
        "reminder_added": False,
        "reminder_message": "",
        "related_questions": [],
        # NEW END - 개인화 초기값
    }


def add_to_history(state: RAGState, node_name: str) -> RAGState:
    """
    Append executed node name to workflow history.
    """
    state["workflow_history"].append(node_name)
    return state
