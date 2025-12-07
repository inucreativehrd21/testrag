"""
LangGraph RAG State definition.

This module defines the shared state passed between LangGraph nodes.
"""

from enum import Enum
from typing import List, Dict, Any, TypedDict, Literal

from pydantic import BaseModel, Field


# ========== Pydantic Models (for Structured Output) ==========

class IntentType(str, Enum):
    """Question intent classification"""
    IN_SCOPE = "in_scope"
    GREETING = "greeting"
    CHITCHAT = "chitchat"
    NONSENSICAL = "nonsensical"


class IntentClassification(BaseModel):
    """Intent classification result"""
    reasoning: str = Field(description="Classification reasoning (1-2 sentences)")
    intent: IntentType = Field(description="Classified intent")


class RelevanceType(str, Enum):
    """Document relevance evaluation"""
    RELEVANT = "relevant"
    PARTIAL = "partial"
    IRRELEVANT = "irrelevant"


class DocumentRelevance(BaseModel):
    """Single document relevance evaluation result"""
    reasoning: str = Field(description="Evaluation reasoning")
    relevance: RelevanceType = Field(description="Relevance evaluation")


class QueryRewriteAction(str, Enum):
    """Query rewrite action"""
    PRESERVE = "preserve"
    REWRITE = "rewrite"


class RewrittenQuery(BaseModel):
    """Query rewrite result"""
    reasoning: str = Field(description="Rewrite reasoning")
    action: QueryRewriteAction = Field(description="Whether to rewrite")
    rewritten_query: str = Field(description="Rewritten query (used only when action is rewrite)")


class HallucinationType(str, Enum):
    """Hallucination evaluation"""
    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    NOT_SURE = "not_sure"


class HallucinationGrade(BaseModel):
    """Hallucination verification result"""
    reasoning: str = Field(description="Judgment reasoning")
    grade: HallucinationType = Field(description="Hallucination status")
    confidence: float = Field(description="Confidence score (0.0-1.0)", ge=0.0, le=1.0)


class UsefulnessType(str, Enum):
    """Answer usefulness evaluation"""
    USEFUL = "useful"
    NOT_USEFUL = "not_useful"


class UsefulnessGrade(BaseModel):
    """Answer usefulness evaluation result"""
    reasoning: str = Field(description="Evaluation reasoning")
    grade: UsefulnessType = Field(description="Usefulness evaluation")


# ========== RAG State Definition ==========

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

        # Personalization fields
        user_id (str): 사용자 식별자
        user_context (Dict[str, Any]): 사용자 컨텍스트 (Django에서 전달)
        related_selections (List[Dict[str, Any]]): 현재 질문과 관련된 선택 항목
        forgotten_candidates (List[Dict[str, Any]]): 사용자가 잊었을 가능성 있는 항목
        reminder_added (bool): 상기 메시지 추가 여부

        # Question suggestion field
        related_questions (List[str]): 관련 질문 추천 리스트
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

    # Personalization fields
    user_id: str
    user_context: Dict[str, Any]
    related_selections: List[Dict[str, Any]]
    forgotten_candidates: List[Dict[str, Any]]
    reminder_added: bool

    # Question suggestion field
    related_questions: List[str]

    # Chat history
    chat_history: List[Dict[str, Any]]


def create_initial_state(question: str, user_id: str = "", user_context: Dict[str, Any] = None) -> RAGState:
    """
    Initialize the RAG state.

    Args:
        question: User question
        user_id: User identifier (for personalization)
        user_context: User context from Django (learning_goals, interested_topics, etc.)
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
        # Personalization initial values
        "user_id": user_id,
        "user_context": user_context or {},
        "related_selections": [],
        "forgotten_candidates": [],
        "reminder_added": False,
        # Question suggestion initial value
        "related_questions": [],
        # Chat history (optional, filled by caller)
        "chat_history": [],
    }


def add_to_history(state: RAGState, node_name: str) -> RAGState:
    """
    Append executed node name to workflow history.
    """
    state["workflow_history"].append(node_name)
    return state
