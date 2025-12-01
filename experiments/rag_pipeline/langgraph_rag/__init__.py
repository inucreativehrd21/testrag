"""
LangGraph Adaptive RAG System

이 패키지는 LangGraph를 사용하여 Adaptive, Corrective, Self-RAG 기능을 구현합니다.

주요 컴포넌트:
- state: RAG 상태 정의
- config: 설정 관리
- tools: 웹 검색 등 외부 도구
- nodes: LangGraph 노드 함수들
- graph: LangGraph StateGraph 구성
- main: 실행 진입점

사용법:
    from langgraph_rag import run_rag_graph

    result = run_rag_graph("git rebase란 무엇인가요?")
    print(result["generation"])
"""

from .config import RAGConfig, get_config
from .graph import create_rag_graph, run_rag_graph, visualize_graph
from .state import RAGState, create_initial_state
from .tools import WebSearchTool, get_web_search_tool

__version__ = "1.0.0"
__author__ = "Claude Code"

__all__ = [
    # Config
    "RAGConfig",
    "get_config",
    # Graph
    "create_rag_graph",
    "run_rag_graph",
    "visualize_graph",
    # State
    "RAGState",
    "create_initial_state",
    # Tools
    "WebSearchTool",
    "get_web_search_tool",
]
