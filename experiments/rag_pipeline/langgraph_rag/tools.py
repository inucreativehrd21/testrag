"""
LangGraph RAG 외부 도구

이 모듈은 RAG 시스템이 사용하는 외부 도구를 정의합니다.
주요 도구: Tavily 웹 검색
"""

import os
from typing import List, Dict, Any

# ### 수정 시작 ###
from langchain_core.tools import tool
# ### 수정 완료 ###


class WebSearchTool:
    """
    웹 검색 도구 (Tavily API 사용)

    RAG 시스템이 로컬 벡터 DB에서 답변을 찾지 못할 때,
    실시간 웹 검색으로 fallback합니다.

    Attributes:
        api_key (str): Tavily API 키
        max_results (int): 최대 검색 결과 수
    """

    def __init__(self, api_key: str = None, max_results: int = 5):
        """
        웹 검색 도구 초기화

        Args:
            api_key (str, optional): Tavily API 키. None이면 환경변수에서 로드
            max_results (int): 최대 검색 결과 수 (기본: 5)
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY", "")
        self.max_results = max_results

        # Tavily API 키 확인
        if not self.api_key:
            print(
                "[WARNING] TAVILY_API_KEY가 설정되지 않았습니다. "
                "웹 검색 기능이 비활성화됩니다."
            )
            self.enabled = False
        else:
            self.enabled = True
            # Tavily 클라이언트 임포트 (설치 필요: pip install tavily-python)
            try:
                from tavily import TavilyClient

                self.client = TavilyClient(api_key=self.api_key)
            except ImportError:
                print(
                    "[ERROR] tavily-python 패키지가 설치되지 않았습니다. "
                    "설치: pip install tavily-python"
                )
                self.enabled = False

    def search(self, query: str) -> List[str]:
        """
        웹 검색 실행

        Args:
            query (str): 검색 쿼리

        Returns:
            List[str]: 검색 결과 텍스트 리스트

        Note:
            - Tavily API가 비활성화된 경우 빈 리스트 반환
            - 각 검색 결과는 "제목 + 내용" 형식으로 결합
        """
        if not self.enabled:
            print("[WARNING] 웹 검색이 비활성화되어 있습니다.")
            return []

        try:
            # Tavily 검색 실행
            response = self.client.search(
                query=query,
                max_results=self.max_results,
                search_depth="advanced",  # 고급 검색 모드
                include_answer=True,  # 답변 요약 포함
            )

            # 검색 결과 파싱
            documents = []

            # 1. Tavily의 답변 요약 추가 (있는 경우)
            if "answer" in response and response["answer"]:
                documents.append(f"[Tavily Summary] {response['answer']}")

            # 2. 개별 검색 결과 추가
            for result in response.get("results", []):
                title = result.get("title", "")
                content = result.get("content", "")
                url = result.get("url", "")

                # 제목 + 내용 결합
                doc_text = f"{title}\n\n{content}\n\nSource: {url}"
                documents.append(doc_text)

            print(f"[WebSearch] {len(documents)}개 검색 결과 반환")
            return documents

        except Exception as e:
            print(f"[ERROR] 웹 검색 실패: {e}")
            return []

    def search_with_metadata(self, query: str) -> tuple[List[str], List[Dict[str, Any]]]:
        """
        웹 검색 실행 (메타데이터 포함)

        Args:
            query (str): 검색 쿼리

        Returns:
            tuple[List[str], List[Dict]]: (문서 텍스트 리스트, 메타데이터 리스트)

        Note:
            - 메타데이터에는 url, title, domain 포함
            - RAG 시스템의 출처 표시에 사용
        """
        if not self.enabled:
            print("[WARNING] 웹 검색이 비활성화되어 있습니다.")
            return [], []

        try:
            # Tavily 검색 실행
            response = self.client.search(
                query=query,
                max_results=self.max_results,
                search_depth="advanced",
                include_answer=True,
            )

            documents = []
            metadatas = []

            # 1. Tavily 답변 요약
            if "answer" in response and response["answer"]:
                documents.append(f"[Tavily Summary] {response['answer']}")
                metadatas.append(
                    {
                        "url": "https://tavily.com",
                        "title": "Tavily AI Summary",
                        "domain": "tavily",
                        "length": len(response["answer"]),
                    }
                )

            # 2. 개별 검색 결과
            for result in response.get("results", []):
                title = result.get("title", "")
                content = result.get("content", "")
                url = result.get("url", "")

                doc_text = f"{title}\n\n{content}"
                documents.append(doc_text)

                metadatas.append(
                    {
                        "url": url,
                        "title": title,
                        "domain": "websearch",
                        "length": len(doc_text),
                    }
                )

            print(f"[WebSearch] {len(documents)}개 검색 결과 반환 (메타데이터 포함)")
            return documents, metadatas

        except Exception as e:
            print(f"[ERROR] 웹 검색 실패: {e}")
            return [], []


# 전역 웹 검색 도구 인스턴스 (싱글톤)
_web_search_tool = None


def get_web_search_tool() -> WebSearchTool:
    """
    전역 웹 검색 도구 인스턴스 반환 (싱글톤)

    Returns:
        WebSearchTool: 웹 검색 도구
    """
    global _web_search_tool
    if _web_search_tool is None:
        _web_search_tool = WebSearchTool()
    return _web_search_tool


# ### 수정 시작 ###
# ========== LangChain Tool (bind_tools용) ==========

@tool(parse_docstring=True)
def web_search(thought: str, query: str) -> str:
    """
    웹에서 최신 정보를 검색합니다. 로컬 문서에서 답변을 찾지 못했거나 최신 정보가 필요할 때 사용하세요.

    Args:
        thought: 웹 검색을 수행하는 이유 (1-2문장)
        query: 검색할 쿼리

    Returns:
        검색 결과 텍스트
    """
    web_tool = get_web_search_tool()
    if not web_tool.enabled:
        return "웹 검색이 비활성화되어 있습니다."

    documents = web_tool.search(query)
    if not documents:
        return "검색 결과가 없습니다."

    return "\n\n---\n\n".join(documents)


@tool(parse_docstring=True)
def answer_directly(thought: str) -> str:
    """
    충분한 정보가 있어 바로 답변할 수 있을 때 호출합니다.
    추가 검색 없이 현재 컨텍스트로 답변을 생성합니다.

    Args:
        thought: 바로 답변할 수 있는 이유 (1-2문장)

    Returns:
        빈 문자열 (답변은 generate 노드에서 생성)
    """
    return ""


def get_rag_tools() -> List:
    """
    RAG 시스템에서 사용할 LangChain Tool 리스트 반환

    Returns:
        List: [web_search, answer_directly] 도구들
    """
    return [web_search, answer_directly]
# ### 수정 완료 ###
