"""
LangGraph RAG 설정 관리

이 모듈은 RAG 시스템의 모든 설정을 관리합니다.
기존 enhanced.yaml 설정을 로드하고, LangGraph 특화 설정을 추가합니다.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


class RAGConfig:
    """
    RAG 시스템 설정 클래스

    기존 enhanced.yaml 설정을 로드하고,
    LangGraph 특화 설정을 추가로 관리합니다.

    Attributes:
        config_path (Path): 설정 파일 경로
        config (Dict): 로드된 YAML 설정
        langsmith_enabled (bool): LangSmith 추적 활성화 여부
        max_retry_count (int): 최대 재시도 횟수
        tavily_api_key (str): Tavily 웹 검색 API 키
    """

    def __init__(self, config_path: str = None):
        """
        설정 초기화

        Args:
            config_path (str, optional): 설정 파일 경로.
                                         None이면 기본 enhanced.yaml 사용
        """
        # Base project dir (experiments/rag_pipeline)
        self.base_dir = Path(__file__).resolve().parent.parent

        if config_path is None:
            # 기본 경로: experiments/rag_pipeline/config/enhanced.yaml
            resolved_config = self.base_dir / "config" / "enhanced.yaml"
        else:
            candidate = Path(config_path)
            # Relative paths are resolved from the project base dir to avoid CWD differences
            resolved_config = candidate if candidate.is_absolute() else (self.base_dir / candidate)

        self.config_path = resolved_config.resolve()
        self.config = self._load_config()

        # LangGraph 특화 설정
        self.langsmith_enabled = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
        self.max_retry_count = 3  # 최대 3회 재시도
        self.tavily_api_key = os.getenv("TAVILY_API_KEY", "")

        # OpenAI API 키 확인
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        if not self.openai_api_key:
            print("[WARNING] OPENAI_API_KEY가 설정되지 않았습니다.")

    def _load_config(self) -> Dict[str, Any]:
        """
        YAML 설정 파일 로드

        Returns:
            Dict: 로드된 설정

        Raises:
            FileNotFoundError: 설정 파일이 없는 경우
        """
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"설정 파일을 찾을 수 없습니다: {self.config_path}"
            )

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return config

    # ========== 기존 설정 접근 메서드 ==========

    @property
    def artifacts_dir(self) -> Path:
        """artifacts 디렉토리 경로"""
        return self.base_dir / self.config["project"]["artifacts_dir"]

    @property
    def chroma_db_path(self) -> Path:
        """ChromaDB 경로"""
        return self.artifacts_dir / "chroma_db"

    @property
    def chunks_path(self) -> Path:
        """chunks.parquet 경로"""
        return self.artifacts_dir / "chunks.parquet"

    @property
    def embedding_model(self) -> str:
        """임베딩 모델 이름"""
        return self.config["embedding"]["model_name"]

    @property
    def embedding_device(self) -> str:
        """임베딩 디바이스 (cuda/cpu)"""
        return self.config["embedding"]["device"]

    @property
    def embedding_batch_size(self) -> int:
        """임베딩 배치 크기"""
        return self.config["embedding"]["batch_size"]

    @property
    def hybrid_dense_top_k(self) -> int:
        """Hybrid search - Dense 검색 top_k"""
        return self.config["retrieval"]["hybrid_dense_top_k"]

    @property
    def hybrid_sparse_top_k(self) -> int:
        """Hybrid search - Sparse 검색 top_k"""
        return self.config["retrieval"]["hybrid_sparse_top_k"]

    @property
    def rerank_top_k(self) -> int:
        """최종 reranking 후 top_k"""
        return self.config["retrieval"]["rerank_top_k"]

    @property
    def reranker_stage1_model(self) -> str:
        """1차 reranker 모델"""
        return self.config["retrieval"]["rerankers"]["stage1"]["model_name"]

    @property
    def reranker_stage1_device(self) -> str:
        """1차 reranker 디바이스"""
        return self.config["retrieval"]["rerankers"]["stage1"]["device"]

    @property
    def reranker_stage2_model(self) -> str:
        """2차 reranker 모델"""
        return self.config["retrieval"]["rerankers"]["stage2"]["model_name"]

    @property
    def reranker_stage2_device(self) -> str:
        """2차 reranker 디바이스"""
        return self.config["retrieval"]["rerankers"]["stage2"]["device"]

    @property
    def llm_provider(self) -> str:
        """LLM 제공자"""
        return self.config["llm"]["provider"]

    @property
    def llm_model(self) -> str:
        """LLM 모델 이름"""
        return self.config["llm"]["model_name"]

    @property
    def llm_temperature(self) -> float:
        """LLM temperature"""
        return self.config["llm"]["temperature"]

    @property
    def llm_max_tokens(self) -> int:
        """LLM 최대 토큰"""
        return self.config["llm"]["max_new_tokens"]

    @property
    def context_quality_enabled(self) -> bool:
        """Context quality filter 활성화 여부"""
        return self.config.get("context_quality", {}).get("enabled", False)

    @property
    def context_quality_threshold(self) -> float:
        """Context quality threshold"""
        return self.config.get("context_quality", {}).get("threshold", 0.6)

    @property
    def context_quality_model(self) -> str:
        """Context quality 평가 모델"""
        return self.config.get("context_quality", {}).get("evaluator_model", "gpt-4o-mini")

    # ========== LangGraph 특화 설정 ==========

    @property
    def use_langsmith(self) -> bool:
        """LangSmith 추적 사용 여부"""
        return self.langsmith_enabled

    @property
    def max_retries(self) -> int:
        """최대 재시도 횟수 (무한 루프 방지)"""
        return self.max_retry_count

    @property
    def web_search_enabled(self) -> bool:
        """웹 검색 기능 활성화 여부 (Tavily API 키 존재 여부로 판단)"""
        return bool(self.tavily_api_key)

    def print_config_summary(self):
        """설정 요약 출력 (디버깅용)"""
        print("\n" + "=" * 80)
        print("LangGraph RAG 설정 요약")
        print("=" * 80)
        print(f"설정 파일: {self.config_path}")
        print(f"\n[임베딩]")
        print(f"  모델: {self.embedding_model}")
        print(f"  디바이스: {self.embedding_device}")
        print(f"  배치 크기: {self.embedding_batch_size}")
        print(f"\n[검색]")
        print(f"  Hybrid Dense top_k: {self.hybrid_dense_top_k}")
        print(f"  Hybrid Sparse top_k: {self.hybrid_sparse_top_k}")
        print(f"  Rerank top_k: {self.rerank_top_k}")
        print(f"\n[Reranking]")
        print(f"  Stage 1: {self.reranker_stage1_model} ({self.reranker_stage1_device})")
        print(f"  Stage 2: {self.reranker_stage2_model} ({self.reranker_stage2_device})")
        print(f"\n[LLM]")
        print(f"  모델: {self.llm_model}")
        print(f"  Temperature: {self.llm_temperature}")
        print(f"  Max tokens: {self.llm_max_tokens}")
        print(f"\n[Context Quality Filter]")
        print(f"  활성화: {self.context_quality_enabled}")
        if self.context_quality_enabled:
            print(f"  Threshold: {self.context_quality_threshold}")
            print(f"  평가 모델: {self.context_quality_model}")
        print(f"\n[LangGraph]")
        print(f"  LangSmith 추적: {self.use_langsmith}")
        print(f"  최대 재시도: {self.max_retries}")
        print(f"  웹 검색: {'활성화' if self.web_search_enabled else '비활성화'}")
        print("=" * 80 + "\n")


# 전역 설정 인스턴스 (싱글톤 패턴)
_config_instance = None


def get_config(config_path: str = None) -> RAGConfig:
    """
    전역 설정 인스턴스 반환 (싱글톤)

    Args:
        config_path (str, optional): 설정 파일 경로

    Returns:
        RAGConfig: 설정 인스턴스
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = RAGConfig(config_path)
    return _config_instance
