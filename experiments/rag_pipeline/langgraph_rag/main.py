#!/usr/bin/env python3
"""
LangGraph RAG 메인 실행 파일

사용법:
    python main.py "git rebase란 무엇인가요?"
    python main.py "Python async/await 사용법" --config config/enhanced.yaml
    python main.py "질문" --log-level DEBUG
    python main.py "질문" --visualize  # 그래프 구조 출력
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Allow running as a script (python main.py) by adjusting import context
if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    __package__ = "langgraph_rag"

# LangSmith 추적 설정 (선택사항)
def setup_langsmith():
    """
    LangSmith 추적 설정

    환경변수:
    - LANGSMITH_TRACING: "true"로 설정 시 추적 활성화
    - LANGSMITH_API_KEY: LangSmith API 키
    - LANGSMITH_PROJECT: 프로젝트 이름 (기본: "langgraph-rag")

    사용법:
        export LANGSMITH_TRACING=true
        export LANGSMITH_API_KEY=your_api_key
        export LANGSMITH_PROJECT=my-rag-project
        python main.py "질문"
    """
    if os.getenv("LANGSMITH_TRACING", "false").lower() == "true":
        api_key = os.getenv("LANGSMITH_API_KEY", "")
        project = os.getenv("LANGSMITH_PROJECT", "langgraph-rag")

        if not api_key:
            print("[WARNING] LANGSMITH_API_KEY가 설정되지 않았습니다.")
            print("LangSmith 추적을 비활성화합니다.")
            os.environ["LANGSMITH_TRACING"] = "false"
        else:
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGSMITH_PROJECT"] = project
            print(f"[INFO] LangSmith 추적 활성화: 프로젝트 '{project}'")
            print(f"[INFO] 추적 결과: https://smith.langchain.com/")
    else:
        print("[INFO] LangSmith 추적 비활성화")


def setup_logging(level: str = "INFO"):
    """로깅 설정"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="LangGraph Adaptive RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python main.py "git rebase란 무엇인가요?"
  python main.py "Python async/await 사용법은?" --log-level DEBUG
  python main.py "질문" --config ../config/enhanced.yaml
  python main.py "질문" --visualize  # 그래프 구조만 출력

LangSmith 추적 (선택사항):
  export LANGSMITH_TRACING=true
  export LANGSMITH_API_KEY=your_api_key
  export LANGSMITH_PROJECT=my-rag-project
  python main.py "질문"
        """,
    )

    parser.add_argument(
        "question",
        type=str,
        nargs="?",
        help="질문 (생략 시 대화형 모드)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="설정 파일 경로 (기본: ../config/enhanced.yaml)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="로그 레벨 (기본: INFO)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="그래프 구조만 출력하고 종료",
    )
    parser.add_argument(
        "--show-workflow",
        action="store_true",
        help="워크플로우 히스토리 출력",
    )

    args = parser.parse_args()

    # 로깅 설정
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # LangSmith 설정
    setup_langsmith()

    # 그래프 시각화만 하고 종료
    if args.visualize:
        from .graph import visualize_graph

        logger.info("그래프 구조 출력 중...")
        visualize_graph()
        return

    # 질문이 없으면 대화형 모드
    if not args.question:
        print("\n" + "=" * 80)
        print("LangGraph Adaptive RAG System - 대화형 모드")
        print("=" * 80)
        print("종료: 'exit', 'quit', 'q' 입력")
        print("=" * 80 + "\n")

        from .graph import run_rag_graph

        while True:
            try:
                question = input("\n질문: ").strip()

                if question.lower() in ["exit", "quit", "q"]:
                    print("종료합니다.")
                    break

                if not question:
                    print("질문을 입력하세요.")
                    continue

                # RAG 실행
                result = run_rag_graph(question, args.config)

                # 답변 출력
                print("\n" + "=" * 80)
                print("답변:")
                print("=" * 80)
                print(result["generation"])
                print("=" * 80)

                # 워크플로우 히스토리 출력 (옵션)
                if args.show_workflow:
                    print(f"\n[Workflow] {' → '.join(result['workflow_history'])}")

            except KeyboardInterrupt:
                print("\n\n종료합니다.")
                break
            except Exception as e:
                logger.error(f"오류 발생: {e}", exc_info=True)
                print(f"\n[ERROR] {e}")

    else:
        # 단일 질문 모드
        from .graph import run_rag_graph

        try:
            # RAG 실행
            result = run_rag_graph(args.question, args.config)

            # 답변 출력
            print("\n" + "=" * 80)
            print("답변:")
            print("=" * 80)
            print(result["generation"])
            print("=" * 80)

            # 워크플로우 히스토리 출력 (옵션)
            if args.show_workflow:
                print(f"\n[Workflow] {' → '.join(result['workflow_history'])}")
                print(f"[Retry Count] {result['retry_count']}")
                print(f"[Document Relevance] {result['document_relevance']}")
                print(f"[Hallucination Grade] {result['hallucination_grade']}")
                print(f"[Answer Usefulness] {result['answer_usefulness']}")

        except Exception as e:
            logger.error(f"오류 발생: {e}", exc_info=True)
            print(f"\n[ERROR] {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
