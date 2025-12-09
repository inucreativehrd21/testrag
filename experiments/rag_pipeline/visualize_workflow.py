#!/usr/bin/env python3
"""
LangGraph RAG Workflow 시각화 스크립트

사용법:
    python visualize_workflow.py
    python visualize_workflow.py --output my_graph.png
    python visualize_workflow.py --no-personalization
"""

import argparse
import logging
from langgraph_rag.graph import create_rag_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_styled_mermaid(app, enable_personalization: bool = True) -> str:
    """
    스타일이 적용된 Mermaid 다이어그램 생성

    노드별 색상 테마:
    - 입구/출구: #E8F5E9 (연한 초록)
    - 의도 분류/라우팅: #E3F2FD (연한 파랑)
    - 검색/Reranking: #FFF3E0 (연한 주황)
    - 평가: #F3E5F5 (연한 보라)
    - 생성: #FFEBEE (연한 빨강)
    - 개인화: #E1F5FE (하늘색)
    - 웹 검색: #FFF9C4 (연한 노랑)
    """
    base_mermaid = app.get_graph().draw_mermaid()

    # 커스텀 스타일 추가
    style_config = """
%%{init: {'theme':'base', 'themeVariables': {'primaryColor':'#ffffff','primaryTextColor':'#000000','primaryBorderColor':'#2196F3','lineColor':'#2196F3','secondaryColor':'#f5f5f5','tertiaryColor':'#e8f5e9','fontSize':'14px'}}}%%
"""

    # 노드별 스타일 정의
    node_styles = """
    classDef entryExit fill:#E8F5E9,stroke:#4CAF50,stroke-width:3px,color:#000
    classDef intent fill:#E3F2FD,stroke:#2196F3,stroke-width:2px,color:#000
    classDef retrieve fill:#FFF3E0,stroke:#FF9800,stroke-width:2px,color:#000
    classDef grade fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px,color:#000
    classDef generate fill:#FFEBEE,stroke:#F44336,stroke-width:2px,color:#000
    classDef personalize fill:#E1F5FE,stroke:#03A9F4,stroke-width:2px,color:#000
    classDef websearch fill:#FFF9C4,stroke:#FBC02D,stroke-width:2px,color:#000

    class __start__,__end__ entryExit
    class intent_classifier,query_router intent
    class hybrid_retrieve,rerank_stage1,rerank_stage2 retrieve
    class grade_documents,hallucination_check,answer_grading grade
    class generate,transform_query generate
    class web_search websearch
"""

    if enable_personalization:
        node_styles += "    class load_user_context,suggest_related_questions personalize\n"

    # Mermaid 다이어그램에 스타일 추가
    lines = base_mermaid.split('\n')

    # flowchart 정의 찾아서 방향을 TD로 변경 (위→아래)
    styled_lines = [style_config]
    for line in lines:
        if line.strip().startswith('graph'):
            styled_lines.append('flowchart TD')
        else:
            styled_lines.append(line)

    # 스타일 정의를 마지막에 추가
    styled_lines.append(node_styles)

    return '\n'.join(styled_lines)


def main():
    parser = argparse.ArgumentParser(description="LangGraph RAG Workflow 시각화")
    parser.add_argument(
        "--output",
        "-o",
        default="rag_workflow4.png",
        help="출력 PNG 파일 경로 (기본값: rag_workflow4.png)"
    )
    parser.add_argument(
        "--no-personalization",
        action="store_true",
        help="개인화 노드 제외"
    )
    args = parser.parse_args()

    enable_personalization = not args.no_personalization

    logger.info(f"워크플로우 시각화 시작 (개인화: {enable_personalization})")

    try:
        app = create_rag_graph(enable_personalization=enable_personalization)

        # 스타일이 적용된 Mermaid 다이어그램 생성
        styled_mermaid = create_styled_mermaid(app, enable_personalization)

        # Mermaid 다이어그램 출력
        print("\n=== LangGraph Styled Mermaid Diagram ===\n")
        print(styled_mermaid)
        print("\n" + "=" * 80 + "\n")

        # PNG 저장
        try:
            # draw_mermaid_png는 내부적으로 draw_mermaid()를 호출하므로
            # 스타일을 적용하려면 파일로 저장 후 외부 도구 사용 필요
            app.get_graph().draw_mermaid_png(output_file_path=args.output)
            logger.info(f"✓ 기본 그래프 PNG 저장 완료: {args.output}")

            # 스타일된 버전을 .mmd 파일로 저장
            mmd_path = args.output.replace('.png', '.mmd')
            with open(mmd_path, 'w', encoding='utf-8') as f:
                f.write(styled_mermaid)
            logger.info(f"✓ 스타일된 Mermaid 파일 저장: {mmd_path}")
            logger.info("  → https://mermaid.live 에서 열어보세요!")

        except Exception as e:
            logger.warning(f"⚠ PNG 저장 실패 (graphviz 미설치 가능): {e}")
            logger.info("graphviz 설치: sudo apt-get install graphviz")

            # 스타일된 버전을 .mmd 파일로 저장
            mmd_path = args.output.replace('.png', '.mmd')
            with open(mmd_path, 'w', encoding='utf-8') as f:
                f.write(styled_mermaid)
            logger.info(f"✓ 스타일된 Mermaid 파일 저장: {mmd_path}")
            logger.info("  → https://mermaid.live 에서 열어보세요!")

    except Exception as e:
        logger.error(f"❌ 시각화 실패: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
