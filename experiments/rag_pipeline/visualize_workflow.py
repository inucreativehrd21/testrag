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


def main():
    parser = argparse.ArgumentParser(description="LangGraph RAG Workflow 시각화")
    parser.add_argument(
        "--output", 
        "-o", 
        default="rag_workflow.png",
        help="출력 PNG 파일 경로 (기본값: rag_workflow.png)"
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
        
        # Mermaid 다이어그램 출력
        print("\n=== LangGraph Mermaid Diagram ===\n")
        print(app.get_graph().draw_mermaid())
        print("\n" + "=" * 80 + "\n")
        
        # PNG 저장
        try:
            app.get_graph().draw_mermaid_png(output_file_path=args.output)
            logger.info(f"✓ 그래프 PNG 저장 완료: {args.output}")
        except Exception as e:
            logger.warning(f"⚠ PNG 저장 실패 (graphviz 미설치 가능): {e}")
            logger.info("graphviz 설치: sudo apt-get install graphviz")
            
    except Exception as e:
        logger.error(f"❌ 시각화 실패: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
