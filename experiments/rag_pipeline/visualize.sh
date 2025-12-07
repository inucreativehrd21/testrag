#!/bin/bash
# LangGraph RAG Workflow 시각화 스크립트

set -e

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== LangGraph RAG Workflow 시각화 ===${NC}\n"

# graphviz 설치 확인
if ! command -v dot &> /dev/null; then
    echo -e "${YELLOW}⚠ graphviz가 설치되지 않았습니다.${NC}"
    echo "설치 명령어: sudo apt-get install graphviz"
    echo ""
fi

# Python 스크립트 실행
python3 visualize_workflow.py "$@"

echo -e "\n${GREEN}✓ 완료${NC}"
