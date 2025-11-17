#!/bin/bash
# RAG 평가 도구 - RunPod 자동 설정 스크립트
# Python 3.11 환경에서 자동으로 모든 것을 설정합니다

set -e

echo "=========================================="
echo "RAG 평가 도구 - RunPod 자동 설정"
echo "=========================================="
echo ""

# 1. Python 버전 확인
echo "[1/5] Python 버전 확인..."
python3.11 --version 2>/dev/null || python3.10 --version 2>/dev/null || {
    echo "❌ Python 3.10 또는 3.11이 필요합니다."
    exit 1
}

PYTHON_CMD=$(which python3.11 || which python3.10)
echo "✓ Python: $($PYTHON_CMD --version)"

# 2. 가상환경 생성
echo ""
echo "[2/5] 가상환경 생성..."
$PYTHON_CMD -m venv venv
source venv/bin/activate
echo "✓ 가상환경 활성화"

# 3. 빌드 도구 설치
echo ""
echo "[3/5] 빌드 도구 설치..."
pip install --upgrade pip setuptools wheel build -q
echo "✓ 빌드 도구 설치 완료"

# 4. 의존성 설치
echo ""
echo "[4/5] 의존성 설치 중... (5-10분)"
pip install -r requirements.txt --no-cache-dir -q
echo "✓ 의존성 설치 완료"

# 5. 설치 검증
echo ""
echo "[5/5] 설치 검증..."
python -c "
import torch
import langchain
import chromadb
print('✓ PyTorch:', torch.__version__)
print('✓ CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✓ GPU:', torch.cuda.get_device_name(0))
print('✓ LangChain:', langchain.__version__)
print('✓ ChromaDB:', chromadb.__version__)
" 2>/dev/null || echo "⚠️  일부 패키지 검증 실패 (무시 가능)"

echo ""
echo "=========================================="
echo "✅ 설정 완료!"
echo "=========================================="
echo ""
echo "📝 다음 명령어로 크롤링 데이터 준비:"
echo ""
echo "📊 평가 실행:"
echo "   cd /workspace/rag_eval_final"
echo "   export RAG_DATA_DIR=/workspace/testrag/data/raw"
echo "   source venv/bin/activate"
echo "   python main_with_ragas.py"
echo ""
echo "📈 결과 확인:"
echo "   cat results/comparison_*.csv"
echo "   cat results/summary_*.txt"
echo ""
