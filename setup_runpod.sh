#!/bin/bash
# 런팟 환경 설정 스크립트
# GPU 할당 확인 후 실행하세요

set -e  # 에러 발생 시 중단

echo "========================================="
echo "런팟 RAG 환경 설정 시작"
echo "========================================="
echo ""

# GPU 확인
echo "1. GPU 확인..."
nvidia-smi
echo ""

# Python 버전 확인
echo "2. Python 버전 확인..."
python --version
echo ""

# pip 업그레이드
echo "3. pip 업그레이드..."
pip install --upgrade pip
echo ""

# PyTorch 설치 (CUDA 12.4)
echo "4. PyTorch 설치 (CUDA 12.4)..."
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124
echo ""

# PyTorch 설치 확인
echo "5. PyTorch GPU 확인..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"
echo ""

# 나머지 패키지 설치
echo "6. 나머지 패키지 설치..."
pip install -r requirements.txt --no-deps  # PyTorch는 이미 설치했으므로
pip install -r requirements.txt  # 의존성 설치
echo ""

# 설치 확인
echo "7. 주요 패키지 버전 확인..."
python -c "
import torch
import transformers
from FlagEmbedding import BGEM3FlagModel
import chromadb
import openai
import ragas

print('✓ PyTorch:', torch.__version__)
print('✓ Transformers:', transformers.__version__)
print('✓ FlagEmbedding: installed')
print('✓ ChromaDB:', chromadb.__version__)
print('✓ OpenAI:', openai.__version__)
print('✓ RAGAS:', ragas.__version__)
print('')
print('✓ CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✓ GPU Name:', torch.cuda.get_device_name(0))
    print('✓ GPU Memory:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), 'GB')
"
echo ""

echo "========================================="
echo "✓ 설치 완료!"
echo "========================================="
echo ""
echo "다음 단계:"
echo "1. export OPENAI_API_KEY='your-api-key'"
echo "2. cd experiments/rag_pipeline"
echo "3. python answerer_v2_optimized.py 'test query' --config config/enhanced.yaml"
echo ""
