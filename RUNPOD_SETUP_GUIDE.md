# RunPod 설정 가이드 (최종 버전)

**LangGraph RAG 시스템 포함 - 2025-12-01**

이 가이드는 RunPod에서 Git pull 후 RAG 시스템을 설정하는 전체 과정을 다룹니다.

---

## 전제 조건

- RunPod GPU Pod 실행 중 (RTX 4090, A100, RTX 5090 등)
- Git 설치 완료
- Python 3.10+ 설치 완료

---

## 1. Git Pull 및 프로젝트 확인

### 1-1. 저장소 클론 또는 Pull

```bash
# 처음인 경우 (클론)
cd /workspace
git clone https://github.com/inucreativehrd21/testrag.git
cd testrag

# 이미 있는 경우 (풀)
cd /workspace/testrag
git pull
```

### 1-2. 프로젝트 구조 확인

```bash
ls -la

# 확인해야 할 주요 폴더/파일:
# - README.md (메인 문서)
# - experiments/rag_pipeline/
# - experiments/rag_pipeline/langgraph_rag/
# - crawler/
# - docs/
```

---

## 2. 환경변수 설정

### 2-1. 필수 환경변수

```bash
# OPENAI API 키 설정 (필수)
export OPENAI_API_KEY="your_openai_api_key_here"

# 확인
echo $OPENAI_API_KEY
```

### 2-2. 선택 환경변수 (LangGraph RAG 사용 시)

```bash
# Tavily API 키 (웹 검색 기능)
export TAVILY_API_KEY="your_tavily_api_key_here"

# LangSmith 추적 (디버깅/모니터링)
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="your_langsmith_api_key_here"
export LANGSMITH_PROJECT="my-rag-project"

# 확인
echo $TAVILY_API_KEY
echo $LANGSMITH_TRACING
```

### 2-3. 환경변수 영구 설정 (권장)

```bash
# ~/.bashrc에 추가
cat >> ~/.bashrc << 'EOF'

# RAG System Environment Variables
export OPENAI_API_KEY="your_openai_api_key_here"
export TAVILY_API_KEY="your_tavily_api_key_here"
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="your_langsmith_api_key_here"
export LANGSMITH_PROJECT="runpod-rag"
EOF

# 즉시 적용
source ~/.bashrc
```

---

## 3. 의존성 설치

### 3-1. 통합 의존성 설치 (Optimized RAG + LangGraph RAG)

**중요:** 프로젝트 루트의 `requirements.txt`에 모든 의존성이 통합되었습니다.

```bash
cd /workspace/testrag

# 한 번에 모든 의존성 설치 (Optimized RAG + LangGraph RAG)
pip install -r requirements.txt

# 예상 시간: 10-15분

# 포함된 주요 패키지:
# - Optimized RAG: FlagEmbedding, transformers, sentence-transformers
# - LangGraph RAG: langgraph (0.2.45), langsmith (0.1.147), tavily-python (0.5.0)
# - 공통: langchain (0.3.7), chromadb (0.5.5), openai (1.109.1)
```

### 3-3. GPU 확인

```bash
# GPU 사용 가능 여부 확인
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('GPU name:', torch.cuda.get_device_name(0))"

# 예상 출력:
# CUDA available: True
# GPU name: NVIDIA RTX 4090
```

---

## 4. 데이터 준비 (선택 - 이미 있다면 스킵)

### 4-1. 데이터가 이미 있는지 확인

```bash
# 크롤링된 데이터 확인
ls -la data/raw/git/
ls -la data/raw/python/

# 파일이 있다면 -> 다음 단계로
# 파일이 없다면 -> 크롤링 실행
```

### 4-2. 크롤링 실행 (필요한 경우)

```bash
cd /workspace/testrag/crawler
python run_crawl_extended.py

# 예상 시간: 10-30분 (문서 양에 따라)
# 출력: data/raw/{git,python,docker,aws}/
```

---

## 5. 벡터 인덱스 구축

### 5-1. 인덱스가 이미 있는지 확인

```bash
# ChromaDB 확인
ls -la experiments/rag_pipeline/artifacts/chroma_db/

# 디렉토리가 있고 파일이 많으면 -> 스킵 가능
# 비어있거나 없으면 -> 인덱스 구축 필요
```

### 5-2. 데이터 준비 및 인덱스 구축

```bash
cd /workspace/testrag/experiments/rag_pipeline

# Step 1: 데이터 준비
python data_prep.py --config config/enhanced.yaml

# 출력: artifacts/rag_chunks.parquet
# 예상 시간: 2-5분

# Step 2: 벡터 인덱스 빌드
python index_builder.py --config config/enhanced.yaml

# 출력: artifacts/chroma_db/
# 예상 시간: 10-30분 (GPU 사용, 문서 양에 따라)
```

### 5-3. GPU 설정 확인 (enhanced.yaml)

```bash
cat config/enhanced.yaml | grep -A5 "embedding:"

# 확인:
# embedding:
#   device: cuda  # <- GPU 사용
#   batch_size: 32  # <- GPU 메모리에 따라 조정
```

**메모리 부족 시:**
```yaml
embedding:
  device: cuda
  batch_size: 16  # 32에서 감소
```

---

## 6. RAG 시스템 실행

이제 2가지 RAG 시스템 중 선택해서 실행할 수 있습니다.

### 옵션 A: Optimized RAG (빠른 응답 - 5초)

```bash
cd /workspace/testrag/experiments/rag_pipeline

# 실행
python answerer_v2_optimized.py --config config/enhanced.yaml

# 예상 출력:
# [INFO] Loading models...
# [INFO] Models loaded successfully
# [질문 입력 대기]
```

**특징:**
- 응답 속도: ~5초
- Context Precision: 0.85
- Hybrid Search + 2-Stage Reranking

---

### 옵션 B: LangGraph RAG (고품질 - 7-10초)

```bash
cd /workspace/testrag/experiments/rag_pipeline/langgraph_rag

# 단일 질문
python -m langgraph_rag.main "git rebase란 무엇인가요?"

# 대화형 모드
python -m langgraph_rag.main

# LangSmith 추적 활성화 (디버깅)
export LANGSMITH_TRACING=true
python -m langgraph_rag.main "질문"
```

**특징:**
- 응답 속도: 7-10초
- Context Precision: 0.92 (+8%)
- Adaptive/Corrective/Self-RAG
- LangSmith 추적 지원
- 웹 검색 Fallback

**LangSmith 대시보드:**
https://smith.langchain.com/

---

## 7. 서비스 실행 (FastAPI)

### 7-1. FastAPI 서버 실행

```bash
cd /workspace/testrag/experiments/rag_pipeline

# 서버 실행 (Optimized RAG 사용)
python serve.py

# 예상 출력:
# INFO:     Uvicorn running on http://0.0.0.0:8000
# INFO:     Application startup complete.
```

### 7-2. 포트 포워딩 (RunPod)

RunPod 대시보드에서:
1. Pod 설정 → Ports
2. 8000 포트 추가
3. Public URL 확인 (예: `https://xxxxx-8000.proxy.runpod.net`)

### 7-3. API 테스트

```bash
# 다른 터미널 또는 로컬에서
curl -X POST "https://xxxxx-8000.proxy.runpod.net/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "git rebase란 무엇인가요?"}'
```

---

## 8. 문제 해결

### 문제 1: GPU 메모리 부족

**에러:**
```
RuntimeError: CUDA out of memory
```

**해결책:**
```bash
# config/enhanced.yaml 수정
embedding:
  batch_size: 16  # 기본 32에서 감소

rerankers:
  stage2:
    enabled: false  # 2단계 리랭커 비활성화
```

---

### 문제 2: ChromaDB 찾을 수 없음

**에러:**
```
ValueError: Collection 'rag_chunks' not found
```

**해결책:**
```bash
cd /workspace/testrag/experiments/rag_pipeline
python data_prep.py --config config/enhanced.yaml
python index_builder.py --config config/enhanced.yaml
```

---

### 문제 3: OPENAI_API_KEY 없음

**에러:**
```
openai.AuthenticationError: No API key provided
```

**해결책:**
```bash
export OPENAI_API_KEY="your_api_key_here"

# 영구 설정
echo 'export OPENAI_API_KEY="your_api_key_here"' >> ~/.bashrc
source ~/.bashrc
```

---

### 문제 4: LangSmith 추적 안됨

**확인:**
```bash
echo $LANGSMITH_TRACING  # true여야 함
echo $LANGSMITH_API_KEY  # API 키 확인
```

**해결책:**
```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="your_api_key_here"
export LANGSMITH_PROJECT="runpod-rag"
```

---

## 9. 성능 최적화 (RunPod)

### 9-1. GPU 모델별 권장 설정

#### RTX 4090 / RTX 5090
```yaml
# config/enhanced.yaml
embedding:
  device: cuda
  batch_size: 32

rerankers:
  stage1:
    enabled: true
  stage2:
    enabled: true  # 메모리 충분
```

#### A100 40GB
```yaml
embedding:
  device: cuda
  batch_size: 64  # 더 큰 배치 가능

rerankers:
  stage1:
    enabled: true
  stage2:
    enabled: true
```

#### RTX 3090 / 4080 (24GB)
```yaml
embedding:
  device: cuda
  batch_size: 24

rerankers:
  stage1:
    enabled: true
  stage2:
    enabled: false  # 메모리 절약
```

---

### 9-2. 추가 최적화

```bash
# GPU 진단
cd /workspace/testrag/experiments/rag_pipeline
python diagnose_gpu.py

# Smoke 테스트
python smoke_test.py

# RAGAS 평가
python run_ragas_evaluation.py
```

---

## 10. 빠른 체크리스트

RunPod 설정 완료 여부 확인:

- [ ] Git pull 완료
- [ ] 환경변수 설정 (`OPENAI_API_KEY`)
- [ ] 의존성 설치 (`pip install -r requirements.txt`)
- [ ] LangGraph 의존성 설치 (선택)
- [ ] 데이터 확인 (`data/raw/`)
- [ ] 벡터 인덱스 확인 (`artifacts/chroma_db/`)
- [ ] GPU 확인 (`torch.cuda.is_available()`)
- [ ] RAG 실행 테스트
- [ ] FastAPI 서버 실행 (선택)
- [ ] 포트 포워딩 설정 (선택)

---

## 11. 주요 파일 위치

### 설정 파일
- `config/enhanced.yaml` - RAG 메인 설정
- `experiments/rag_pipeline/config/enhanced.yaml` - RAG 상세 설정

### 실행 파일
- `experiments/rag_pipeline/answerer_v2_optimized.py` - Optimized RAG
- `experiments/rag_pipeline/langgraph_rag/main.py` - LangGraph RAG
- `experiments/rag_pipeline/serve.py` - FastAPI 서버

### 데이터
- `data/raw/` - 크롤링된 원본 데이터
- `experiments/rag_pipeline/artifacts/chroma_db/` - 벡터 DB
- `experiments/rag_pipeline/artifacts/ragas_evals/` - 평가 결과

---

## 12. 참고 문서

- [README.md](README.md) - 메인 프로젝트 문서
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - 구조 빠른 참조
- [FINAL_PROJECT_OVERVIEW.md](FINAL_PROJECT_OVERVIEW.md) - 프로젝트 개요
- [docs/TROUBLESHOOTING_RTX5090.md](docs/TROUBLESHOOTING_RTX5090.md) - GPU 문제 해결
- [langgraph_rag/README.md](experiments/rag_pipeline/langgraph_rag/README.md) - LangGraph 상세 가이드

---

## 요약: 최소 설정 (빠른 시작)

```bash
# 1. Pull
cd /workspace/testrag
git pull

# 2. 환경변수
export OPENAI_API_KEY="your_key_here"

# 3. 의존성 (이미 했다면 스킵)
pip install -r requirements.txt

# 4. 데이터/인덱스 확인 (이미 있다면 스킵)
ls -la experiments/rag_pipeline/artifacts/chroma_db/

# 5. 실행
cd experiments/rag_pipeline
python answerer_v2_optimized.py --config config/enhanced.yaml

# 또는 LangGraph RAG
cd langgraph_rag
python -m langgraph_rag.main
```

---

**작성:** Claude Code
**날짜:** 2025-12-01
**버전:** Final (LangGraph RAG 포함)
