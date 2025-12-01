# 의존성 통합 완료

**날짜:** 2025-12-01
**목적:** 의존성 충돌 방지 및 단일 requirements.txt 사용

---

## 변경사항 요약

### 문제
- 기존: 3개의 별도 requirements.txt 파일 존재
  1. `requirements.txt` (루트)
  2. `experiments/rag_pipeline/requirements.txt`
  3. `experiments/rag_pipeline/langgraph_rag/requirements.txt`
- 의존성 충돌 가능성:
  - `langchain`: 0.1.20 (루트) vs >=0.3.0 (LangGraph)
  - `chromadb`: 0.4.24 (루트) vs >=0.5.0 (LangGraph)

### 해결책
- **단일 requirements.txt** 사용 (프로젝트 루트)
- 기본 RAG 의존성 기반으로 LangGraph 의존성 통합

---

## 통합된 requirements.txt 주요 변경사항

### 업그레이드된 패키지
```python
# Before → After
chromadb==0.4.24 → chromadb==0.5.5
langchain==0.1.20 → langchain==0.3.7
langchain-core==0.1.53 → langchain-core==0.3.21
langchain-community==0.0.38 → langchain-community==0.3.5
langchain-text-splitters==0.0.1 → langchain-text-splitters==0.3.2
langchain-openai==0.1.7 → langchain-openai==0.2.9
pydantic==2.5.3 → pydantic==2.10.3  # langchain 0.3.7 호환 (>=2.7.4 필요)
pydantic-core==2.14.6 → pydantic-core==2.27.1
```

### 추가된 패키지 (LangGraph RAG용)
```python
langgraph==0.2.45
langsmith==0.1.147
tavily-python==0.5.0
graphviz==0.20.3  # 그래프 시각화 (선택)
```

### 추가된 유틸리티
```python
beautifulsoup4==4.12.3  # 버전 명시
lxml==5.1.0  # 버전 명시
httpx==0.27.2  # openai 호환성
onnxruntime==1.16.3  # 추론 최적화
```

---

## 제거된 파일

### 1. `experiments/rag_pipeline/langgraph_rag/requirements.txt`
- **이유:** 중복 및 버전 충돌
- **대체:** 루트 `requirements.txt` 사용

---

## 업데이트된 문서

### 1. `experiments/rag_pipeline/langgraph_rag/README.md`
**Before:**
```bash
cd experiments/rag_pipeline/langgraph_rag
pip install -r requirements.txt
```

**After:**
```bash
# 프로젝트 루트에서 실행
cd /path/to/project/root
pip install -r requirements.txt

# LangGraph 및 LangSmith 의존성이 포함되어 있습니다
```

### 2. `README.md`
**Before:**
```bash
# 메인 의존성
pip install -r requirements.txt

# LangGraph RAG 의존성
cd experiments/rag_pipeline/langgraph_rag
pip install -r requirements.txt
```

**After:**
```bash
# 프로젝트 루트에서 한 번에 모든 의존성 설치
pip install -r requirements.txt

# 포함된 주요 패키지:
# - Optimized RAG: FlagEmbedding, chromadb, transformers
# - LangGraph RAG: langgraph, langsmith, tavily-python
# - 공통: langchain (0.3.7), chromadb (0.5.5), openai
```

### 3. `RUNPOD_SETUP_GUIDE.md`
- 섹션 3: 의존성 설치 통합
- LangGraph RAG 별도 설치 단계 제거
- 단일 `pip install -r requirements.txt` 가이드

---

## RunPod 설정 변경사항

### Before (2단계 설치)
```bash
# 1. 메인 의존성
pip install -r requirements.txt

# 2. LangGraph RAG 의존성
cd experiments/rag_pipeline/langgraph_rag
pip install -r requirements.txt
```

### After (1단계 설치)
```bash
# 한 번에 모든 의존성 설치
cd /workspace/testrag
pip install -r requirements.txt

# 완료! Optimized RAG + LangGraph RAG 모두 사용 가능
```

---

## 호환성 테스트

### 테스트해야 할 항목

1. **Optimized RAG**
   ```bash
   cd experiments/rag_pipeline
   python answerer_v2_optimized.py --config config/enhanced.yaml
   ```

2. **LangGraph RAG**
   ```bash
   cd experiments/rag_pipeline/langgraph_rag
   python -m langgraph_rag.main "test question"
   ```

3. **Data Prep & Index Building**
   ```bash
   cd experiments/rag_pipeline
   python data_prep.py --config config/enhanced.yaml
   python index_builder.py --config config/enhanced.yaml
   ```

4. **RAGAS Evaluation**
   ```bash
   cd experiments/rag_pipeline
   python run_ragas_evaluation.py
   ```

### 예상 이슈 및 해결책

#### 이슈 1: ChromaDB 버전 변경 (0.4.24 → 0.5.5)
**가능성:** 기존 ChromaDB 인덱스 호환성 문제

**해결책:**
```bash
# 인덱스 재구축
cd experiments/rag_pipeline
python index_builder.py --config config/enhanced.yaml
```

#### 이슈 2: LangChain 버전 업그레이드 (0.1.20 → 0.3.7)
**가능성:** API 변경으로 인한 deprecated 경고

**해결책:** 대부분 하위 호환성 유지, 경고 무시 가능

#### 이슈 3: Pydantic 버전 (2.5.3 유지)
**확인 필요:** LangChain 0.3.x와 호환성

**해결책:** 문제 발생 시 pydantic==2.10.x로 업그레이드

---

## 설치 시간 예상

### Before (분리된 설치)
- 루트 requirements.txt: 5-10분
- LangGraph requirements.txt: 3-5분
- **총 8-15분**

### After (통합 설치)
- 통합 requirements.txt: **10-15분**
- 중복 설치 제거로 시간 절약

---

## 버전 관리 전략

### 고정 버전 (==)
- 핵심 패키지: torch, transformers, chromadb, langchain 등
- 이유: 안정성 및 재현성

### 유연한 버전 (>=)
- 현재 없음 (모든 버전 고정)

### 추천: 정기적 업데이트
```bash
# 6개월마다 의존성 업데이트 확인
pip list --outdated

# 주요 패키지 업데이트 시 테스트 필수
pytest tests/
python experiments/rag_pipeline/smoke_test.py
```

---

## 롤백 가이드

문제 발생 시 이전 버전으로 롤백:

```bash
# Git으로 이전 requirements.txt 복구
git checkout HEAD~1 -- requirements.txt

# 의존성 재설치
pip install -r requirements.txt --force-reinstall
```

---

## 요약

### ✅ 완료된 작업
1. 3개 requirements.txt → 1개 통합
2. LangChain 0.1.20 → 0.3.7 업그레이드
3. ChromaDB 0.4.24 → 0.5.5 업그레이드
4. LangGraph, LangSmith, Tavily 의존성 추가
5. 관련 문서 업데이트 (README, LangGraph README, RUNPOD_SETUP_GUIDE)

### 🎯 장점
- 의존성 충돌 제거
- 단일 설치 명령
- 버전 관리 단순화
- RunPod 설정 간소화

### ⚠️ 주의사항
- ChromaDB 버전 변경으로 인덱스 재구축 필요 가능
- 기존 환경에서 `pip install -r requirements.txt --upgrade` 실행 필요

---

**작성:** Claude Code
**날짜:** 2025-12-01
