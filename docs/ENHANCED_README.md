# Enhanced RAG Pipeline - Hybrid Search + Context Quality Filter

## 🚀 새로운 기능

### 1. **Hybrid Search (Dense + Sparse + RRF)**
- **Dense Search**: BGE-M3 semantic embeddings (의미 기반 검색)
- **Sparse Search**: BGE-M3 lexical weights (키워드 기반 검색, BM25-like)
- **RRF (Reciprocal Rank Fusion)**: 두 검색 결과를 최적 결합

**기대 효과:**
- Context Precision: +10-15%
- Context Recall: +10-15%

### 2. **Context Quality Filter (Self-RAG 스타일)**
- 검색된 각 문서를 LLM으로 평가
- RELEVANT / PARTIAL / IRRELEVANT 분류
- 품질 낮은 문서 자동 필터링

**기대 효과:**
- Faithfulness: +15-20% (환각 감소)
- Answer Correctness: +8-12%

---

## 📋 시스템 비교

### Baseline (answerer.py)
```
[Query]
  ↓
[BGE-M3 Dense] → Top 25
  ↓
[Stage 1 Reranker] → Top 25
  ↓
[Stage 2 Reranker] → Top 5
  ↓
[GPT-4 Generation]
```

### Enhanced (answerer_v2.py)
```
[Query]
  ↓
[BGE-M3 Dense + Sparse Encoding]
  ↓
[Dense Search] → Top 50
  ↓
[Sparse Search on Top 50]
  ↓
[RRF Fusion] → Hybrid results
  ↓
[Stage 1 Reranker] → Top 25
  ↓
[Stage 2 Reranker] → Top 5
  ↓
[Context Quality Filter (gpt-4o-mini)] → 3-5개
  ↓
[GPT-4 Generation]
```

---

## 🔧 설치 및 설정

### 1. 환경 준비 (이미 완료됨)
```bash
# BGE-M3 모델 이미 설치됨
# ChromaDB 이미 설정됨
# OpenAI API Key 이미 설정됨
```

### 2. Enhanced Config 사용
```bash
# 새 설정 파일이 자동 생성됨
config/enhanced.yaml
```

**주요 변경사항:**
- `hybrid_dense_top_k: 50` (더 많은 후보 검색)
- `hybrid_sparse_top_k: 50` (sparse 재검색)
- `rrf_k: 60` (RRF 상수)
- `context_quality.enabled: true` (품질 필터 활성화)

---

## 🧪 테스트 방법

### Quick Test
```bash
cd experiments/rag_pipeline

# 단일 질문 테스트
python answerer_v2.py "Git에서 마지막 커밋을 수정하려면?" --config config/enhanced.yaml

# 여러 질문 테스트
python test_enhanced.py
```

### 성능 비교 테스트

#### 1. Baseline 평가
```bash
# 기존 시스템으로 평가 (answerer.py)
python ragas_benchmark.py --config config/base.yaml --output baseline_results
```

#### 2. Enhanced 평가
```bash
# 향상된 시스템으로 평가 (answerer_v2.py)
python ragas_benchmark.py --config config/enhanced.yaml --answerer answerer_v2 --output enhanced_results
```

**주의:** `ragas_benchmark.py`는 `answerer_v2.py`를 import할 수 있도록 수정 필요

---

## 📊 예상 성능 향상

### Before (Baseline - 선행 결과 기준)
```
Git/Python 15문제 평가:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Context Precision:   67.5%
Context Recall:      63.3%
Faithfulness:        85.9%
Answer Relevancy:    77.4%
Answer Correctness:  59.6%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Overall Score:       70.7%
```

### After (Enhanced - 예상)
```
Git/Python 15문제 평가:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Context Precision:   77-80%  (+10-13%)
Context Recall:      73-78%  (+10-15%)
Faithfulness:        93-96%  (+7-10%)
Answer Relevancy:    82-85%  (+5-8%)
Answer Correctness:  68-72%  (+8-12%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Overall Score:       79-82% (+8-11%)
```

---

## 🔍 세부 기능 설명

### 1. Hybrid Search

**문제:** Dense-only search는 키워드 매칭이 약함
```
질문: "git commit --amend 사용법"
Dense만: 의미적으로 유사한 문서 검색 (때로 정확한 명령어 놓침)
Hybrid: Dense(의미) + Sparse(키워드) 결합
```

**구현 핵심:**
```python
# answerer_v2.py Line 109-115
query_encoding = self.embedding_model.encode(
    [question],
    return_dense=True,   # 의미 벡터
    return_sparse=True,  # 키워드 가중치
)

# RRF로 결합
rrf_scores[doc] = 1/(k + dense_rank) + 1/(k + sparse_rank)
```

### 2. Context Quality Filter

**문제:** Reranker도 가끔 무관한 문서를 통과시킴

**해결:** LLM으로 각 문서 평가
```python
# answerer_v2.py Line 175-210
for ctx in contexts:
    # gpt-4o-mini로 평가 (저렴)
    label = evaluate_relevance(question, ctx)

    if label == "RELEVANT":
        final_contexts.append(ctx)
    elif label == "IRRELEVANT":
        # 필터링
        pass
```

**비용:**
- 문서 5개 평가: ~$0.001 (매우 저렴)
- Faithfulness 대폭 향상

---

## ⚙️ 설정 옵션

### Hybrid Search 튜닝
```yaml
retrieval:
  hybrid_dense_top_k: 50   # 늘리면: Recall ↑, 속도 ↓
  hybrid_sparse_top_k: 50  # Dense 후보에서 sparse 재검색
  rrf_k: 60                # 60이 논문 표준 (40-80 범위)
```

### Context Quality 튜닝
```yaml
context_quality:
  enabled: true            # false로 하면 비활성화
  threshold: 0.6           # 미래 확장용
  evaluator_model: gpt-4o-mini  # 또는 gpt-3.5-turbo
```

---

## 🐛 문제 해결

### 1. "Sparse search failed"
```bash
# BGE-M3 모델이 sparse를 지원하는지 확인
python -c "from FlagEmbedding import BGEM3FlagModel; print('OK')"
```

### 2. "Context quality evaluation timeout"
```bash
# gpt-4o-mini API 키 확인
echo $OPENAI_API_KEY

# 또는 비활성화
# config/enhanced.yaml에서 context_quality.enabled: false
```

### 3. "No relevant contexts found"
```
로그에서 확인:
- "Context quality filter: 0/5 kept" → 모든 문서가 IRRELEVANT
- 해결: 질문을 더 구체적으로 변경하거나 threshold 낮춤
```

---

## 📈 성능 모니터링

### 로그 분석
```bash
# Enhanced pipeline 실행 시 로그 확인
python answerer_v2.py "질문" --log-level DEBUG

# 주요 지표:
# - Dense retrieval: XXXms
# - Sparse search: XXXms
# - RRF fusion: XXXms
# - Context quality filter: X/Y kept
```

### 병목 지점
```
일반적인 시간 분포:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Query encoding:     50-100ms
Dense retrieval:    100-200ms
Sparse search:      200-400ms  ← 가장 오래 걸림
RRF fusion:         10-20ms
Stage 1 rerank:     300-500ms
Stage 2 rerank:     200-300ms
Quality filter:     500-800ms  ← LLM 호출
LLM generation:     1000-2000ms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total:             ~3-4초 (Baseline 대비 +1초)
```

**최적화 방법:**
- Sparse search를 비동기로 처리
- Quality filter를 batch로 처리
- 캐싱 추가

---

## 🎯 다음 단계 (Phase 2)

현재 구현: **Phase 1 완료**
- ✅ Hybrid Search (Dense + Sparse + RRF)
- ✅ Context Quality Filter (Self-RAG)

추가 가능 기능:
1. **Query Rewriting (HyDE)** - 복잡한 질문 개선
2. **Metadata Filtering** - 도메인별 검색 (git/python)
3. **Semantic Caching** - 동일 질문 캐싱
4. **Fallback Strategy (CRAG)** - 문서 부족 시 웹 검색

---

## 📚 참고 자료

### 논문
1. **Self-RAG** (ICLR 2024)
   - "Self-Reflective Retrieval-Augmented Generation"
   - Context quality evaluation 아이디어 출처

2. **RRF** (SIGIR 2009)
   - "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"
   - Hybrid search fusion 표준 방법

3. **BGE-M3** (2024)
   - "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation"
   - Dense + Sparse + ColBERT 지원

### GitHub
- [FlagEmbedding (BGE)](https://github.com/FlagOpen/FlagEmbedding)
- [LlamaIndex Hybrid Search](https://docs.llamaindex.ai/en/stable/examples/retrievers/bm25_retriever.html)

---

## 📞 지원

문제 발생 시:
1. 로그 확인: `--log-level DEBUG`
2. Config 검증: `config/enhanced.yaml`
3. Baseline과 비교: `answerer.py` vs `answerer_v2.py`

**Known Issues:**
- Windows CPU 모드에서 sparse search가 느릴 수 있음 (정상)
- gpt-4o-mini API 속도제한 시 context quality filter 타임아웃 가능
