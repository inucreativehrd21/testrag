# Git/Python RAG 챗봇 최적화 - Executive Summary

**프로젝트**: Git & Python 개발 학습 도우미 RAG 챗봇 최적화
**기간**: 2025-11-19 (1일 집중 최적화)
**목표**: RAGAS 성능 지표 개선 및 사용자 경험 향상

---

## 🎯 핵심 문제 및 해결

### 초기 상황
```
RAGAS 평가 결과 (16 questions):
❌ Faithfulness: 45.31% (심각한 환각 문제)
❌ Answer Correctness: 44.03% (가장 심각)
⚠️  Context Precision: 65.44%
⚠️  Context Recall: 67.77%
```

### 근본 원인
1. **도메인 과확장**: Git, Python, Docker, AWS 동시 지원 → Docker/AWS 문서 부족
2. **Hyperparameter 미검증**: Top-k, chunking 등이 실제 데이터 기반 아님
3. **답변 스타일 문제**: 형식적, 장황함 (15-20줄), 출처 표시 부실

---

## 🚀 구현한 최적화

### 1. 도메인 집중화
- **Before**: Git, Python, Docker, AWS (4개)
- **After**: Git, Python만 (2개)
- **근거**: Docker/AWS 문서 부족으로 Advanced 질문 답변 불가

### 2. Enhanced RAG Pipeline (answerer_v2.py)

#### Hybrid Search (Dense + Sparse + RRF)
```python
# Dense Search (의미적 유사도)
dense_results = collection.query(query_dense, n=50)

# Sparse Search (키워드 매칭, BM25-like)
sparse_results = sparse_search(query_sparse, top_k=50)

# RRF Fusion (두 방식 통합)
rrf_score = 1/(k + dense_rank) + 1/(k + sparse_rank)
```
**참고**: "Reciprocal Rank Fusion outperforms Condorcet" (SIGIR 2009)

#### Context Quality Filter (Self-RAG Style)
```python
# gpt-4o-mini로 각 컨텍스트 평가
# RELEVANT / PARTIAL / IRRELEVANT 분류
# IRRELEVANT는 LLM에 전달 안 함
```
**참고**: "Self-RAG" (Asai et al., 2023, arXiv:2310.11511)

#### Two-Stage Reranking (기존 확인)
- Stage 1: BAAI/bge-reranker-v2-m3 (빠른 필터링)
- Stage 2: BAAI/bge-reranker-large (정밀 순위)

### 3. 데이터 기반 Hyperparameter 최적화

#### 문서 분포 분석 (analyze_documents.py)
```
Git/Python 12,796 chunks 분석:
- Mean: 829 chars
- Median: 899 chars ← 핵심 지표
- P75: 969 chars
- P95: 1012 chars
```

#### 최적화 결과

| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| **chunk_size** | 1024 | **900** | Median=899 기준 (50% 문서 최적화) |
| **chunk_overlap** | 150 (14.6%) | **180** (20%) | 표준 overlap ratio |
| **rerank_top_k** | 5 | **10** | 초기 검색의 20% (이상적 비율) |
| **hybrid_top_k** | 50 | **50** ✓ | sqrt(12796)=113보다 작지만 효율적 |

**예상 효과**:
- Index Size: -12% (청크 수 감소)
- Search Speed: +10-15%
- Context Recall: +10-15%

### 4. System Prompt 최적화 (prompts/system_v2.txt)

#### 적용 이론
1. **Constitutional AI** (Anthropic 2022): "검색된 문서에만 기반" 원칙
2. **Few-shot Prompting** (Brown et al. 2020): 3개 완벽한 예시로 스타일 학습
3. **Context Grounding** (Self-RAG 2023): RELEVANT 문서만 사용
4. **Chain-of-Thought** (Wei et al. 2022): 내부 분해 → 간결한 결과
5. **Minimal Citation**: 끝에 한 번만 인용

#### Before vs After

**Before** (형식적, 장황함):
```
요약: ...
세부 단계:
1) ...
[증거 요약]
[출처 인용]
[자체 검증]
```
→ 15-20줄

**After** (자연스럽고 간결함):
```
*args와 **kwargs는 함수에 가변 개수의 인자를 전달할 때 사용합니다.

• *args: 위치 인자를 튜플로 받음
• **kwargs: 키워드 인자를 딕셔너리로 받음

예시: ...

📚 참고: function-arguments.md
```
→ 5-8줄

### 5. 출처 표시 개선
**Before**: "근거 1, 근거 2" (파일명 없음)
**After**: "function-arguments.md" (실제 파일명)

```python
# ChromaDB 메타데이터 조회 및 포맷팅
context_block = f"[문서 {i}] {domain} | {filename}\n{text}"
```

---

## 📊 예상 성능 개선

| Metric | Before | Target | Improvement |
|--------|--------|--------|-------------|
| **Faithfulness** | 45% | **93%+** | +106% 🎯 |
| **Answer Correctness** | 44% | **75%+** | +70% 🎯 |
| **Context Precision** | 65% | **80%+** | +23% |
| **Context Recall** | 68% | **85%+** | +25% |
| **Answer Relevancy** | 70% | **80%+** | +14% |

**추가 효과**:
- 답변 길이: 70% 감소 (15줄 → 5-8줄)
- 가독성: 대폭 향상
- Search Speed: +10-15%

---

## 🛠️ 기술 스택

**Core**:
- Embedding: BAAI/bge-m3 (Dense + Sparse)
- Vector DB: ChromaDB (12,796 chunks)
- Reranking: bge-reranker-v2-m3 + bge-reranker-large
- LLM: GPT-4.1 (답변) + GPT-4o-mini (평가)

**분석 도구**:
- pandas, numpy (데이터 분석)
- matplotlib, seaborn (시각화)

---

## 📁 생성된 파일

**코어**:
1. `answerer_v2.py` - Enhanced RAG Pipeline
2. `config/enhanced.yaml` - 최적화된 설정
3. `prompts/system_v2.txt` - 개선된 System Prompt

**분석**:
4. `analyze_documents.py` - 문서 분포 분석 + 시각화
5. `test_enhanced.py` - Quick 테스트
6. `compare_pipelines.py` - Baseline vs Enhanced 비교

**문서**:
7. `ENHANCED_README.md` - Enhanced Pipeline 가이드
8. `OPTIMIZATION_GUIDE.md` - Hyperparameter 최적화 가이드
9. `PROMPT_OPTIMIZATION.md` - System Prompt 최적화 가이드
10. `PROJECT_SUMMARY.md` - 상세 보고서
11. `EXECUTIVE_SUMMARY.md` - 이 요약본

---

## ✅ 다음 단계

### 즉시 실행
1. **재인덱싱**: `python data_prep.py && python index_builder.py`
2. **Quick Test**: 3-5개 질문 테스트
3. **RAGAS 평가**: 최종 성능 측정

### 단기 (1-2일)
4. Baseline vs Enhanced 비교
5. 에러 케이스 분석
6. 성능 벤치마크

### 중기 (1주)
7. Phase 2: HyDE, Metadata Filtering, Semantic Caching
8. 사용자 피드백 수집

---

## 🔬 주요 참고 논문

1. **Self-RAG** (Asai et al., 2023) - arXiv:2310.11511
2. **Constitutional AI** (Anthropic, 2022) - arXiv:2212.08073
3. **Chain-of-Thought** (Wei et al., 2022) - arXiv:2201.11903
4. **Few-shot Learning** (Brown et al., 2020) - arXiv:2005.14165
5. **RRF** (SIGIR 2009) - Reciprocal Rank Fusion

---

## 💡 핵심 인사이트

1. **데이터 기반 의사결정**: 직감 < 실제 데이터 분석 (chunk_size 최적화)
2. **Few-shot > 지침**: 100줄 지침 < 3개 완벽한 예시
3. **Hybrid Search**: Dense + Sparse 상호보완 → Recall ↑
4. **Constitutional AI**: "금지" < "원칙 제시" → Faithfulness ↑
5. **사용자 경험**: 기술적 완성도 < 간결하고 자연스러운 답변

---

## 📞 프로젝트 정보

**위치**: `c:\develop1\test\experiments\rag_pipeline\`
**상태**: ✅ 구현 완료, ⏳ 검증 대기 중
**날짜**: 2025-11-20
**버전**: 1.0

---

**요약**:
Git/Python RAG 챗봇을 도메인 집중화, Enhanced Pipeline (Hybrid Search + Context Quality Filter), 데이터 기반 Hyperparameter 최적화, System Prompt 개선을 통해 종합적으로 최적화. Faithfulness 45% → 93%+, Answer Correctness 44% → 75%+ 목표. 5개 주요 LLM 프롬프팅 이론 및 논문 기반 구현.

**다음 단계**: 재인덱싱 → Quick Test → RAGAS 평가 → 목표 달성 검증 🚀
