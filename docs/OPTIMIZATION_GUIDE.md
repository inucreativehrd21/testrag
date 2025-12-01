# 📊 RAG Hyperparameter Optimization Guide

## 목적
Git/Python으로 도메인을 좁힌 후, 실제 문서 데이터를 기반으로:
1. **Top-k 파라미터** 검증 및 최적화
2. **Chunking 크기** 최적화
3. **데이터 기반 의사결정**

---

## 🚀 빠른 시작 (Runpod에서 실행)

### 1. 분석 스크립트 실행

```bash
cd /workspace/rag_pipeline  # 또는 작업 디렉토리

# 간단 버전 (시각화 없음, 텍스트만)
python analyze_documents_simple.py

# 완전 버전 (시각화 포함)
python analyze_documents.py
```

### 2. 결과 확인

```bash
# 통계 확인
cat artifacts/analysis/statistics.json

# 시각화 확인 (완전 버전 실행 시)
# artifacts/analysis/document_analysis.png 다운로드
```

---

## 📋 생성된 파일

### `analyze_documents.py`
- **기능**: 완전한 시각화 + 통계 분석
- **의존성**: matplotlib, seaborn 필요
- **출력**:
  - `artifacts/analysis/document_analysis.png` (6개 차트)
  - `artifacts/analysis/statistics.json` (수치 데이터)

**차트 종류:**
1. 전체 문서 길이 분포 (Histogram)
2. 도메인별 길이 분포 (Git vs Python)
3. Box Plot (도메인별 비교)
4. 누적 분포 함수 (CDF)
5. 시간에 따른 길이 변화
6. 추천 파라미터 요약 표

### `analyze_documents_simple.py`
- **기능**: 통계 분석 + ASCII 히스토그램
- **의존성**: pandas, numpy만 필요 (기본 설치됨)
- **출력**:
  - 터미널에 전체 분석 결과 출력
  - `artifacts/analysis/statistics.json` (수치 데이터)

---

## 🔍 분석 내용

### 1. 데이터셋 통계
```
✓ 총 청크 개수
✓ 도메인별 분포 (Git vs Python)
✓ 평균 길이, 중앙값, 표준편차
✓ Percentiles (P25, P50, P75, P90, P95, P99)
```

### 2. Top-k 추천
```
알고리즘:
- sqrt(N): 매우 보수적 (대규모 데이터셋용)
- 10% of N: 관대함 (높은 Recall)
- 5% of N: 균형
- 3% of N: 효율적 (추천)

출력:
- hybrid_dense_top_k
- hybrid_sparse_top_k
- rerank_top_k
- rrf_k (표준값 60 유지)
```

**근거:**
- 문서가 1,000개라면 3% = 30개 검색
- 문서가 5,000개라면 3% = 150개 (상한 100 적용)
- Hybrid Search는 30-70개 후보가 최적

### 3. Chunking 파라미터 추천
```
기준: P75 (75th Percentile)

로직:
- P75 < 900 chars  → chunk_size 줄이기 (1024 → P75)
- P75 > 1100 chars → chunk_size 늘리기 (1024 → P75)
- P75 ≈ 900-1100   → 현재 유지 (1024)

Overlap: chunk_size의 20% (표준)
```

**이유:**
- P75 기준 = 75%의 문서를 한 청크에 담을 수 있음
- 너무 큰 청크 = 검색 정밀도 하락
- 너무 작은 청크 = 컨텍스트 부족

---

## 📊 예상 결과 (예시)

```
================================================================================
OVERALL STATISTICS
================================================================================
Total Chunks            :           2,847
Total Characters        :       3,456,891
Mean Length             :        1,214.32 chars
Median Length           :        1,089.00 chars
Std Deviation           :          456.78 chars
Min Length              :           45 chars
Max Length              :        4,523 chars

Percentiles:
  P25                 :          834 chars
  P50                 :        1,089 chars
  P75                 :        1,398 chars  ← 주목!
  P90                 :        1,856 chars
  P95                 :        2,134 chars
  P99                 :        3,012 chars

================================================================================
HYPERPARAMETER RECOMMENDATIONS
================================================================================

1. RETRIEVAL TOP-K PARAMETERS
   Dataset Size: 2,847 chunks

   Mathematical Baselines:
     • sqrt(N):          53  (conservative)
     • 10% of N:        284  (generous)
     • 5% of N:         142  (balanced)
     • 3% of N:          85  (efficient)

   Current Config:
     • hybrid_dense_top_k:  50
     • hybrid_sparse_top_k: 50
     • rerank_top_k:        5
     • rrf_k:               60

   🎯 RECOMMENDED CONFIG:
     • hybrid_dense_top_k:   70  ← HIGHER (3% rule + safety margin)
     • hybrid_sparse_top_k:  70  ← HIGHER
     • rerank_top_k:         10  ← ADJUST (15% of 70)
     • rrf_k:                60  ← OK

   Rationale:
     - With 2,847 chunks, 3% = 85 is efficient
     - Capped at 70 for performance
     - Reranking top-k should be 10-20% of initial

2. CHUNKING PARAMETERS
   Current Config:
     • chunk_size:    1024 chars
     • chunk_overlap: 150 chars (14.6%)

   Distribution Analysis:
     • 75% of docs ≤ 1,398 chars
     • 90% of docs ≤ 1,856 chars
     • 95% of docs ≤ 2,134 chars

   🎯 RECOMMENDED CONFIG:
     • chunk_size:    1400 chars  ← INCREASE
     • chunk_overlap:  280 chars  (20%)

   Rationale:
     - P75=1,398 > 1100, current 1024 may be too small
     - 20% overlap is standard for context preservation
     - Covers 75% of documents optimally
     - 📈 36% more context per chunk

================================================================================
CONFIGURATION SUMMARY
================================================================================

Parameter                      Current         Recommended     Action
---------------------------------------------------------------------------
hybrid_dense_top_k             50              70              → Change to 70
hybrid_sparse_top_k            50              70              → Change to 70
rerank_top_k                   5               10              → Change to 10
rrf_k                          60              60              ✓ OK
chunk_size                     1024            1400            → Increase
chunk_overlap                  150             280             → Adjust
================================================================================
```

---

## 🎯 의사결정 플로우

### Case 1: 청크가 많고 (3,000+), P75 ≈ 1,000
```yaml
# Recommended
retrieval:
  hybrid_dense_top_k: 70-100
  hybrid_sparse_top_k: 70-100
  rerank_top_k: 10-15
  rrf_k: 60

chunking:
  chunk_size: 1024  # 유지
  chunk_overlap: 200
```

### Case 2: 청크가 적고 (1,000-), P75 < 800
```yaml
# Recommended
retrieval:
  hybrid_dense_top_k: 30-50
  hybrid_sparse_top_k: 30-50
  rerank_top_k: 5-8
  rrf_k: 60

chunking:
  chunk_size: 800  # 감소
  chunk_overlap: 160
```

### Case 3: 청크가 많고, P75 > 1,200
```yaml
# Recommended
retrieval:
  hybrid_dense_top_k: 70-100
  hybrid_sparse_top_k: 70-100
  rerank_top_k: 10-15
  rrf_k: 60

chunking:
  chunk_size: 1200-1400  # 증가
  chunk_overlap: 240-280
```

---

## ⚙️ Config 업데이트 방법

### 1. 분석 결과 확인
```bash
python analyze_documents.py
# 또는
python analyze_documents_simple.py
```

### 2. `config/enhanced.yaml` 수정
```yaml
retrieval:
  # 분석 결과의 "RECOMMENDED CONFIG" 값 반영
  hybrid_dense_top_k: <분석 결과>
  hybrid_sparse_top_k: <분석 결과>
  rerank_top_k: <분석 결과>
  rrf_k: 60

chunking:
  primary:
    chunk_size: <분석 결과>
    chunk_overlap: <분석 결과>
```

### 3. 청킹 파라미터가 변경되었다면 재인덱싱 필요!
```bash
# 주의: chunk_size 변경 시만 실행
python data_prep.py --config config/enhanced.yaml
python index_builder.py --config config/enhanced.yaml
```

### 4. Top-k만 변경했다면 재인덱싱 불필요
```bash
# 그냥 바로 테스트 가능
python answerer_v2.py "질문" --config config/enhanced.yaml
```

---

## 🔬 상세 분석 (statistics.json)

```json
{
  "dataset": {
    "total_chunks": 2847,
    "total_characters": 3456891,
    "domains": {
      "git": 1523,
      "python": 1324
    }
  },
  "statistics": {
    "mean": 1214.32,
    "median": 1089.0,
    "std": 456.78,
    "min": 45,
    "max": 4523,
    "percentiles": {
      "p25": 834.0,
      "p50": 1089.0,
      "p75": 1398.0,
      "p90": 1856.0,
      "p95": 2134.0,
      "p99": 3012.0
    }
  },
  "recommendations": {
    "retrieval": {
      "hybrid_dense_top_k": 70,
      "hybrid_sparse_top_k": 70,
      "rerank_top_k": 10,
      "rrf_k": 60
    },
    "chunking": {
      "chunk_size": 1400,
      "chunk_overlap": 280,
      "action": "INCREASE",
      "reason": "P75=1398 > 1100, current 1024 may be too small"
    }
  }
}
```

---

## 📈 성능 영향 예측

### Top-k 조정 (50 → 70)
```
Impact:
  Latency:        +15-20% (더 많은 문서 처리)
  Context Recall: +5-10% (더 많은 후보)
  Precision:      유지 (Reranking이 필터링)

Trade-off:
  ✓ Recall 향상
  ✗ 약간 느려짐 (허용 범위)
```

### Chunking 증가 (1024 → 1400)
```
Impact:
  Index Size:     -20-25% (청크 개수 감소)
  Context per chunk: +36% (더 풍부한 컨텍스트)
  Retrieval Speed: +10-15% (청크 수 감소)
  Precision:      약간 감소 (청크가 커짐)

Trade-off:
  ✓ 더 적은 청크 = 빠른 검색
  ✓ 더 많은 컨텍스트 = 더 나은 답변
  ✗ 청크 크기 증가 = 약간의 정밀도 손실 (미미)

Overall: 🎯 추천
```

---

## 🛠️ Troubleshooting

### "ModuleNotFoundError: matplotlib"
```bash
# Runpod에서 설치
pip install matplotlib seaborn

# 또는 simple 버전 사용
python analyze_documents_simple.py
```

### "FileNotFoundError: chunks.parquet"
```bash
# 데이터 준비 먼저 실행
python data_prep.py --config config/enhanced.yaml
python index_builder.py --config config/enhanced.yaml
```

### "분석 결과가 이상해요"
```
체크리스트:
1. Git/Python 문서만 필터링되었는지 확인
2. domain 컬럼이 chunks.parquet에 있는지 확인
3. 전체 청크 수가 합리적인지 확인 (500-10,000 범위)
```

---

## 🎓 이론적 배경

### Top-k 선택 기준

**Rule of Thumb (경험 법칙):**
1. **sqrt(N)**: 정보 검색 이론의 고전적 기준
   - 예: 10,000 문서 → sqrt(10000) = 100
   - 매우 보수적, 대규모 데이터셋용

2. **3-5% of N**: 현대 RAG 시스템 표준
   - 예: 3,000 문서 → 3% = 90
   - 효율성과 성능의 균형

3. **Reranking ratio**: 10-20%
   - 초기 검색의 10-20%만 최종 선택
   - 예: 70개 검색 → 10개 rerank

**참고 논문:**
- "Reciprocal Rank Fusion outperforms Condorcet" (SIGIR 2009)
- "Lost in the Middle" (2023) - 너무 많은 컨텍스트의 문제

### Chunking 크기 선택

**원칙:**
1. **Percentile-based**: P75 기준이 최적
   - 75%의 문서를 완전히 담음
   - 너무 작지도, 크지도 않음

2. **Overlap ratio**: 15-25%
   - 20%가 표준
   - 문장/단락 경계 보존

3. **Domain-specific**: 도메인 특성 고려
   - Git: 명령어 설명 (짧음, 500-1000)
   - Python: 튜토리얼/예제 (김, 1000-2000)

**참고:**
- LangChain Chunking Guide
- LlamaIndex Text Splitters

---

## ✅ 체크리스트

분석 및 최적화 전:
- [ ] chunks.parquet 파일 존재 확인
- [ ] Git/Python 도메인 필터링 확인
- [ ] 현재 config 백업

분석 실행:
- [ ] `analyze_documents.py` 또는 `_simple.py` 실행
- [ ] 결과 저장: `statistics.json`
- [ ] 시각화 확인: `document_analysis.png` (선택)

Config 업데이트:
- [ ] `config/enhanced.yaml` 수정
- [ ] chunk_size 변경 시: 재인덱싱 필요 체크
- [ ] top-k만 변경 시: 바로 테스트 가능

검증:
- [ ] 간단한 질문으로 테스트
- [ ] 응답 품질 확인
- [ ] 속도 벤치마크 (선택)

---

## 🚀 다음 단계

1. **분석 실행** → `python analyze_documents.py`
2. **추천값 확인** → `statistics.json`
3. **Config 업데이트** → `config/enhanced.yaml`
4. **재인덱싱** (chunk_size 변경 시)
5. **테스트** → `python test_enhanced.py`
6. **RAGAS 평가** → 성능 측정

**목표:**
- Context Recall: 70% → 80%+
- Faithfulness: 86% → 93%+
- Answer Correctness: 60% → 70%+
