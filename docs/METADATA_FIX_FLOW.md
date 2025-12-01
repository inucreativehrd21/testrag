# 🔧 메타데이터 출처 표시 수정 및 재인덱싱 Flow

## ❌ 현재 문제

답변에서 출처가 **"근거 1, 근거 2"**로 표시되며, 실제 파일명이 나오지 않습니다.

## ✅ 해결 방법

`answerer_v2.py`를 수정하여 ChromaDB에서 **메타데이터를 함께 조회**하도록 변경했습니다.

---

## 🚀 실행 Flow (순서대로)

### **Step 1: 기존 answerer_v2.py 백업 및 교체**

```bash
cd c:\develop1\test\experiments\rag_pipeline

# 기존 파일 백업
copy answerer_v2.py answerer_v2_old.py

# 수정된 버전으로 교체
copy answerer_v2_fixed.py answerer_v2.py
```

**또는 직접 교체:**
- `answerer_v2.py` 삭제
- `answerer_v2_fixed.py` 이름을 `answerer_v2.py`로 변경

---

### **Step 2: 재인덱싱 (필수!)**

```bash
# 현재 디렉토리 확인
cd c:\develop1\test\experiments\rag_pipeline

# 1. 데이터 준비 (chunk_size=900, overlap=180)
python data_prep.py --config config/enhanced.yaml

# 2. 인덱스 구축 (메타데이터 포함)
python index_builder.py --config config/enhanced.yaml
```

**예상 소요 시간:**
- data_prep.py: 약 1-2분
- index_builder.py: 약 2-3분
- **총 약 3-5분**

**출력 확인:**
```
✓ Loaded xxx files
✓ Created xxx chunks
✓ Saved to artifacts/chunks.parquet
✓ Indexed xxx chunks into artifacts/chroma_db
```

---

### **Step 3: 테스트 (간단한 질문)**

```bash
# Python 질문 테스트
python answerer_v2.py "Python에서 얕은 복사와 깊은 복사의 차이는?" --config config/enhanced.yaml
```

**기대 출력:**
```
얕은 복사와 깊은 복사는 리스트 등 복합 객체를 복사할 때 내부 구조를 어떻게 다루는지에 따라 다릅니다.

• 얕은 복사(shallow copy): copy()를 사용하며...
• 깊은 복사(deep copy): deepcopy()를 사용하며...

예시:
import copy
...

📚 참고: python | chunk abc123de  ← 여기 메타데이터 포함!
```

**출처 표시 확인:**
- ❌ Before: "📚 참고: 근거 1, 근거 4"
- ✅ After: "📚 참고: python | chunk abc123de" (또는 실제 파일명)

---

### **Step 4: RAGAS 평가 실행**

```bash
# OpenAI API 키 설정 (PowerShell)
$env:OPENAI_API_KEY="your-api-key"

# RAGAS 평가 실행 (Python 40 + Git 40 = 80개 질문)
python run_ragas_evaluation.py
```

**예상 소요 시간: 11-15분**

**출력:**
```
STEP 1: Running RAG Pipeline
[1/80] python_001 (python, easy)
...
[80/80] git_040 (git, hard)

STEP 2: Preparing RAGAS Dataset
✓ Dataset prepared: 80 samples

STEP 3: Running RAGAS Evaluation
✓ RAGAS evaluation complete

STEP 4: Saving Results
✓ Detailed results saved
✓ Report saved

RAGAS EVALUATION RESULTS
context_precision       :  XX.XX%
context_recall          :  XX.XX%
faithfulness            :  XX.XX%
answer_relevancy        :  XX.XX%
answer_correctness      :  XX.XX%
```

---

## 📁 주요 변경사항 (answerer_v2_fixed.py)

### **1. retrieve() 함수 시그니처 변경**

**Before:**
```python
def retrieve(self, question: str) -> List[str]:
    ...
    return final_contexts  # 텍스트만 반환
```

**After:**
```python
def retrieve(self, question: str) -> Tuple[List[str], List[Dict]]:
    ...
    return final_contexts, context_metadatas  # 텍스트 + 메타데이터 반환
```

---

### **2. Dense retrieval 시 메타데이터 조회**

**Before:**
```python
dense_results = self.collection.query(
    query_embeddings=[query_dense],
    n_results=dense_top_k
)
dense_docs = dense_results["documents"][0]
dense_ids = dense_results["ids"][0]
# metadatas 조회 안 함 ❌
```

**After:**
```python
dense_results = self.collection.query(
    query_embeddings=[query_dense],
    n_results=dense_top_k,
    include=["documents", "metadatas"]  # ← 메타데이터 포함!
)
dense_docs = dense_results["documents"][0]
dense_ids = dense_results["ids"][0]
dense_metas = dense_results["metadatas"][0]  # ← 메타데이터 가져오기 ✅
```

---

### **3. 최종 contexts에 대응하는 metadatas 매칭**

```python
# Step 7: Get metadatas for final contexts
context_metadatas = []
for ctx in final_contexts:
    # Find matching metadata
    try:
        idx = dense_docs.index(ctx)
        meta = dense_metas[idx]
        meta["chunk_id"] = dense_ids[idx]
        context_metadatas.append(meta)
    except ValueError:
        context_metadatas.append({"domain": "unknown", "chunk_id": "unknown"})

return final_contexts, context_metadatas
```

---

### **4. answer() 함수에서 메타데이터 활용**

**Before:**
```python
contexts = self.retrieve(question)  # 텍스트만
context_block = "\n\n".join(f"근거 {i+1}: {chunk}" for i, chunk in enumerate(contexts))
# ❌ "근거 1, 근거 2"로 표시
```

**After:**
```python
contexts, metadatas = self.retrieve(question)  # 텍스트 + 메타데이터

context_block = "\n\n".join(
    f"[문서 {i+1}] {meta.get('domain', 'unknown')} | chunk {meta.get('chunk_id', 'unknown')[-8:]}\n{ctx}"
    for i, (ctx, meta) in enumerate(zip(contexts, metadatas))
)
# ✅ "[문서 1] python | chunk abc123de" 형식
```

---

## 🔍 메타데이터 구조 (ChromaDB)

현재 `index_builder.py`에서 저장하는 메타데이터:

```python
metadatas = batch[["domain", "length"]].to_dict(orient="records")

# 예시:
{
    "domain": "python",
    "length": 856
}
```

**chunk_id 형식:**
- `python_abc123de` (domain_hash)
- `git_xyz789ab`

---

## ⚠️ 중요 사항

### **1. 반드시 재인덱싱 필요**

`answerer_v2.py`를 수정했으므로, ChromaDB에서 메타데이터를 조회할 수 있어야 합니다.
기존 인덱스에는 메타데이터가 이미 있지만, `chunk_size=900`으로 재인덱싱이 필요합니다.

### **2. run_ragas_evaluation.py 수정 필요 없음**

`run_ragas_evaluation.py`는 `answerer_v2.py`의 `answer()` 메서드만 호출하므로, 수정 불필요합니다.

### **3. test_enhanced.py 수정 필요 없음**

마찬가지로 `answer()` 메서드만 사용하므로 수정 불필요합니다.

---

## ✅ 검증 체크리스트

### 재인덱싱 전
- [ ] `answerer_v2.py` 백업 완료
- [ ] `answerer_v2_fixed.py` → `answerer_v2.py`로 교체

### 재인덱싱
- [ ] `python data_prep.py --config config/enhanced.yaml` 실행
- [ ] `python index_builder.py --config config/enhanced.yaml` 실행
- [ ] 에러 없이 완료 (3-5분 소요)
- [ ] `artifacts/chroma_db/` 생성 확인

### 테스트
- [ ] 간단한 질문 테스트 실행
- [ ] 답변에서 **"[문서 1] python | chunk xxx"** 형식 확인
- [ ] ❌ "근거 1, 근거 2"가 아닌 ✅ 실제 메타데이터 확인

### RAGAS 평가
- [ ] OpenAI API 키 설정
- [ ] `python run_ragas_evaluation.py` 실행
- [ ] 80개 질문 모두 처리 (11-15분 소요)
- [ ] 결과 파일 생성 확인

---

## 🚨 Troubleshooting

### **"Tuple unpacking error"**

**증상:**
```python
contexts = self.retrieve(question)  # 에러!
# TypeError: cannot unpack non-iterable list object
```

**원인:** 이전 코드에서 `retrieve()`가 리스트를 반환했지만, 수정 후 튜플을 반환

**해결:**
```python
# Before (에러)
contexts = self.retrieve(question)

# After (수정)
contexts, metadatas = self.retrieve(question)
```

`answerer_v2.py`를 `answerer_v2_fixed.py`로 완전히 교체했는지 확인하세요.

---

### **"Metadata shows 'unknown'**

**증상:** 출처가 "unknown | chunk unknown"으로 표시

**원인:** ChromaDB에 메타데이터가 없거나, 매칭 실패

**해결:**
1. 재인덱싱 실행 (Step 2)
2. `index_builder.py`가 올바르게 실행되었는지 확인
3. ChromaDB 컬렉션 확인:
   ```python
   import chromadb
   client = chromadb.PersistentClient(path="artifacts/chroma_db")
   collection = client.get_collection("rag_chunks")
   result = collection.get(limit=1, include=["metadatas"])
   print(result["metadatas"])  # domain, length 확인
   ```

---

### **"dense_docs.index(ctx) ValueError"**

**증상:** Context를 찾을 수 없음

**원인:** RRF, reranking, quality filter를 거치면서 텍스트가 약간 변경되었을 수 있음

**해결:** 이미 코드에 try-except로 처리되어 있으며, fallback으로 "unknown" 반환

---

## 📊 기대 결과

### **Before (수정 전)**
```
얕은 복사와 깊은 복사는...

📚 참고: 근거 1, 근거 2, 근거 3, 근거 4
```

### **After (수정 후)**
```
얕은 복사와 깊은 복사는...

📚 참고: python | chunk a1b2c3d4, python | chunk e5f6g7h8
```

**또는 더 개선된 버전 (system_v2.txt 활용):**
```
📚 참고: python-copy.md, shallow-deep-copy.md
```

---

## 🎯 최종 목표

1. ✅ 출처 표시 개선: "근거 1" → "python | chunk xxx"
2. ✅ 메타데이터 활용: domain, chunk_id 표시
3. ✅ RAGAS 평가 실행: Python 40 + Git 40 = 80개
4. ✅ 성능 목표 달성: Faithfulness 93%+, Answer Correctness 75%+

---

**이제 위 Flow대로 실행하세요!** 🚀

1. answerer_v2.py 교체
2. 재인덱싱 (data_prep + index_builder)
3. 테스트 (간단한 질문)
4. RAGAS 평가 실행
5. 결과 확인 및 분석
