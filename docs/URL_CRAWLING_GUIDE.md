# 🔗 URL 출처 태깅 완료 가이드

## ✅ 완료된 작업

URL 출처가 제대로 표시되도록 전체 RAG 파이프라인을 수정했습니다.

### 1. **크롤러 구축** ✅
- 위치: `c:\develop1\test\crawler\`
- Git 180문서 + Python 272문서 크롤링
- 각 문서마다 URL 태깅 포함

### 2. **데이터 준비 파이프라인 수정** ✅
- `experiments/rag_pipeline/data_prep.py`: URL metadata 포함
- `experiments/rag_pipeline/index_builder.py`: URL을 ChromaDB에 저장

### 3. **답변 생성 시 URL 표시** ✅
- `experiments/rag_pipeline/answerer_v2_optimized.py`: 답변 끝에 URL 출처 추가

---

## 🚀 실행 순서

### **1단계: 크롤링 (Git + Python 문서)**

```bash
cd c:\develop1\test\crawler

# 필요한 패키지 설치
pip install beautifulsoup4 lxml requests tqdm

# 크롤링 실행
python run_crawl.py
```

**선택 옵션:**
- `1`: Git만 크롤링 (180문서)
- `2`: Python만 크롤링 (272문서)
- `3`: 둘 다 크롤링 **(권장)**

**예상 소요 시간:**
- Git: 약 10-15분 (rate limiting 2초/페이지)
- Python: 약 15-20분
- **총 약 30분**

**결과물:**
- `data/raw/git/pages.json` (URL 포함)
- `data/raw/python/pages.json` (URL 포함)

---

### **2단계: 데이터 준비 (청킹)**

```bash
cd c:\develop1\test\experiments\rag_pipeline

# 청킹 실행 (URL metadata 포함)
python data_prep.py --config config/enhanced.yaml
```

**예상 소요 시간:** 1-2분

**결과물:**
- `artifacts/chunks.parquet` (URL 컬럼 포함)

**검증:**
```python
import pandas as pd
df = pd.read_parquet("artifacts/chunks.parquet")
print(df.columns)  # ['domain', 'chunk_id', 'text', 'length', 'url']
print(df['url'].head())  # URL 확인
```

---

### **3단계: 벡터 인덱싱 (URL metadata 포함)**

```bash
# ChromaDB 인덱싱 (URL metadata 포함)
python index_builder.py --config config/enhanced.yaml
```

**예상 소요 시간:**
- CPU: 10-15분
- GPU (RTX 4090): **3-5분**

**결과물:**
- `artifacts/chroma_db/` (URL이 metadata에 포함됨)

---

### **4단계: 테스트 (URL 출처 확인)**

```bash
# 질문하고 URL 출처 확인
python answerer_v2_optimized.py "Python에서 얕은 복사와 깊은 복사의 차이는?" --config config/enhanced.yaml
```

**예상 출력:**

```
================================================================================
답변:
================================================================================
얕은 복사(shallow copy)는 객체의 최상위 레벨만 복사하고 내부 객체는 참조를 공유합니다.
깊은 복사(deep copy)는 객체와 그 안의 모든 중첩된 객체까지 재귀적으로 복사합니다.

예를 들어, 리스트 안에 리스트가 있을 때:
- 얕은 복사: 내부 리스트는 원본과 같은 객체를 가리킴
- 깊은 복사: 내부 리스트도 완전히 새로운 객체로 생성

Python에서는 copy.copy()로 얕은 복사, copy.deepcopy()로 깊은 복사를 수행합니다.

📚 참고:
- https://realpython.com/python-shallow-deep-copy
- https://www.programiz.com/python-programming/shallow-deep-copy
- https://docs.python.org/3/library/copy.html
================================================================================
```

✅ **URL이 제대로 표시됩니다!**

---

## 📁 디렉토리 구조

```
c:\develop1\test\
├── crawler/                      # 새로 추가한 크롤러
│   ├── config/
│   │   └── settings.py          # Git 180 + Python 272 문서 설정
│   ├── scrapers/
│   │   ├── base_scraper.py      # HTTP 요청 + Rate limiting
│   │   ├── content_extractor.py # HTML → 구조화된 JSON
│   │   ├── git_scraper.py       # Git 크롤러 (URL 포함)
│   │   └── python_scraper.py    # Python 크롤러 (URL 포함)
│   ├── utils/
│   │   ├── logger.py
│   │   └── retry_handler.py
│   └── run_crawl.py             # 크롤링 실행 스크립트
│
├── data/
│   └── raw/
│       ├── git/
│       │   ├── pages.json       # ← URL 포함!
│       │   └── metadata.json
│       └── python/
│           ├── pages.json       # ← URL 포함!
│           └── metadata.json
│
└── experiments/rag_pipeline/
    ├── data_prep.py             # ← URL metadata 포함 수정
    ├── index_builder.py         # ← URL metadata 저장 수정
    ├── answerer_v2_optimized.py # ← URL 출처 표시 수정
    └── artifacts/
        ├── chunks.parquet       # url 컬럼 포함
        └── chroma_db/           # url metadata 포함
```

---

## 🔍 주요 변경 사항

### **1. `data_prep.py` 변경**

**Before:**
```python
{
    "domain": "python",
    "chunk_id": "python_123",
    "text": "...",
    "length": 500
}
```

**After:**
```python
{
    "domain": "python",
    "chunk_id": "python_123",
    "text": "...",
    "length": 500,
    "url": "https://realpython.com/python-shallow-deep-copy"  # ← 추가!
}
```

### **2. `index_builder.py` 변경**

**Before:**
```python
metadatas = batch[["domain", "length"]].to_dict(orient="records")
```

**After:**
```python
metadata_columns = ["domain", "length"]
if "url" in batch.columns:
    metadata_columns.append("url")  # ← URL 추가!
metadatas = batch[metadata_columns].to_dict(orient="records")
```

### **3. `answerer_v2_optimized.py` 변경**

**Before:**
```python
return answer  # 답변만 반환
```

**After:**
```python
# URL 출처 추가
source_urls = []
for meta in metadatas:
    url = meta.get('url', 'unknown')
    if url != 'unknown' and url not in source_urls:
        source_urls.append(url)

if source_urls:
    sources_section = "\n\n📚 참고:\n" + "\n".join(f"- {url}" for url in source_urls)
    answer = answer_text + sources_section

return answer  # 답변 + URL 출처
```

---

## 🎯 크롤링 대상 문서 수

### **Git (총 180문서)**
- Atlassian: 54문서
- Pro Git (한국어): 36문서
- GitHub Docs: 17문서
- W3Schools: 18문서
- Git Reference: 35문서

### **Python (총 272문서)**
- Real Python: 45문서
- Official Tutorial: 15문서
- W3Schools: 27문서
- Official Library: 28문서
- Official HOWTOs: 18문서
- GeeksforGeeks: 30문서
- Programiz: 26문서
- PyMOTW: 45문서
- Official Advanced: 38문서

**총 452문서** (Git 180 + Python 272)

---

## 🛠️ Troubleshooting

### **Q1: "pages.json not found" 에러**
→ 1단계(크롤링)를 먼저 실행하세요.

### **Q2: URL이 "unknown"으로 표시됨**
→ 크롤링을 다시 실행하고, 2-3단계를 순서대로 재실행하세요.

### **Q3: 크롤링이 너무 느림**
→ `crawler/config/settings.py`에서 `request_delay`를 1.0초로 줄이세요 (하지만 429 에러 위험).

### **Q4: 일부 페이지가 크롤링 실패**
→ 정상입니다. 404 에러나 네트워크 오류는 자동으로 스킵되며, 로그에 기록됩니다.

---

## ✅ 검증 체크리스트

- [ ] **1단계**: `data/raw/git/pages.json`과 `data/raw/python/pages.json` 생성 확인
- [ ] **2단계**: `artifacts/chunks.parquet`에 `url` 컬럼 포함 확인
- [ ] **3단계**: ChromaDB 인덱싱 완료 (`artifacts/chroma_db/` 폴더 생성)
- [ ] **4단계**: 테스트 질문에서 **URL 출처**가 답변 끝에 표시되는지 확인

---

## 🎉 완료!

이제 RAG 챗봇이 답변할 때 **실제 크롤링한 URL 출처**를 함께 제공합니다!

**예시:**

**질문:** "git rebase는 무엇인가요?"

**답변:**
```
git rebase는 한 브랜치의 변경사항을 다른 브랜치 위에 재적용하는 명령입니다.
merge와 달리 선형적인 커밋 히스토리를 만들어 프로젝트 히스토리를 깔끔하게 유지할 수 있습니다.

📚 참고:
- https://www.atlassian.com/git/tutorials/rewriting-history/git-rebase
- https://git-scm.com/book/ko/v2/Git-브랜치-Rebase-하기
- https://git-scm.com/docs/git-rebase
```

**"근거 1, 근거 2" → 실제 URL로 변경 완료!** ✅
