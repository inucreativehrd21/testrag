# RAG 파이프라인 검증 리포트 (확장 크롤링 1,001개 대응)

## ✅ 검증 완료 항목

### 1. **data_prep.py** - URL 메타데이터 포함 ✓

**확인 사항:**
- ✅ `_load_documents_with_metadata()` 메서드가 pages.json에서 URL을 읽음
- ✅ 각 청크에 URL을 포함하여 저장 (78번 줄)
- ✅ chunks.parquet에 'url' 컬럼 저장

**코드:**
```python
chunk_rows.append({
    "domain": domain,
    "chunk_id": f"{domain}_{len(chunk_rows)}",
    "text": chunk,
    "length": len(chunk),
    "url": chunk_url  # ← URL 추가!
})
```

**대용량 데이터 처리:**
- 메모리 사용: 1,001개 문서 → 예상 7,000-10,000 청크 → 약 500MB-1GB RAM
- 처리 시간: 2-3분 예상
- **문제 없음** ✓

---

### 2. **index_builder.py** - URL 메타데이터 ChromaDB 저장 ✓

**확인 사항:**
- ✅ chunks.parquet에서 'url' 컬럼 읽음
- ✅ ChromaDB metadata에 URL 포함 (90-93번 줄)

**코드:**
```python
metadata_columns = ["domain", "length"]
if "url" in batch.columns:
    metadata_columns.append("url")  # ← URL 추가
metadatas = batch[metadata_columns].to_dict(orient="records")
```

**대용량 데이터 처리:**
- 배치 크기: 512 청크/배치 (72번 줄)
- 예상 배치 수: 7,000 청크 ÷ 512 = ~14 배치
- GPU 인덱싱 시간: **5-7분** 예상
- CPU 인덱싱 시간: 15-20분 예상
- **문제 없음** ✓

---

### 3. **answerer_v2_optimized.py** - URL 출처 표시 ✓

**확인 사항:**
- ✅ retrieve() 메서드가 metadatas 반환
- ✅ answer() 메서드가 URL 출처 추가 (397-408번 줄)
- ✅ 중복 URL 제거

**코드:**
```python
# Add source URLs at the end
source_urls = []
for meta in metadatas:
    url = meta.get('url', 'unknown')
    if url != 'unknown' and url not in source_urls:
        source_urls.append(url)

if source_urls:
    sources_section = "\n\n📚 참고:\n" + "\n".join(f"- {url}" for url in source_urls)
    answer = answer_text + sources_section
```

**출력 예시:**
```
답변 내용...

📚 참고:
- https://www.atlassian.com/git/tutorials/rewriting-history/git-rebase
- https://git-scm.com/book/ko/v2/Git-브랜치-Rebase-하기
- https://docs.gitlab.com/ee/topics/git/git_rebase.html
```

**문제 없음** ✓

---

### 4. **config/enhanced.yaml** - 대용량 데이터 처리 설정 ✓

**확인 사항:**
- ✅ chunk_size: 900 (데이터 기반 최적화)
- ✅ batch_size: 32 (적절한 크기)
- ✅ hybrid search top_k: 50+50 (충분한 후보)
- ✅ rerank_top_k: 10 (최종 컨텍스트)

**대용량 데이터 처리:**
- 7,000-10,000 청크 인덱싱: **문제 없음**
- 검색 속도: 50+50 후보 검색 → 10개 rerank → 매우 빠름
- **문제 없음** ✓

---

## ⚠️ 주의사항 및 개선 제안

### 1. **메모리 사용량**

**현재 상태:**
- data_prep: ~500MB-1GB RAM
- index_builder: ~2-4GB RAM (GPU), ~4-8GB RAM (CPU)
- ChromaDB 크기: ~3-4GB 디스크

**권장사항:**
- 최소 16GB RAM 권장
- GPU 사용 시 최소 8GB VRAM 권장

---

### 2. **index_builder.py 배치 크기**

**현재 설정:**
```python
batch_size = 512  # ChromaDB 배치 크기 (72번 줄)
```

**문제점:**
- 7,000-10,000 청크 인덱싱 시 메모리 부족 가능성

**개선 제안:**
메모리가 부족한 경우 배치 크기를 줄이세요:

```python
# index_builder.py 72번 줄
batch_size = 256  # 512 → 256으로 줄임 (메모리 부족 시)
```

---

### 3. **크롤러 content_extractor 검증 필요**

**확인 필요:**
확장 크롤러가 실제로 URL을 올바르게 추출하는지 검증 필요

**검증 방법:**
```bash
# 1. 테스트 크롤링 (소규모)
cd c:\develop1\test\crawler
python -c "
from scrapers.git_scraper_extended import GitDocsScraperExtended
scraper = GitDocsScraperExtended()
# 테스트: atlassian만 크롤링
from config.settings_extended import TARGET_URLS
config = TARGET_URLS['git']['atlassian']
docs = scraper._scrape_source('atlassian', config)
print(f'수집: {len(docs)}개')
print(f'URL 샘플: {docs[0].get(\"url\", \"NO URL\")}')
"

# 2. URL 태깅 확인
python -c "
import json
with open('data/raw/git/pages.json') as f:
    docs = json.load(f)
urls_found = sum(1 for d in docs if d.get('url', 'unknown') != 'unknown')
print(f'총 문서: {len(docs)}개')
print(f'URL 태깅: {urls_found}/{len(docs)} ({urls_found/len(docs)*100:.1f}%)')
print(f'샘플 URL: {docs[0].get(\"url\", \"NO URL\")}')
"
```

---

### 4. **ChromaDB 설정 최적화**

**현재:** 기본 설정 사용

**개선 제안:**
대용량 데이터에 최적화된 ChromaDB 설정 추가

**수정 파일:** `index_builder.py`

```python
# 59번 줄 수정
# Before:
client = chromadb.PersistentClient(path=str(self.chroma_path))

# After:
client = chromadb.PersistentClient(
    path=str(self.chroma_path),
    settings=chromadb.Settings(
        anonymized_telemetry=False,
        allow_reset=True,
        # 대용량 데이터 최적화
        chroma_db_impl="duckdb+parquet",  # 더 빠른 백엔드
        chroma_server_cors_allow_origins=["*"]
    )
)
```

---

## 📊 예상 처리 시간 (1,001개 문서)

| 단계 | CPU | GPU (RTX 4090) |
|------|-----|----------------|
| **1. 크롤링** | 40-50분 | 40-50분 |
| **2. data_prep** | 2-3분 | 2-3분 |
| **3. index_builder** | 15-20분 | **5-7분** |
| **4. 답변 생성** | 5-10초/질문 | 3-5초/질문 |
| **총합** | ~60분 | **~50분** |

---

## 🔍 실행 전 체크리스트

### 크롤링 전:
- [ ] 최소 5GB 디스크 여유 공간 확인
- [ ] 인터넷 연결 안정성 확인
- [ ] `crawler/config/settings_extended.py` 로드 확인

### 데이터 준비 전:
- [ ] `data/raw/git/pages.json` 존재 확인
- [ ] `data/raw/python/pages.json` 존재 확인
- [ ] 각 pages.json에 URL 태깅 확인

### 인덱싱 전:
- [ ] `artifacts/chunks.parquet` 존재 확인
- [ ] chunks.parquet에 'url' 컬럼 존재 확인
- [ ] 최소 16GB RAM 확인
- [ ] GPU 사용 시 최소 8GB VRAM 확인

### 답변 생성 전:
- [ ] `artifacts/chroma_db/` 존재 확인
- [ ] OPENAI_API_KEY 환경변수 설정 확인

---

## 🎯 결론

### ✅ 준비 완료
- data_prep.py: URL 메타데이터 포함 ✓
- index_builder.py: URL 메타데이터 저장 ✓
- answerer_v2_optimized.py: URL 출처 표시 ✓
- 대용량 데이터 처리 가능 ✓

### ⚠️ 확인 필요
1. 크롤러 content_extractor.py가 URL을 올바르게 추출하는지 검증
2. 메모리 부족 시 index_builder.py 배치 크기 조정
3. ChromaDB 설정 최적화 (선택사항)

### 🚀 다음 단계
1. 확장 크롤링 실행 (`run_crawl_extended.py`)
2. URL 태깅 확인 (위 검증 방법 사용)
3. 데이터 파이프라인 실행 (data_prep → index_builder)
4. 테스트 질문으로 URL 출처 확인

---

**작성일:** 2025-11-28
**버전:** Pipeline Verification v1
