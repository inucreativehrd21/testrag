"""
RAG 파이프라인 검증 스크립트
확장 크롤링(1,001개) 데이터가 올바르게 처리되는지 확인
"""
import json
import sys
from pathlib import Path

print("=" * 80)
print("RAG 파이프라인 검증 시작")
print("=" * 80)

# 1. 크롤링 데이터 검증
print("\n[1/4] 크롤링 데이터 검증...")

git_pages = Path("data/raw/git/pages.json")
python_pages = Path("data/raw/python/pages.json")

if not git_pages.exists():
    print(f"  [WARNING]  Git 크롤링 데이터 없음: {git_pages}")
    print("  -> 먼저 crawler/run_crawl_extended.py를 실행하세요")
    sys.exit(1)

if not python_pages.exists():
    print(f"  [WARNING]  Python 크롤링 데이터 없음: {python_pages}")
    print("  -> 먼저 crawler/run_crawl_extended.py를 실행하세요")
    sys.exit(1)

# Git 문서 확인
with open(git_pages, 'r', encoding='utf-8') as f:
    git_docs = json.load(f)

git_total = len(git_docs)
git_with_url = sum(1 for d in git_docs if d.get('url', 'unknown') != 'unknown')
git_url_pct = (git_with_url / git_total * 100) if git_total > 0 else 0

print(f"\n  [OK] Git 문서:")
print(f"    - 총 문서: {git_total}개")
print(f"    - URL 태깅: {git_with_url}/{git_total} ({git_url_pct:.1f}%)")
if git_docs:
    print(f"    - 샘플 URL: {git_docs[0].get('url', 'NO URL')}")

# Python 문서 확인
with open(python_pages, 'r', encoding='utf-8') as f:
    python_docs = json.load(f)

python_total = len(python_docs)
python_with_url = sum(1 for d in python_docs if d.get('url', 'unknown') != 'unknown')
python_url_pct = (python_with_url / python_total * 100) if python_total > 0 else 0

print(f"\n  [OK] Python 문서:")
print(f"    - 총 문서: {python_total}개")
print(f"    - URL 태깅: {python_with_url}/{python_total} ({python_url_pct:.1f}%)")
if python_docs:
    print(f"    - 샘플 URL: {python_docs[0].get('url', 'NO URL')}")

total_docs = git_total + python_total
total_with_url = git_with_url + python_with_url
total_url_pct = (total_with_url / total_docs * 100) if total_docs > 0 else 0

print(f"\n  [SUMMARY] 총합:")
print(f"    - 총 문서: {total_docs}개")
print(f"    - URL 태깅: {total_with_url}/{total_docs} ({total_url_pct:.1f}%)")

if total_url_pct < 95:
    print(f"\n  [WARNING] URL 태깅률이 95% 미만입니다!")
    print(f"  -> 크롤링을 다시 실행하거나 content_extractor.py를 확인하세요")

# 2. chunks.parquet 검증
print("\n[2/4] 청크 데이터 검증...")

chunks_file = Path("experiments/rag_pipeline/artifacts/chunks.parquet")

if not chunks_file.exists():
    print(f"  [WARNING] 청크 데이터 없음: {chunks_file}")
    print("  -> data_prep.py를 실행하세요:")
    print("     cd experiments/rag_pipeline")
    print("     python data_prep.py --config config/enhanced.yaml")
else:
    try:
        import pandas as pd
        df = pd.read_parquet(chunks_file)

        print(f"  [OK] 청크 파일 존재: {chunks_file}")
        print(f"    - 총 청크: {len(df)}개")
        print(f"    - 컬럼: {list(df.columns)}")

        if 'url' not in df.columns:
            print(f"\n  [ERROR] 'url' 컬럼이 없습니다!")
            print(f"  -> data_prep.py를 다시 실행하세요")
        else:
            url_count = df['url'].notna().sum()
            url_pct = (url_count / len(df) * 100) if len(df) > 0 else 0
            print(f"    - URL 포함: {url_count}/{len(df)} ({url_pct:.1f}%)")

            if len(df) > 0:
                print(f"    - 샘플 청크:")
                print(f"      domain: {df.iloc[0]['domain']}")
                print(f"      length: {df.iloc[0]['length']}")
                print(f"      url: {df.iloc[0]['url']}")
    except ImportError:
        print(f"  [WARNING]  pandas를 설치하세요: pip install pandas pyarrow")
    except Exception as e:
        print(f"  [ERROR] 오류: {e}")

# 3. ChromaDB 검증
print("\n[3/4] ChromaDB 인덱스 검증...")

chroma_dir = Path("experiments/rag_pipeline/artifacts/chroma_db")

if not chroma_dir.exists():
    print(f"  [WARNING]  ChromaDB 인덱스 없음: {chroma_dir}")
    print("  -> index_builder.py를 실행하세요:")
    print("     cd experiments/rag_pipeline")
    print("     python index_builder.py --config config/enhanced.yaml")
else:
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_collection(name="rag_chunks")

        count = collection.count()
        print(f"  [OK] ChromaDB 인덱스 존재: {chroma_dir}")
        print(f"    - 인덱싱된 청크: {count}개")

        # 샘플 확인
        results = collection.get(limit=1, include=["metadatas"])
        if results and results['metadatas']:
            meta = results['metadatas'][0]
            print(f"    - 샘플 메타데이터:")
            print(f"      domain: {meta.get('domain', 'N/A')}")
            print(f"      length: {meta.get('length', 'N/A')}")
            print(f"      url: {meta.get('url', 'N/A')}")

            if 'url' not in meta:
                print(f"\n  [ERROR] 오류: metadata에 'url'이 없습니다!")
                print(f"  -> index_builder.py를 다시 실행하세요")
    except ImportError:
        print(f"  [WARNING]  chromadb를 설치하세요: pip install chromadb")
    except Exception as e:
        print(f"  [ERROR] 오류: {e}")

# 4. 설정 파일 검증
print("\n[4/4] 설정 파일 검증...")

config_file = Path("experiments/rag_pipeline/config/enhanced.yaml")

if not config_file.exists():
    print(f"  [ERROR] 설정 파일 없음: {config_file}")
else:
    try:
        import yaml
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        print(f"  [OK] 설정 파일 존재: {config_file}")
        print(f"    - 도메인: {config['data']['domains']}")
        print(f"    - 청크 크기: {config['chunking']['primary']['chunk_size']}")
        print(f"    - 임베딩 모델: {config['embedding']['model_name']}")
        print(f"    - LLM 모델: {config['llm']['model_name']}")

        # Context Quality Filter 확인
        if config.get('context_quality', {}).get('enabled', False):
            print(f"    - Context Quality Filter: [OK] 활성화")
        else:
            print(f"    - Context Quality Filter: [DISABLED] 비활성화")
    except ImportError:
        print(f"  [WARNING]  PyYAML을 설치하세요: pip install PyYAML")
    except Exception as e:
        print(f"  [ERROR] 오류: {e}")

# 최종 결과
print("\n" + "=" * 80)
print("검증 완료!")
print("=" * 80)

print("\n다음 단계:")
if not git_pages.exists() or not python_pages.exists():
    print("  1. 확장 크롤링 실행:")
    print("     cd crawler")
    print("     python run_crawl_extended.py")
elif not chunks_file.exists():
    print("  1. 데이터 준비:")
    print("     cd experiments/rag_pipeline")
    print("     python data_prep.py --config config/enhanced.yaml")
elif not chroma_dir.exists():
    print("  1. 벡터 인덱싱:")
    print("     cd experiments/rag_pipeline")
    print("     python index_builder.py --config config/enhanced.yaml")
else:
    print("  [OK] 모든 준비 완료!")
    print("\n  테스트 질문:")
    print("     cd experiments/rag_pipeline")
    print("     python answerer_v2_optimized.py \"git rebase와 merge의 차이는?\" --config config/enhanced.yaml")
    print("     python answerer_v2_optimized.py \"Python async/await 사용법은?\" --config config/enhanced.yaml")

print("\n" + "=" * 80)
