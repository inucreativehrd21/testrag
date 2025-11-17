#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 평가 도구 - RAGAS 벤치마크 통합 버전
크롤링 데이터 자동 로드 + RAGAS 평가 (모든 기능 통합)

사용법:
    python main.py                                      # 기본 (data/raw/ 자동 감지)
    RAG_DATA_DIR=/path/to/data python main.py          # 커스텀 경로
    python main.py --config custom_config.yaml         # 커스텀 설정
    python main.py --no-ragas                          # RAGAS 비활성화 (빠른 실행)
"""

import os
import sys
import json
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import warnings

import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

# PyTorch & ML
import torch
try:
    # LangChain 0.1.x
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # LangChain 0.0.x 호환
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from FlagEmbedding import BGEM3FlagModel
import chromadb

# RAGAS (벤치마크)
try:
    from ragas import evaluate
    from ragas.metrics import (
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("⚠️  RAGAS 미설치. 기본 메트릭만 사용됩니다.")

# 경고 제거
warnings.filterwarnings('ignore')
load_dotenv()

os.environ["CHROMA_TELEMETRY_IMPL"] = "none"

# ============================================================================
# 로깅 설정
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# RAG 평가 엔진 (RAGAS 통합)
# ============================================================================

class RAGEvaluationEngine:
    """크롤링 데이터 기반 RAG 평가 엔진 (RAGAS 벤치마크 포함)"""
    
    def __init__(self, config_path: str = "config.yaml", data_dir: str = None, 
                 use_ragas: bool = True):
        """
        초기화
        
        Args:
            config_path: 설정 YAML 경로
            data_dir: 크롤링 데이터 디렉토리 (None이면 자동 감지)
            use_ragas: RAGAS 벤치마크 사용 여부
        """
        # 설정 로드
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # RAGAS 설정
        self.use_ragas = use_ragas and RAGAS_AVAILABLE
        if use_ragas and not RAGAS_AVAILABLE:
            logger.warning("RAGAS를 사용할 수 없습니다. 기본 메트릭만 사용됩니다.")
        
        # 디바이스 확인
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"✓ Device: {self.device}")
        
        if torch.cuda.is_available():
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        if self.use_ragas:
            logger.info(f"✓ RAGAS 벤치마크: 활성화")
        else:
            logger.info(f"✓ RAGAS 벤치마크: 비활성화 (빠른 모드)")
        
        # 데이터 디렉토리
        self.data_dir = Path(data_dir or os.getenv("RAG_DATA_DIR", "data/raw"))
        
        # 결과 저장소
        self.results = []
        self.ragas_results = []
        self.crawled_documents = {}
        self.test_queries = []
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.embedding_model = None
        # 크롤링 데이터 로드
        self._load_crawled_data()
    
    # ========================================================================
    # 1단계: 크롤링 데이터 로드
    # ========================================================================
    
    def _load_crawled_data(self):
        """
        기존 레포의 크롤링 데이터 자동 로드
        
        지원 구조:
        data/raw/
        ├── git/pages.json
        ├── python/pages.json
        ├── docker/pages.json
        └── aws/pages.json
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"1단계: 크롤링 데이터 로드")
        logger.info(f"{'='*70}")
        logger.info(f"경로: {self.data_dir}")
        
        if not self.data_dir.exists():
            logger.warning(f"크롤링 디렉토리 없음: {self.data_dir}")
            logger.info("샘플 데이터로 진행합니다.")
            self._load_sample_data()
            return
        
        # 각 도메인 폴더 순회
        for domain_dir in sorted(self.data_dir.iterdir()):
            if not domain_dir.is_dir():
                continue
            
            domain_name = domain_dir.name
            
            # pages.json 우선 로드
            pages_file = domain_dir / "pages.json"
            if pages_file.exists():
                self._load_pages_json(domain_name, pages_file)
                continue
            
            # pages.json 없으면 디렉토리의 모든 파일 로드
            self._load_directory_files(domain_name, domain_dir)
        
        if not self.crawled_documents:
            logger.warning("크롤링 데이터를 찾을 수 없습니다. 샘플 데이터 사용.")
            self._load_sample_data()
        else:
            logger.info(f"\n✓ 총 {len(self.crawled_documents)}개 도메인 로드 완료")
            for domain, text in self.crawled_documents.items():
                logger.info(f"  - {domain}: {len(text):,} 글자")
    
    def _load_pages_json(self, domain_name: str, pages_file: Path):
        """pages.json 파일 로드 및 동적 형식 감지"""
        try:
            with open(pages_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            texts = self._extract_texts(data)
            
            if texts:
                combined = "\n\n".join(texts)
                self.crawled_documents[domain_name] = combined
                logger.info(f"✓ {domain_name}: {len(combined):,} 글자 (pages.json)")
        
        except Exception as e:
            logger.warning(f"pages.json 로드 실패 ({domain_name}): {e}")
    
    def _load_directory_files(self, domain_name: str, domain_dir: Path):
        """디렉토리의 모든 파일 로드"""
        all_texts = []
        
        for file in domain_dir.glob("*"):
            if file.name.startswith('.'):
                continue
            
            # TXT 파일
            if file.suffix == '.txt':
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        all_texts.append(f.read())
                except Exception as e:
                    logger.warning(f"파일 로드 실패: {file.name}, {e}")
            
            # JSON 파일
            elif file.suffix == '.json' and file.name != 'pages.json':
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        texts = self._extract_texts(data)
                        all_texts.extend(texts)
                except Exception as e:
                    logger.warning(f"JSON 파일 로드 실패: {file.name}, {e}")
        
        if all_texts:
            combined = "\n\n".join(all_texts)
            self.crawled_documents[domain_name] = combined
            logger.info(f"✓ {domain_name}: {len(combined):,} 글자 (디렉토리 파일)")
    
    def _extract_texts(self, data: Any) -> List[str]:
        """JSON 데이터에서 텍스트 추출 (형식 자동 감지)"""
        texts = []
        
        # 리스트
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    for key in ['content', 'text', 'body', 'page_content']:
                        if key in item and item[key]:
                            texts.append(str(item[key]))
                            break
                    else:
                        if item:
                            texts.append(str(item))
                else:
                    texts.append(str(item))
        
        # 딕셔너리
        elif isinstance(data, dict):
            for key in ['pages', 'documents', 'data', 'items']:
                if key in data and isinstance(data[key], list):
                    for item in data[key]:
                        if isinstance(item, dict):
                            for content_key in ['content', 'text', 'body']:
                                if content_key in item:
                                    texts.append(str(item[content_key]))
                                    break
                        else:
                            texts.append(str(item))
                    break
            else:
                for key in ['content', 'text', 'body']:
                    if key in data:
                        texts.append(str(data[key]))
                        break
        
        else:
            texts.append(str(data))
        
        return [t for t in texts if t and len(t.strip()) > 0]
    
    def _load_sample_data(self):
        """샘플 데이터 로드 (테스트용)"""
        self.crawled_documents = {
            "git": "Git은 분산 버전 관리 시스템입니다. 주요 명령어: git init 저장소 초기화, git add 파일 스테이징, git commit 커밋 생성, git push 원격 푸시, git pull 원격에서 가져오기, git branch 브랜치 관리, git merge 브랜치 병합. Git의 장점은 오프라인 작업 가능, 빠른 속도, 강력한 브랜칭입니다.",
            "python": "Python은 고급 프로그래밍 언어입니다. 특징: 간단한 문법, 동적 타이핑, 풍부한 라이브러리, 크로스 플랫폼. 패키지 관리는 pip를 사용합니다. pip install package_name으로 설치합니다. 가상환경은 python -m venv로 생성합니다. Python은 데이터 과학, 웹 개발, 머신러닝에 널리 사용됩니다."
        }
        logger.info("✓ 샘플 데이터 로드됨 (2개 도메인)")
    
    # ========================================================================
    # 2단계: 테스트 쿼리 생성
    # ========================================================================
    
    def _generate_test_queries(self):
        """로드된 도메인 기반 테스트 쿼리 자동 생성"""
        logger.info(f"\n{'='*70}")
        logger.info(f"2단계: 테스트 쿼리 생성")
        logger.info(f"{'='*70}")
        
        queries = []
        
        # 도메인별 쿼리
        domain_queries = {
            'git': [
                "Git의 주요 명령어는 무엇인가?",
                "Git에서 커밋이란?",
                "브랜치 병합 방법",
                "Git rebase 사용법"
            ],
            'python': [
                "Python의 특징을 설명해줄래?",
                "패키지 설치 방법",
                "가상환경 생성 방법",
                "Python 라이브러리"
            ],
            'docker': [
                "Docker란 무엇인가?",
                "Docker 이미지와 컨테이너",
                "Docker 네트워크",
                "Docker Compose 사용법"
            ],
            'aws': [
                "AWS 주요 서비스",
                "EC2 인스턴스 생성",
                "S3 버킷",
                "Lambda 함수"
            ],
            'kubernetes': [
                "Kubernetes 기본 개념",
                "Pod와 Deployment",
                "Service 종류",
                "Ingress 설정"
            ]
        }
        
        # 로드된 도메인에 맞는 쿼리 추가
        for domain in self.crawled_documents.keys():
            if domain in domain_queries:
                queries.extend(domain_queries[domain])
            else:
                queries.append(f"{domain}의 주요 개념은?")
                queries.append(f"{domain} 사용 방법")
        
        if not queries:
            queries = ["주요 개념", "사용 방법", "장점"]
        
        self.test_queries = queries[:self.config['evaluation'].get('sample_size', 5)]
        logger.info(f"✓ {len(self.test_queries)}개 쿼리 생성")
        for i, q in enumerate(self.test_queries, 1):
            logger.info(f"  {i}. {q}")
    
    # ========================================================================
    # 3단계: 평가 실행
    # ========================================================================
    
    def evaluate(self):
        """전체 평가 실행"""
        self._generate_test_queries()
        
        logger.info(f"\n{'='*70}")
        logger.info(f"3단계: 평가 실행")
        logger.info(f"{'='*70}")
        
        # 전체 조합 수 계산
        total = (
            len(self.config['chunking_strategies']) *
            len(self.config['retrieval_configs']) *
            len(self.test_queries)
        )
        
        logger.info(f"총 조합: {total}개")
        logger.info(f"  - 청킹 전략: {len(self.config['chunking_strategies'])}개")
        logger.info(f"  - 검색 구성: {len(self.config['retrieval_configs'])}개")
        logger.info(f"  - 테스트 쿼리: {len(self.test_queries)}개")
        logger.info(f"  - RAGAS 평가: {'활성화' if self.use_ragas else '비활성화'}")
        logger.info("")
        
        # 전체 크롤링 데이터 결합
        all_text = self._combine_all_documents()
        
        if not all_text or len(all_text.strip()) == 0:
            logger.error("평가할 데이터가 없습니다.")
            return
        
        # 진행 바
        pbar = tqdm(total=total, desc="평가 진행중", unit="조합")
        
        # 모든 조합 평가
        for chunking_name, chunking_config in self.config['chunking_strategies'].items():
            for retrieval_name, retrieval_config in self.config['retrieval_configs'].items():
                for query in self.test_queries:
                    result = self._evaluate_single(
                        all_text,
                        chunking_config,
                        retrieval_config,
                        query
                    )
                    
                    if result:
                        self.results.append(result)
                    
                    pbar.update(1)
        
        pbar.close()
        logger.info(f"✓ {len(self.results)}개 조합 평가 완료\n")
    
    def _combine_all_documents(self) -> str:
        """모든 크롤링 데이터 결합"""
        all_texts = []
        for domain, text in self.crawled_documents.items():
            if text and len(text.strip()) > 0:
                all_texts.append(f"[{domain.upper()}]\n{text}")
        return "\n\n".join(all_texts)
    
    def _evaluate_single(self, text: str, chunking_config: Dict,
                        retrieval_config: Dict, query: str) -> Dict:
        """단일 조합 평가 (기본 메트릭)"""
        try:
            # 1. 청킹
            chunks = self._chunk_text(
                text,
                chunking_config['chunk_size'],
                chunking_config['chunk_overlap']
            )
            
            if not chunks:
                return None
            
            # 2. 임베딩
            dense_embed, sparse_embed = self._embed_chunks(chunks)
            
            # 3. 벡터DB
            collection = self._build_vectordb(chunks, dense_embed)
            
            # 4. 검색
            retrieved_docs = self._retrieve(query, collection, retrieval_config)
            
            # 5. 기본 메트릭
            result = {
                'chunking_strategy': chunking_config['name'],
                'chunk_size': chunking_config['chunk_size'],
                'chunk_overlap': chunking_config['chunk_overlap'],
                'num_chunks': len(chunks),
                'retrieval_config': retrieval_config['name'],
                'retrieval_type': retrieval_config.get('retrieval_type', 'dense'),
                'top_k': retrieval_config.get('top_k', 5),
                'query': query,
                'retrieved_count': len(retrieved_docs),
                'retrieval_score': len(retrieved_docs) / len(chunks) if chunks else 0,
                'retrieved_text': " ".join(retrieved_docs),  # RAGAS용
                'timestamp': datetime.now().isoformat()
            }
            
            # 6. RAGAS 평가 (선택사항)
            if self.use_ragas and retrieved_docs:
                ragas_metrics = self._evaluate_with_ragas(query, retrieved_docs, text)
                result.update(ragas_metrics)
            
            return result
        
        except Exception as e:
            logger.warning(f"평가 오류: {e}")
            return None
    
    def _evaluate_with_ragas(self, query: str, retrieved_docs: List[str], 
                             context: str) -> Dict[str, float]:
        """RAGAS 벤치마크를 사용한 평가"""
        try:
            # RAGAS 평가용 데이터셋 구성
            eval_data = {
                'question': [query],
                'contexts': [[" ".join(retrieved_docs)]],
                'answer': [""]  # 생성 답변 (여기서는 비어있음)
            }
            
            dataset = Dataset.from_dict(eval_data)
            
            # 평가 실행
            scores = evaluate(
                dataset=dataset,
                metrics=[
                    context_precision,
                    context_recall,
                    faithfulness,
                    answer_relevancy
                ]
            )
            
            return {
                'ragas_context_precision': float(scores['context_precision']),
                'ragas_context_recall': float(scores['context_recall']),
                'ragas_faithfulness': float(scores['faithfulness']),
                'ragas_answer_relevancy': float(scores['answer_relevancy'])
            }
        
        except Exception as e:
            logger.warning(f"RAGAS 평가 오류: {e}")
            return {
                'ragas_context_precision': 0.0,
                'ragas_context_recall': 0.0,
                'ragas_faithfulness': 0.0,
                'ragas_answer_relevancy': 0.0
            }
    
    def _chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """문서 청킹"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return splitter.split_text(text)
    
    def _embed_chunks(self, chunks: List[str]) -> Tuple:
        """청크 임베딩"""
        if self.embedding_model is None:
            self.embedding_model = BGEM3FlagModel(
                'BAAI/bge-m3',
                use_fp16=True,
                device=self.device
            )
    
            # 동적 배치 크기 계산 (← 핵심 수정)
            if len(chunks) > 1000:
                batch_size = 8
            elif len(chunks) > 500:
                batch_size = 16
            else:
                batch_size = 32
        
        embeddings = self.embedding_model.encode(
            chunks,
            batch_size=batch_size,
            max_length=512,
            return_dense=True,
            return_sparse=True,
        )
        
        return embeddings['dense_vecs'], embeddings['lexical_weights']
    
    def _build_vectordb(self, chunks: List[str], embeddings: np.ndarray) -> chromadb.Collection:
        try:
            # 텔레메트리 비활성화 상태에서 클라이언트 생성
            client = chromadb.Client()
            try:
                client.delete_collection(name="rag_eval")
            except:
                pass
            collection = client.get_or_create_collection(name="rag_eval")
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings.tolist(),
                metadatas=[{"source": f"chunk_{i}"} for i in range(len(chunks))]
            )
            return collection
        except Exception as e:
            logger.warning(f"ChromaDB 오류: {e}")
            return None
    
    def _retrieve(self, query: str, collection: chromadb.Collection,
                 retrieval_config: Dict) -> List[str]:
        """문서 검색"""
        top_k = retrieval_config.get('top_k', 5)
        
        try:
            results = collection.query(
                query_texts=[query],
                n_results=top_k
            )
            return results['documents'][0][:top_k] if results['documents'] else []
        except Exception as e:
            logger.warning(f"검색 오류: {e}")
            return []
    
    # ========================================================================
    # 4단계: 결과 저장 및 리포트
    # ========================================================================
    
    def save_results(self):
        """결과 저장 (기본 메트릭 + RAGAS)"""
        if not self.results:
            logger.warning("저장할 결과 없음")
            return
        
        logger.info(f"\n{'='*70}")
        logger.info(f"4단계: 결과 저장")
        logger.info(f"{'='*70}")
        
        df = pd.DataFrame(self.results)
        
        # CSV 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = self.results_dir / f"comparison_{timestamp}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"✓ CSV 저장: {csv_path}")
        
        # 요약 통계 (기본 메트릭)
        summary_cols = ['retrieved_count', 'retrieval_score', 'num_chunks']
        summary = df.groupby(['chunking_strategy', 'retrieval_config']).agg({
            'retrieved_count': 'mean',
            'retrieval_score': 'mean',
            'num_chunks': 'first'
        }).round(4)
        
        # RAGAS 메트릭 추가
        ragas_cols = [col for col in df.columns if col.startswith('ragas_')]
        if ragas_cols:
            ragas_summary = df.groupby(['chunking_strategy', 'retrieval_config'])[ragas_cols].mean().round(4)
            summary = pd.concat([summary, ragas_summary], axis=1)
        
        summary_path = self.results_dir / f"summary_{timestamp}.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("RAG 평가 요약 보고서\n")
            f.write("="*70 + "\n\n")
            f.write(f"평가 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"로드된 도메인: {', '.join(self.crawled_documents.keys())}\n")
            f.write(f"총 크롤링 크기: {sum(len(t) for t in self.crawled_documents.values()):,} 글자\n")
            f.write(f"RAGAS 평가: {'활성화' if self.use_ragas else '비활성화'}\n\n")
            f.write(summary.to_string())
            f.write("\n\n" + "="*70 + "\n")
            f.write("상세 통계\n")
            f.write("="*70 + "\n\n")
            f.write(f"총 평가 조합: {len(self.results)}\n")
            f.write(f"청킹 전략: {len(self.config['chunking_strategies'])}\n")
            f.write(f"검색 구성: {len(self.config['retrieval_configs'])}\n")
            f.write(f"테스트 쿼리: {len(set(df['query']))}\n\n")
            
            # 최고 성능
            if 'ragas_context_precision' in df.columns:
                # RAGAS 메트릭 기준 최고 성능
                best_metric = 'ragas_context_precision'
            else:
                # 기본 메트릭 기준
                best_metric = 'retrieval_score'
            
            best_idx = df[best_metric].idxmax()
            best = df.iloc[best_idx]
            f.write("최고 성능 조합:\n")
            f.write(f"  청킹: {best['chunking_strategy']}\n")
            f.write(f"  검색: {best['retrieval_config']}\n")
            f.write(f"  성능 점수: {best[best_metric]:.4f}\n")
        
        logger.info(f"✓ 요약 저장: {summary_path}")
        
        # 콘솔 출력
        print("\n" + "="*70)
        print("평가 결과 요약")
        print("="*70)
        print(summary)
        if 'ragas_context_precision' in df.columns:
            print("\n✓ RAGAS 벤치마크 메트릭:")
            print(f"  - context_precision (평균): {df['ragas_context_precision'].mean():.4f}")
            print(f"  - context_recall (평균): {df['ragas_context_recall'].mean():.4f}")
            print(f"  - faithfulness (평균): {df['ragas_faithfulness'].mean():.4f}")
            print(f"  - answer_relevancy (평균): {df['ragas_answer_relevancy'].mean():.4f}")
        print("\n최고 성능 조합:")
        print(f"  청킹: {best['chunking_strategy']}")
        print(f"  검색: {best['retrieval_config']}")
        print(f"  성능: {best[best_metric]:.4f}")
        print("="*70 + "\n")


# ============================================================================
# 메인 함수
# ============================================================================

def main():
    """메인 실행"""
    parser = argparse.ArgumentParser(description="RAG 평가 도구 (RAGAS 벤치마크 통합)")
    parser.add_argument('--config', default='config.yaml', help='설정 파일 경로')
    parser.add_argument('--data-dir', default=None, help='크롤링 데이터 디렉토리')
    parser.add_argument('--no-ragas', action='store_true', help='RAGAS 평가 비활성화')
    args = parser.parse_args()
    
    try:
        engine = RAGEvaluationEngine(
            args.config, 
            args.data_dir,
            use_ragas=not args.no_ragas
        )
        engine.evaluate()
        engine.save_results()
        
        logger.info("✓ 평가 완료!")
        return 0
    
    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
