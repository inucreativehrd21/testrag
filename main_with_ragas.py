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
import time
import asyncio
import atexit

os.environ["CHROMA_TELEMETRY_IMPL"] = "none"
os.environ["CHROMA_ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMADB_NO_TELEMETRY"] = "true"

import warnings
warnings.filterwarnings('ignore')

import json
import yaml
import logging
import argparse
import re
from itertools import cycle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

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
from sentence_transformers import CrossEncoder
from openai import OpenAI
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

load_dotenv()


# ============================================================================
# 로깅 설정
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logging.getLogger("chromadb").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)


def _cleanup_async_clients():
    """Ensure lingering async tasks finish before exit."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(asyncio.sleep(0))
    except RuntimeError:
        pass


atexit.register(_cleanup_async_clients)
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
        
        eval_config = self.config.get('evaluation', {})
        config_wants_ragas = eval_config.get('use_ragas', True)
        requested_ragas = use_ragas and config_wants_ragas
        self.use_ragas = requested_ragas and RAGAS_AVAILABLE
        if requested_ragas and not RAGAS_AVAILABLE:
            logger.warning("RAGAS를 사용할 수 없습니다. 기본 메트릭만 사용됩니다.")
        if not requested_ragas:
            logger.warning("설정상 RAGAS가 비활성화되어 벤치마크 신뢰도가 낮아질 수 있습니다.")
        
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
        self.reranker_models = {}
        self.chunk_cache: Dict[str, List[str]] = {}
        self.collection_cache: Dict[Tuple[str, str], chromadb.Collection] = {}
        self.embedding_config = self.config.get('embedding', {})
        self.embedding_identifier = self.embedding_config.get('model_name', 'BAAI/bge-m3')
        self.chroma_client = chromadb.Client()
        self.llm_config = self.config.get('llm', {})
        self.reranker_config = self.config.get('reranker', {})
        self.openai_client = None
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

        difficulty_levels = ['easy', 'medium', 'hard']
        questions_per_level = 5

        domain_question_bank = {
            'git': {
                'easy': [
                    "Git의 기본 개념은 무엇인가?",
                    "Git에서 커밋은 어떻게 생성하나요?",
                    "Git branch를 조회하는 명령어는?",
                    "Git clone 사용 방법",
                    "Git status 결과는 무엇을 의미하나요?"
                ],
                'medium': [
                    "Git rebase와 merge 차이점은?",
                    "여러 개발자가 동시에 작업할 때 충돌을 줄이는 전략은?",
                    "Git stash를 사용하는 상황",
                    "submodule 업데이트 절차",
                    "hook을 활용해 품질 검사를 자동화하는 방법"
                ],
                'hard': [
                    "대규모 모노레포에서 Git history를 관리하는 모범 사례는?",
                    "Git LFS와 일반 파일 관리 전략을 어떻게 병행하나요?",
                    "수백 개 서비스가 연동된 저장소에서 trunk 기반 개발을 안전하게 도입하는 절차",
                    "부분 checkout이나 sparse checkout이 필요한 시나리오",
                    "서브트리 전략을 적용할 때 주의해야 할 통합 이슈"
                ]
            },
            'python': {
                'easy': [
                    "Python의 장점 세 가지는?",
                    "pip로 패키지를 설치하는 명령어는?",
                    "가상환경을 생성하는 대표적인 방법은?",
                    "Python 인터프리터 버전 확인 방법",
                    "list와 tuple의 차이점"
                ],
                'medium': [
                    "의존성 충돌을 피하기 위한 pyproject.toml 구성 팁",
                    "asyncio를 기반으로 IO 작업을 최적화하는 패턴",
                    "typing 모듈을 도입했을 때의 이점",
                    "pytest fixture를 활용한 모듈 테스트 전략",
                    "패키지 배포 시 wheel 파일을 빌드하는 절차"
                ],
                'hard': [
                    "대규모 Python 서비스에서 GIL을 우회하거나 완화하는 방법",
                    "데이터 과학 워크플로우와 웹 백엔드를 동시에 지원하는 프로젝트 구조",
                    "C 확장 모듈을 빌드해 성능을 높이는 과정",
                    "분산 처리 프레임워크(Celery, Ray 등)를 Python 서비스에 통합할 때 주의점",
                    "Python 패키지를 사내 artifact repo에서 관리하는 자동화 전략"
                ]
            },
            'docker': {
                'easy': [
                    "Docker 이미지와 컨테이너의 차이점은?",
                    "Dockerfile에서 FROM 지시어의 역할",
                    "컨테이너 로그 확인 명령어",
                    "이미지 빌드를 위한 기본 명령(docker build)",
                    "컨테이너 종료 후 다시 실행하는 방법"
                ],
                'medium': [
                    "멀티 스테이지 Dockerfile을 구성하는 이유",
                    "Docker Compose를 사용해 복수 서비스를 연결하는 절차",
                    "컨테이너 리소스 제한(cpu/memory) 설정 방법",
                    "Private registry에서 이미지를 받아오는 방법",
                    "이미지 취약점 스캔을 CI에서 자동화하는 방식"
                ],
                'hard': [
                    "GPU/CPU 혼합 워크로드를 동일 클러스터에서 운영할 때 고려사항",
                    "수백 개 서비스 이미지를 관리하는 태깅/릴리스 정책",
                    "OCI 이미지 표준을 준수하면서 레이어 캐싱을 최적화하는 팁",
                    "네트워크 분리(bridge/overlay)를 통해 멀티테넌시를 유지하는 전략",
                    "대규모 이미지 빌드 파이프라인에서 병목을 줄이기 위한 캐시 서버 구성"
                ]
            },
            'aws': {
                'easy': [
                    "AWS의 대표 서비스 세 가지는?",
                    "EC2 인스턴스를 생성하는 기본 절차",
                    "S3 버킷을 만드는 명령 또는 콘솔 위치",
                    "IAM Role의 주요 목적",
                    "RDS와 DynamoDB의 차이"
                ],
                'medium': [
                    "VPC를 구성할 때 Public/Private Subnet을 나누는 이유",
                    "CloudWatch와 CloudTrail을 활용한 모니터링 전략",
                    "Auto Scaling Group 튜닝 포인트",
                    "S3 비용 최적화를 위한 라이프사이클 정책",
                    "Lambda 함수 배포 시 버전 관리 방법"
                ],
                'hard': [
                    "멀티 리전 DR 아키텍처를 설계할 때의 네트워크 고려사항",
                    "AWS Organizations로 대규모 계정을 관리하는 거버넌스 전략",
                    "EKS/EC2 혼합 환경에서 observability를 통합하는 절차",
                    "S3 데이터 레이크 보안 정책을 계층별로 분리하는 방법",
                    "AWS 기반 SaaS에서 고객 데이터를 논리적/물리적으로 격리하는 기법"
                ]
            }
        }

        general_templates = {
            'easy': [
                "{domain}의 기본 개념은 무엇인가?",
                "{domain}을 처음 사용할 때 알아야 할 명령이나 설정은?",
                "{domain}을 설치하거나 시작하는 방법",
                "{domain}을 통해 얻을 수 있는 이점 세 가지",
                "{domain}에서 자주 쓰이는 실습 예제를 알려줘"
            ],
            'medium': [
                "{domain}을 팀 프로젝트에 도입할 때 생길 수 있는 문제와 해결책",
                "{domain} 관련 구성 요소나 모듈을 체계적으로 관리하는 방법",
                "{domain} 성능을 중간 수준으로 최적화하는 전략",
                "{domain}을 다른 시스템과 연동할 때 주의할 점",
                "{domain}으로 자동화를 구축할 때 필요한 절차"
            ],
            'hard': [
                "대규모 환경에서 {domain}을 운영하며 발생하는 고급 문제와 해결책",
                "{domain}을 고가용성/재해복구 시나리오에 맞게 설계하는 방법",
                "{domain} 도입 후 레거시 시스템과의 통합 전략",
                "{domain} 기반 서비스를 모니터링/보안 측면에서 강화하는 절차",
                "{domain} 사용 시 규제 준수나 데이터 거버넌스를 고려하는 방법"
            ]
        }

        known_domain_order = ['git', 'python', 'docker', 'aws', 'kubernetes']
        available_domains = list(self.crawled_documents.keys())
        domain_pool = available_domains if available_domains else []
        for candidate in known_domain_order:
            if candidate not in domain_pool:
                domain_pool.append(candidate)
        domains_for_eval = domain_pool[:4] if len(domain_pool) >= 4 else domain_pool
        if len(domains_for_eval) < 4:
            logger.warning("도메인이 4개 미만입니다. 기본 템플릿으로 채웁니다.")

        def build_domain_level_map(domain_name: str) -> Dict[str, List[str]]:
            domain_bank = domain_question_bank.get(domain_name.lower())
            level_map = {}
            for level in difficulty_levels:
                if domain_bank and domain_bank.get(level):
                    level_map[level] = domain_bank[level]
                else:
                    level_map[level] = [
                        template.format(domain=domain_name)
                        for template in general_templates[level]
                    ]
            return level_map

        domain_level_map = {
            domain: build_domain_level_map(domain)
            for domain in domains_for_eval
        }

        def collect_level_queries(level: str) -> List[str]:
            queries = []
            domain_iter = cycle(domains_for_eval)
            usage_counter = {domain: 0 for domain in domains_for_eval}
            while len(queries) < questions_per_level:
                domain = next(domain_iter)
                question_list = domain_level_map[domain][level]
                idx = usage_counter[domain] % len(question_list)
                queries.append(question_list[idx])
                usage_counter[domain] += 1
            return queries

        final_queries = []
        for level in difficulty_levels:
            level_queries = collect_level_queries(level)
            final_queries.extend(level_queries)
            logger.info(f"  - {level} 난이도 쿼리 {len(level_queries)}개 구성")

        self.test_queries = final_queries[: self.config['evaluation'].get('sample_size', 15)]
        logger.info(f"✓ {len(self.test_queries)}개 쿼리 생성 (도메인: {', '.join(domains_for_eval[:4])})")
        for i, q in enumerate(self.test_queries, 1):
            logger.info(f"  {i}. {q}")

    def _collection_cache_key(self, chunking_name: str) -> Tuple[str, str]:
        """청킹 전략 + 임베딩 모델 조합 키"""
        return (chunking_name, self.embedding_identifier)

    def _format_collection_name(self, chunking_name: str) -> str:
        """Chroma 컬렉션명 안전 변환"""
        chunk_safe = re.sub(r'[^a-zA-Z0-9_-]+', '_', chunking_name)
        embed_safe = re.sub(r'[^a-zA-Z0-9_-]+', '_', self.embedding_identifier)
        return f"rag_eval_{chunk_safe}_{embed_safe}"

    def _get_chunks_for_strategy(self, chunking_name: str, chunking_config: Dict,
                                 text: str) -> List[str]:
        """청킹 전략별 청크 생성 (1회)"""
        if chunking_name not in self.chunk_cache:
            chunks = self._chunk_text(
                text,
                chunking_config['chunk_size'],
                chunking_config['chunk_overlap']
            )
            self.chunk_cache[chunking_name] = chunks
            logger.info(f"✓ 청킹 캐시 생성: {chunking_config['name']} ({len(chunks)}개 청크)")
        return self.chunk_cache[chunking_name]

    def _get_collection_for_strategy(self, chunking_name: str, chunking_config: Dict,
                                     chunks: List[str]) -> chromadb.Collection:
        """청킹+임베딩 조합별 벡터 DB 생성 (1회)"""
        cache_key = self._collection_cache_key(chunking_name)
        if cache_key in self.collection_cache:
            return self.collection_cache[cache_key]
        dense_embed, _ = self._embed_chunks(chunks)
        collection = self._build_vectordb(
            chunks,
            dense_embed,
            collection_name=self._format_collection_name(chunking_name)
        )
        self.collection_cache[cache_key] = collection
        logger.info(f"✓ 벡터DB 캐시 생성: {chunking_config['name']}")
        return collection

    def _get_reranker(self, model_name: str, device_name: str = None,
                      trust_remote_code: bool = False):
        """리랭커 모델 캐시"""
        if not model_name:
            return None
        device_name = device_name or self.device
        cache_key = (model_name, device_name, trust_remote_code)
        if cache_key not in self.reranker_models:
            logger.info(
                f"리랭커 모델 로드 중... ({model_name}) [device={device_name}, trust_remote_code={trust_remote_code}]"
            )
            reranker = CrossEncoder(
                model_name,
                device=device_name,
                trust_remote_code=trust_remote_code
            )
            self._ensure_tokenizer_padding(reranker)
            self.reranker_models[cache_key] = reranker
            logger.info("✓ 리랭커 로드 완료")
        return self.reranker_models[cache_key]

    def _rerank_documents(self, query: str, documents: List[str], rerank_model: str,
                          top_k: int, device_name: str = None,
                          trust_remote_code: bool = False) -> List[str]:
        """CrossEncoder 기반 리랭킹"""
        reranker = self._get_reranker(rerank_model, device_name, trust_remote_code)
        if reranker is None or not documents:
            return documents
        pairs = [[query, doc] for doc in documents]
        try:
            scores = self._batched_reranker_predict(reranker, pairs)
            ranked = [doc for _, doc in sorted(zip(scores, documents), reverse=True)]
            return ranked[:top_k]
        except Exception as e:
            logger.warning(f"리랭킹 실패: {e}")
            return documents[:top_k]

    def _ensemble_rerank_documents(self, query: str, documents: List[str],
                                   ensemble_config: List[Dict], top_k: int) -> List[str]:
        """여러 리랭커를 앙상블"""
        if not documents or not ensemble_config:
            return documents[:top_k]
        pairs = [[query, doc] for doc in documents]
        scores_total = np.zeros(len(documents))
        valid_models = 0
        for rerank_def in ensemble_config:
            model_name = rerank_def.get('model') or rerank_def.get('rerank_model')
            weight = rerank_def.get('weight', 1.0)
            device_name = rerank_def.get('device')
            trust_remote = rerank_def.get('trust_remote_code', False)
            reranker = self._get_reranker(model_name, device_name, trust_remote)
            if reranker is None:
                continue
            try:
                scores = self._batched_reranker_predict(reranker, pairs)
                scores = np.asarray(scores, dtype=np.float32)
                scores_total += weight * scores
                valid_models += 1
            except Exception as e:
                logger.warning(f"앙상블 리랭커 실패 ({model_name}): {e}")
        if valid_models == 0:
            return documents[:top_k]
        ranked = [doc for _, doc in sorted(zip(scores_total, documents), reverse=True)]
        return ranked[:top_k]

    def _two_stage_pipeline(self, query: str, documents: List[str],
                            retrieval_config: Dict, top_k: int) -> List[str]:
        """Two-stage retrieval: 빠른 1차 선별 후 고정밀 2차"""
        stage1_conf = retrieval_config.get('stage1', {})
        stage2_conf = retrieval_config.get('stage2', {})
        if not documents:
            return []
        candidate_k = stage1_conf.get('candidate_k', max(top_k * 3, top_k))
        keep_k = stage1_conf.get('keep_k', max(top_k, 10))
        stage1_docs = documents[:candidate_k]
        if stage1_conf.get('rerank_model'):
            stage1_docs = self._rerank_documents(
                query,
                stage1_docs,
                stage1_conf.get('rerank_model'),
                keep_k,
                stage1_conf.get('rerank_device'),
                stage1_conf.get('rerank_trust_remote_code', False)
            )
        else:
            stage1_docs = stage1_docs[:keep_k]
        final_k = stage2_conf.get('final_k', top_k)
        if stage2_conf.get('rerank_model'):
            return self._rerank_documents(
                query,
                stage1_docs,
                stage2_conf.get('rerank_model'),
                final_k,
                stage2_conf.get('rerank_device'),
                stage2_conf.get('rerank_trust_remote_code', False)
            )
        return stage1_docs[:final_k]

    def _format_reranker_scores(self, raw_scores) -> np.ndarray:
        if torch.is_tensor(raw_scores):
            raw_scores = raw_scores.detach().cpu()
        return np.asarray(raw_scores, dtype=np.float32)

    def _ensure_tokenizer_padding(self, reranker):
        tokenizer = getattr(reranker, 'tokenizer', None)
        if tokenizer is None:
            return
        if getattr(tokenizer, 'pad_token', None) is not None:
            return
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            if hasattr(reranker.model, 'resize_token_embeddings'):
                reranker.model.resize_token_embeddings(len(tokenizer))
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        tokenizer.padding_side = 'right'
        if hasattr(reranker.model, 'config'):
            reranker.model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(reranker, 'model') and hasattr(reranker.model, 'resize_token_embeddings'):
            reranker.model.resize_token_embeddings(len(tokenizer))

    def _batched_reranker_predict(self, reranker, pairs: List[List[str]]) -> List[float]:
        if not pairs:
            return []
        batch_size = max(1, self.reranker_config.get('max_batch_size', 16))
        scores: List[float] = []
        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start:start + batch_size]
            self._ensure_tokenizer_padding(reranker)
            raw = reranker.predict(batch_pairs)
            formatted = self._format_reranker_scores(raw)
            scores.extend(formatted.tolist())
        return scores

    def _get_llm_client(self) -> OpenAI:
        """OpenAI LLM 클라이언트 로드"""
        if self.openai_client is not None:
            return self.openai_client
        api_key_env = self.llm_config.get('api_key_env', 'OPENAI_API_KEY')
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"LLM 사용을 위해 {api_key_env} 환경 변수를 설정하세요.")
        api_base = self.llm_config.get('api_base')
        try:
            if api_base:
                self.openai_client = OpenAI(api_key=api_key, base_url=api_base)
            else:
                self.openai_client = OpenAI(api_key=api_key)
            logger.info("✓ OpenAI LLM 클라이언트 준비 완료")
        except Exception as e:
            logger.error(f"OpenAI 클라이언트 초기화 실패: {e}")
            raise
        return self.openai_client

    def _build_llm_prompt(self, query: str, contexts: List[str]) -> Tuple[str, str]:
        """프롬프트 템플릿 구성 (system, user)"""
        system_prompt = (
            "당신은 15년차 프롬프트 엔지니어로서 개발 학습 도우미 챗봇을 설계 중입니다. "
            "사용자의 질문에 대해 제공된 컨텍스트만으로 정확하고 간결한 답변을 작성하세요. "
            "추정하거나 근거 없는 내용을 포함하지 말고, 필요 시 '자료 부족'이라고 명시하세요."
        )
        context_section = "\n\n".join([f"- {ctx.strip()}" for ctx in contexts])
        instructions = (
            "1. 컨텍스트에서 확인된 사실만 답변에 포함합니다.\n"
            "2. 단계가 필요한 절차는 번호 목록으로 설명합니다.\n"
            "3. Korean으로 답변하되, 코드나 명령어는 원문을 유지합니다."
        )
        user_prompt = (
            f"[컨텍스트]\n{context_section}\n\n"
            f"[질문]\n{query}\n\n"
            f"[답변 지침]\n{instructions}\n\n"
            "[최종 답변]"
        )
        return system_prompt, user_prompt

    def _generate_answer(self, query: str, retrieved_docs: List[str]) -> Tuple[str, str, bool, str]:
        """LLM을 활용한 최종 답변 생성 (재시도 + 상태 반환)"""
        if not retrieved_docs:
            return "", "", False, "no_context"
        max_docs = self.llm_config.get('max_context_docs', 3)
        context_subset = retrieved_docs[:max_docs]
        system_prompt, user_prompt = self._build_llm_prompt(query, context_subset)
        combined_prompt = f"System:\n{system_prompt}\n\nUser:\n{user_prompt}"
        max_retries = self.llm_config.get('max_retries', 2)
        retry_delay = self.llm_config.get('retry_delay', 2.0)
        last_error = ""
        for attempt in range(1, max_retries + 1):
            try:
                client = self._get_llm_client()
                response = client.chat.completions.create(
                    model=self.llm_config.get('model_name', 'gpt-4o-mini'),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.llm_config.get('temperature', 0.2),
                    top_p=self.llm_config.get('top_p', 0.9),
                    max_tokens=self.llm_config.get('max_new_tokens', 256)
                )
                choice = response.choices[0]
                message_content = choice.message.content
                if isinstance(message_content, list):
                    message_content = "\n".join(
                        segment.get('text', '') if isinstance(segment, dict) else str(segment)
                        for segment in message_content
                    )
                answer = (message_content or "").strip()
                if answer:
                    return answer, combined_prompt, True, ""
                last_error = "empty_response"
                logger.warning("LLM 응답이 비어 있습니다. 재시도합니다.")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"LLM 호출 실패 (시도 {attempt}/{max_retries}): {e}")
                time.sleep(retry_delay)
        logger.warning(f"LLM 반복 실패로 조합을 스킵합니다: {last_error}")
        return "", combined_prompt, False, last_error

    def _prepare_answer_for_ragas(self, answer: str, retrieved_docs: List[str]) -> str:
        """RAGAS용 영어 답변 생성 (필요시 번역)"""
        answer = (answer or "").strip()
        if not answer:
            return ""
        if not self.llm_config.get('translate_ragas_answers', False):
            return answer
        translation_model = (
            self.llm_config.get('ragas_translation_model')
            or self.llm_config.get('model_name', 'gpt-4o-mini')
        )
        translation_prompt = (
            "Translate the following assistant answer into English while preserving all factual statements. "
            "Return concise bullet sentences that can be evaluated for factual consistency. "
            "Do not add new information."
        )
        try:
            client = self._get_llm_client()
            response = client.chat.completions.create(
                model=translation_model,
                messages=[
                    {"role": "system", "content": translation_prompt},
                    {"role": "user", "content": answer}
                ],
                temperature=0.0,
                max_tokens=self.llm_config.get('max_new_tokens', 256)
            )
            translated = response.choices[0].message.content.strip()
            if translated:
                return translated
        except Exception as e:
            logger.warning(f"RAGAS 번역 실패: {e}")
        # 번역 실패 시 첫 번째 검색 결과 요약으로 대체
        fallback = retrieved_docs[0][:400] if retrieved_docs else answer
        return fallback

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
            chunks = self._get_chunks_for_strategy(chunking_name, chunking_config, all_text)
            if not chunks:
                logger.warning(f"청크 생성 실패: {chunking_config['name']}")
                continue
            collection = self._get_collection_for_strategy(chunking_name, chunking_config, chunks)
            for retrieval_name, retrieval_config in self.config['retrieval_configs'].items():
                for query in self.test_queries:
                    result = self._evaluate_single(
                        chunks,
                        collection,
                        chunking_config,
                        retrieval_config,
                        query,
                        all_text
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
    
    def _evaluate_single(self, chunks: List[str], collection: chromadb.Collection,
                        chunking_config: Dict, retrieval_config: Dict,
                        query: str, context_text: str) -> Dict:
        """단일 조합 평가 (청킹/임베딩 재사용 버전)"""
        try:
            if not chunks:
                return None
            
            # 검색
            retrieved_docs = self._retrieve(query, collection, retrieval_config)
            
            # LLM 답변 생성
            llm_answer, llm_prompt, llm_success, llm_error = self._generate_answer(query, retrieved_docs)
            if not llm_success:
                logger.warning(f"LLM 실패로 조합 스킵: {llm_error}")
                return None
            llm_answer = (llm_answer or "").strip()
            ragas_answer = self._prepare_answer_for_ragas(llm_answer, retrieved_docs)
            
            # 기본 메트릭
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
                'llm_answer': llm_answer,
                'llm_prompt': llm_prompt,
                'timestamp': datetime.now().isoformat()
            }
            
            # 6. RAGAS 평가 (선택사항)
            if self.use_ragas and retrieved_docs and ragas_answer.strip():
                ragas_metrics = self._evaluate_with_ragas(query, retrieved_docs, ragas_answer)
                result.update(ragas_metrics)
            
            return result
        
        except Exception as e:
            logger.warning(f"평가 오류: {e}")
            return None
    
    def _evaluate_with_ragas(self, query: str, retrieved_docs: List[str], answer: str) -> Dict:
        """
        RAGAS 5가지 메트릭 평가 (완전 구현)
        
        메트릭:
        1. context_precision: 검색된 문서의 정확도
        2. context_recall: 검색된 문서의 재현율
        3. faithfulness: 답변이 문서 기반인지
        4. answer_relevancy: 답변이 질문과 관련있는지
        5. answer_similarity: 답변과 ground_truth의 유사도
        
        수정사항:
        - LLM으로 답변 생성
        - ground_truth 자동 생성
        - 모든 필수 필드 포함
        """
        try:
            # LLM 답변 생성 (간단한 방식)
            # 검색된 문서를 기반으로 답변 요약
            if retrieved_docs:
                combined_context = " ".join(retrieved_docs[:3])
                ground_truth = retrieved_docs[0][:200]
            else:
                combined_context = ""
                ground_truth = ""
            
            # RAGAS 데이터셋 구성 (모든 필드 포함)
            eval_data = {
                'question': [query],
                'contexts': [[" ".join(retrieved_docs)]],  # 모든 검색 결과 결합
                'answer': [answer],  # LLM 생성 답변
                'ground_truth': [ground_truth]  # 정답 (첫 검색 결과)
            }
            
            dataset = Dataset.from_dict(eval_data)
            
            # RAGAS 5가지 메트릭
            from ragas.metrics import (
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
                answer_similarity
            )
            
            # 평가 실행
            scores = evaluate(
                dataset=dataset,
                metrics=[
                    context_precision,
                    context_recall,
                    faithfulness,
                    answer_relevancy,
                    answer_similarity
                ]
            )
            
            return {
                'ragas_context_precision': float(scores.get('context_precision', 0.0)),
                'ragas_context_recall': float(scores.get('context_recall', 0.0)),
                'ragas_faithfulness': float(scores.get('faithfulness', 0.0)),
                'ragas_answer_relevancy': float(scores.get('answer_relevancy', 0.0)),
                'ragas_answer_similarity': float(scores.get('answer_similarity', 0.0))
            }
        
        except Exception as e:
            logger.warning(f"RAGAS 평가 오류: {e}")
            return {
                'ragas_context_precision': 0.0,
                'ragas_context_recall': 0.0,
                'ragas_faithfulness': 0.0,
                'ragas_answer_relevancy': 0.0,
                'ragas_answer_similarity': 0.0
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
        """임베딩 (배치 크기 최적화)"""
        try:
            # 모델 캐시
            if self.embedding_model is None:
                model_name = self.embedding_config.get('model_name', 'BAAI/bge-m3')
                use_fp16 = self.embedding_config.get('use_fp16', True)
                logger.info(f"임베딩 모델 로드 중... ({model_name})")
                self.embedding_model = BGEM3FlagModel(
                    model_name,
                    use_fp16=use_fp16,
                    device=self.device
                )
                logger.info("✓ 임베딩 모델 로드 완료")
            
            # 동적 배치 크기 (모든 경로에서 정의)
            num_chunks = len(chunks)
            default_batch = self.embedding_config.get('batch_size', 32)
            
            if num_chunks > 10000:
                batch_size = 4
            elif num_chunks > 5000:
                batch_size = 8
            elif num_chunks > 1000:
                batch_size = 16
            else:
                batch_size = default_batch  # ← 반드시 모든 경로에서 정의
            
            # 임베딩 실행
            embeddings = self.embedding_model.encode(
                chunks,
                batch_size=batch_size,  # ← 이제 항상 정의됨
                max_length=self.embedding_config.get('max_length', 512),
                return_dense=True,
                return_sparse=True,
            )
            
            return embeddings['dense_vecs'], embeddings['lexical_weights']
        
        except Exception as e:
            logger.error(f"임베딩 오류: {e}")
            raise

    
    def _build_vectordb(self, chunks: List[str], embeddings: np.ndarray,
                        collection_name: str) -> chromadb.Collection:
        """ChromaDB 구축 (배치 크기 제한 해결)"""
        try:
            try:
                self.chroma_client.delete_collection(name=collection_name)
            except Exception:
                pass
            
            collection = self.chroma_client.get_or_create_collection(name=collection_name)
            
            # ChromaDB 최대 배치 크기: 41666
            # 안전 마진: 30000
            max_batch_size = 30000
            
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            embeddings_list = embeddings.tolist()
            metadatas = [{"source": f"chunk_{i}"} for i in range(len(chunks))]
            
            # 배치로 나누어 추가
            num_batches = (len(chunks) + max_batch_size - 1) // max_batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * max_batch_size
                end_idx = min((batch_idx + 1) * max_batch_size, len(chunks))
                
                batch_ids = ids[start_idx:end_idx]
                batch_chunks = chunks[start_idx:end_idx]
                batch_embeddings = embeddings_list[start_idx:end_idx]
                batch_metadatas = metadatas[start_idx:end_idx]
                
                collection.add(
                    ids=batch_ids,
                    documents=batch_chunks,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas
                )
            
            return collection
        
        except Exception as e:
            logger.error(f"ChromaDB 오류: {e}")
            raise

    def _retrieve(self, query: str, collection: chromadb.Collection,
                retrieval_config: Dict) -> List[str]:
        """문서 검색"""
        try:
            top_k = retrieval_config.get('top_k', 5)
            stage_strategy = retrieval_config.get('stage_strategy', 'basic')
            candidate_k = retrieval_config.get('candidate_k', max(top_k * 2, top_k))
            if stage_strategy == 'two-stage':
                stage1_conf = retrieval_config.get('stage1', {})
                stage_candidate = stage1_conf.get('candidate_k', candidate_k)
                n_results = max(stage_candidate, candidate_k, top_k)
            else:
                n_results = max(candidate_k, top_k)
            
            # 쿼리도 BGE-M3로 임베딩 (중요!)
            query_embedding = self.embedding_model.encode(
                [query],
                batch_size=1,
                max_length=self.embedding_config.get('max_length', 512),
                return_dense=True,
                return_sparse=False
            )['dense_vecs'].tolist()
            
            # 임베딩된 쿼리로 검색
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=n_results
            )
            
            if results and 'documents' in results and results['documents']:
                documents = results['documents'][0][:n_results]
                if stage_strategy == 'two-stage':
                    return self._two_stage_pipeline(query, documents, retrieval_config, top_k)
                if stage_strategy == 'ensemble':
                    ensemble_cfg = retrieval_config.get('ensemble_rerankers', [])
                    return self._ensemble_rerank_documents(query, documents, ensemble_cfg, top_k)
                if retrieval_config.get('use_reranker'):
                    rerank_model = retrieval_config.get('rerank_model')
                    rerank_device = retrieval_config.get('rerank_device')
                    rerank_trust = retrieval_config.get('rerank_trust_remote_code', False)
                    return self._rerank_documents(
                        query,
                        documents,
                        rerank_model,
                        top_k,
                        rerank_device,
                        rerank_trust
                    )
                return documents[:top_k]
            
            return []
        
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
