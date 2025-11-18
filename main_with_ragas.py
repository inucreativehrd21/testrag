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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
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
        self.llm_pipeline = None
        self.llm_tokenizer = None
        self.llm_model = None
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
        
        basic_queries = []
        advanced_queries = []
        
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
        
        advanced_domain_queries = {
            'git': [
                "여러 팀이 동시에 작업하는 저장소에서 Git flow와 trunk 기반 전략을 어떻게 조합하면 충돌을 줄일 수 있을까?",
                "서브모듈과 서브트리 전략을 비교하고, 수백 개 마이크로서비스 저장소를 통합 관리할 때 어떤 접근을 취해야 할까?"
            ],
            'python': [
                "대규모 Python 서비스에서 패키지 버전 충돌과 가상환경 관리를 자동화하려면 어떤 빌드/배포 파이프라인이 필요할까?",
                "데이터 과학 워크로드와 웹 백엔드가 공존할 때 공용 라이브러리의 호환성을 유지하는 방법은?"
            ],
            'docker': [
                "GPU가 필요한 학습 파이프라인과 CPU 위주의 API 서버를 동일 클러스터에서 운영할 때 Docker 리소스 제약을 어떻게 설계해야 할까?",
                "대규모 이미지 빌드를 최적화하고 취약점 스캔을 자동화하기 위한 CI/CD 파이프라인 구성은?"
            ],
            'aws': [
                "AWS 상에서 멀티 AZ/Region 아키텍처를 구성할 때 트래픽 라우팅과 데이터 일관성을 동시에 확보하려면?",
                "대규모 S3 데이터 레이크를 운영하며 권한분리와 비용 최적화를 달성하기 위한 구체적 전략은?"
            ],
            'kubernetes': [
                "Kubernetes에서 StatefulSet과 Operator를 활용해 고가용성 DB를 구성할 때 주의할 장애 시나리오는?",
                "수천 개의 Pod가 있는 클러스터에서 HPA와 VPA를 동시에 운용하며 메모리 스파이크를 완화하는 방법은?"
            ]
        }
        
        general_basic = ["주요 개념", "사용 방법", "장점"]
        general_advanced = [
            "실제 장애 상황을 가정하고 근본 원인 분석부터 복구까지의 절차를 설명해줘.",
            "기존 레거시 시스템과 연동하면서 발생하는 병목과 이를 완화하는 설계 패턴은 무엇일까?"
        ]
        
        for domain in self.crawled_documents.keys():
            if domain in domain_queries:
                basic_queries.extend(domain_queries[domain])
            else:
                basic_queries.append(f"{domain}의 주요 개념은?")
                basic_queries.append(f"{domain} 사용 방법")
            
            if domain in advanced_domain_queries:
                advanced_queries.extend(advanced_domain_queries[domain])
            else:
                advanced_queries.append(f"{domain}을(를) 기존 인프라에 통합할 때 예상되는 병목과 해결 전략은?")
                advanced_queries.append(f"{domain} 기반 서비스를 장애 허용 아키텍처로 설계하려면?")
        
        if not basic_queries:
            basic_queries = general_basic.copy()
        if not advanced_queries:
            advanced_queries = general_advanced.copy()
        
        merged_queries = []
        basic_idx = 0
        advanced_idx = 0
        while basic_idx < len(basic_queries) or advanced_idx < len(advanced_queries):
            if advanced_idx < len(advanced_queries):
                merged_queries.append(advanced_queries[advanced_idx])
                advanced_idx += 1
            if basic_idx < len(basic_queries):
                merged_queries.append(basic_queries[basic_idx])
                basic_idx += 1
        
        sample_size = self.config['evaluation'].get('sample_size', 5)
        self.test_queries = merged_queries[:sample_size]
        logger.info(f"✓ {len(self.test_queries)}개 쿼리 생성")
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

    def _get_reranker(self, model_name: str):
        """리랭커 모델 캐시"""
        if not model_name:
            return None
        if model_name not in self.reranker_models:
            logger.info(f"리랭커 모델 로드 중... ({model_name})")
            self.reranker_models[model_name] = CrossEncoder(
                model_name,
                device=self.device
            )
            logger.info("✓ 리랭커 로드 완료")
        return self.reranker_models[model_name]

    def _rerank_documents(self, query: str, documents: List[str], rerank_model: str,
                          top_k: int) -> List[str]:
        """CrossEncoder 기반 리랭킹"""
        reranker = self._get_reranker(rerank_model)
        if reranker is None or not documents:
            return documents
        pairs = [[query, doc] for doc in documents]
        try:
            scores = reranker.predict(pairs)
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
            reranker = self._get_reranker(model_name)
            if reranker is None:
                continue
            try:
                scores = np.array(reranker.predict(pairs))
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
                keep_k
            )
        else:
            stage1_docs = stage1_docs[:keep_k]
        final_k = stage2_conf.get('final_k', top_k)
        if stage2_conf.get('rerank_model'):
            return self._rerank_documents(
                query,
                stage1_docs,
                stage2_conf.get('rerank_model'),
                final_k
            )
        return stage1_docs[:final_k]

    def _get_llm_pipeline(self):
        """LLM 파이프라인 로드"""
        if self.llm_pipeline is not None:
            return self.llm_pipeline
        model_name = self.llm_config.get('model_name', 'google/flan-t5-base')
        logger.info(f"LLM 로드 중... ({model_name})")
        try:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            device = 0 if self.device == 'cuda' else -1
            self.llm_pipeline = pipeline(
                'text2text-generation',
                model=self.llm_model,
                tokenizer=self.llm_tokenizer,
                device=device
            )
            logger.info("✓ LLM 로드 완료")
        except Exception as e:
            logger.warning(f"LLM 로드 실패: {e}")
            self.llm_pipeline = None
        return self.llm_pipeline

    def _build_llm_prompt(self, query: str, contexts: List[str]) -> str:
        """프롬프트 템플릿 구성"""
        system_prompt = (
            "당신은 15년차 풀스택 엔지니어로서 개발 학습 도우미 멘토를 맡았습니다. "
            "사용자의 질문에 대해 제공된 컨텍스트만으로 정확하고 간결한 답변을 작성하세요. "
            "추정하거나 근거 없는 내용을 포함하지 말고, 필요 시 '자료 부족'이라고 명시하세요."
        )
        context_section = "\n\n".join([f"- {ctx.strip()}" for ctx in contexts])
        instructions = (
            "1. 컨텍스트에서 확인된 사실만 답변에 포함합니다.\n"
            "2. 단계가 필요한 절차는 번호 목록으로 설명합니다.\n"
            "3. Korean으로 답변하되, 코드나 명령어는 원문을 유지합니다."
        )
        prompt = (
            f"{system_prompt}\n\n"
            f"[컨텍스트]\n{context_section}\n\n"
            f"[질문]\n{query}\n\n"
            f"[답변 지침]\n{instructions}\n\n"
            "[최종 답변]"
        )
        return prompt

    def _generate_answer(self, query: str, retrieved_docs: List[str]) -> Tuple[str, str]:
        """LLM을 활용한 최종 답변 생성"""
        if not retrieved_docs:
            return "관련 정보를 찾을 수 없습니다.", ""
        max_docs = self.llm_config.get('max_context_docs', 3)
        context_subset = retrieved_docs[:max_docs]
        prompt = self._build_llm_prompt(query, context_subset)
        llm_pipe = self._get_llm_pipeline()
        if llm_pipe is None:
            return context_subset[0][:200], prompt
        generation_kwargs = {
            'max_new_tokens': self.llm_config.get('max_new_tokens', 256),
            'temperature': self.llm_config.get('temperature', 0.2),
            'top_p': self.llm_config.get('top_p', 0.9),
            'do_sample': self.llm_config.get('temperature', 0.2) > 0
        }
        try:
            outputs = llm_pipe(prompt, **generation_kwargs)
            if outputs and len(outputs) > 0:
                answer = outputs[0]['generated_text'].strip()
            else:
                answer = context_subset[0][:200]
        except Exception as e:
            logger.warning(f"LLM 답변 생성 실패: {e}")
            answer = context_subset[0][:200]
        return answer, prompt
    
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
            llm_answer, llm_prompt = self._generate_answer(query, retrieved_docs)
            
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
            if self.use_ragas and retrieved_docs:
                ragas_metrics = self._evaluate_with_ragas(query, retrieved_docs, llm_answer)
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
                    return self._rerank_documents(query, documents, rerank_model, top_k)
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
