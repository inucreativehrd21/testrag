#!/usr/bin/env python3
"""
RAG 파이프라인 성능 측정 스크립트

TTFT (Time To First Token): 요청부터 첫 번째 토큰 생성까지의 시간
TPS (Tokens Per Second): 초당 생성되는 토큰 수

사용법:
    python test_performance.py
    python test_performance.py --questions 10
    python test_performance.py --output performance_results.json
    python test_performance.py --compare-models  # 여러 모델 비교
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import tiktoken
from langgraph_rag.graph import create_rag_graph
from langgraph_rag.state import create_initial_state
from langgraph_rag import config as rag_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 비교할 모델 목록
COMPARISON_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",  # GPT-4 Turbo (4.1)
    "gpt-3.5-turbo",
]


class PerformanceMetrics:
    """성능 메트릭 측정 및 저장"""

    def __init__(self, model_name: str = None):
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        self.results = []
        self.current_model = model_name

    def count_tokens(self, text: str) -> int:
        """텍스트의 토큰 수 계산"""
        return len(self.encoding.encode(text))

    def measure_rag_performance(
        self,
        question: str,
        user_id: str = "test_user",
        enable_personalization: bool = True,
        model_name: str = None
    ) -> Dict[str, Any]:
        """
        RAG 파이프라인 성능 측정

        Args:
            question: 사용자 질문
            user_id: 사용자 ID
            enable_personalization: 개인화 활성화 여부
            model_name: 테스트할 LLM 모델 (None이면 config 기본값 사용)

        Returns:
            dict: {
                "question": str,
                "model_name": str,
                "ttft": float,  # Time to first token (초)
                "total_time": float,  # 전체 처리 시간 (초)
                "answer_length": int,  # 답변 길이 (문자)
                "answer_tokens": int,  # 답변 토큰 수
                "tps": float,  # Tokens per second
                "nodes_executed": List[str],  # 실행된 노드 목록
                "node_timings": Dict[str, float],  # 각 노드별 시간
            }
        """
        # 모델 이름 결정
        if model_name is None:
            config = rag_config.get_config()
            model_name = config.llm_model

        logger.info(f"\n{'='*80}")
        logger.info(f"질문 측정 시작: {question}")
        logger.info(f"모델: {model_name}")
        logger.info(f"{'='*80}")

        # 모델 설정 임시 변경
        config = rag_config.get_config()
        original_model = config.config["llm"]["model_name"]
        config.config["llm"]["model_name"] = model_name

        # 초기 상태 생성
        initial_state = create_initial_state(question, user_id=user_id)
        app = create_rag_graph(enable_personalization=enable_personalization)

        # 시작 시간
        start_time = time.time()
        ttft = None
        first_token_received = False
        node_timings = {}
        current_node = None
        node_start_time = start_time

        try:
            final_state = None

            # 그래프 스트리밍 실행
            for state in app.stream(initial_state):
                current_time = time.time()

                for node_name, node_state in state.items():
                    # 노드 실행 시간 기록
                    if current_node is not None:
                        node_timings[current_node] = current_time - node_start_time

                    current_node = node_name
                    node_start_time = current_time

                    # 첫 번째 토큰 시간 측정 (generate 노드에서 답변이 생성되는 시점)
                    if not first_token_received and node_name == "generate":
                        if node_state.get("generation"):
                            ttft = current_time - start_time
                            first_token_received = True
                            logger.info(f"⚡ TTFT: {ttft:.3f}초 (첫 답변 생성)")

                    final_state = node_state

            # 마지막 노드 시간 기록
            if current_node is not None:
                node_timings[current_node] = time.time() - node_start_time

            # 전체 처리 시간
            total_time = time.time() - start_time

            # 답변 분석
            answer = final_state.get("generation", "")
            answer_length = len(answer)
            answer_tokens = self.count_tokens(answer)

            # TPS 계산 (답변 생성 시간 기준)
            # generate 노드부터 종료까지의 시간으로 계산
            generation_time = total_time - (ttft or total_time)
            tps = answer_tokens / generation_time if generation_time > 0 else 0

            # 결과 정리
            result = {
                "question": question,
                "model_name": model_name,
                "ttft": ttft or total_time,  # TTFT가 측정되지 않으면 전체 시간 사용
                "total_time": total_time,
                "answer_length": answer_length,
                "answer_tokens": answer_tokens,
                "tps": tps,
                "nodes_executed": final_state.get("workflow_history", []),
                "node_timings": node_timings,
                "timestamp": datetime.now().isoformat(),
                "personalization_enabled": enable_personalization,
            }

            # 로그 출력
            logger.info(f"\n📊 성능 측정 결과:")
            logger.info(f"  🤖 모델: {model_name}")
            logger.info(f"  ⚡ TTFT: {result['ttft']:.3f}초")
            logger.info(f"  ⏱️  전체 시간: {result['total_time']:.3f}초")
            logger.info(f"  📝 답변 길이: {answer_length}자 ({answer_tokens} 토큰)")
            logger.info(f"  🚀 TPS: {tps:.2f} tokens/sec")
            logger.info(f"  🔄 실행 노드: {' → '.join(result['nodes_executed'])}")

            # 노드별 시간 출력
            logger.info(f"\n📌 노드별 실행 시간:")
            for node, duration in sorted(node_timings.items(), key=lambda x: x[1], reverse=True):
                percentage = (duration / total_time) * 100
                logger.info(f"  • {node}: {duration:.3f}초 ({percentage:.1f}%)")

            self.results.append(result)
            return result

        except Exception as e:
            logger.error(f"❌ 성능 측정 실패: {e}", exc_info=True)
            raise

        finally:
            # 원래 모델로 복원
            config.config["llm"]["model_name"] = original_model

    def run_batch_test(
        self,
        questions: List[str],
        user_id: str = "test_user",
        enable_personalization: bool = True,
        model_name: str = None
    ) -> List[Dict[str, Any]]:
        """여러 질문에 대해 배치 테스트 실행"""
        logger.info(f"\n{'='*80}")
        logger.info(f"배치 테스트 시작: {len(questions)}개 질문")
        if model_name:
            logger.info(f"모델: {model_name}")
        logger.info(f"{'='*80}\n")

        for i, question in enumerate(questions, 1):
            logger.info(f"\n[{i}/{len(questions)}] 테스트 중...")
            try:
                self.measure_rag_performance(question, user_id, enable_personalization, model_name)
            except Exception as e:
                logger.error(f"질문 '{question}' 측정 실패: {e}")
                continue

        return self.results

    def run_model_comparison(
        self,
        questions: List[str],
        models: List[str],
        user_id: str = "test_user",
        enable_personalization: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        여러 모델을 비교 테스트

        Args:
            questions: 테스트할 질문 목록
            models: 비교할 모델 목록
            user_id: 사용자 ID
            enable_personalization: 개인화 활성화 여부

        Returns:
            dict: {model_name: [results]}
        """
        all_results = {}

        logger.info(f"\n{'='*80}")
        logger.info(f"🔬 모델 비교 테스트 시작")
        logger.info(f"  질문 수: {len(questions)}개")
        logger.info(f"  비교 모델: {', '.join(models)}")
        logger.info(f"{'='*80}\n")

        for model_idx, model in enumerate(models, 1):
            logger.info(f"\n{'#'*80}")
            logger.info(f"[{model_idx}/{len(models)}] {model} 테스트 시작")
            logger.info(f"{'#'*80}\n")

            model_results = []

            for q_idx, question in enumerate(questions, 1):
                logger.info(f"\n[질문 {q_idx}/{len(questions)}]")
                try:
                    result = self.measure_rag_performance(
                        question=question,
                        user_id=user_id,
                        enable_personalization=enable_personalization,
                        model_name=model
                    )
                    model_results.append(result)
                except Exception as e:
                    logger.error(f"❌ 질문 '{question}' 측정 실패 (모델: {model}): {e}")
                    continue

            all_results[model] = model_results

            # 각 모델별 중간 요약
            if model_results:
                avg_ttft = sum(r["ttft"] for r in model_results) / len(model_results)
                avg_tps = sum(r["tps"] for r in model_results) / len(model_results)
                avg_total = sum(r["total_time"] for r in model_results) / len(model_results)

                logger.info(f"\n📊 {model} 중간 요약:")
                logger.info(f"  평균 TTFT: {avg_ttft:.3f}초")
                logger.info(f"  평균 TPS: {avg_tps:.2f} tokens/sec")
                logger.info(f"  평균 전체 시간: {avg_total:.3f}초")

        return all_results

    def get_model_comparison_summary(self, comparison_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        모델 비교 결과 요약

        Args:
            comparison_results: run_model_comparison의 반환값

        Returns:
            dict: 모델별 통계 요약
        """
        summary = {}

        logger.info(f"\n{'='*80}")
        logger.info(f"📊 모델 비교 최종 요약")
        logger.info(f"{'='*80}\n")

        for model, results in comparison_results.items():
            if not results:
                continue

            model_summary = {
                "model_name": model,
                "total_questions": len(results),
                "avg_ttft": sum(r["ttft"] for r in results) / len(results),
                "min_ttft": min(r["ttft"] for r in results),
                "max_ttft": max(r["ttft"] for r in results),
                "avg_total_time": sum(r["total_time"] for r in results) / len(results),
                "avg_tps": sum(r["tps"] for r in results) / len(results),
                "avg_answer_tokens": sum(r["answer_tokens"] for r in results) / len(results),
            }

            summary[model] = model_summary

            logger.info(f"🤖 {model}")
            logger.info(f"  질문 수: {model_summary['total_questions']}개")
            logger.info(f"  평균 TTFT: {model_summary['avg_ttft']:.3f}초")
            logger.info(f"  최소 TTFT: {model_summary['min_ttft']:.3f}초")
            logger.info(f"  최대 TTFT: {model_summary['max_ttft']:.3f}초")
            logger.info(f"  평균 전체 시간: {model_summary['avg_total_time']:.3f}초")
            logger.info(f"  평균 TPS: {model_summary['avg_tps']:.2f} tokens/sec")
            logger.info(f"  평균 답변 토큰: {model_summary['avg_answer_tokens']:.0f} tokens")
            logger.info("")

        # 모델 간 비교 표 출력
        logger.info(f"{'='*80}")
        logger.info("🏆 모델 순위 (TTFT 기준)")
        logger.info(f"{'='*80}")

        sorted_models = sorted(summary.items(), key=lambda x: x[1]["avg_ttft"])
        for rank, (model, stats) in enumerate(sorted_models, 1):
            logger.info(f"  {rank}위: {model:20s} | TTFT: {stats['avg_ttft']:.3f}초 | TPS: {stats['avg_tps']:.2f}")

        logger.info(f"{'='*80}\n")

        return summary

    def get_summary_stats(self) -> Dict[str, Any]:
        """전체 결과 통계 요약"""
        if not self.results:
            return {}

        ttfts = [r["ttft"] for r in self.results]
        total_times = [r["total_time"] for r in self.results]
        tpss = [r["tps"] for r in self.results]

        summary = {
            "total_questions": len(self.results),
            "avg_ttft": sum(ttfts) / len(ttfts),
            "min_ttft": min(ttfts),
            "max_ttft": max(ttfts),
            "avg_total_time": sum(total_times) / len(total_times),
            "min_total_time": min(total_times),
            "max_total_time": max(total_times),
            "avg_tps": sum(tpss) / len(tpss),
            "min_tps": min(tpss),
            "max_tps": max(tpss),
        }

        logger.info(f"\n{'='*80}")
        logger.info(f"📊 전체 성능 통계 요약")
        logger.info(f"{'='*80}")
        logger.info(f"  총 질문 수: {summary['total_questions']}개")
        logger.info(f"\n  ⚡ TTFT:")
        logger.info(f"     평균: {summary['avg_ttft']:.3f}초")
        logger.info(f"     최소: {summary['min_ttft']:.3f}초")
        logger.info(f"     최대: {summary['max_ttft']:.3f}초")
        logger.info(f"\n  ⏱️  전체 시간:")
        logger.info(f"     평균: {summary['avg_total_time']:.3f}초")
        logger.info(f"     최소: {summary['min_total_time']:.3f}초")
        logger.info(f"     최대: {summary['max_total_time']:.3f}초")
        logger.info(f"\n  🚀 TPS:")
        logger.info(f"     평균: {summary['avg_tps']:.2f} tokens/sec")
        logger.info(f"     최소: {summary['min_tps']:.2f} tokens/sec")
        logger.info(f"     최대: {summary['max_tps']:.2f} tokens/sec")
        logger.info(f"{'='*80}\n")

        return summary

    def save_results(self, output_path: str, comparison_results: Dict[str, List[Dict[str, Any]]] = None):
        """
        결과를 JSON 파일로 저장

        Args:
            output_path: 저장 경로
            comparison_results: 모델 비교 결과 (선택사항)
        """
        if comparison_results:
            # 모델 비교 결과 저장
            output_data = {
                "comparison_summary": self.get_model_comparison_summary(comparison_results),
                "detailed_results_by_model": comparison_results,
            }
        else:
            # 단일 테스트 결과 저장
            output_data = {
                "summary": self.get_summary_stats(),
                "detailed_results": self.results,
            }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ 결과 저장 완료: {output_path}")


# 테스트용 질문 샘플
DEFAULT_TEST_QUESTIONS = [
    "git rebase와 git merge의 차이는?",
    "Python의 데코레이터는 어떻게 작동하나요?",
    "async/await를 사용한 비동기 프로그래밍 설명해줘",
    "git stash는 언제 사용하나요?",
    "Python의 제너레이터와 이터레이터 차이는?",
    "git cherry-pick 명령어 사용법",
    "Python 리스트 컴프리헨션 예시",
    "git reset --soft, --mixed, --hard 차이",
    "Python의 *args와 **kwargs 설명",
    "git fetch와 git pull 차이점",
]


def main():
    parser = argparse.ArgumentParser(description="RAG 파이프라인 성능 측정")
    parser.add_argument(
        "--questions",
        "-q",
        type=int,
        default=5,
        help="테스트할 질문 개수 (기본값: 5)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="performance_results.json",
        help="결과 저장 파일 경로 (기본값: performance_results.json)"
    )
    parser.add_argument(
        "--no-personalization",
        action="store_true",
        help="개인화 노드 제외"
    )
    parser.add_argument(
        "--custom-questions",
        nargs="+",
        help="커스텀 질문 목록 (공백으로 구분)"
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="여러 모델 비교 테스트 (gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=COMPARISON_MODELS,
        help=f"비교할 모델 목록 (기본값: {' '.join(COMPARISON_MODELS)})"
    )
    args = parser.parse_args()

    # 질문 목록 준비
    if args.custom_questions:
        test_questions = args.custom_questions
    else:
        test_questions = DEFAULT_TEST_QUESTIONS[:args.questions]

    enable_personalization = not args.no_personalization

    # 성능 측정 시작
    metrics = PerformanceMetrics()

    try:
        if args.compare_models:
            # 모델 비교 테스트
            comparison_results = metrics.run_model_comparison(
                questions=test_questions,
                models=args.models,
                user_id="test_user",
                enable_personalization=enable_personalization
            )

            # 결과 저장
            metrics.save_results(args.output, comparison_results=comparison_results)

        else:
            # 단일 모델 테스트
            metrics.run_batch_test(
                questions=test_questions,
                user_id="test_user",
                enable_personalization=enable_personalization
            )

            # 통계 요약
            metrics.get_summary_stats()

            # 결과 저장
            metrics.save_results(args.output)

        logger.info("\n✅ 성능 측정 완료!")

    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
