"""
기본 스크래퍼 클래스
모든 스크래퍼의 공통 기능 제공

알고리즘 개요:
1. RateLimiter: 도메인별 요청 속도 제한 (429 에러 방지)
   - 슬라이딩 윈도우 방식: 최근 1분간 요청 수 추적
   - 지수 백오프: 429 에러 발생 시 대기 시간 2배씩 증가
   - 도메인 분리: 각 도메인마다 독립적인 제한 적용

2. BaseScraper: HTTP 요청 및 재시도 로직
   - Rate Limiter 적용 후 요청
   - 재시도 전략: 지수 백오프 (1초, 2초, 4초, ...)
   - 429 에러 특별 처리: Retry-After 헤더 확인
"""
import requests
import time
from typing import Dict, List, Optional
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from collections import defaultdict
from datetime import datetime, timedelta
from config.settings_extended import CRAWL_CONFIG
from utils.logger import get_logger
from utils.retry_handler import retry_on_error, get_circuit_breaker

logger = get_logger(__name__)


class RateLimiter:
    """
    도메인별 Rate Limiter (요청 속도 제한)

    역할:
    - 각 도메인마다 분당 요청 수 제한 (기본 15개)
    - 429 Too Many Requests 에러 발생 시 자동 백오프
    - 슬라이딩 윈도우 방식으로 정확한 속도 제한

    알고리즘:
    1. 슬라이딩 윈도우:
       - 최근 1분간의 요청 타임스탬프를 리스트로 저장
       - 1분이 지난 타임스탬프는 자동 제거
       - 남은 타임스탬프 개수로 속도 제한 판단

    2. 지수 백오프:
       - 429 에러 발생 시 backoff_count 증가
       - 대기 시간 = base_delay * (2 ^ backoff_count)
       - 성공 시 backoff_count 리셋
    """

    def __init__(self):
        """
        초기화

        자료구조:
        - domain_requests: {도메인: [타임스탬프1, 타임스탬프2, ...]}
          예: {'github.com': [datetime1, datetime2, ...]}
        - backoff_delays: {도메인: 백오프 횟수}
          예: {'github.com': 2}  # 2번 백오프 (대기 시간 = 60 * 2^2 = 240초)
        """
        self.domain_requests = defaultdict(list)  # 도메인별 요청 타임스탬프 리스트
        self.requests_per_minute = CRAWL_CONFIG['rate_limit']['requests_per_minute']  # 분당 최대 요청 수 (기본 15)
        self.backoff_delays = defaultdict(int)  # 도메인별 백오프 횟수 (0부터 시작)
        
    def wait_if_needed(self, url: str):
        """
        필요시 대기 (슬라이딩 윈도우 + 지수 백오프)

        알고리즘:
        1. URL에서 도메인 추출 (예: https://github.com/xxx → github.com)
        2. 슬라이딩 윈도우 업데이트:
           - 현재 시각 기준 1분 이전 타임스탬프 제거
           - 남은 타임스탬프가 최근 1분간 요청 수
        3. Rate limit 체크:
           - 최근 1분간 요청 수 >= 분당 최대 요청 수?
           - Yes → 가장 오래된 요청 이후 1분 경과까지 대기
        4. 백오프 체크:
           - backoff_count > 0?
           - Yes → 지수 백오프 대기 (60 * 2^count, 최대 300초)
        5. 현재 요청 타임스탬프 기록

        Args:
            url: 요청할 URL (도메인 추출용)
        """
        domain = urlparse(url).netloc  # 도메인 추출 (예: github.com)
        now = datetime.now()  # 현재 시각

        # === 1. 슬라이딩 윈도우 업데이트 ===
        # 1분 이전 타임스탬프 제거 (필터링)
        cutoff = now - timedelta(minutes=1)  # 1분 전 시각
        self.domain_requests[domain] = [
            ts for ts in self.domain_requests[domain] if ts > cutoff  # 1분 이내 타임스탬프만 유지
        ]

        # === 2. Rate limit 체크 ===
        # 최근 1분간 요청 수가 한계에 도달했는지 확인
        if len(self.domain_requests[domain]) >= self.requests_per_minute:
            # 슬라이딩 윈도우 대기:
            # 가장 오래된 요청이 1분 경과할 때까지 대기
            oldest = self.domain_requests[domain][0]  # 가장 오래된 타임스탬프
            wait_time = 60 - (now - oldest).total_seconds()  # 1분 - 경과 시간 = 남은 대기 시간
            if wait_time > 0:
                logger.warning(f"Rate limit reached for {domain}. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)  # 대기

        # === 3. 백오프 체크 (429 에러로 인한 추가 대기) ===
        if self.backoff_delays[domain] > 0:
            # 지수 백오프: delay = base * (2 ^ count)
            # 예: count=1 → 60*2=120초, count=2 → 60*4=240초
            backoff_delay = min(
                CRAWL_CONFIG['rate_limit']['backoff_base_delay'] * (2 ** self.backoff_delays[domain]),  # 지수 증가
                CRAWL_CONFIG['rate_limit']['max_backoff_delay']  # 최대값 제한 (5분)
            )
            logger.warning(f"Backoff delay for {domain}: {backoff_delay}s")
            time.sleep(backoff_delay)  # 백오프 대기

        # === 4. 현재 요청 기록 ===
        self.domain_requests[domain].append(now)  # 타임스탬프 추가
    
    def on_429_error(self, url: str):
        """
        429 에러 발생 시 backoff 증가

        알고리즘:
        1. URL에서 도메인 추출
        2. 해당 도메인의 backoff_count를 1 증가
        3. 다음 요청 시 대기 시간이 2배로 증가

        예시:
        - 1번째 429: backoff=1 → 다음 대기 = 60*2^1 = 120초
        - 2번째 429: backoff=2 → 다음 대기 = 60*2^2 = 240초
        - 3번째 429: backoff=3 → 다음 대기 = 60*2^3 = 300초 (최대값)

        Args:
            url: 429 에러가 발생한 URL
        """
        domain = urlparse(url).netloc  # 도메인 추출
        self.backoff_delays[domain] += 1  # 백오프 카운트 증가
        logger.error(f"429 Too Many Requests for {domain}. Backoff level: {self.backoff_delays[domain]}")

    def on_success(self, url: str):
        """
        성공 시 backoff 리셋

        알고리즘:
        1. URL에서 도메인 추출
        2. 요청 성공 시 해당 도메인의 backoff_count를 0으로 리셋
        3. 다음 요청부터는 정상 속도로 진행

        Args:
            url: 성공한 URL
        """
        domain = urlparse(url).netloc  # 도메인 추출
        if self.backoff_delays[domain] > 0:  # 백오프 상태였다면
            logger.info(f"Request successful. Resetting backoff for {domain}")
            self.backoff_delays[domain] = 0  # 백오프 리셋


class BaseScraper:
    """
    기본 스크래퍼 (모든 스크래퍼의 부모 클래스)

    역할:
    - HTTP 요청 및 재시도 로직 제공
    - Rate Limiter 적용
    - 429 에러 특별 처리
    - BeautifulSoup 객체 생성

    특징:
    - 클래스 레벨 RateLimiter 사용 (모든 인스턴스가 공유)
      → 여러 스크래퍼가 동시에 실행되어도 도메인별 제한 공유
    - requests.Session 사용 (연결 재사용으로 성능 향상)
    """

    # 클래스 레벨 Rate Limiter (모든 인스턴스가 공유)
    # 이를 통해 GitScraper, PythonScraper 등이 동시 실행되어도
    # 같은 도메인에 대해서는 하나의 Rate Limiter로 제한됨
    _rate_limiter = RateLimiter()

    def __init__(self):
        """
        초기화

        설정:
        - session: HTTP 연결 재사용 (성능 향상)
        - User-Agent: 봇이 아닌 일반 브라우저로 위장
        - request_delay: 각 요청 후 대기 시간 (2초)
        - max_retries: 최대 재시도 횟수 (5회)
        - timeout: 요청 타임아웃 (30초)
        """
        self.session = requests.Session()  # HTTP 세션 (연결 재사용)
        self.session.headers.update({  # HTTP 헤더 설정
            'User-Agent': CRAWL_CONFIG['user_agent']  # 브라우저로 위장
        })
        self.request_delay = CRAWL_CONFIG['request_delay']  # 기본 딜레이 (2초)
        self.max_retries = CRAWL_CONFIG['max_retries']  # 최대 재시도 (5회)
        self.timeout = CRAWL_CONFIG['timeout']  # 타임아웃 (30초)
    
    def get_html(self, url: str) -> Optional[str]:
        """
        HTML 가져오기 (Rate Limiting + 재시도 + 서킷 브레이커 포함)

        안정화 개선사항:
        - 서킷 브레이커 패턴 적용 (연속 실패 시 빠른 실패)
        - 더 견고한 예외 처리 (메모리 오류, 키보드 인터럽트 등)
        - 재시도 로직 개선 (최대 지연 시간 제한)
        
        알고리즘:
        1. 서킷 브레이커 상태 확인 (OPEN이면 즉시 실패)
        2. Rate Limiter 대기 (필요시)
        3. HTTP GET 요청
        4. 응답 코드 체크:
           - 429 → Retry-After 헤더 확인 후 대기, 재시도
           - 200~299 → 성공, HTML 반환
           - 그 외 → HTTP 에러, 재시도
        5. 재시도 전략: 지수 백오프 (1초, 2초, 4초, 8초, 16초, 최대 60초)
        6. 최대 5회 재시도 후 실패 시 None 반환

        Args:
            url: 크롤링할 URL

        Returns:
            HTML 텍스트 (성공 시) 또는 None (실패 시)
        """
        domain = urlparse(url).netloc
        circuit_breaker = get_circuit_breaker(
            f"scraper_{domain}",
            failure_threshold=5,
            recovery_timeout=60.0
        )
        
        # === 최대 5회 재시도 루프 ===
        for attempt in range(self.max_retries):  # 0, 1, 2, 3, 4
            try:
                # === 0. 서킷 브레이커 체크 ===
                if circuit_breaker.state == 'OPEN':
                    logger.warning(f"서킷 브레이커 OPEN 상태: {domain} - 빠른 실패")
                    return None
                
                # === 1. Rate Limiter 적용 ===
                # 슬라이딩 윈도우 + 백오프 체크
                self._rate_limiter.wait_if_needed(url)

                # === 2. HTTP GET 요청 ===
                logger.info(f"페이지 요청 (시도 {attempt + 1}/{self.max_retries}): {url}")
                response = self.session.get(
                    url,  # 요청 URL
                    timeout=self.timeout,  # 타임아웃 (30초)
                    verify=CRAWL_CONFIG['verify_ssl']  # SSL 인증서 검증
                )

                # === 3. 429 에러 특별 처리 ===
                # Too Many Requests → 서버가 요청 제한 중
                if response.status_code == 429:
                    self._rate_limiter.on_429_error(url)  # 백오프 카운트 증가
                    retry_after = response.headers.get('Retry-After')  # 서버가 지정한 대기 시간

                    if retry_after:
                        # 서버가 대기 시간 지정 → 그대로 따름 (최대 300초)
                        wait_time = min(int(retry_after), 300)
                        logger.warning(f"429 Too Many Requests. Server says wait {wait_time}s")
                        time.sleep(wait_time)
                    else:
                        # 서버가 대기 시간 미지정 → 지수 백오프 적용
                        base_delay = CRAWL_CONFIG['rate_limit']['backoff_base_delay']  # 60초
                        wait_time = min(base_delay * (2 ** attempt), 300)  # 최대 300초
                        logger.warning(f"429 Too Many Requests. Waiting {wait_time}s...")
                        time.sleep(wait_time)

                    continue  # 재시도 (for 루프 처음으로)

                # === 4. HTTP 에러 체크 ===
                # 4xx, 5xx 에러 발생 시 HTTPError 예외 발생
                response.raise_for_status()

                # === 5. 성공 처리 ===
                # 백오프 카운트 리셋 (다음 요청부터 정상 속도)
                self._rate_limiter.on_success(url)
                
                # 서킷 브레이커 성공 기록
                circuit_breaker._on_success()

                # 기본 딜레이 (2초) - 서버 부하 방지
                time.sleep(self.request_delay)

                # HTML 텍스트 반환
                return response.text

            # === 예외 처리 1: HTTP 에러 (4xx, 5xx) ===
            except requests.HTTPError as e:
                circuit_breaker._on_failure()
                logger.warning(f"HTTP 에러 {attempt + 1}/{self.max_retries}: {url} - {e}")
                
                # 404, 403 등은 재시도 불필요 (영구적 오류)
                if response.status_code in [400, 403, 404, 410]:
                    logger.error(f"영구적 HTTP 에러: {response.status_code} - {url}")
                    return None
                
                if attempt < self.max_retries - 1:  # 마지막 시도가 아니면
                    # 지수 백오프: 1초, 2초, 4초, 8초, 16초 (최대 60초)
                    wait_time = min(2 ** attempt, 60)
                    logger.info(f"재시도 전 {wait_time}초 대기...")
                    time.sleep(wait_time)
                else:  # 마지막 시도도 실패
                    logger.error(f"최종 실패 (HTTP 에러): {url}")
                    return None

            # === 예외 처리 2: 네트워크 에러 (타임아웃, 연결 실패 등) ===
            except requests.RequestException as e:
                circuit_breaker._on_failure()
                logger.warning(f"요청 에러 {attempt + 1}/{self.max_retries}: {url} - {e}")
                if attempt < self.max_retries - 1:  # 마지막 시도가 아니면
                    # 지수 백오프: 1초, 2초, 4초, 8초, 16초 (최대 60초)
                    wait_time = min(2 ** attempt, 60)
                    logger.info(f"재시도 전 {wait_time}초 대기...")
                    time.sleep(wait_time)
                else:  # 마지막 시도도 실패
                    logger.error(f"최종 실패 (요청 에러): {url}")
                    return None
            
            # === 예외 처리 3: 치명적 오류 (즉시 실패) ===
            except (MemoryError, KeyboardInterrupt) as e:
                logger.critical(f"치명적 오류 발생: {type(e).__name__} - 즉시 중단")
                raise
            
            # === 예외 처리 4: 예상치 못한 오류 ===
            except Exception as e:
                circuit_breaker._on_failure()
                logger.error(f"예상치 못한 오류 {attempt + 1}/{self.max_retries}: {url} - {type(e).__name__}: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = min(2 ** attempt, 60)
                    logger.info(f"재시도 전 {wait_time}초 대기...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"최종 실패 (예상치 못한 오류): {url}")
                    return None
        
        # 모든 재시도 실패
        return None
    
    def get_soup(self, url: str) -> Optional[BeautifulSoup]:
        """
        BeautifulSoup 객체 생성

        알고리즘:
        1. get_html()로 HTML 텍스트 가져오기
        2. HTML이 있으면 BeautifulSoup로 파싱
        3. lxml 파서 사용 (빠르고 유연함)

        Args:
            url: 파싱할 URL

        Returns:
            BeautifulSoup 객체 (성공 시) 또는 None (실패 시)
        """
        html = self.get_html(url)  # HTML 텍스트 가져오기
        if html:
            return BeautifulSoup(html, 'lxml')  # lxml 파서로 파싱
        return None  # HTML 가져오기 실패
