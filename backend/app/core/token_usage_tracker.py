"""
Token Usage Tracker
계약서 분석 시 LLM 토큰 사용량 및 비용 추적

사용법:
    tracker = TokenUsageTracker(contract_id="contract_123")
    tracker.record_usage("clause_analyzer", "gpt-4.1-mini", input_tokens=1000, output_tokens=500)
    tracker.record_usage("hyde", "gemini-2.5-flash-lite", input_tokens=500, output_tokens=200)
    summary = tracker.get_summary()
    tracker.save_log()
"""

import os
import json
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from pathlib import Path
from contextlib import contextmanager
import logging

# 로그 디렉토리 설정
TOKEN_LOG_DIR = Path(__file__).parent.parent.parent / "logs" / "token_usage"
TOKEN_LOG_DIR.mkdir(parents=True, exist_ok=True)

# 가격 정보 JSON 파일 경로
PRICING_JSON_PATH = Path(__file__).parent / "model_pricing.json"


def load_model_pricing() -> tuple[dict, dict]:
    """
    JSON 파일에서 모델 가격 정보를 로드합니다.

    Returns:
        tuple: (MODEL_PRICING dict, DEFAULT_PRICING dict)
    """
    try:
        with open(PRICING_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        # provider별 가격을 flat한 딕셔너리로 변환
        pricing = {}

        # OpenAI 모델
        for model, prices in data.get("openai", {}).items():
            pricing[model] = {**prices, "provider": "openai"}

        # Google 모델
        for model, prices in data.get("google", {}).items():
            pricing[model] = {**prices, "provider": "google"}

        # 기본 가격
        default = data.get("default", {"input": 0.50, "output": 2.00, "cached_input": 0.125})
        default["provider"] = "unknown"

        return pricing, default

    except FileNotFoundError:
        logging.warning(f"Pricing file not found: {PRICING_JSON_PATH}, using defaults")
        return {}, {"input": 0.50, "output": 2.00, "cached_input": 0.125, "provider": "unknown"}
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse pricing JSON: {e}")
        return {}, {"input": 0.50, "output": 2.00, "cached_input": 0.125, "provider": "unknown"}


# 모델별 가격 로드
MODEL_PRICING, DEFAULT_PRICING = load_model_pricing()


@dataclass
class LLMCallRecord:
    """단일 LLM 호출 기록"""
    timestamp: str
    module: str  # e.g., "clause_analyzer", "hyde", "crag"
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    cached_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    duration_ms: float = 0.0
    success: bool = True
    error_message: str = ""


@dataclass
class TokenUsageSummary:
    """토큰 사용량 요약"""
    contract_id: str
    analysis_start: str
    analysis_end: str
    total_duration_ms: float

    # 토큰 집계
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_tokens: int = 0
    total_tokens: int = 0

    # 비용 집계
    total_input_cost_usd: float = 0.0
    total_output_cost_usd: float = 0.0
    total_cached_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    total_cost_krw: float = 0.0

    # 호출 집계
    total_llm_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0

    # 모듈별 상세
    by_module: Dict[str, Any] = field(default_factory=dict)
    by_model: Dict[str, Any] = field(default_factory=dict)
    by_provider: Dict[str, Any] = field(default_factory=dict)

    # 개별 호출 기록
    call_records: List[Dict] = field(default_factory=list)


class TokenUsageTracker:
    """
    계약서 분석 세션의 토큰 사용량 추적기

    Thread-safe하게 여러 모듈에서 동시에 사용량을 기록할 수 있음
    """

    # 현재 활성 트래커 (contract_id -> tracker)
    _active_trackers: Dict[str, "TokenUsageTracker"] = {}
    _lock = threading.Lock()

    def __init__(self, contract_id: str, save_to_file: bool = True):
        self.contract_id = contract_id
        self.save_to_file = save_to_file
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.calls: List[LLMCallRecord] = []
        self._call_lock = threading.Lock()

        # 로거 설정
        self.logger = logging.getLogger(f"token_tracker.{contract_id}")
        self.logger.setLevel(logging.INFO)

        # 파일 핸들러
        if save_to_file:
            log_file = TOKEN_LOG_DIR / f"token_{contract_id}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.log_file = log_file

        self.logger.info(f"Token tracking started for contract: {contract_id}")

    @classmethod
    def get_tracker(cls, contract_id: str) -> Optional["TokenUsageTracker"]:
        """활성 트래커 조회"""
        with cls._lock:
            return cls._active_trackers.get(contract_id)

    @classmethod
    def set_active(cls, tracker: "TokenUsageTracker"):
        """활성 트래커 등록"""
        with cls._lock:
            cls._active_trackers[tracker.contract_id] = tracker

    @classmethod
    def remove_active(cls, contract_id: str):
        """활성 트래커 제거"""
        with cls._lock:
            cls._active_trackers.pop(contract_id, None)

    def record_usage(
        self,
        module: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        duration_ms: float = 0.0,
        success: bool = True,
        error_message: str = ""
    ) -> LLMCallRecord:
        """
        LLM 호출 사용량 기록

        Args:
            module: 호출 모듈 (e.g., "clause_analyzer", "hyde")
            model: 모델명 (e.g., "gpt-4.1-mini")
            input_tokens: 입력 토큰 수
            output_tokens: 출력 토큰 수
            cached_tokens: 캐시된 입력 토큰 수
            duration_ms: 호출 소요 시간 (ms)
            success: 성공 여부
            error_message: 에러 메시지

        Returns:
            LLMCallRecord
        """
        # 모델 가격 조회
        pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)
        provider = pricing["provider"]

        # 비용 계산
        regular_input = input_tokens - cached_tokens
        input_cost = (regular_input / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        cached_cost = (cached_tokens / 1_000_000) * pricing.get("cached_input", pricing["input"] * 0.25)
        total_cost = input_cost + output_cost + cached_cost

        record = LLMCallRecord(
            timestamp=datetime.now().isoformat(),
            module=module,
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            input_cost_usd=round(input_cost, 6),
            output_cost_usd=round(output_cost, 6),
            cached_cost_usd=round(cached_cost, 6),
            total_cost_usd=round(total_cost, 6),
            duration_ms=duration_ms,
            success=success,
            error_message=error_message
        )

        with self._call_lock:
            self.calls.append(record)

        # 로그 출력
        status = "OK" if success else "FAIL"
        self.logger.info(
            f"[{module}] {model} | "
            f"in:{input_tokens:,} out:{output_tokens:,} cached:{cached_tokens:,} | "
            f"${total_cost:.6f} | {duration_ms:.0f}ms | {status}"
        )

        if not success and error_message:
            self.logger.error(f"[{module}] Error: {error_message}")

        return record

    def get_summary(self) -> TokenUsageSummary:
        """사용량 요약 생성"""
        self.end_time = datetime.now()
        total_duration = (self.end_time - self.start_time).total_seconds() * 1000

        summary = TokenUsageSummary(
            contract_id=self.contract_id,
            analysis_start=self.start_time.isoformat(),
            analysis_end=self.end_time.isoformat(),
            total_duration_ms=total_duration
        )

        # 모듈/모델/프로바이더별 집계
        by_module: Dict[str, Dict] = {}
        by_model: Dict[str, Dict] = {}
        by_provider: Dict[str, Dict] = {}

        for call in self.calls:
            # 토큰 집계
            summary.total_input_tokens += call.input_tokens
            summary.total_output_tokens += call.output_tokens
            summary.total_cached_tokens += call.cached_tokens

            # 비용 집계
            summary.total_input_cost_usd += call.input_cost_usd
            summary.total_output_cost_usd += call.output_cost_usd
            summary.total_cached_cost_usd += call.cached_cost_usd
            summary.total_cost_usd += call.total_cost_usd

            # 호출 집계
            summary.total_llm_calls += 1
            if call.success:
                summary.successful_calls += 1
            else:
                summary.failed_calls += 1

            # 모듈별 집계
            if call.module not in by_module:
                by_module[call.module] = {
                    "calls": 0, "input_tokens": 0, "output_tokens": 0,
                    "cached_tokens": 0, "total_cost_usd": 0.0, "duration_ms": 0.0
                }
            by_module[call.module]["calls"] += 1
            by_module[call.module]["input_tokens"] += call.input_tokens
            by_module[call.module]["output_tokens"] += call.output_tokens
            by_module[call.module]["cached_tokens"] += call.cached_tokens
            by_module[call.module]["total_cost_usd"] += call.total_cost_usd
            by_module[call.module]["duration_ms"] += call.duration_ms

            # 모델별 집계
            if call.model not in by_model:
                by_model[call.model] = {
                    "calls": 0, "input_tokens": 0, "output_tokens": 0,
                    "cached_tokens": 0, "total_cost_usd": 0.0
                }
            by_model[call.model]["calls"] += 1
            by_model[call.model]["input_tokens"] += call.input_tokens
            by_model[call.model]["output_tokens"] += call.output_tokens
            by_model[call.model]["cached_tokens"] += call.cached_tokens
            by_model[call.model]["total_cost_usd"] += call.total_cost_usd

            # 프로바이더별 집계
            if call.provider not in by_provider:
                by_provider[call.provider] = {
                    "calls": 0, "input_tokens": 0, "output_tokens": 0,
                    "total_cost_usd": 0.0
                }
            by_provider[call.provider]["calls"] += 1
            by_provider[call.provider]["input_tokens"] += call.input_tokens
            by_provider[call.provider]["output_tokens"] += call.output_tokens
            by_provider[call.provider]["total_cost_usd"] += call.total_cost_usd

        # 총계 계산
        summary.total_tokens = summary.total_input_tokens + summary.total_output_tokens
        summary.total_cost_krw = round(summary.total_cost_usd * 1450, 2)  # USD to KRW

        # 비용 반올림
        summary.total_input_cost_usd = round(summary.total_input_cost_usd, 6)
        summary.total_output_cost_usd = round(summary.total_output_cost_usd, 6)
        summary.total_cached_cost_usd = round(summary.total_cached_cost_usd, 6)
        summary.total_cost_usd = round(summary.total_cost_usd, 6)

        # 상세 데이터 할당
        summary.by_module = by_module
        summary.by_model = by_model
        summary.by_provider = by_provider
        summary.call_records = [asdict(c) for c in self.calls]

        return summary

    def save_log(self) -> Optional[str]:
        """JSON 로그 파일 저장"""
        if not self.save_to_file:
            return None

        summary = self.get_summary()

        # JSON 저장
        json_file = TOKEN_LOG_DIR / f"token_{self.contract_id}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(summary), f, ensure_ascii=False, indent=2)

        # 요약 로그 출력
        self.logger.info("=" * 60)
        self.logger.info("TOKEN USAGE SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Contract ID: {self.contract_id}")
        self.logger.info(f"Total Duration: {summary.total_duration_ms:.0f}ms")
        self.logger.info(f"Total LLM Calls: {summary.total_llm_calls} (success: {summary.successful_calls}, fail: {summary.failed_calls})")
        self.logger.info("-" * 60)
        self.logger.info(f"Total Input Tokens: {summary.total_input_tokens:,}")
        self.logger.info(f"Total Output Tokens: {summary.total_output_tokens:,}")
        self.logger.info(f"Total Cached Tokens: {summary.total_cached_tokens:,}")
        self.logger.info(f"Total Tokens: {summary.total_tokens:,}")
        self.logger.info("-" * 60)
        self.logger.info(f"Input Cost: ${summary.total_input_cost_usd:.6f}")
        self.logger.info(f"Output Cost: ${summary.total_output_cost_usd:.6f}")
        self.logger.info(f"Cached Cost: ${summary.total_cached_cost_usd:.6f}")
        self.logger.info(f"TOTAL COST: ${summary.total_cost_usd:.6f} ({summary.total_cost_krw:,.0f} KRW)")
        self.logger.info("-" * 60)
        self.logger.info("BY MODEL:")
        for model, stats in summary.by_model.items():
            self.logger.info(f"  {model}: {stats['calls']} calls, ${stats['total_cost_usd']:.6f}")
        self.logger.info("-" * 60)
        self.logger.info("BY MODULE:")
        for module, stats in summary.by_module.items():
            self.logger.info(f"  {module}: {stats['calls']} calls, ${stats['total_cost_usd']:.6f}")
        self.logger.info("=" * 60)
        self.logger.info(f"Log saved: {json_file}")

        return str(json_file)

    def print_summary(self):
        """콘솔에 요약 출력"""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("TOKEN USAGE SUMMARY")
        print("=" * 60)
        print(f"Contract ID: {self.contract_id}")
        print(f"Duration: {summary.total_duration_ms/1000:.2f}s")
        print(f"LLM Calls: {summary.total_llm_calls}")
        print("-" * 60)
        print(f"Input Tokens:  {summary.total_input_tokens:>10,}")
        print(f"Output Tokens: {summary.total_output_tokens:>10,}")
        print(f"Cached Tokens: {summary.total_cached_tokens:>10,}")
        print(f"Total Tokens:  {summary.total_tokens:>10,}")
        print("-" * 60)
        print(f"TOTAL COST: ${summary.total_cost_usd:.4f} ({summary.total_cost_krw:,.0f} KRW)")
        print("=" * 60 + "\n")

    def __enter__(self):
        TokenUsageTracker.set_active(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save_log()
        TokenUsageTracker.remove_active(self.contract_id)
        # 핸들러 정리
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


@contextmanager
def track_token_usage(contract_id: str, save_to_file: bool = True):
    """
    토큰 사용량 추적 컨텍스트 매니저

    사용법:
        with track_token_usage("contract_123") as tracker:
            # AI 파이프라인 실행
            result = pipeline.analyze(contract)
        # 자동으로 로그 저장됨
    """
    tracker = TokenUsageTracker(contract_id, save_to_file)
    try:
        TokenUsageTracker.set_active(tracker)
        yield tracker
    finally:
        tracker.save_log()
        TokenUsageTracker.remove_active(contract_id)
        for handler in tracker.logger.handlers[:]:
            handler.close()
            tracker.logger.removeHandler(handler)


def get_current_tracker(contract_id: str) -> Optional[TokenUsageTracker]:
    """현재 활성 트래커 조회"""
    return TokenUsageTracker.get_tracker(contract_id)


def record_llm_usage(
    contract_id: str,
    module: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    duration_ms: float = 0.0,
    success: bool = True,
    error_message: str = ""
):
    """
    간편 사용량 기록 함수

    활성 트래커가 있으면 기록, 없으면 무시
    """
    tracker = TokenUsageTracker.get_tracker(contract_id)
    if tracker:
        tracker.record_usage(
            module=module,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            duration_ms=duration_ms,
            success=success,
            error_message=error_message
        )


# 테스트
if __name__ == "__main__":
    print("Testing TokenUsageTracker...")

    with track_token_usage("test_contract_001") as tracker:
        # 시뮬레이션: 여러 모듈에서 LLM 호출
        tracker.record_usage(
            module="clause_analyzer",
            model="gpt-4.1-mini",
            input_tokens=5000,
            output_tokens=1500,
            duration_ms=2500
        )

        tracker.record_usage(
            module="hyde",
            model="gemini-2.5-flash-lite",
            input_tokens=2000,
            output_tokens=800,
            duration_ms=800
        )

        tracker.record_usage(
            module="crag",
            model="gpt-4.1-mini",
            input_tokens=8000,
            output_tokens=2000,
            cached_tokens=3000,
            duration_ms=3000
        )

        tracker.record_usage(
            module="constitutional_ai",
            model="gpt-4.1-mini",
            input_tokens=4000,
            output_tokens=1000,
            duration_ms=1800
        )

        tracker.record_usage(
            module="judge",
            model="gpt-4.1-mini",
            input_tokens=3000,
            output_tokens=500,
            duration_ms=1200
        )

    print("\nTest completed! Check logs/token_usage/ for output files.")
