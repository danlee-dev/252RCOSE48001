"""
Pipeline Logger - 분석 파이프라인 단계별 로깅

각 AI 모듈의 실행 결과를 파일로 저장하여 디버깅 지원
"""

import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path
from dataclasses import dataclass, asdict, field
import traceback

# 로그 디렉토리 설정
LOG_DIR = Path(__file__).parent.parent.parent / "logs" / "pipeline"
LOG_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class StepLog:
    """단계별 로그 데이터"""
    step_name: str
    status: str  # "started", "success", "error"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_ms: float = 0.0
    input_summary: str = ""
    output_summary: str = ""
    error_message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class PipelineLogger:
    """
    파이프라인 단계별 로거

    사용법:
        logger = PipelineLogger(contract_id="contract_123")
        logger.log_step("HyDE", "started", input_summary="Query: ...")
        # ... 처리 ...
        logger.log_step("HyDE", "success", output_summary="Generated 3 docs", duration_ms=150.5)
        logger.save()
    """

    def __init__(self, contract_id: str, save_to_file: bool = True):
        """
        Args:
            contract_id: 계약서 ID
            save_to_file: 파일 저장 여부
        """
        self.contract_id = contract_id
        self.save_to_file = save_to_file
        self.start_time = datetime.now()
        self.steps: list[StepLog] = []

        # Python 로거 설정
        self.logger = logging.getLogger(f"pipeline.{contract_id}")
        self.logger.setLevel(logging.DEBUG)

        # 파일 핸들러 추가
        if save_to_file:
            log_file = LOG_DIR / f"{contract_id}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.log_file = log_file
        else:
            self.log_file = None

    def log_step(
        self,
        step_name: str,
        status: str,
        input_summary: str = "",
        output_summary: str = "",
        duration_ms: float = 0.0,
        error_message: str = "",
        details: Dict[str, Any] = None
    ):
        """
        단계 로그 기록

        Args:
            step_name: 단계 이름 (e.g., "HyDE", "RAPTOR", "CRAG")
            status: 상태 ("started", "success", "error")
            input_summary: 입력 요약
            output_summary: 출력 요약
            duration_ms: 처리 시간 (ms)
            error_message: 에러 메시지
            details: 추가 세부 정보
        """
        step = StepLog(
            step_name=step_name,
            status=status,
            input_summary=input_summary,
            output_summary=output_summary,
            duration_ms=duration_ms,
            error_message=error_message,
            details=details or {}
        )
        self.steps.append(step)

        # 콘솔/파일 로그
        if status == "started":
            self.logger.info(f"[{step_name}] Started - {input_summary[:100] if input_summary else ''}")
        elif status == "success":
            self.logger.info(f"[{step_name}] Success ({duration_ms:.1f}ms) - {output_summary[:200] if output_summary else ''}")
        elif status == "error":
            self.logger.error(f"[{step_name}] Error ({duration_ms:.1f}ms) - {error_message}")

    def log_error(self, step_name: str, exception: Exception, duration_ms: float = 0.0):
        """예외 로깅"""
        error_msg = str(exception)
        tb = traceback.format_exc()

        self.log_step(
            step_name=step_name,
            status="error",
            error_message=error_msg,
            duration_ms=duration_ms,
            details={"traceback": tb}
        )
        self.logger.error(f"[{step_name}] Traceback:\n{tb}")

    def log_llm_call(
        self,
        step_name: str,
        model: str,
        prompt: str,
        response: str,
        temperature: float = 0.0,
        max_tokens: int = 0,
        duration_ms: float = 0.0,
        extra: Dict[str, Any] = None
    ):
        """
        LLM 호출 상세 로깅 (프롬프트/응답 전문 포함)

        Args:
            step_name: 단계 이름
            model: 사용된 모델명
            prompt: LLM에 전송된 프롬프트 전문
            response: LLM 응답 전문
            temperature: temperature 값
            max_tokens: max_tokens 값
            duration_ms: 처리 시간
            extra: 추가 정보
        """
        llm_log = {
            "type": "llm_call",
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "prompt": prompt,
            "response": response,
            **(extra or {})
        }

        self.log_step(
            step_name=f"{step_name}_LLM",
            status="success",
            output_summary=f"Model: {model}, Prompt: {len(prompt)} chars, Response: {len(response)} chars",
            duration_ms=duration_ms,
            details=llm_log
        )

        # 파일에 전문 기록
        self.logger.debug(f"[{step_name}_LLM] === PROMPT ===\n{prompt}")
        self.logger.debug(f"[{step_name}_LLM] === RESPONSE ===\n{response}")

    def log_retrieval(
        self,
        step_name: str,
        query: str,
        results: list,
        scores: list = None,
        retrieval_type: str = "vector",
        duration_ms: float = 0.0,
        extra: Dict[str, Any] = None
    ):
        """
        검색 결과 상세 로깅 (검색된 문서 전문 포함)

        Args:
            step_name: 단계 이름
            query: 검색 쿼리
            results: 검색된 문서 리스트 (각 문서는 dict 또는 str)
            scores: 유사도 점수 리스트
            retrieval_type: 검색 유형 ("vector", "graph", "hybrid")
            duration_ms: 처리 시간
            extra: 추가 정보
        """
        retrieval_log = {
            "type": "retrieval",
            "retrieval_type": retrieval_type,
            "query": query,
            "result_count": len(results),
            "results": results,
            "scores": scores or [],
            **(extra or {})
        }

        self.log_step(
            step_name=f"{step_name}_Retrieval",
            status="success",
            input_summary=f"Query: {query[:100]}..." if len(query) > 100 else f"Query: {query}",
            output_summary=f"Retrieved {len(results)} docs via {retrieval_type}",
            duration_ms=duration_ms,
            details=retrieval_log
        )

        # 파일에 전문 기록
        self.logger.debug(f"[{step_name}_Retrieval] === QUERY ===\n{query}")
        for i, doc in enumerate(results):
            score_str = f" (score: {scores[i]:.4f})" if scores and i < len(scores) else ""
            doc_content = doc if isinstance(doc, str) else json.dumps(doc, ensure_ascii=False, indent=2)
            self.logger.debug(f"[{step_name}_Retrieval] === RESULT {i+1}{score_str} ===\n{doc_content}")

    def log_detail(
        self,
        step_name: str,
        category: str,
        data: Any,
        description: str = ""
    ):
        """
        상세 데이터 로깅 (요약 없이 전체 데이터 저장)

        Args:
            step_name: 단계 이름
            category: 데이터 카테고리
            data: 저장할 데이터 (dict, list, str 등)
            description: 설명
        """
        detail_log = {
            "type": "detail",
            "category": category,
            "description": description,
            "data": data
        }

        self.log_step(
            step_name=f"{step_name}_{category}",
            status="success",
            output_summary=description,
            details=detail_log
        )

        # 파일에 전문 기록
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, ensure_ascii=False, indent=2)
        else:
            data_str = str(data)
        self.logger.debug(f"[{step_name}_{category}] {description}\n{data_str}")

    def get_summary(self) -> Dict[str, Any]:
        """로그 요약 반환"""
        total_duration = sum(s.duration_ms for s in self.steps)
        success_steps = [s for s in self.steps if s.status == "success"]
        error_steps = [s for s in self.steps if s.status == "error"]

        return {
            "contract_id": self.contract_id,
            "start_time": self.start_time.isoformat(),
            "total_duration_ms": total_duration,
            "total_steps": len(self.steps),
            "success_count": len(success_steps),
            "error_count": len(error_steps),
            "steps": [asdict(s) for s in self.steps]
        }

    def save(self) -> Optional[str]:
        """JSON 형식으로 저장"""
        if not self.save_to_file:
            return None

        summary = self.get_summary()
        json_file = LOG_DIR / f"{self.contract_id}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Pipeline log saved: {json_file}")
        return str(json_file)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.log_error("Pipeline", exc_val)
        self.save()
        # 핸들러 정리
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


def get_pipeline_logger(contract_id: str) -> PipelineLogger:
    """파이프라인 로거 팩토리 함수"""
    return PipelineLogger(contract_id)
