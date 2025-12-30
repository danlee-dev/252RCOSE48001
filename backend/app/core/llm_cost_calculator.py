"""
LLM Cost Calculator
API 비용 계산 유틸리티
"""

import json
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class UsageLog:
    """API 사용량 로그"""
    model: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0


class LLMCostCalculator:
    """LLM API 비용 계산기"""

    def __init__(self):
        pricing_path = Path(__file__).parent / "llm_pricing.json"
        with open(pricing_path, "r", encoding="utf-8") as f:
            self.pricing_data = json.load(f)

        # Flatten models for easy lookup
        self.models = {}
        for provider, models in self.pricing_data["models"].items():
            for model_name, model_info in models.items():
                self.models[model_name] = model_info

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0
    ) -> Dict[str, float]:
        """
        단일 요청 비용 계산

        Args:
            model: 모델명 (e.g., "gemini-2.5-flash-lite", "gpt-5-mini")
            input_tokens: 입력 토큰 수
            output_tokens: 출력 토큰 수
            cached_input_tokens: 캐시된 입력 토큰 수

        Returns:
            비용 정보 딕셔너리
        """
        if model not in self.models:
            raise ValueError(f"Unknown model: {model}")

        model_info = self.models[model]

        # Per million tokens to per token
        input_rate = model_info["input"] / 1_000_000
        output_rate = model_info["output"] / 1_000_000
        cached_rate = model_info.get("cached_input", model_info["input"] * 0.5) / 1_000_000

        # Calculate costs
        regular_input_tokens = input_tokens - cached_input_tokens
        input_cost = regular_input_tokens * input_rate
        cached_cost = cached_input_tokens * cached_rate
        output_cost = output_tokens * output_rate
        total_cost = input_cost + cached_cost + output_cost

        return {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_input_tokens": cached_input_tokens,
            "input_cost_usd": round(input_cost, 6),
            "cached_cost_usd": round(cached_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(total_cost, 6),
            "total_cost_krw": round(total_cost * 1450, 2)  # Approximate KRW
        }

    def calculate_batch_cost(self, usage_logs: list[UsageLog]) -> Dict[str, any]:
        """
        배치 요청 비용 계산

        Args:
            usage_logs: 사용량 로그 리스트

        Returns:
            총 비용 및 모델별 비용
        """
        total_cost = 0.0
        model_costs = {}

        for log in usage_logs:
            cost = self.calculate_cost(
                log.model,
                log.input_tokens,
                log.output_tokens,
                log.cached_input_tokens
            )
            total_cost += cost["total_cost_usd"]

            if log.model not in model_costs:
                model_costs[log.model] = {
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_cost_usd": 0.0
                }

            model_costs[log.model]["total_input_tokens"] += log.input_tokens
            model_costs[log.model]["total_output_tokens"] += log.output_tokens
            model_costs[log.model]["total_cost_usd"] += cost["total_cost_usd"]

        return {
            "total_requests": len(usage_logs),
            "total_cost_usd": round(total_cost, 4),
            "total_cost_krw": round(total_cost * 1450, 0),
            "by_model": model_costs
        }

    def estimate_monthly_cost(
        self,
        contracts_per_month: int,
        avg_input_tokens: int = 10000,
        avg_output_tokens: int = 2000
    ) -> Dict[str, any]:
        """
        월간 비용 추정 (하이브리드 접근법)

        Args:
            contracts_per_month: 월간 처리 계약서 수
            avg_input_tokens: 평균 입력 토큰
            avg_output_tokens: 평균 출력 토큰

        Returns:
            월간 비용 추정
        """
        # Hybrid approach: Gemini for retrieval, GPT for reasoning
        gemini_model = "gemini-2.5-flash-lite"
        openai_model = "gpt-5-mini"

        # Retrieval/summarization phase (Gemini)
        gemini_cost = self.calculate_cost(
            gemini_model,
            input_tokens=avg_input_tokens,
            output_tokens=avg_output_tokens // 2
        )

        # Reasoning/response phase (OpenAI)
        openai_cost = self.calculate_cost(
            openai_model,
            input_tokens=avg_input_tokens // 2,
            output_tokens=avg_output_tokens
        )

        per_contract = gemini_cost["total_cost_usd"] + openai_cost["total_cost_usd"]
        monthly_total = per_contract * contracts_per_month

        return {
            "contracts_per_month": contracts_per_month,
            "cost_per_contract_usd": round(per_contract, 4),
            "monthly_total_usd": round(monthly_total, 2),
            "monthly_total_krw": round(monthly_total * 1450, 0),
            "breakdown": {
                "gemini_retrieval": {
                    "model": gemini_model,
                    "per_contract_usd": gemini_cost["total_cost_usd"],
                    "monthly_usd": round(gemini_cost["total_cost_usd"] * contracts_per_month, 2)
                },
                "openai_reasoning": {
                    "model": openai_model,
                    "per_contract_usd": openai_cost["total_cost_usd"],
                    "monthly_usd": round(openai_cost["total_cost_usd"] * contracts_per_month, 2)
                }
            },
            "within_budget": {
                "gemini_10usd": contracts_per_month <= (10 / gemini_cost["total_cost_usd"]),
                "openai_10usd": contracts_per_month <= (10 / openai_cost["total_cost_usd"]),
                "max_contracts_with_20usd": int(20 / per_contract)
            }
        }

    def get_model_info(self, model: str) -> Optional[Dict]:
        """모델 정보 조회"""
        return self.models.get(model)

    def list_models(self) -> list[str]:
        """사용 가능한 모델 목록"""
        return list(self.models.keys())


# 테스트
if __name__ == "__main__":
    calc = LLMCostCalculator()

    print("Available models:", calc.list_models())
    print()

    # 단일 요청 비용
    cost = calc.calculate_cost("gemini-2.5-flash-lite", 10000, 2000)
    print("Single request (Gemini 2.5 Flash Lite):")
    print(f"  Total: ${cost['total_cost_usd']:.4f} ({cost['total_cost_krw']:.0f} KRW)")
    print()

    cost = calc.calculate_cost("gpt-5-mini", 10000, 2000)
    print("Single request (GPT-5 Mini):")
    print(f"  Total: ${cost['total_cost_usd']:.4f} ({cost['total_cost_krw']:.0f} KRW)")
    print()

    # 월간 비용 추정
    monthly = calc.estimate_monthly_cost(1000)
    print("Monthly estimate (1000 contracts, hybrid approach):")
    print(f"  Gemini: ${monthly['breakdown']['gemini_retrieval']['monthly_usd']:.2f}")
    print(f"  OpenAI: ${monthly['breakdown']['openai_reasoning']['monthly_usd']:.2f}")
    print(f"  Total: ${monthly['monthly_total_usd']:.2f} ({monthly['monthly_total_krw']:.0f} KRW)")
    print(f"  Max contracts with $20 budget: {monthly['within_budget']['max_contracts_with_20usd']}")
