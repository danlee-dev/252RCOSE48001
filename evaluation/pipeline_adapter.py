"""
Pipeline Adapter for Evaluation

Connects evaluation framework to actual DocScanner AI pipeline.
Provides both DocScanner analysis and baseline LLM analysis for comparison.
"""

import os
import sys
import json
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from dotenv import load_dotenv
load_dotenv()


@dataclass
class AnalysisResult:
    """Standardized analysis result for evaluation"""
    contract_id: str
    source: str  # "docscanner" or "baseline_gpt4" or "baseline_gemini"

    # Core results
    violations: List[Dict[str, Any]] = field(default_factory=list)
    risk_level: str = "Low"
    risk_score: float = 0.0

    # Underpayment
    annual_underpayment: int = 0
    total_underpayment: int = 0

    # Metadata
    processing_time: float = 0.0
    token_usage: Dict[str, Any] = field(default_factory=dict)
    raw_response: Dict[str, Any] = field(default_factory=dict)


class DocScannerAdapter:
    """Adapter for DocScanner AI Pipeline"""

    def __init__(self, use_lite: bool = False):
        """
        Args:
            use_lite: If True, disable heavy modules (RAPTOR, reasoning trace)
        """
        self.use_lite = use_lite
        self._pipeline = None

    def _get_pipeline(self):
        """Lazy initialization of pipeline"""
        if self._pipeline is None:
            from app.ai.pipeline import AdvancedAIPipeline, PipelineConfig

            # Initialize ES client with API key
            es_client = None
            try:
                from elasticsearch import Elasticsearch
                es_url = os.getenv('ES_URL')
                es_api_key = os.getenv('ES_API_KEY')
                if es_url:
                    if es_api_key:
                        es_client = Elasticsearch(es_url, api_key=es_api_key)
                    else:
                        es_client = Elasticsearch(es_url)
                    if not es_client.ping():
                        print("[Warning] ES not reachable, running without ES")
                        es_client = None
            except Exception as e:
                print(f"[Warning] ES init error: {e}")

            # Initialize Neo4j driver
            neo4j_driver = None
            try:
                from neo4j import GraphDatabase
                neo4j_uri = os.getenv('NEO4J_URI')
                neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
                neo4j_password = os.getenv('NEO4J_PASSWORD')
                if neo4j_uri and neo4j_password:
                    neo4j_driver = GraphDatabase.driver(
                        neo4j_uri,
                        auth=(neo4j_user, neo4j_password)
                    )
            except Exception as e:
                print(f"[Warning] Neo4j init error: {e}")

            config = PipelineConfig(
                enable_llm_clause_analysis=True,
                enable_pii_masking=True,
                enable_raptor=not self.use_lite,
                enable_constitutional_ai=True,
                enable_judge=True,
                enable_reasoning_trace=not self.use_lite,
                enable_dspy=False,
                enable_hyde=True,
                enable_crag=es_client is not None,  # Only enable CRAG if ES available
            )

            self._pipeline = AdvancedAIPipeline(
                config=config,
                es_client=es_client,
                neo4j_driver=neo4j_driver
            )

        return self._pipeline

    def analyze(self, contract_text: str, contract_id: str = None) -> AnalysisResult:
        """
        Run DocScanner AI pipeline on contract

        Args:
            contract_text: Contract text to analyze
            contract_id: Optional contract ID

        Returns:
            AnalysisResult with standardized format
        """
        pipeline = self._get_pipeline()

        # Run pipeline
        result = pipeline.analyze(contract_text, contract_id=contract_id)
        result_dict = result.to_dict()

        # Extract violations
        violations = []
        if result_dict.get("stress_test") and result_dict["stress_test"].get("violations"):
            for v in result_dict["stress_test"]["violations"]:
                violations.append({
                    "type": v.get("type", ""),
                    "severity": v.get("severity", "MEDIUM"),
                    "description": v.get("description", ""),
                    "legal_basis": v.get("legal_basis", ""),
                    "current_value": v.get("current_value"),
                    "legal_standard": v.get("legal_standard"),
                    "clause_number": v.get("clause_number"),
                    "suggestion": v.get("suggestion", ""),
                })

        return AnalysisResult(
            contract_id=contract_id or result_dict.get("contract_id", ""),
            source="docscanner",
            violations=violations,
            risk_level=result_dict.get("risk_level", "Low"),
            risk_score=result_dict.get("risk_score", 0.0),
            annual_underpayment=result_dict.get("stress_test", {}).get("annual_underpayment", 0) if result_dict.get("stress_test") else 0,
            total_underpayment=result_dict.get("stress_test", {}).get("total_underpayment", 0) if result_dict.get("stress_test") else 0,
            processing_time=result_dict.get("processing_time", 0.0),
            token_usage=result_dict.get("token_usage", {}),
            raw_response=result_dict
        )


class BaselineLLMAdapter:
    """Adapter for baseline LLM analysis (pure GPT-4 / Gemini)"""

    ANALYSIS_PROMPT = """당신은 한국 노동법 전문가입니다. 다음 근로계약서를 분석하여 노동법 위반 사항을 찾아주세요.

[계약서]
{contract_text}

다음 JSON 형식으로 응답하세요:
{{
    "violations": [
        {{
            "type": "위반 유형 (예: 최저임금 위반, 연장근로수당 미지급 등)",
            "severity": "HIGH/MEDIUM/LOW",
            "description": "위반 내용 설명",
            "legal_basis": "관련 법조항",
            "current_value": "현재 계약서의 값",
            "legal_standard": "법적 기준",
            "suggestion": "수정 제안"
        }}
    ],
    "risk_level": "High/Medium/Low",
    "risk_score": 0.0-1.0,
    "annual_underpayment": 연간 예상 체불액 (숫자만),
    "summary": "전체 분석 요약"
}}

중요한 출력 규칙:
- 동일한 계약서 조항에서 여러 위반이 발견되더라도, 해당 조항의 위반들을 하나로 통합하여 출력하세요.
- 예: "제5조 임금" 조항에서 최저임금 위반과 주휴수당 미지급이 동시에 발생하면, 이를 하나의 violation으로 통합하고 type에 "임금 관련 복합 위반" 등으로 표기하세요.
- 조항 번호가 다른 경우에만 별도의 violation으로 분리하세요.

법적 기준:
- 2025년 최저임금: 시급 10,030원, 월 209시간 기준 2,096,270원
- 주휴수당: 주 15시간 이상 근무 시 지급 의무
- 연장근로수당: 통상임금의 50% 가산
- 야간근로수당(22:00-06:00): 통상임금의 50% 가산
- 휴일근로수당: 통상임금의 50% 가산 (8시간 초과 시 100%)
- 연차휴가: 1년 미만 매월 1일, 1년 이상 15일+
- 퇴직금: 1년 이상 근무 시 30일분 평균임금"""

    def __init__(self, provider: str = "openai", model: str = None):
        """
        Args:
            provider: "openai" or "gemini"
            model: Model name (default: gpt-4o or gemini-2.5-flash)
        """
        self.provider = provider
        self.model = model or ("gpt-4o" if provider == "openai" else "gemini-2.5-flash")
        self._client = None

    def _get_client(self):
        if self._client is None:
            if self.provider == "openai":
                from openai import OpenAI
                self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            else:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
                self._client = genai.GenerativeModel(self.model)
        return self._client

    def analyze(self, contract_text: str, contract_id: str = None) -> AnalysisResult:
        """
        Run baseline LLM analysis on contract

        Args:
            contract_text: Contract text to analyze
            contract_id: Optional contract ID

        Returns:
            AnalysisResult with standardized format
        """
        import time
        start_time = time.time()

        prompt = self.ANALYSIS_PROMPT.format(contract_text=contract_text[:8000])

        client = self._get_client()

        try:
            if self.provider == "openai":
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                raw_text = response.choices[0].message.content
                token_usage = {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens
                }
            else:
                response = client.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.0,
                        "response_mime_type": "application/json"
                    }
                )
                raw_text = response.text
                token_usage = {}

            # Parse JSON response
            data = json.loads(raw_text)

            violations = data.get("violations", [])

            return AnalysisResult(
                contract_id=contract_id or "",
                source=f"baseline_{self.provider}",
                violations=violations,
                risk_level=data.get("risk_level", "Low"),
                risk_score=float(data.get("risk_score", 0.0)),
                annual_underpayment=int(data.get("annual_underpayment", 0)),
                total_underpayment=int(data.get("annual_underpayment", 0)),
                processing_time=time.time() - start_time,
                token_usage=token_usage,
                raw_response=data
            )

        except Exception as e:
            return AnalysisResult(
                contract_id=contract_id or "",
                source=f"baseline_{self.provider}",
                violations=[],
                risk_level="Unknown",
                risk_score=0.0,
                processing_time=time.time() - start_time,
                raw_response={"error": str(e)}
            )


class EvaluationPipelineRunner:
    """
    Runs both DocScanner and baseline LLMs for comparison evaluation
    """

    def __init__(self, use_lite: bool = True):
        self.docscanner = DocScannerAdapter(use_lite=use_lite)
        self.baseline_gpt4 = BaselineLLMAdapter(provider="openai", model="gpt-4o")
        self.baseline_gemini = BaselineLLMAdapter(provider="gemini", model="gemini-2.5-flash")

    def run_comparison(
        self,
        contract_text: str,
        contract_id: str = None,
        run_docscanner: bool = True,
        run_gpt4: bool = True,
        run_gemini: bool = True
    ) -> Dict[str, AnalysisResult]:
        """
        Run all analyzers and return comparison results

        Args:
            contract_text: Contract text to analyze
            contract_id: Contract ID
            run_docscanner: Whether to run DocScanner pipeline
            run_gpt4: Whether to run GPT-4 baseline
            run_gemini: Whether to run Gemini baseline

        Returns:
            Dict mapping source to AnalysisResult
        """
        results = {}

        if run_docscanner:
            print(f"  Running DocScanner pipeline...")
            try:
                results["docscanner"] = self.docscanner.analyze(contract_text, contract_id)
            except Exception as e:
                print(f"    DocScanner error: {e}")
                results["docscanner"] = AnalysisResult(
                    contract_id=contract_id or "",
                    source="docscanner",
                    raw_response={"error": str(e)}
                )

        if run_gpt4:
            print(f"  Running GPT-4o baseline...")
            results["baseline_gpt4"] = self.baseline_gpt4.analyze(contract_text, contract_id)

        if run_gemini:
            print(f"  Running Gemini baseline...")
            results["baseline_gemini"] = self.baseline_gemini.analyze(contract_text, contract_id)

        return results

    def run_dataset(
        self,
        dataset_path: str,
        limit: int = None,
        run_docscanner: bool = True,
        run_baselines: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run evaluation on dataset

        Args:
            dataset_path: Path to contract_dataset.json
            limit: Max number of contracts to process
            run_docscanner: Whether to run DocScanner
            run_baselines: Whether to run baseline LLMs

        Returns:
            List of comparison results with ground truth
        """
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        contracts = dataset.get("contracts", [])
        if limit:
            contracts = contracts[:limit]

        all_results = []

        for i, contract in enumerate(contracts):
            print(f"\n[{i+1}/{len(contracts)}] Processing {contract['id']}...")

            comparison = self.run_comparison(
                contract_text=contract["contract_text"],
                contract_id=contract["id"],
                run_docscanner=run_docscanner,
                run_gpt4=run_baselines,
                run_gemini=run_baselines
            )

            all_results.append({
                "contract_id": contract["id"],
                "ground_truth": contract.get("ground_truth", {}),
                "results": {
                    source: {
                        "violations": r.violations,
                        "risk_level": r.risk_level,
                        "risk_score": r.risk_score,
                        "annual_underpayment": r.annual_underpayment,
                        "processing_time": r.processing_time,
                        "token_usage": r.token_usage
                    }
                    for source, r in comparison.items()
                }
            })

        return all_results


def run_quick_test():
    """Quick test to verify pipeline adapter works"""
    test_contract = """
    근로계약서

    1. 근로계약기간: 2025년 1월 1일 ~ 2025년 12월 31일
    2. 근무장소: 서울시 강남구 테헤란로 123
    3. 업무내용: 일반 사무
    4. 근로시간: 09:00 ~ 18:00 (휴게시간 12:00~13:00)
    5. 임금: 월 180만원 (주휴수당 포함)
    6. 연차휴가: 회사 규정에 따름
    """

    print("=" * 60)
    print("Pipeline Adapter Quick Test")
    print("=" * 60)

    # Test baseline only (DocScanner requires ES/Neo4j)
    print("\n[1] Testing GPT-4o Baseline...")
    baseline = BaselineLLMAdapter(provider="openai", model="gpt-4o-mini")
    result = baseline.analyze(test_contract, "test_001")
    print(f"  Violations found: {len(result.violations)}")
    print(f"  Risk level: {result.risk_level}")
    print(f"  Processing time: {result.processing_time:.2f}s")

    print("\n[2] Testing Gemini Baseline...")
    baseline_gemini = BaselineLLMAdapter(provider="gemini")
    result_gemini = baseline_gemini.analyze(test_contract, "test_001")
    print(f"  Violations found: {len(result_gemini.violations)}")
    print(f"  Risk level: {result_gemini.risk_level}")
    print(f"  Processing time: {result_gemini.processing_time:.2f}s")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_quick_test()
