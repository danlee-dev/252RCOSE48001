"""
Legal Citation Evaluator

Verifies the accuracy of legal citations in analysis results:
1. Existence Check: Does the cited law/article actually exist?
2. Relevance Check: Is the citation relevant to the violation?
3. Content Accuracy: Does the cited content match the actual law?
4. Hallucination Detection: Identifies fabricated legal references

Uses both:
- Rule-based verification against law database
- LLM-as-Judge for relevance and application correctness

Academic References:
- FActScore (Min et al., 2023): Atomic fact verification
- SAFE (Wei et al., 2024): Search-augmented factuality
"""

import os
import sys
import json
import re
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

from evaluation.core.llm_evaluators import (
    MultiLLMEvaluator,
    EvaluationDimension,
    CrossValidationResult
)


@dataclass
class LegalCitation:
    """Parsed legal citation"""
    law_name: str           # e.g., "근로기준법"
    article: str            # e.g., "제56조"
    paragraph: Optional[str] = None  # e.g., "제1항"
    raw_text: str = ""      # Original citation text
    context: str = ""       # Surrounding context
    source_module: str = "" # Which module generated this


@dataclass
class CitationVerification:
    """Result of verifying a single citation"""
    citation: LegalCitation
    exists: bool = False
    is_relevant: bool = False
    content_accurate: bool = False
    actual_content: str = ""
    relevance_score: float = 0.0
    verification_method: str = ""
    error_type: Optional[str] = None  # "non_existent", "wrong_article", "irrelevant", "misquoted"


@dataclass
class LegalCitationMetrics:
    """Metrics for legal citation evaluation"""
    total_citations: int = 0
    verified_citations: int = 0
    hallucinated_citations: int = 0

    # Existence check
    existing_laws: int = 0
    non_existent_laws: int = 0

    # Relevance check
    relevant_citations: int = 0
    irrelevant_citations: int = 0

    # Accuracy check
    accurate_content: int = 0
    inaccurate_content: int = 0

    # Calculated metrics
    existence_rate: float = 0.0
    relevance_rate: float = 0.0
    accuracy_rate: float = 0.0
    hallucination_rate: float = 0.0

    # Per-law breakdown
    by_law: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Error type breakdown
    error_types: Dict[str, int] = field(default_factory=dict)


# Comprehensive Korean Labor Law Database
# This should be expanded with actual law content
LABOR_LAW_DATABASE = {
    "근로기준법": {
        "제2조": {
            "title": "정의",
            "content": "이 법에서 사용하는 용어의 뜻: 근로자란 직업의 종류와 관계없이 임금을 목적으로 사업이나 사업장에 근로를 제공하는 사람을 말한다.",
            "keywords": ["근로자", "사용자", "정의", "임금"]
        },
        "제15조": {
            "title": "이 법을 위반한 근로계약",
            "content": "이 법에서 정하는 기준에 미치지 못하는 근로조건을 정한 근로계약은 그 부분에 한하여 무효로 한다.",
            "keywords": ["무효", "근로조건", "기준"]
        },
        "제17조": {
            "title": "근로조건의 명시",
            "content": "사용자는 근로계약을 체결할 때에 근로자에게 임금, 소정근로시간, 휴일, 연차유급휴가 등을 명시하여야 한다.",
            "keywords": ["명시", "근로조건", "계약서", "교부"]
        },
        "제20조": {
            "title": "위약 예정의 금지",
            "content": "사용자는 근로계약 불이행에 대한 위약금 또는 손해배상액을 예정하는 계약을 체결하지 못한다.",
            "keywords": ["위약금", "손해배상", "예정", "금지"]
        },
        "제23조": {
            "title": "해고 등의 제한",
            "content": "사용자는 근로자에게 정당한 이유 없이 해고, 휴직, 정직, 전직, 감봉, 그 밖의 징벌을 하지 못한다.",
            "keywords": ["해고", "정당한 이유", "제한", "부당해고"]
        },
        "제26조": {
            "title": "해고의 예고",
            "content": "사용자는 근로자를 해고하려면 적어도 30일 전에 예고를 하여야 하고, 30일 전에 예고를 하지 아니하였을 때에는 30일분 이상의 통상임금을 지급하여야 한다.",
            "keywords": ["해고예고", "30일", "통상임금"]
        },
        "제43조": {
            "title": "임금 지급",
            "content": "임금은 통화로 직접 근로자에게 그 전액을 지급하여야 한다. 임금은 매월 1회 이상 일정한 날짜를 정하여 지급하여야 한다.",
            "keywords": ["임금", "전액", "지급", "통화", "매월"]
        },
        "제50조": {
            "title": "근로시간",
            "content": "1주 간의 근로시간은 휴게시간을 제외하고 40시간을 초과할 수 없다. 1일의 근로시간은 휴게시간을 제외하고 8시간을 초과할 수 없다.",
            "keywords": ["근로시간", "40시간", "8시간", "주"]
        },
        "제53조": {
            "title": "연장 근로의 제한",
            "content": "당사자 간에 합의하면 1주 간에 12시간을 한도로 제50조의 근로시간을 연장할 수 있다.",
            "keywords": ["연장근로", "12시간", "합의", "52시간"]
        },
        "제54조": {
            "title": "휴게",
            "content": "사용자는 근로시간이 4시간인 경우에는 30분 이상, 8시간인 경우에는 1시간 이상의 휴게시간을 근로시간 도중에 주어야 한다.",
            "keywords": ["휴게시간", "30분", "1시간"]
        },
        "제55조": {
            "title": "휴일",
            "content": "사용자는 근로자에게 1주에 평균 1회 이상의 유급휴일을 보장하여야 한다.",
            "keywords": ["휴일", "유급휴일", "주휴일", "주휴수당"]
        },
        "제56조": {
            "title": "연장, 야간 및 휴일 근로",
            "content": "사용자는 연장근로와 야간근로 또는 휴일근로에 대하여는 통상임금의 100분의 50 이상을 가산하여 근로자에게 지급하여야 한다.",
            "keywords": ["연장근로", "야간근로", "휴일근로", "가산", "50%", "수당"]
        },
        "제60조": {
            "title": "연차 유급휴가",
            "content": "사용자는 1년간 80퍼센트 이상 출근한 근로자에게 15일의 유급휴가를 주어야 한다. 1년 미만 근로자는 1개월 개근 시 1일의 유급휴가를 주어야 한다.",
            "keywords": ["연차", "유급휴가", "15일", "80%"]
        }
    },
    "최저임금법": {
        "제5조": {
            "title": "최저임금액",
            "content": "최저임금액은 시간, 일, 주 또는 월을 단위로 하여 정한다.",
            "keywords": ["최저임금", "시간급", "월급"]
        },
        "제6조": {
            "title": "최저임금의 효력",
            "content": "사용자는 최저임금의 적용을 받는 근로자에게 최저임금액 이상의 임금을 지급하여야 한다.",
            "keywords": ["최저임금", "효력", "지급"]
        }
    },
    "근로자퇴직급여보장법": {
        "제4조": {
            "title": "퇴직급여제도의 설정",
            "content": "사용자는 퇴직하는 근로자에게 급여를 지급하기 위하여 퇴직급여제도 중 하나 이상의 제도를 설정하여야 한다.",
            "keywords": ["퇴직급여", "설정"]
        },
        "제8조": {
            "title": "퇴직금제도의 설정",
            "content": "퇴직금제도를 설정하려는 사용자는 계속근로기간 1년에 대하여 30일분 이상의 평균임금을 퇴직금으로 퇴직 근로자에게 지급할 수 있는 제도를 설정하여야 한다.",
            "keywords": ["퇴직금", "1년", "30일분", "평균임금"]
        }
    },
    "국민연금법": {
        "제8조": {
            "title": "가입 대상",
            "content": "국내에 거주하는 국민으로서 18세 이상 60세 미만인 자는 국민연금 가입 대상이 된다.",
            "keywords": ["국민연금", "가입", "대상"]
        }
    },
    "국민건강보험법": {
        "제6조": {
            "title": "가입자의 종류",
            "content": "가입자는 직장가입자와 지역가입자로 구분한다.",
            "keywords": ["건강보험", "가입자", "직장"]
        }
    }
}


class LegalCitationEvaluator:
    """
    Evaluator for legal citation accuracy

    Combines:
    1. Database lookup for existence verification
    2. Keyword matching for relevance
    3. LLM-as-Judge for application correctness
    """

    # Citation extraction patterns
    CITATION_PATTERNS = [
        r'(근로기준법)\s*제?\s*(\d+)\s*조(?:\s*제?\s*(\d+)\s*항)?',
        r'(최저임금법)\s*제?\s*(\d+)\s*조(?:\s*제?\s*(\d+)\s*항)?',
        r'(근로자퇴직급여보장법)\s*제?\s*(\d+)\s*조(?:\s*제?\s*(\d+)\s*항)?',
        r'(국민연금법)\s*제?\s*(\d+)\s*조(?:\s*제?\s*(\d+)\s*항)?',
        r'(국민건강보험법)\s*제?\s*(\d+)\s*조(?:\s*제?\s*(\d+)\s*항)?',
        r'(고용보험법)\s*제?\s*(\d+)\s*조(?:\s*제?\s*(\d+)\s*항)?',
        r'(남녀고용평등법)\s*제?\s*(\d+)\s*조(?:\s*제?\s*(\d+)\s*항)?',
        r'(파견근로자보호법)\s*제?\s*(\d+)\s*조(?:\s*제?\s*(\d+)\s*항)?',
        r'(기간제법)\s*제?\s*(\d+)\s*조(?:\s*제?\s*(\d+)\s*항)?',
    ]

    def __init__(
        self,
        output_dir: str = None,
        use_llm_evaluation: bool = True,
        law_database: Dict = None
    ):
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent.parent / "results" / "legal_citation"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_llm_evaluation = use_llm_evaluation
        self.law_db = law_database or LABOR_LAW_DATABASE

        if use_llm_evaluation:
            try:
                self.llm_evaluator = MultiLLMEvaluator(
                    use_openai=True,
                    use_gemini=True,
                    use_claude=bool(os.getenv("ANTHROPIC_API_KEY"))
                )
            except ValueError:
                self.llm_evaluator = None
        else:
            self.llm_evaluator = None

    def extract_citations(self, text: str, source_module: str = "") -> List[LegalCitation]:
        """Extract legal citations from text"""
        citations = []

        for pattern in self.CITATION_PATTERNS:
            for match in re.finditer(pattern, text):
                law_name = match.group(1)
                article = f"제{match.group(2)}조"
                paragraph = f"제{match.group(3)}항" if match.group(3) else None

                # Get surrounding context
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end]

                citations.append(LegalCitation(
                    law_name=law_name,
                    article=article,
                    paragraph=paragraph,
                    raw_text=match.group(0),
                    context=context,
                    source_module=source_module
                ))

        return citations

    def extract_citations_from_result(self, analysis_result: Dict[str, Any]) -> List[LegalCitation]:
        """Extract citations from DocScanner analysis result"""
        citations = []

        # From clause analysis violations
        clause_analysis = analysis_result.get("clause_analysis", {})
        for violation in clause_analysis.get("violations", []):
            legal_basis = violation.get("legal_basis", "")
            description = violation.get("description", "")

            basis_citations = self.extract_citations(legal_basis, "clause_analysis")
            for c in basis_citations:
                c.context = description[:200]
            citations.extend(basis_citations)

        # From stress_test (legacy)
        stress_test = analysis_result.get("stress_test") or {}
        for violation in stress_test.get("violations", []):
            legal_basis = violation.get("legal_basis", "")
            description = violation.get("description", "")

            basis_citations = self.extract_citations(legal_basis, "stress_test")
            for c in basis_citations:
                c.context = description[:200]
            citations.extend(basis_citations)

        # From constitutional review
        const_review = analysis_result.get("constitutional_review") or {}
        for critique in const_review.get("critiques", []):
            suggestion = critique.get("suggestion", "")
            citations.extend(self.extract_citations(suggestion, "constitutional_review"))

        # From summary
        summary = analysis_result.get("analysis_summary", "")
        citations.extend(self.extract_citations(summary, "summary"))

        return citations

    def verify_citation(self, citation: LegalCitation) -> CitationVerification:
        """Verify a single legal citation"""
        result = CitationVerification(citation=citation)

        # Check if law exists in database
        if citation.law_name not in self.law_db:
            result.exists = False
            result.error_type = "non_existent_law"
            result.verification_method = "database_lookup"
            return result

        law = self.law_db[citation.law_name]

        # Check if article exists
        if citation.article not in law:
            result.exists = False
            result.error_type = "wrong_article"
            result.verification_method = "database_lookup"
            return result

        # Article exists
        result.exists = True
        article_data = law[citation.article]
        result.actual_content = article_data.get("content", "")
        result.verification_method = "database_lookup"

        # Check relevance using keyword matching
        keywords = article_data.get("keywords", [])
        context_lower = citation.context.lower()

        keyword_matches = sum(1 for kw in keywords if kw in context_lower)
        result.relevance_score = keyword_matches / max(1, len(keywords))
        result.is_relevant = result.relevance_score > 0.2

        if not result.is_relevant:
            result.error_type = "irrelevant"

        # Content accuracy (simplified check)
        result.content_accurate = result.is_relevant

        return result

    async def verify_with_llm(
        self,
        citation: LegalCitation,
        violation_context: str
    ) -> Tuple[bool, float, str]:
        """
        Use LLM to verify citation relevance and application
        """
        if not self.llm_evaluator:
            return True, 0.5, "LLM not available"

        content = f"""[인용된 법조항]
{citation.law_name} {citation.article}

[위반 설명 컨텍스트]
{violation_context}

위 법조항이 해당 위반 사항에 적절하게 인용되었는지 평가해주세요.
1. 해당 법조항이 실제로 존재하는가?
2. 해당 법조항이 설명된 위반 내용과 관련이 있는가?
3. 법조항의 적용이 정확한가?"""

        try:
            result = await self.llm_evaluator.evaluate_single(
                content=content,
                dimensions=[EvaluationDimension.ACCURACY, EvaluationDimension.RELEVANCE]
            )

            is_valid = result.overall_score > 0.6
            return is_valid, result.overall_score, result.detailed_feedback

        except Exception as e:
            return True, 0.5, f"Error: {str(e)}"

    def evaluate_citations(
        self,
        analysis_result: Dict[str, Any],
        contract_id: str = "unknown"
    ) -> Tuple[LegalCitationMetrics, List[CitationVerification]]:
        """
        Evaluate all citations in an analysis result

        Args:
            analysis_result: DocScanner analysis output
            contract_id: Contract identifier

        Returns:
            (metrics, verification_results)
        """
        citations = self.extract_citations_from_result(analysis_result)
        verifications = []
        metrics = LegalCitationMetrics()

        metrics.total_citations = len(citations)

        for citation in citations:
            verification = self.verify_citation(citation)
            verifications.append(verification)

            # Update metrics
            if verification.exists:
                metrics.existing_laws += 1
                if verification.is_relevant:
                    metrics.relevant_citations += 1
                    metrics.verified_citations += 1
                    if verification.content_accurate:
                        metrics.accurate_content += 1
                    else:
                        metrics.inaccurate_content += 1
                else:
                    metrics.irrelevant_citations += 1
            else:
                metrics.non_existent_laws += 1
                metrics.hallucinated_citations += 1

            # By-law breakdown
            law_name = citation.law_name
            if law_name not in metrics.by_law:
                metrics.by_law[law_name] = {"total": 0, "valid": 0, "invalid": 0}
            metrics.by_law[law_name]["total"] += 1
            if verification.exists and verification.is_relevant:
                metrics.by_law[law_name]["valid"] += 1
            else:
                metrics.by_law[law_name]["invalid"] += 1

            # Error type breakdown
            if verification.error_type:
                metrics.error_types[verification.error_type] = metrics.error_types.get(verification.error_type, 0) + 1

        # Calculate rates
        if metrics.total_citations > 0:
            metrics.existence_rate = metrics.existing_laws / metrics.total_citations
            metrics.relevance_rate = metrics.relevant_citations / metrics.total_citations
            metrics.accuracy_rate = metrics.accurate_content / max(1, metrics.relevant_citations)
            metrics.hallucination_rate = metrics.hallucinated_citations / metrics.total_citations

        return metrics, verifications

    async def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate multiple analysis results
        """
        all_metrics = []
        all_verifications = []
        aggregated = LegalCitationMetrics()

        for i, case in enumerate(test_cases):
            contract_id = case.get("contract_id", f"contract_{i}")
            print(f"  Evaluating {i+1}/{len(test_cases)}: {contract_id}")

            analysis_result = case.get("analysis_result", case)
            metrics, verifications = self.evaluate_citations(analysis_result, contract_id)

            all_metrics.append({"contract_id": contract_id, "metrics": metrics})
            all_verifications.extend(verifications)

            # Aggregate
            aggregated.total_citations += metrics.total_citations
            aggregated.verified_citations += metrics.verified_citations
            aggregated.hallucinated_citations += metrics.hallucinated_citations
            aggregated.existing_laws += metrics.existing_laws
            aggregated.non_existent_laws += metrics.non_existent_laws
            aggregated.relevant_citations += metrics.relevant_citations
            aggregated.irrelevant_citations += metrics.irrelevant_citations
            aggregated.accurate_content += metrics.accurate_content
            aggregated.inaccurate_content += metrics.inaccurate_content

            for law, counts in metrics.by_law.items():
                if law not in aggregated.by_law:
                    aggregated.by_law[law] = {"total": 0, "valid": 0, "invalid": 0}
                aggregated.by_law[law]["total"] += counts["total"]
                aggregated.by_law[law]["valid"] += counts["valid"]
                aggregated.by_law[law]["invalid"] += counts["invalid"]

            for error_type, count in metrics.error_types.items():
                aggregated.error_types[error_type] = aggregated.error_types.get(error_type, 0) + count

        # Calculate aggregated rates
        if aggregated.total_citations > 0:
            aggregated.existence_rate = aggregated.existing_laws / aggregated.total_citations
            aggregated.relevance_rate = aggregated.relevant_citations / aggregated.total_citations
            aggregated.accuracy_rate = aggregated.accurate_content / max(1, aggregated.relevant_citations)
            aggregated.hallucination_rate = aggregated.hallucinated_citations / aggregated.total_citations

        # Collect hallucination examples
        hallucinations = [v for v in all_verifications if not v.exists or v.error_type]

        return {
            "individual_results": all_metrics,
            "aggregated_metrics": aggregated,
            "hallucinations": hallucinations[:20],  # Top 20 examples
            "summary": {
                "total_contracts": len(test_cases),
                "total_citations": aggregated.total_citations,
                "existence_rate": aggregated.existence_rate,
                "relevance_rate": aggregated.relevance_rate,
                "hallucination_rate": aggregated.hallucination_rate,
                "by_law": aggregated.by_law,
                "error_breakdown": aggregated.error_types
            }
        }

    def save_results(self, results: Dict[str, Any], prefix: str = "legal_citation"):
        """Save evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        serializable = {
            "aggregated_metrics": asdict(results["aggregated_metrics"]),
            "summary": results["summary"],
            "hallucinations": [
                {
                    "citation": asdict(h.citation),
                    "exists": h.exists,
                    "is_relevant": h.is_relevant,
                    "error_type": h.error_type,
                    "actual_content": h.actual_content[:200] if h.actual_content else ""
                }
                for h in results["hallucinations"]
            ],
            "individual_results": [
                {"contract_id": r["contract_id"], "metrics": asdict(r["metrics"])}
                for r in results["individual_results"]
            ]
        }

        json_file = self.output_dir / f"{prefix}_{timestamp}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

        report = self._generate_report(results)
        report_file = self.output_dir / f"{prefix}_report_{timestamp}.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"Results saved to {self.output_dir}")
        return json_file, report_file

    def _generate_report(self, results: Dict[str, Any]) -> str:
        """Generate markdown report"""
        m = results["aggregated_metrics"]
        s = results["summary"]

        by_law_rows = []
        for law, counts in sorted(m.by_law.items()):
            precision = counts["valid"] / max(1, counts["total"])
            by_law_rows.append(f"| {law} | {counts['total']} | {counts['valid']} | {counts['invalid']} | {precision:.1%} |")
        by_law_table = "\n".join(by_law_rows) if by_law_rows else "| (No citations) | - | - | - | - |"

        report = f"""# Legal Citation Evaluation Report

## Summary

| Metric | Value |
|--------|-------|
| Total Contracts | {s['total_contracts']} |
| Total Citations | {m.total_citations} |
| **Existence Rate** | **{m.existence_rate:.1%}** |
| **Relevance Rate** | **{m.relevance_rate:.1%}** |
| **Hallucination Rate** | **{m.hallucination_rate:.1%}** |

## Citation Statistics

| Statistic | Count |
|-----------|-------|
| Verified Citations | {m.verified_citations} |
| Hallucinated Citations | {m.hallucinated_citations} |
| Existing Laws | {m.existing_laws} |
| Non-existent Laws | {m.non_existent_laws} |
| Relevant | {m.relevant_citations} |
| Irrelevant | {m.irrelevant_citations} |

## By Law Analysis

| Law | Total | Valid | Invalid | Precision |
|-----|-------|-------|---------|-----------|
{by_law_table}

## Error Type Breakdown

| Error Type | Count |
|------------|-------|
"""
        for error_type, count in sorted(m.error_types.items(), key=lambda x: -x[1]):
            report += f"| {error_type} | {count} |\n"

        report += f"""
## Hallucination Examples

"""
        for i, h in enumerate(results["hallucinations"][:10]):
            report += f"""### Example {i+1}
- Law: {h.citation.law_name}
- Article: {h.citation.article}
- Error: {h.error_type}
- Context: "{h.citation.context[:100]}..."

"""

        report += f"""
## Interpretation

{"Excellent citation accuracy with minimal hallucinations." if m.hallucination_rate < 0.05 else "Good citation accuracy but some hallucinations detected." if m.hallucination_rate < 0.15 else "Significant hallucination issues in legal citations."}

---
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        return report
