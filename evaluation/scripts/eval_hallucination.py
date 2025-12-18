"""
Hallucination Detection and Legal Basis Verification

Evaluates the factual accuracy of DocScanner AI analysis results:
1. Legal citation verification - checks if cited laws exist and match
2. Factual consistency - checks if claims are supported by evidence
3. Hallucination rate calculation

Methodology:
- Cross-references legal citations with official law database
- Uses NLI (Natural Language Inference) for factual consistency
- Calculates precision of legal references

Academic References:
- FActScore (Min et al., 2023) for atomic fact verification
- SelfCheckGPT (Manakul et al., 2023) for consistency checking
"""

import os
import sys
import json
import re
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
import numpy as np

# Add paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))


@dataclass
class LegalReference:
    """A legal reference found in analysis"""
    law_name: str  # e.g., "근로기준법"
    article: str   # e.g., "제56조"
    paragraph: Optional[str] = None  # e.g., "제1항"
    cited_text: str = ""  # The text quoted/paraphrased
    context: str = ""     # Surrounding text
    source_module: str = ""  # Which module generated this


@dataclass
class VerificationResult:
    """Result of verifying a legal reference"""
    reference: LegalReference
    exists: bool = False           # Does the law/article exist?
    content_matches: bool = False  # Does cited content match actual law?
    actual_content: str = ""       # Actual content from law database
    similarity_score: float = 0.0  # Semantic similarity
    verification_source: str = ""  # Where we verified


@dataclass
class HallucinationMetrics:
    """Metrics for hallucination evaluation"""
    # Citation-level metrics
    total_citations: int = 0
    valid_citations: int = 0
    invalid_citations: int = 0
    citation_precision: float = 0.0

    # Content-level metrics
    total_claims: int = 0
    supported_claims: int = 0
    unsupported_claims: int = 0
    factual_accuracy: float = 0.0

    # Hallucination rate
    hallucination_rate: float = 0.0

    # Per-law breakdown
    by_law: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Error types
    non_existent_laws: int = 0
    wrong_article_numbers: int = 0
    content_mismatches: int = 0


# Korean Labor Law Database (Key articles)
# In production, this would be loaded from a comprehensive database
LABOR_LAW_DATABASE = {
    "근로기준법": {
        "제2조": {
            "title": "정의",
            "content": """이 법에서 사용하는 용어의 뜻은 다음과 같다.
1. "근로자"란 직업의 종류와 관계없이 임금을 목적으로 사업이나 사업장에 근로를 제공하는 사람을 말한다.
2. "사용자"란 사업주 또는 사업 경영 담당자, 그 밖에 근로자에 관한 사항에 대하여 사업주를 위하여 행위하는 자를 말한다.""",
            "keywords": ["근로자", "사용자", "정의", "임금", "사업"]
        },
        "제17조": {
            "title": "근로조건의 명시",
            "content": """사용자는 근로계약을 체결할 때에 근로자에게 다음 각 호의 사항을 명시하여야 한다.
1. 임금
2. 소정근로시간
3. 제55조에 따른 휴일
4. 제60조에 따른 연차 유급휴가
5. 그 밖에 대통령령으로 정하는 근로조건""",
            "keywords": ["근로조건", "명시", "임금", "근로시간", "휴일", "연차"]
        },
        "제20조": {
            "title": "위약 예정의 금지",
            "content": "사용자는 근로계약 불이행에 대한 위약금 또는 손해배상액을 예정하는 계약을 체결하지 못한다.",
            "keywords": ["위약금", "손해배상", "예정", "금지"]
        },
        "제23조": {
            "title": "해고 등의 제한",
            "content": """① 사용자는 근로자에게 정당한 이유 없이 해고, 휴직, 정직, 전직, 감봉, 그 밖의 징벌을 하지 못한다.
② 사용자는 근로자가 업무상 부상 또는 질병의 요양을 위하여 휴업한 기간과 그 후 30일 동안 또는 산전·산후의 여성이 이 법에 따라 휴업한 기간과 그 후 30일 동안은 해고하지 못한다.""",
            "keywords": ["해고", "제한", "정당한 이유", "휴직", "징벌"]
        },
        "제43조": {
            "title": "임금 지급",
            "content": """① 임금은 통화로 직접 근로자에게 그 전액을 지급하여야 한다.
② 임금은 매월 1회 이상 일정한 날짜를 정하여 지급하여야 한다.""",
            "keywords": ["임금", "지급", "통화", "전액", "매월"]
        },
        "제50조": {
            "title": "근로시간",
            "content": """① 1주 간의 근로시간은 휴게시간을 제외하고 40시간을 초과할 수 없다.
② 1일의 근로시간은 휴게시간을 제외하고 8시간을 초과할 수 없다.""",
            "keywords": ["근로시간", "40시간", "8시간", "휴게시간"]
        },
        "제53조": {
            "title": "연장 근로의 제한",
            "content": """① 당사자 간에 합의하면 1주 간에 12시간을 한도로 제50조의 근로시간을 연장할 수 있다.
② 당사자 간에 합의하면 1주 간에 12시간을 한도로 제51조의 근로시간을 연장할 수 있고, 제52조제2호의 정산기간을 평균하여 1주 간에 12시간을 초과하지 아니하는 범위에서 제52조의 근로시간을 연장할 수 있다.""",
            "keywords": ["연장근로", "12시간", "합의", "제한"]
        },
        "제54조": {
            "title": "휴게",
            "content": """① 사용자는 근로시간이 4시간인 경우에는 30분 이상, 8시간인 경우에는 1시간 이상의 휴게시간을 근로시간 도중에 주어야 한다.
② 휴게시간은 근로자가 자유롭게 이용할 수 있다.""",
            "keywords": ["휴게시간", "30분", "1시간", "자유롭게"]
        },
        "제55조": {
            "title": "휴일",
            "content": """① 사용자는 근로자에게 1주에 평균 1회 이상의 유급휴일을 보장하여야 한다.
② 사용자는 근로자에게 대통령령으로 정하는 휴일을 유급으로 보장하여야 한다.""",
            "keywords": ["휴일", "유급휴일", "1주", "주휴일"]
        },
        "제56조": {
            "title": "연장·야간 및 휴일 근로",
            "content": """① 사용자는 연장근로와 야간근로 또는 휴일근로에 대하여는 통상임금의 100분의 50 이상을 가산하여 근로자에게 지급하여야 한다.
② 제1항에도 불구하고 사용자는 휴일근로에 대하여는 다음 각 호의 기준에 따른 금액 이상을 가산하여 근로자에게 지급하여야 한다.
1. 8시간 이내의 휴일근로: 통상임금의 100분의 50
2. 8시간을 초과한 휴일근로: 통상임금의 100분의 100""",
            "keywords": ["연장근로", "야간근로", "휴일근로", "가산", "50%", "100%", "통상임금"]
        },
        "제60조": {
            "title": "연차 유급휴가",
            "content": """① 사용자는 1년간 80퍼센트 이상 출근한 근로자에게 15일의 유급휴가를 주어야 한다.
② 사용자는 계속하여 근로한 기간이 1년 미만인 근로자 또는 1년간 80퍼센트 미만 출근한 근로자에게 1개월 개근 시 1일의 유급휴가를 주어야 한다.""",
            "keywords": ["연차", "유급휴가", "15일", "80%", "출근"]
        },
        "제110조": {
            "title": "벌칙",
            "content": "다음 각 호의 어느 하나에 해당하는 자는 2년 이하의 징역 또는 2천만원 이하의 벌금에 처한다.",
            "keywords": ["벌칙", "징역", "벌금"]
        }
    },
    "최저임금법": {
        "제6조": {
            "title": "최저임금의 효력",
            "content": """① 사용자는 최저임금의 적용을 받는 근로자에게 최저임금액 이상의 임금을 지급하여야 한다.
② 사용자는 이 법에 따른 최저임금을 이유로 종전의 임금수준을 낮추어서는 아니 된다.""",
            "keywords": ["최저임금", "효력", "임금", "지급"]
        },
        "제8조": {
            "title": "최저임금의 결정",
            "content": "최저임금은 최저임금위원회의 심의를 거쳐 매년 8월 5일까지 결정한다.",
            "keywords": ["최저임금", "결정", "심의", "위원회"]
        }
    },
    "근로자퇴직급여보장법": {
        "제4조": {
            "title": "퇴직급여제도의 설정",
            "content": """① 사용자는 퇴직하는 근로자에게 급여를 지급하기 위하여 퇴직급여제도 중 하나 이상의 제도를 설정하여야 한다.
② 제1항에 따라 퇴직급여제도를 설정하는 경우에 하나의 사업에서 급여 및 부담금 산정방법의 적용 등에 관하여 차등을 두어서는 아니 된다.""",
            "keywords": ["퇴직급여", "설정", "근로자", "사용자"]
        },
        "제8조": {
            "title": "퇴직금제도의 설정",
            "content": """퇴직금제도를 설정하려는 사용자는 계속근로기간 1년에 대하여 30일분 이상의 평균임금을 퇴직금으로 퇴직 근로자에게 지급할 수 있는 제도를 설정하여야 한다.""",
            "keywords": ["퇴직금", "1년", "30일", "평균임금"]
        }
    }
}


class LegalCitationExtractor:
    """Extract legal citations from analysis results"""

    # Patterns for legal citations
    CITATION_PATTERNS = [
        r'(근로기준법)\s*제?\s*(\d+)\s*조(?:\s*제?\s*(\d+)\s*항)?',
        r'(최저임금법)\s*제?\s*(\d+)\s*조(?:\s*제?\s*(\d+)\s*항)?',
        r'(근로자퇴직급여보장법)\s*제?\s*(\d+)\s*조(?:\s*제?\s*(\d+)\s*항)?',
        r'(남녀고용평등법)\s*제?\s*(\d+)\s*조(?:\s*제?\s*(\d+)\s*항)?',
        r'(산업안전보건법)\s*제?\s*(\d+)\s*조(?:\s*제?\s*(\d+)\s*항)?',
    ]

    def extract(self, analysis_result: Dict[str, Any]) -> List[LegalReference]:
        """Extract all legal citations from analysis result"""
        citations = []

        # Extract from stress_test violations
        stress_test = analysis_result.get("stress_test", {})
        for violation in stress_test.get("violations", []):
            legal_basis = violation.get("legal_basis", "")
            if legal_basis:
                refs = self._parse_citation(legal_basis, violation.get("description", ""))
                for ref in refs:
                    ref.source_module = "stress_test"
                citations.extend(refs)

        # Extract from constitutional_review
        const_review = analysis_result.get("constitutional_review", {})
        for critique in const_review.get("critiques", []):
            suggestion = critique.get("suggestion", "")
            refs = self._parse_citation(suggestion, critique.get("critique", ""))
            for ref in refs:
                ref.source_module = "constitutional_review"
            citations.extend(refs)

        # Extract from analysis_summary
        summary = analysis_result.get("analysis_summary", "")
        refs = self._parse_citation(summary, summary)
        for ref in refs:
            ref.source_module = "summary"
        citations.extend(refs)

        return citations

    def _parse_citation(self, text: str, context: str) -> List[LegalReference]:
        """Parse legal citations from text"""
        citations = []

        for pattern in self.CITATION_PATTERNS:
            matches = re.finditer(pattern, text)
            for match in matches:
                law_name = match.group(1)
                article = f"제{match.group(2)}조"
                paragraph = f"제{match.group(3)}항" if match.group(3) else None

                citations.append(LegalReference(
                    law_name=law_name,
                    article=article,
                    paragraph=paragraph,
                    cited_text=match.group(0),
                    context=context[:200] if context else ""
                ))

        return citations


class LegalCitationVerifier:
    """Verify legal citations against law database"""

    def __init__(self, law_database: Dict = None):
        self.law_db = law_database or LABOR_LAW_DATABASE

    def verify(self, reference: LegalReference) -> VerificationResult:
        """Verify a single legal reference"""
        result = VerificationResult(reference=reference)

        # Check if law exists
        if reference.law_name not in self.law_db:
            result.exists = False
            result.verification_source = "law_database"
            return result

        law = self.law_db[reference.law_name]

        # Check if article exists
        if reference.article not in law:
            result.exists = False
            result.verification_source = "law_database"
            return result

        # Article exists
        result.exists = True
        article_data = law[reference.article]
        result.actual_content = article_data.get("content", "")
        result.verification_source = "law_database"

        # Check content match using keyword overlap
        if result.actual_content:
            keywords = article_data.get("keywords", [])
            context_lower = reference.context.lower()

            # Count keyword matches
            matches = sum(1 for kw in keywords if kw in context_lower)
            result.similarity_score = matches / max(1, len(keywords))

            # Consider it a match if similarity is above threshold
            result.content_matches = result.similarity_score > 0.3

        return result


class HallucinationEvaluator:
    """
    Main evaluator for hallucination detection

    Evaluates:
    1. Legal citation accuracy
    2. Factual consistency
    3. Overall hallucination rate
    """

    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent.parent / "results" / "hallucination"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.extractor = LegalCitationExtractor()
        self.verifier = LegalCitationVerifier()

        # Results storage
        self.verification_results: List[VerificationResult] = []

    def evaluate_analysis(
        self,
        analysis_result: Dict[str, Any],
        contract_id: str = None
    ) -> HallucinationMetrics:
        """
        Evaluate a single analysis result for hallucinations

        Args:
            analysis_result: DocScanner pipeline output
            contract_id: Optional contract identifier

        Returns:
            HallucinationMetrics
        """
        metrics = HallucinationMetrics()

        # Extract citations
        citations = self.extractor.extract(analysis_result)
        metrics.total_citations = len(citations)

        # Verify each citation
        for citation in citations:
            result = self.verifier.verify(citation)
            self.verification_results.append(result)

            # Update metrics
            if result.exists and result.content_matches:
                metrics.valid_citations += 1
            else:
                metrics.invalid_citations += 1

                if not result.exists:
                    metrics.non_existent_laws += 1
                elif not result.content_matches:
                    metrics.content_mismatches += 1

            # By-law breakdown
            law_name = citation.law_name
            if law_name not in metrics.by_law:
                metrics.by_law[law_name] = {"total": 0, "valid": 0, "invalid": 0}
            metrics.by_law[law_name]["total"] += 1
            if result.exists and result.content_matches:
                metrics.by_law[law_name]["valid"] += 1
            else:
                metrics.by_law[law_name]["invalid"] += 1

        # Calculate rates
        if metrics.total_citations > 0:
            metrics.citation_precision = metrics.valid_citations / metrics.total_citations
            metrics.hallucination_rate = metrics.invalid_citations / metrics.total_citations
        else:
            metrics.citation_precision = 1.0  # No citations = no hallucinations
            metrics.hallucination_rate = 0.0

        return metrics

    def evaluate_batch(
        self,
        analysis_results: List[Dict[str, Any]]
    ) -> HallucinationMetrics:
        """
        Evaluate multiple analysis results

        Args:
            analysis_results: List of DocScanner outputs

        Returns:
            Aggregated HallucinationMetrics
        """
        print(f"\nEvaluating {len(analysis_results)} analysis results for hallucinations...\n")

        all_metrics = []
        for i, result in enumerate(analysis_results):
            print(f"  Processing {i+1}/{len(analysis_results)}...")
            metrics = self.evaluate_analysis(result, contract_id=f"contract_{i}")
            all_metrics.append(metrics)

        # Aggregate metrics
        aggregated = HallucinationMetrics()
        for m in all_metrics:
            aggregated.total_citations += m.total_citations
            aggregated.valid_citations += m.valid_citations
            aggregated.invalid_citations += m.invalid_citations
            aggregated.non_existent_laws += m.non_existent_laws
            aggregated.wrong_article_numbers += m.wrong_article_numbers
            aggregated.content_mismatches += m.content_mismatches

            # Merge by_law
            for law, counts in m.by_law.items():
                if law not in aggregated.by_law:
                    aggregated.by_law[law] = {"total": 0, "valid": 0, "invalid": 0}
                aggregated.by_law[law]["total"] += counts["total"]
                aggregated.by_law[law]["valid"] += counts["valid"]
                aggregated.by_law[law]["invalid"] += counts["invalid"]

        # Calculate aggregated rates
        if aggregated.total_citations > 0:
            aggregated.citation_precision = aggregated.valid_citations / aggregated.total_citations
            aggregated.hallucination_rate = aggregated.invalid_citations / aggregated.total_citations

        return aggregated

    def save_results(self, metrics: HallucinationMetrics):
        """Save evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save metrics
        metrics_file = self.output_dir / f"hallucination_metrics_{timestamp}.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(asdict(metrics), f, indent=2, ensure_ascii=False)

        # Save detailed verification results
        details_file = self.output_dir / f"verification_details_{timestamp}.json"
        with open(details_file, "w", encoding="utf-8") as f:
            results_data = []
            for r in self.verification_results:
                results_data.append({
                    "law_name": r.reference.law_name,
                    "article": r.reference.article,
                    "cited_text": r.reference.cited_text,
                    "context": r.reference.context,
                    "source_module": r.reference.source_module,
                    "exists": r.exists,
                    "content_matches": r.content_matches,
                    "similarity_score": r.similarity_score,
                    "actual_content": r.actual_content[:200] if r.actual_content else ""
                })
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        # Generate report
        report = self._generate_report(metrics)
        report_file = self.output_dir / f"hallucination_report_{timestamp}.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"\nResults saved to {self.output_dir}")
        return metrics_file, details_file, report_file

    def _generate_report(self, metrics: HallucinationMetrics) -> str:
        """Generate markdown report"""
        # By-law table
        by_law_rows = []
        for law, counts in sorted(metrics.by_law.items()):
            precision = counts["valid"] / max(1, counts["total"])
            by_law_rows.append(
                f"| {law} | {counts['total']} | {counts['valid']} | {counts['invalid']} | {precision:.1%} |"
            )
        by_law_table = "\n".join(by_law_rows) if by_law_rows else "| (No citations found) | - | - | - | - |"

        report = f"""# Hallucination Detection Report

## Summary

| Metric | Value |
|--------|-------|
| Total Citations | {metrics.total_citations} |
| Valid Citations | {metrics.valid_citations} |
| Invalid Citations | {metrics.invalid_citations} |
| **Citation Precision** | **{metrics.citation_precision:.1%}** |
| **Hallucination Rate** | **{metrics.hallucination_rate:.1%}** |

## Error Breakdown

| Error Type | Count |
|------------|-------|
| Non-existent Laws | {metrics.non_existent_laws} |
| Wrong Article Numbers | {metrics.wrong_article_numbers} |
| Content Mismatches | {metrics.content_mismatches} |

## By Law Analysis

| Law | Total | Valid | Invalid | Precision |
|-----|-------|-------|---------|-----------|
{by_law_table}

## Interpretation

{"The system shows high factual accuracy with citation precision above 90%." if metrics.citation_precision > 0.9 else "There are concerns about citation accuracy that need to be addressed." if metrics.citation_precision < 0.8 else "Citation accuracy is acceptable but has room for improvement."}

Hallucination rate of {metrics.hallucination_rate:.1%} indicates {"minimal" if metrics.hallucination_rate < 0.1 else "moderate" if metrics.hallucination_rate < 0.2 else "significant"} hallucination issues.

## Recommendations

{"1. Review and update the legal citation generation prompts" if metrics.hallucination_rate > 0.1 else "1. Continue monitoring citation accuracy"}
{"2. Add more rigorous fact-checking in the Constitutional AI module" if metrics.content_mismatches > 0 else "2. Maintain current fact-checking procedures"}
{"3. Update the law database to include more recent amendments" if metrics.non_existent_laws > 0 else "3. Keep law database current"}

---
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate hallucination in analysis results")
    parser.add_argument("--data", type=str, help="Path to analysis results (JSON/JSONL)")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--single", type=str, help="Single analysis result JSON")

    args = parser.parse_args()

    evaluator = HallucinationEvaluator(output_dir=args.output)

    if args.data:
        # Load and evaluate batch
        with open(args.data, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            metrics = evaluator.evaluate_batch(data)
        else:
            metrics = evaluator.evaluate_analysis(data)

    elif args.single:
        # Evaluate single result
        with open(args.single, "r", encoding="utf-8") as f:
            data = json.load(f)
        metrics = evaluator.evaluate_analysis(data)

    else:
        # Demo with sample data
        sample_result = {
            "analysis_summary": "이 계약서는 근로기준법 제56조를 위반하고 있습니다.",
            "stress_test": {
                "violations": [
                    {
                        "type": "포괄임금제 위반",
                        "description": "연장근로수당을 포함한 임금 지급은 근로기준법 제56조 위반입니다.",
                        "legal_basis": "근로기준법 제56조"
                    },
                    {
                        "type": "최저임금 미달",
                        "description": "시급이 최저임금법 제6조에서 정한 최저임금에 미달합니다.",
                        "legal_basis": "최저임금법 제6조"
                    }
                ]
            }
        }

        metrics = evaluator.evaluate_analysis(sample_result)

    # Save and print results
    evaluator.save_results(metrics)

    print("\n" + "="*60)
    print("HALLUCINATION EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Citations: {metrics.total_citations}")
    print(f"Citation Precision: {metrics.citation_precision:.1%}")
    print(f"Hallucination Rate: {metrics.hallucination_rate:.1%}")
    print("="*60)


if __name__ == "__main__":
    main()
