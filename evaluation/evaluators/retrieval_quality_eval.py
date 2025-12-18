"""
Retrieval Quality Evaluator

Evaluates the quality of document retrieval (HyDE + CRAG):
1. Retrieval Relevance: Are retrieved documents relevant to the query?
2. HyDE Effectiveness: Does HyDE improve retrieval quality?
3. CRAG Correction: Does CRAG improve retrieval through correction?
4. Context Quality: Is the retrieved context helpful for analysis?

Metrics:
- nDCG@K (Normalized Discounted Cumulative Gain)
- MRR (Mean Reciprocal Rank)
- Precision@K, Recall@K
- Context Relevance Score (LLM-as-Judge)

Academic References:
- HyDE (Gao et al., 2023): Hypothetical Document Embeddings
- CRAG (Yan et al., 2024): Corrective RAG
- RAGAS (Es et al., 2023): Context relevance evaluation
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import numpy as np
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
class RetrievalQuery:
    """A retrieval query with expected relevant documents"""
    query_id: str
    query_text: str
    query_type: str  # "violation_check", "legal_basis", "precedent"
    expected_relevant_docs: List[str] = field(default_factory=list)
    relevance_scores: Dict[str, int] = field(default_factory=dict)  # doc_id -> relevance (0-3)


@dataclass
class RetrievalResult:
    """Result of a retrieval operation"""
    query_id: str
    retrieved_docs: List[Dict[str, Any]]  # [{doc_id, content, score}]
    retrieval_method: str  # "baseline", "hyde", "crag"
    processing_time_ms: float = 0.0
    hyde_documents: List[str] = field(default_factory=list)  # Generated hypothetical docs
    crag_iterations: int = 0
    crag_quality_grades: List[str] = field(default_factory=list)


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval evaluation"""
    total_queries: int = 0

    # Standard IR metrics
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    mrr: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    map_score: float = 0.0  # Mean Average Precision

    # LLM-based scores
    context_relevance: float = 0.0
    context_completeness: float = 0.0
    context_faithfulness: float = 0.0

    # Efficiency
    avg_latency_ms: float = 0.0
    avg_docs_retrieved: float = 0.0

    # By query type
    by_query_type: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class ComparativeMetrics:
    """Metrics comparing different retrieval methods"""
    baseline_metrics: RetrievalMetrics
    hyde_metrics: RetrievalMetrics
    crag_metrics: RetrievalMetrics

    # Improvements
    hyde_ndcg_improvement: float = 0.0
    crag_ndcg_improvement: float = 0.0
    hyde_mrr_improvement: float = 0.0
    crag_mrr_improvement: float = 0.0

    # Statistical significance
    hyde_p_value: float = 1.0
    crag_p_value: float = 1.0


class RetrievalQualityEvaluator:
    """
    Evaluator for retrieval quality

    Compares:
    1. Baseline vector search
    2. HyDE-enhanced retrieval
    3. CRAG with correction
    """

    def __init__(
        self,
        output_dir: str = None,
        use_llm_evaluation: bool = True,
        k_values: List[int] = None
    ):
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent.parent / "results" / "retrieval_quality"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_llm_evaluation = use_llm_evaluation
        self.k_values = k_values or [5, 10]

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

    @staticmethod
    def dcg_at_k(relevances: List[int], k: int) -> float:
        """Calculate DCG@K"""
        relevances = relevances[:k]
        dcg = sum(
            rel / np.log2(i + 2)
            for i, rel in enumerate(relevances)
        )
        return dcg

    @staticmethod
    def ndcg_at_k(relevances: List[int], ideal_relevances: List[int], k: int) -> float:
        """Calculate nDCG@K"""
        dcg = RetrievalQualityEvaluator.dcg_at_k(relevances, k)
        idcg = RetrievalQualityEvaluator.dcg_at_k(
            sorted(ideal_relevances, reverse=True), k
        )
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: set, k: int) -> float:
        """Calculate Precision@K"""
        retrieved_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc in retrieved_k if doc in relevant)
        return relevant_retrieved / k if k > 0 else 0.0

    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: set, k: int) -> float:
        """Calculate Recall@K"""
        retrieved_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc in retrieved_k if doc in relevant)
        return relevant_retrieved / len(relevant) if relevant else 0.0

    @staticmethod
    def mrr(retrieved_lists: List[List[str]], relevant_sets: List[set]) -> float:
        """Calculate Mean Reciprocal Rank"""
        reciprocal_ranks = []
        for retrieved, relevant in zip(retrieved_lists, relevant_sets):
            for i, doc in enumerate(retrieved):
                if doc in relevant:
                    reciprocal_ranks.append(1.0 / (i + 1))
                    break
            else:
                reciprocal_ranks.append(0.0)
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    @staticmethod
    def average_precision(retrieved: List[str], relevant: set) -> float:
        """Calculate Average Precision for a single query"""
        if not relevant:
            return 0.0

        precisions = []
        relevant_count = 0

        for i, doc in enumerate(retrieved):
            if doc in relevant:
                relevant_count += 1
                precisions.append(relevant_count / (i + 1))

        return np.mean(precisions) if precisions else 0.0

    def calculate_metrics(
        self,
        queries: List[RetrievalQuery],
        results: List[RetrievalResult]
    ) -> RetrievalMetrics:
        """Calculate retrieval metrics"""
        metrics = RetrievalMetrics()
        metrics.total_queries = len(queries)

        all_ndcg_5 = []
        all_ndcg_10 = []
        all_precision_5 = []
        all_precision_10 = []
        all_recall_5 = []
        all_recall_10 = []
        all_ap = []
        all_latencies = []
        all_doc_counts = []

        query_map = {q.query_id: q for q in queries}

        for result in results:
            query = query_map.get(result.query_id)
            if not query:
                continue

            # Get retrieved doc IDs and relevances
            retrieved_ids = [d.get("doc_id", d.get("id", str(i)))
                          for i, d in enumerate(result.retrieved_docs)]

            # Get relevance scores for retrieved docs
            relevances = [
                query.relevance_scores.get(doc_id, 0)
                for doc_id in retrieved_ids
            ]

            # Ideal relevances (all relevant docs)
            ideal_relevances = list(query.relevance_scores.values())

            # Relevant set
            relevant_set = set(query.expected_relevant_docs)

            # Calculate metrics
            if ideal_relevances:
                all_ndcg_5.append(self.ndcg_at_k(relevances, ideal_relevances, 5))
                all_ndcg_10.append(self.ndcg_at_k(relevances, ideal_relevances, 10))

            if relevant_set:
                all_precision_5.append(self.precision_at_k(retrieved_ids, relevant_set, 5))
                all_precision_10.append(self.precision_at_k(retrieved_ids, relevant_set, 10))
                all_recall_5.append(self.recall_at_k(retrieved_ids, relevant_set, 5))
                all_recall_10.append(self.recall_at_k(retrieved_ids, relevant_set, 10))
                all_ap.append(self.average_precision(retrieved_ids, relevant_set))

            all_latencies.append(result.processing_time_ms)
            all_doc_counts.append(len(result.retrieved_docs))

        # Aggregate
        metrics.ndcg_at_5 = np.mean(all_ndcg_5) if all_ndcg_5 else 0.0
        metrics.ndcg_at_10 = np.mean(all_ndcg_10) if all_ndcg_10 else 0.0
        metrics.precision_at_5 = np.mean(all_precision_5) if all_precision_5 else 0.0
        metrics.precision_at_10 = np.mean(all_precision_10) if all_precision_10 else 0.0
        metrics.recall_at_5 = np.mean(all_recall_5) if all_recall_5 else 0.0
        metrics.recall_at_10 = np.mean(all_recall_10) if all_recall_10 else 0.0
        metrics.map_score = np.mean(all_ap) if all_ap else 0.0

        # MRR
        retrieved_lists = [[d.get("doc_id", str(i)) for i, d in enumerate(r.retrieved_docs)]
                         for r in results]
        relevant_sets = [set(query_map[r.query_id].expected_relevant_docs)
                        for r in results if r.query_id in query_map]
        if retrieved_lists and relevant_sets:
            metrics.mrr = self.mrr(retrieved_lists, relevant_sets)

        metrics.avg_latency_ms = np.mean(all_latencies) if all_latencies else 0.0
        metrics.avg_docs_retrieved = np.mean(all_doc_counts) if all_doc_counts else 0.0

        return metrics

    async def evaluate_context_relevance(
        self,
        query: str,
        retrieved_context: str
    ) -> Dict[str, float]:
        """
        Use LLM-as-Judge to evaluate context relevance

        Based on RAGAS context relevance metric
        """
        if not self.llm_evaluator:
            return {"relevance": 0.5, "completeness": 0.5, "faithfulness": 0.5}

        content = f"""[질의 (Query)]
{query}

[검색된 컨텍스트 (Retrieved Context)]
{retrieved_context[:4000]}

위 컨텍스트가 질의에 대해 얼마나 관련성 있고 유용한지 평가해주세요."""

        custom_criteria = {
            "context_relevance": "검색된 컨텍스트가 질의와 관련이 있는가?",
            "context_completeness": "질의에 답하기 위해 필요한 정보가 충분히 포함되어 있는가?",
            "context_noise": "관련 없는 정보가 얼마나 포함되어 있는가? (낮을수록 좋음)"
        }

        try:
            result = await self.llm_evaluator.evaluate_with_cross_validation(
                content=content,
                dimensions=[
                    EvaluationDimension.RELEVANCE,
                    EvaluationDimension.COMPLETENESS
                ],
                custom_criteria=custom_criteria
            )

            return {
                "relevance": result.consensus_score,
                "completeness": result.individual_results[0].scores.get("completeness", {}).score
                              if result.individual_results else 0.5,
                "agreement": result.agreement_rate
            }
        except Exception as e:
            return {"relevance": 0.5, "completeness": 0.5, "error": str(e)}

    def create_test_queries(self) -> List[RetrievalQuery]:
        """Create test queries for retrieval evaluation"""
        queries = []

        # Query 1: Minimum wage check
        queries.append(RetrievalQuery(
            query_id="Q001",
            query_text="최저임금 미달 여부를 확인하기 위한 법적 기준",
            query_type="legal_basis",
            expected_relevant_docs=["최저임금법_제6조", "최저임금법_제5조", "근로기준법_제2조"],
            relevance_scores={
                "최저임금법_제6조": 3,
                "최저임금법_제5조": 2,
                "근로기준법_제2조": 1
            }
        ))

        # Query 2: Overtime pay
        queries.append(RetrievalQuery(
            query_id="Q002",
            query_text="연장근로수당 가산율과 지급 의무",
            query_type="legal_basis",
            expected_relevant_docs=["근로기준법_제56조", "근로기준법_제53조"],
            relevance_scores={
                "근로기준법_제56조": 3,
                "근로기준법_제53조": 2
            }
        ))

        # Query 3: Penalty clause
        queries.append(RetrievalQuery(
            query_id="Q003",
            query_text="위약금 예정 금지 조항의 법적 근거",
            query_type="legal_basis",
            expected_relevant_docs=["근로기준법_제20조"],
            relevance_scores={
                "근로기준법_제20조": 3
            }
        ))

        # Query 4: Working hours
        queries.append(RetrievalQuery(
            query_id="Q004",
            query_text="법정 근로시간과 주 52시간제 관련 규정",
            query_type="legal_basis",
            expected_relevant_docs=["근로기준법_제50조", "근로기준법_제53조"],
            relevance_scores={
                "근로기준법_제50조": 3,
                "근로기준법_제53조": 3
            }
        ))

        # Query 5: Annual leave
        queries.append(RetrievalQuery(
            query_id="Q005",
            query_text="연차유급휴가 부여 기준과 계산 방법",
            query_type="legal_basis",
            expected_relevant_docs=["근로기준법_제60조"],
            relevance_scores={
                "근로기준법_제60조": 3
            }
        ))

        # Query 6: Inclusive wage precedent
        queries.append(RetrievalQuery(
            query_id="Q006",
            query_text="포괄임금제 관련 대법원 판례",
            query_type="precedent",
            expected_relevant_docs=["대법원_2016다243078", "대법원_2019다14341"],
            relevance_scores={
                "대법원_2016다243078": 3,
                "대법원_2019다14341": 2
            }
        ))

        # Query 7: Unfair dismissal
        queries.append(RetrievalQuery(
            query_id="Q007",
            query_text="부당해고의 요건과 구제 방법",
            query_type="legal_basis",
            expected_relevant_docs=["근로기준법_제23조", "근로기준법_제26조"],
            relevance_scores={
                "근로기준법_제23조": 3,
                "근로기준법_제26조": 2
            }
        ))

        return queries

    async def evaluate_comparative(
        self,
        queries: List[RetrievalQuery],
        baseline_results: List[RetrievalResult],
        hyde_results: List[RetrievalResult],
        crag_results: List[RetrievalResult]
    ) -> ComparativeMetrics:
        """
        Compare different retrieval methods
        """
        baseline_metrics = self.calculate_metrics(queries, baseline_results)
        hyde_metrics = self.calculate_metrics(queries, hyde_results)
        crag_metrics = self.calculate_metrics(queries, crag_results)

        comparative = ComparativeMetrics(
            baseline_metrics=baseline_metrics,
            hyde_metrics=hyde_metrics,
            crag_metrics=crag_metrics
        )

        # Calculate improvements
        if baseline_metrics.ndcg_at_10 > 0:
            comparative.hyde_ndcg_improvement = (hyde_metrics.ndcg_at_10 - baseline_metrics.ndcg_at_10) / baseline_metrics.ndcg_at_10
            comparative.crag_ndcg_improvement = (crag_metrics.ndcg_at_10 - baseline_metrics.ndcg_at_10) / baseline_metrics.ndcg_at_10

        if baseline_metrics.mrr > 0:
            comparative.hyde_mrr_improvement = (hyde_metrics.mrr - baseline_metrics.mrr) / baseline_metrics.mrr
            comparative.crag_mrr_improvement = (crag_metrics.mrr - baseline_metrics.mrr) / baseline_metrics.mrr

        return comparative

    def save_results(
        self,
        results: Dict[str, Any],
        prefix: str = "retrieval_quality"
    ):
        """Save evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Convert metrics to dict
        def metrics_to_dict(m: RetrievalMetrics) -> Dict:
            return asdict(m)

        serializable = {
            "summary": {
                "total_queries": results.get("total_queries", 0),
                "baseline_ndcg": results.get("baseline_metrics", {}).ndcg_at_10 if hasattr(results.get("baseline_metrics", {}), "ndcg_at_10") else 0,
                "hyde_ndcg": results.get("hyde_metrics", {}).ndcg_at_10 if hasattr(results.get("hyde_metrics", {}), "ndcg_at_10") else 0,
                "crag_ndcg": results.get("crag_metrics", {}).ndcg_at_10 if hasattr(results.get("crag_metrics", {}), "ndcg_at_10") else 0
            }
        }

        if "comparative" in results:
            comp = results["comparative"]
            serializable["comparative"] = {
                "baseline": metrics_to_dict(comp.baseline_metrics),
                "hyde": metrics_to_dict(comp.hyde_metrics),
                "crag": metrics_to_dict(comp.crag_metrics),
                "improvements": {
                    "hyde_ndcg": comp.hyde_ndcg_improvement,
                    "crag_ndcg": comp.crag_ndcg_improvement,
                    "hyde_mrr": comp.hyde_mrr_improvement,
                    "crag_mrr": comp.crag_mrr_improvement
                }
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

        report = f"""# Retrieval Quality Evaluation Report

## Comparative Analysis

"""
        if "comparative" in results:
            comp = results["comparative"]
            b = comp.baseline_metrics
            h = comp.hyde_metrics
            c = comp.crag_metrics

            report += f"""
| Metric | Baseline | HyDE | CRAG | HyDE Improvement | CRAG Improvement |
|--------|----------|------|------|------------------|------------------|
| nDCG@5 | {b.ndcg_at_5:.3f} | {h.ndcg_at_5:.3f} | {c.ndcg_at_5:.3f} | - | - |
| nDCG@10 | {b.ndcg_at_10:.3f} | {h.ndcg_at_10:.3f} | {c.ndcg_at_10:.3f} | {comp.hyde_ndcg_improvement:+.1%} | {comp.crag_ndcg_improvement:+.1%} |
| MRR | {b.mrr:.3f} | {h.mrr:.3f} | {c.mrr:.3f} | {comp.hyde_mrr_improvement:+.1%} | {comp.crag_mrr_improvement:+.1%} |
| P@5 | {b.precision_at_5:.3f} | {h.precision_at_5:.3f} | {c.precision_at_5:.3f} | - | - |
| P@10 | {b.precision_at_10:.3f} | {h.precision_at_10:.3f} | {c.precision_at_10:.3f} | - | - |
| R@5 | {b.recall_at_5:.3f} | {h.recall_at_5:.3f} | {c.recall_at_5:.3f} | - | - |
| R@10 | {b.recall_at_10:.3f} | {h.recall_at_10:.3f} | {c.recall_at_10:.3f} | - | - |
| MAP | {b.map_score:.3f} | {h.map_score:.3f} | {c.map_score:.3f} | - | - |

## Efficiency

| Method | Avg Latency (ms) | Avg Docs Retrieved |
|--------|-----------------|-------------------|
| Baseline | {b.avg_latency_ms:.1f} | {b.avg_docs_retrieved:.1f} |
| HyDE | {h.avg_latency_ms:.1f} | {h.avg_docs_retrieved:.1f} |
| CRAG | {c.avg_latency_ms:.1f} | {c.avg_docs_retrieved:.1f} |
"""

        report += f"""
## Interpretation

{"HyDE significantly improves retrieval quality." if comp.hyde_ndcg_improvement > 0.1 else "HyDE shows modest improvement." if comp.hyde_ndcg_improvement > 0 else "HyDE does not improve retrieval."}

{"CRAG correction further enhances results." if comp.crag_ndcg_improvement > comp.hyde_ndcg_improvement else "CRAG provides similar performance to HyDE."}

---
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        return report
