"""
Retrieval Evaluation Script

Comprehensive evaluation of search/retrieval performance using standard IR metrics:
- nDCG@K (Normalized Discounted Cumulative Gain)
- MRR (Mean Reciprocal Rank)
- MAP (Mean Average Precision)
- Recall@K
- Precision@K

Supports evaluation of:
1. HyDE query enhancement
2. CRAG retrieval correction
3. Full pipeline retrieval

Academic References:
- Jarvelin & Kekalainen (2002) for nDCG
- Voorhees (1999) for MRR and MAP
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
import numpy as np
from scipy import stats

# Add paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))


@dataclass
class RetrievalQuery:
    """A single retrieval query with ground truth"""
    query_id: str
    query_text: str
    relevant_doc_ids: List[str]  # Ground truth relevant documents
    relevance_scores: Dict[str, int] = field(default_factory=dict)  # doc_id -> relevance (graded)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Result from a retrieval system"""
    query_id: str
    retrieved_doc_ids: List[str]  # Ordered by rank
    scores: List[float] = field(default_factory=list)
    retrieval_method: str = "unknown"
    hyde_query: Optional[str] = None
    processing_time_ms: float = 0.0


@dataclass
class RetrievalMetrics:
    """Comprehensive retrieval metrics"""
    # Primary metrics
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    mrr: float = 0.0
    map_score: float = 0.0

    # Recall/Precision
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0

    # Additional metrics
    hit_rate_at_5: float = 0.0
    hit_rate_at_10: float = 0.0
    avg_relevant_rank: float = 0.0

    # Processing stats
    avg_processing_time_ms: float = 0.0
    total_queries: int = 0

    # Per-query breakdown (for statistical tests)
    per_query_ndcg: List[float] = field(default_factory=list)
    per_query_mrr: List[float] = field(default_factory=list)


class RetrievalMetricsCalculator:
    """
    Calculate standard IR metrics

    All metrics follow standard definitions from IR literature.
    """

    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """
        Recall@K: Fraction of relevant docs found in top-k results

        R@K = |relevant ∩ retrieved[:k]| / |relevant|
        """
        if not relevant:
            return 0.0

        retrieved_k = set(retrieved[:k])
        relevant_set = set(relevant)

        return len(retrieved_k & relevant_set) / len(relevant_set)

    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """
        Precision@K: Fraction of top-k results that are relevant

        P@K = |relevant ∩ retrieved[:k]| / k
        """
        if k == 0:
            return 0.0

        retrieved_k = set(retrieved[:k])
        relevant_set = set(relevant)

        return len(retrieved_k & relevant_set) / k

    @staticmethod
    def hit_rate_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """
        Hit Rate@K: 1 if any relevant doc in top-k, else 0
        """
        retrieved_k = set(retrieved[:k])
        relevant_set = set(relevant)

        return 1.0 if len(retrieved_k & relevant_set) > 0 else 0.0

    @staticmethod
    def mrr(retrieved: List[str], relevant: List[str]) -> float:
        """
        Mean Reciprocal Rank: 1/rank of first relevant document

        MRR = 1/rank(first relevant doc)
        """
        relevant_set = set(relevant)

        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant_set:
                return 1.0 / i

        return 0.0

    @staticmethod
    def average_precision(retrieved: List[str], relevant: List[str]) -> float:
        """
        Average Precision: Mean of precision values at each relevant doc position

        AP = (1/|relevant|) * Σ P@k * rel(k)
        """
        if not relevant:
            return 0.0

        relevant_set = set(relevant)
        num_relevant = 0
        sum_precision = 0.0

        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant_set:
                num_relevant += 1
                precision_at_i = num_relevant / i
                sum_precision += precision_at_i

        if len(relevant_set) == 0:
            return 0.0

        return sum_precision / len(relevant_set)

    @staticmethod
    def dcg_at_k(
        retrieved: List[str],
        relevance_scores: Dict[str, int],
        k: int
    ) -> float:
        """
        Discounted Cumulative Gain@K

        DCG@K = Σ (2^rel_i - 1) / log2(i + 1)
        """
        dcg = 0.0

        for i, doc_id in enumerate(retrieved[:k], 1):
            rel = relevance_scores.get(doc_id, 0)
            dcg += (2**rel - 1) / math.log2(i + 1)

        return dcg

    @staticmethod
    def ndcg_at_k(
        retrieved: List[str],
        relevance_scores: Dict[str, int],
        k: int
    ) -> float:
        """
        Normalized Discounted Cumulative Gain@K

        nDCG@K = DCG@K / IDCG@K
        """
        # Calculate DCG
        dcg = RetrievalMetricsCalculator.dcg_at_k(retrieved, relevance_scores, k)

        # Calculate IDCG (ideal DCG with perfect ranking)
        ideal_ranking = sorted(
            relevance_scores.keys(),
            key=lambda x: relevance_scores.get(x, 0),
            reverse=True
        )
        idcg = RetrievalMetricsCalculator.dcg_at_k(ideal_ranking, relevance_scores, k)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def calculate_all_metrics(
        retrieved: List[str],
        relevant: List[str],
        relevance_scores: Dict[str, int] = None,
        k_values: List[int] = [5, 10]
    ) -> Dict[str, float]:
        """Calculate all metrics for a single query"""
        # If no graded relevance, create binary
        if relevance_scores is None:
            relevance_scores = {doc_id: 1 for doc_id in relevant}

        metrics = {
            "mrr": RetrievalMetricsCalculator.mrr(retrieved, relevant),
            "ap": RetrievalMetricsCalculator.average_precision(retrieved, relevant)
        }

        for k in k_values:
            metrics[f"recall@{k}"] = RetrievalMetricsCalculator.recall_at_k(retrieved, relevant, k)
            metrics[f"precision@{k}"] = RetrievalMetricsCalculator.precision_at_k(retrieved, relevant, k)
            metrics[f"hit_rate@{k}"] = RetrievalMetricsCalculator.hit_rate_at_k(retrieved, relevant, k)
            metrics[f"ndcg@{k}"] = RetrievalMetricsCalculator.ndcg_at_k(retrieved, relevance_scores, k)

        return metrics


class RetrievalEvaluator:
    """
    Main evaluator for retrieval systems

    Supports evaluation of multiple systems for comparison.
    """

    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent.parent / "results" / "retrieval"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.calculator = RetrievalMetricsCalculator()

        # Results storage
        self.results_by_system: Dict[str, List[RetrievalResult]] = defaultdict(list)
        self.queries: List[RetrievalQuery] = []

    def load_queries(self, queries_path: str) -> List[RetrievalQuery]:
        """Load evaluation queries with ground truth"""
        with open(queries_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        queries = []
        for item in data:
            query = RetrievalQuery(
                query_id=item["query_id"],
                query_text=item["query_text"],
                relevant_doc_ids=item.get("relevant_doc_ids", []),
                relevance_scores=item.get("relevance_scores", {}),
                metadata=item.get("metadata", {})
            )
            queries.append(query)

        self.queries = queries
        return queries

    def evaluate_system(
        self,
        system_name: str,
        results: List[RetrievalResult],
        queries: List[RetrievalQuery] = None
    ) -> RetrievalMetrics:
        """
        Evaluate a retrieval system

        Args:
            system_name: Name of the system (e.g., "hyde", "crag", "baseline")
            results: List of retrieval results
            queries: List of queries with ground truth

        Returns:
            Aggregated RetrievalMetrics
        """
        queries = queries or self.queries

        if not queries:
            raise ValueError("No queries provided for evaluation")

        # Build query lookup
        query_lookup = {q.query_id: q for q in queries}

        # Calculate per-query metrics
        all_metrics = []
        per_query_ndcg = []
        per_query_mrr = []
        processing_times = []

        for result in results:
            if result.query_id not in query_lookup:
                continue

            query = query_lookup[result.query_id]

            # Calculate metrics
            metrics = self.calculator.calculate_all_metrics(
                retrieved=result.retrieved_doc_ids,
                relevant=query.relevant_doc_ids,
                relevance_scores=query.relevance_scores,
                k_values=[5, 10]
            )

            all_metrics.append(metrics)
            per_query_ndcg.append(metrics.get("ndcg@10", 0))
            per_query_mrr.append(metrics["mrr"])
            processing_times.append(result.processing_time_ms)

        # Aggregate metrics
        aggregated = RetrievalMetrics()
        aggregated.total_queries = len(all_metrics)

        if all_metrics:
            aggregated.ndcg_at_5 = np.mean([m["ndcg@5"] for m in all_metrics])
            aggregated.ndcg_at_10 = np.mean([m["ndcg@10"] for m in all_metrics])
            aggregated.mrr = np.mean([m["mrr"] for m in all_metrics])
            aggregated.map_score = np.mean([m["ap"] for m in all_metrics])
            aggregated.recall_at_5 = np.mean([m["recall@5"] for m in all_metrics])
            aggregated.recall_at_10 = np.mean([m["recall@10"] for m in all_metrics])
            aggregated.precision_at_5 = np.mean([m["precision@5"] for m in all_metrics])
            aggregated.precision_at_10 = np.mean([m["precision@10"] for m in all_metrics])
            aggregated.hit_rate_at_5 = np.mean([m["hit_rate@5"] for m in all_metrics])
            aggregated.hit_rate_at_10 = np.mean([m["hit_rate@10"] for m in all_metrics])
            aggregated.avg_processing_time_ms = np.mean(processing_times) if processing_times else 0
            aggregated.per_query_ndcg = per_query_ndcg
            aggregated.per_query_mrr = per_query_mrr

        # Store results
        self.results_by_system[system_name] = results

        return aggregated

    def compare_systems(
        self,
        system_a: str,
        system_b: str,
        metrics_a: RetrievalMetrics,
        metrics_b: RetrievalMetrics
    ) -> Dict[str, Any]:
        """
        Compare two systems with statistical significance tests

        Args:
            system_a, system_b: System names
            metrics_a, metrics_b: Metrics from each system

        Returns:
            Comparison results with p-values and effect sizes
        """
        comparison = {
            "system_a": system_a,
            "system_b": system_b,
            "metrics": {}
        }

        # Compare nDCG@10
        if metrics_a.per_query_ndcg and metrics_b.per_query_ndcg:
            ndcg_a = np.array(metrics_a.per_query_ndcg)
            ndcg_b = np.array(metrics_b.per_query_ndcg)

            if len(ndcg_a) == len(ndcg_b) and len(ndcg_a) > 1:
                try:
                    # Paired test
                    diff = ndcg_a - ndcg_b
                    _, p_normal = stats.shapiro(diff)

                    if p_normal > 0.05:
                        t_stat, p_value = stats.ttest_rel(ndcg_a, ndcg_b)
                        test_name = "paired_t_test"
                    else:
                        t_stat, p_value = stats.wilcoxon(ndcg_a, ndcg_b)
                        test_name = "wilcoxon"

                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt((np.var(ndcg_a) + np.var(ndcg_b)) / 2)
                    effect_size = np.mean(diff) / pooled_std if pooled_std > 0 else 0

                    comparison["metrics"]["ndcg@10"] = {
                        "mean_a": float(np.mean(ndcg_a)),
                        "mean_b": float(np.mean(ndcg_b)),
                        "difference": float(np.mean(diff)),
                        "test": test_name,
                        "p_value": float(p_value),
                        "effect_size": float(effect_size),
                        "significant": p_value < 0.05
                    }
                except Exception as e:
                    comparison["metrics"]["ndcg@10"] = {"error": str(e)}

        # Compare MRR
        if metrics_a.per_query_mrr and metrics_b.per_query_mrr:
            mrr_a = np.array(metrics_a.per_query_mrr)
            mrr_b = np.array(metrics_b.per_query_mrr)

            if len(mrr_a) == len(mrr_b) and len(mrr_a) > 1:
                try:
                    diff = mrr_a - mrr_b
                    t_stat, p_value = stats.wilcoxon(mrr_a, mrr_b)

                    pooled_std = np.sqrt((np.var(mrr_a) + np.var(mrr_b)) / 2)
                    effect_size = np.mean(diff) / pooled_std if pooled_std > 0 else 0

                    comparison["metrics"]["mrr"] = {
                        "mean_a": float(np.mean(mrr_a)),
                        "mean_b": float(np.mean(mrr_b)),
                        "difference": float(np.mean(diff)),
                        "p_value": float(p_value),
                        "effect_size": float(effect_size),
                        "significant": p_value < 0.05
                    }
                except Exception as e:
                    comparison["metrics"]["mrr"] = {"error": str(e)}

        return comparison

    def save_results(
        self,
        system_metrics: Dict[str, RetrievalMetrics],
        comparisons: List[Dict[str, Any]] = None
    ):
        """Save evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save metrics
        metrics_data = {
            name: asdict(metrics)
            for name, metrics in system_metrics.items()
        }
        metrics_file = self.output_dir / f"retrieval_metrics_{timestamp}.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2)

        # Save comparisons
        if comparisons:
            comparison_file = self.output_dir / f"comparisons_{timestamp}.json"
            with open(comparison_file, "w", encoding="utf-8") as f:
                json.dump(comparisons, f, indent=2)

        # Generate report
        report = self._generate_report(system_metrics, comparisons)
        report_file = self.output_dir / f"retrieval_report_{timestamp}.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"\nResults saved to {self.output_dir}")

    def _generate_report(
        self,
        system_metrics: Dict[str, RetrievalMetrics],
        comparisons: List[Dict[str, Any]] = None
    ) -> str:
        """Generate markdown report"""
        # Metrics table
        metrics_rows = []
        for name, metrics in system_metrics.items():
            metrics_rows.append(
                f"| {name} | {metrics.ndcg_at_5:.3f} | {metrics.ndcg_at_10:.3f} | "
                f"{metrics.mrr:.3f} | {metrics.map_score:.3f} | "
                f"{metrics.recall_at_10:.3f} | {metrics.precision_at_10:.3f} |"
            )
        metrics_table = "\n".join(metrics_rows)

        # Comparison analysis
        comparison_text = ""
        if comparisons:
            for comp in comparisons:
                for metric_name, data in comp.get("metrics", {}).items():
                    if isinstance(data, dict) and "p_value" in data:
                        sig = "Yes" if data.get("significant") else "No"
                        comparison_text += f"\n### {comp['system_a']} vs {comp['system_b']} ({metric_name})\n"
                        comparison_text += f"- Difference: {data.get('difference', 0):+.3f}\n"
                        comparison_text += f"- p-value: {data.get('p_value', 1):.4f}\n"
                        comparison_text += f"- Significant (p<0.05): {sig}\n"
                        comparison_text += f"- Effect size: {data.get('effect_size', 0):.3f}\n"

        report = f"""# Retrieval Evaluation Report

## Summary

Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## System Performance

| System | nDCG@5 | nDCG@10 | MRR | MAP | Recall@10 | P@10 |
|--------|--------|---------|-----|-----|-----------|------|
{metrics_table}

## Statistical Comparisons
{comparison_text if comparison_text else "No comparisons performed."}

## Interpretation

The results show the retrieval performance of different configurations.
Higher values indicate better retrieval quality.

Key findings:
- nDCG@10 measures ranking quality with graded relevance
- MRR measures how quickly relevant documents are found
- MAP provides an overall ranking quality measure

---
Generated by DocScanner AI Evaluation Framework
"""
        return report


def run_hyde_evaluation(evaluator: RetrievalEvaluator, queries: List[RetrievalQuery]):
    """Run evaluation for HyDE system"""
    print("\nEvaluating HyDE retrieval...")

    try:
        from app.ai.hyde import HyDEGenerator
        from app.ai.crag import GraphGuidedCRAG, RetrievedDocument

        hyde = HyDEGenerator()

        results = []
        for query in queries:
            start_time = datetime.now()

            # Generate HyDE query
            hyde_result = hyde.generate(query.query_text, prompt_type="labor_law")

            # Mock retrieval (in production, use actual search)
            retrieved_ids = []  # Would come from actual search

            result = RetrievalResult(
                query_id=query.query_id,
                retrieved_doc_ids=retrieved_ids,
                retrieval_method="hyde",
                hyde_query=hyde_result.primary_document,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
            results.append(result)

        return evaluator.evaluate_system("hyde", results, queries)

    except Exception as e:
        print(f"HyDE evaluation error: {e}")
        return RetrievalMetrics()


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval performance")
    parser.add_argument("--queries", type=str, help="Path to queries JSON")
    parser.add_argument("--results", type=str, help="Path to retrieval results JSON")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--system", type=str, default="default", help="System name")

    args = parser.parse_args()

    evaluator = RetrievalEvaluator(output_dir=args.output)

    # Load or create sample queries
    if args.queries:
        queries = evaluator.load_queries(args.queries)
    else:
        # Sample queries for demo
        queries = [
            RetrievalQuery(
                query_id="q1",
                query_text="포괄임금제가 유효한 경우는?",
                relevant_doc_ids=["doc_law_56", "doc_precedent_123"],
                relevance_scores={"doc_law_56": 3, "doc_precedent_123": 2, "doc_article_1": 1}
            ),
            RetrievalQuery(
                query_id="q2",
                query_text="최저임금 미달 시 처벌",
                relevant_doc_ids=["doc_minwage_6", "doc_penalty_1"],
                relevance_scores={"doc_minwage_6": 3, "doc_penalty_1": 2}
            )
        ]
        evaluator.queries = queries

    # Load or create sample results
    if args.results:
        with open(args.results, "r", encoding="utf-8") as f:
            results_data = json.load(f)
        results = [
            RetrievalResult(**r) for r in results_data
        ]
    else:
        # Sample results for demo
        results = [
            RetrievalResult(
                query_id="q1",
                retrieved_doc_ids=["doc_law_56", "doc_article_1", "doc_precedent_123", "doc_other"],
                scores=[0.9, 0.7, 0.6, 0.3],
                processing_time_ms=150
            ),
            RetrievalResult(
                query_id="q2",
                retrieved_doc_ids=["doc_minwage_6", "doc_penalty_1", "doc_other2"],
                scores=[0.85, 0.75, 0.4],
                processing_time_ms=120
            )
        ]

    # Evaluate
    metrics = evaluator.evaluate_system(args.system, results, queries)

    # Save results
    evaluator.save_results({args.system: metrics})

    # Print summary
    print("\n" + "="*60)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("="*60)
    print(f"System: {args.system}")
    print(f"Queries: {metrics.total_queries}")
    print(f"nDCG@5:  {metrics.ndcg_at_5:.3f}")
    print(f"nDCG@10: {metrics.ndcg_at_10:.3f}")
    print(f"MRR:     {metrics.mrr:.3f}")
    print(f"MAP:     {metrics.map_score:.3f}")
    print(f"R@10:    {metrics.recall_at_10:.3f}")
    print(f"P@10:    {metrics.precision_at_10:.3f}")
    print("="*60)


if __name__ == "__main__":
    main()
