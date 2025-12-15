"""
Retrieval Evaluation Metrics
정보 검색 성능 평가 지표
"""

import numpy as np
from typing import List, Dict, Any, Set
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """검색 결과"""
    query_id: str
    retrieved_doc_ids: List[str]
    scores: List[float]


@dataclass
class RetrievalMetrics:
    """검색 평가 결과"""
    recall_at_k: Dict[int, float]
    precision_at_k: Dict[int, float]
    mrr: float
    ndcg_at_k: Dict[int, float]
    map_score: float


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Recall@K: 상위 K개 결과 중 관련 문서 포함 비율

    Args:
        retrieved: 검색된 문서 ID 리스트 (순위순)
        relevant: 관련 문서 ID 집합
        k: 상위 K개

    Returns:
        Recall@K 값
    """
    if not relevant:
        return 0.0

    retrieved_at_k = set(retrieved[:k])
    hits = len(retrieved_at_k & relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Precision@K: 상위 K개 결과의 정밀도

    Args:
        retrieved: 검색된 문서 ID 리스트 (순위순)
        relevant: 관련 문서 ID 집합
        k: 상위 K개

    Returns:
        Precision@K 값
    """
    if k == 0:
        return 0.0

    retrieved_at_k = set(retrieved[:k])
    hits = len(retrieved_at_k & relevant)
    return hits / k


def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Reciprocal Rank: 첫 관련 문서 순위의 역수

    Args:
        retrieved: 검색된 문서 ID 리스트 (순위순)
        relevant: 관련 문서 ID 집합

    Returns:
        Reciprocal Rank 값
    """
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def mean_reciprocal_rank(results: List[RetrievalResult],
                         ground_truth: Dict[str, Set[str]]) -> float:
    """
    Mean Reciprocal Rank (MRR)

    Args:
        results: 검색 결과 리스트
        ground_truth: 쿼리별 관련 문서 ID 집합

    Returns:
        MRR 값
    """
    if not results:
        return 0.0

    rr_sum = 0.0
    for result in results:
        relevant = ground_truth.get(result.query_id, set())
        rr_sum += reciprocal_rank(result.retrieved_doc_ids, relevant)

    return rr_sum / len(results)


def dcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Discounted Cumulative Gain at K

    Args:
        retrieved: 검색된 문서 ID 리스트 (순위순)
        relevant: 관련 문서 ID 집합
        k: 상위 K개

    Returns:
        DCG@K 값
    """
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        rel = 1.0 if doc_id in relevant else 0.0
        dcg += rel / np.log2(i + 2)  # i+2 because i starts from 0
    return dcg


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at K

    Args:
        retrieved: 검색된 문서 ID 리스트 (순위순)
        relevant: 관련 문서 ID 집합
        k: 상위 K개

    Returns:
        nDCG@K 값
    """
    dcg = dcg_at_k(retrieved, relevant, k)

    # Ideal DCG: 모든 관련 문서가 상위에 있는 경우
    ideal_k = min(k, len(relevant))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_k))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def average_precision(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Average Precision: 각 관련 문서 위치에서의 정밀도 평균

    Args:
        retrieved: 검색된 문서 ID 리스트 (순위순)
        relevant: 관련 문서 ID 집합

    Returns:
        AP 값
    """
    if not relevant:
        return 0.0

    hits = 0
    sum_precision = 0.0

    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            hits += 1
            sum_precision += hits / (i + 1)

    return sum_precision / len(relevant)


def mean_average_precision(results: List[RetrievalResult],
                           ground_truth: Dict[str, Set[str]]) -> float:
    """
    Mean Average Precision (MAP)

    Args:
        results: 검색 결과 리스트
        ground_truth: 쿼리별 관련 문서 ID 집합

    Returns:
        MAP 값
    """
    if not results:
        return 0.0

    ap_sum = 0.0
    for result in results:
        relevant = ground_truth.get(result.query_id, set())
        ap_sum += average_precision(result.retrieved_doc_ids, relevant)

    return ap_sum / len(results)


def calculate_all_retrieval_metrics(
    results: List[RetrievalResult],
    ground_truth: Dict[str, Set[str]],
    k_values: List[int] = [1, 3, 5, 10, 20]
) -> RetrievalMetrics:
    """
    모든 검색 평가 지표 계산

    Args:
        results: 검색 결과 리스트
        ground_truth: 쿼리별 관련 문서 ID 집합
        k_values: K 값 리스트

    Returns:
        RetrievalMetrics 객체
    """
    recall = {}
    precision = {}
    ndcg = {}

    for k in k_values:
        recall_scores = []
        precision_scores = []
        ndcg_scores = []

        for result in results:
            relevant = ground_truth.get(result.query_id, set())
            recall_scores.append(recall_at_k(result.retrieved_doc_ids, relevant, k))
            precision_scores.append(precision_at_k(result.retrieved_doc_ids, relevant, k))
            ndcg_scores.append(ndcg_at_k(result.retrieved_doc_ids, relevant, k))

        recall[k] = np.mean(recall_scores)
        precision[k] = np.mean(precision_scores)
        ndcg[k] = np.mean(ndcg_scores)

    return RetrievalMetrics(
        recall_at_k=recall,
        precision_at_k=precision,
        mrr=mean_reciprocal_rank(results, ground_truth),
        ndcg_at_k=ndcg,
        map_score=mean_average_precision(results, ground_truth)
    )


def compare_systems(
    baseline_results: List[RetrievalResult],
    proposed_results: List[RetrievalResult],
    ground_truth: Dict[str, Set[str]],
    k: int = 5
) -> Dict[str, Any]:
    """
    두 시스템 성능 비교 및 통계적 유의성 검정

    Args:
        baseline_results: 베이스라인 검색 결과
        proposed_results: 제안 시스템 검색 결과
        ground_truth: 정답 데이터
        k: 비교할 K 값

    Returns:
        비교 결과 및 통계 검정 결과
    """
    from scipy import stats

    baseline_metrics = calculate_all_retrieval_metrics(baseline_results, ground_truth, [k])
    proposed_metrics = calculate_all_retrieval_metrics(proposed_results, ground_truth, [k])

    # 쿼리별 점수 계산
    baseline_scores = []
    proposed_scores = []

    for baseline, proposed in zip(baseline_results, proposed_results):
        relevant = ground_truth.get(baseline.query_id, set())
        baseline_scores.append(recall_at_k(baseline.retrieved_doc_ids, relevant, k))
        proposed_scores.append(recall_at_k(proposed.retrieved_doc_ids, relevant, k))

    baseline_scores = np.array(baseline_scores)
    proposed_scores = np.array(proposed_scores)

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(proposed_scores, baseline_scores)

    # Effect size (Cohen's d)
    diff = proposed_scores - baseline_scores
    cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0

    return {
        "baseline": {
            "recall@k": baseline_metrics.recall_at_k[k],
            "precision@k": baseline_metrics.precision_at_k[k],
            "mrr": baseline_metrics.mrr
        },
        "proposed": {
            "recall@k": proposed_metrics.recall_at_k[k],
            "precision@k": proposed_metrics.precision_at_k[k],
            "mrr": proposed_metrics.mrr
        },
        "improvement": {
            "recall@k": (proposed_metrics.recall_at_k[k] - baseline_metrics.recall_at_k[k]) /
                        baseline_metrics.recall_at_k[k] * 100 if baseline_metrics.recall_at_k[k] > 0 else 0,
        },
        "statistical_test": {
            "test": "paired_t_test",
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant_at_0.05": p_value < 0.05,
            "significant_at_0.01": p_value < 0.01,
            "cohens_d": cohens_d,
            "effect_size": "large" if abs(cohens_d) >= 0.8 else
                          "medium" if abs(cohens_d) >= 0.5 else "small"
        }
    }


# 테스트 코드
if __name__ == "__main__":
    # 테스트 데이터
    results = [
        RetrievalResult("q1", ["d1", "d2", "d3", "d4", "d5"], [0.9, 0.8, 0.7, 0.6, 0.5]),
        RetrievalResult("q2", ["d2", "d1", "d4", "d3", "d6"], [0.95, 0.85, 0.75, 0.65, 0.55]),
    ]

    ground_truth = {
        "q1": {"d1", "d3"},
        "q2": {"d1", "d2", "d5"},
    }

    metrics = calculate_all_retrieval_metrics(results, ground_truth)

    print("Retrieval Metrics:")
    print(f"  Recall@5: {metrics.recall_at_k[5]:.4f}")
    print(f"  Precision@5: {metrics.precision_at_k[5]:.4f}")
    print(f"  MRR: {metrics.mrr:.4f}")
    print(f"  nDCG@5: {metrics.ndcg_at_k[5]:.4f}")
    print(f"  MAP: {metrics.map_score:.4f}")
