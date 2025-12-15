"""
DocScanner AI Evaluation Metrics
평가 지표 모듈
"""

from .retrieval_metrics import (
    RetrievalResult,
    RetrievalMetrics,
    recall_at_k,
    precision_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    mean_average_precision,
    calculate_all_retrieval_metrics,
    compare_systems
)

from .risk_detection_metrics import (
    RiskClause,
    RiskDetectionResult,
    ClauseLevelMetrics,
    SeverityMetrics,
    FinancialMetrics,
    calculate_clause_level_metrics,
    calculate_severity_metrics,
    calculate_financial_metrics,
    calculate_risk_type_accuracy,
    evaluate_redlining_quality,
    generate_evaluation_report
)

__all__ = [
    # Retrieval
    "RetrievalResult",
    "RetrievalMetrics",
    "recall_at_k",
    "precision_at_k",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "mean_average_precision",
    "calculate_all_retrieval_metrics",
    "compare_systems",

    # Risk Detection
    "RiskClause",
    "RiskDetectionResult",
    "ClauseLevelMetrics",
    "SeverityMetrics",
    "FinancialMetrics",
    "calculate_clause_level_metrics",
    "calculate_severity_metrics",
    "calculate_financial_metrics",
    "calculate_risk_type_accuracy",
    "evaluate_redlining_quality",
    "generate_evaluation_report"
]
