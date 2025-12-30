"""
DocScanner AI Evaluation Scripts

This package contains scripts for comprehensive evaluation:
- eval_retrieval.py: IR metrics (NDCG, MRR, MAP, Recall, Precision)
- eval_hallucination.py: Legal citation verification and hallucination detection
- eval_baseline_comparison.py: DocScanner vs pure LLM comparison
"""

from .eval_retrieval import (
    RetrievalEvaluator,
    RetrievalMetrics,
    RetrievalMetricsCalculator,
    RetrievalQuery,
    RetrievalResult
)

from .eval_hallucination import (
    HallucinationEvaluator,
    HallucinationMetrics,
    LegalCitationExtractor,
    LegalCitationVerifier,
    LegalReference,
    VerificationResult
)

from .eval_baseline_comparison import (
    BaselineComparisonEvaluator,
    ComparisonMetrics,
    BaselineLLM,
    EvaluationSample
)

__all__ = [
    # Retrieval
    "RetrievalEvaluator",
    "RetrievalMetrics",
    "RetrievalMetricsCalculator",
    "RetrievalQuery",
    "RetrievalResult",

    # Hallucination
    "HallucinationEvaluator",
    "HallucinationMetrics",
    "LegalCitationExtractor",
    "LegalCitationVerifier",
    "LegalReference",
    "VerificationResult",

    # Baseline Comparison
    "BaselineComparisonEvaluator",
    "ComparisonMetrics",
    "BaselineLLM",
    "EvaluationSample"
]
