"""
DocScanner AI Evaluation Core Module

This module provides tools for rigorous academic evaluation of the DocScanner AI system.
"""

from .eval_logger import (
    EvaluationLogger,
    SearchLogContext,
    SearchQuery,
    SearchResult,
    RetrievedDoc,
    LegalCitation,
    AnalysisResult,
    get_eval_logger,
    init_eval_logger,
    close_eval_logger
)

from .llm_evaluators import (
    BaseLLMEvaluator,
    OpenAIEvaluator,
    GeminiEvaluator,
    ClaudeEvaluator,
    MultiLLMEvaluator,
    EvaluatorType,
    EvaluationDimension,
    EvaluationResult,
    CrossValidationResult
)

__all__ = [
    # Evaluation Logger
    "EvaluationLogger",
    "SearchLogContext",
    "SearchQuery",
    "SearchResult",
    "RetrievedDoc",
    "LegalCitation",
    "AnalysisResult",
    "get_eval_logger",
    "init_eval_logger",
    "close_eval_logger",
    # LLM Evaluators
    "BaseLLMEvaluator",
    "OpenAIEvaluator",
    "GeminiEvaluator",
    "ClaudeEvaluator",
    "MultiLLMEvaluator",
    "EvaluatorType",
    "EvaluationDimension",
    "EvaluationResult",
    "CrossValidationResult"
]
