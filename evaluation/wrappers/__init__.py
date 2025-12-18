"""
Evaluation Wrappers Module

Provides wrappers for DocScanner AI components that enable
evaluation logging without modifying core modules.
"""

from .retrieval_wrapper import (
    HyDEEvalWrapper,
    CRAGEvalWrapper,
    PipelineEvalWrapper,
    EvalConfig,
    create_eval_pipeline
)

__all__ = [
    "HyDEEvalWrapper",
    "CRAGEvalWrapper",
    "PipelineEvalWrapper",
    "EvalConfig",
    "create_eval_pipeline"
]
