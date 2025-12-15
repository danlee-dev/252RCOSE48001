# DocScanner AI Module
# Advanced AI Pipeline for Legal Document Analysis

from .preprocessor import ContractPreprocessor
from .pii_masking import PIIMasker, mask_pii
from .hyde import HyDEGenerator, create_hyde_generator
from .raptor import RAPTORIndexer, ContractRAPTOR
from .constitutional_ai import ConstitutionalAI, get_labor_law_constitution
from .crag import GraphGuidedCRAG, CRAGWorkflow
from .legal_stress_test import LegalStressTest, run_stress_test
from .judge import LLMJudge, create_judge
from .vision_parser import VisionParser, ContractVisionAnalyzer
from .redlining import GenerativeRedlining, redline_contract
from .reasoning_trace import ReasoningTracer, TraceVisualizer
from .dspy_optimizer import DSPyOptimizer, SelfEvolvingPipeline
from .pipeline import AdvancedAIPipeline, PipelineConfig, PipelineResult, create_pipeline, quick_analyze

__all__ = [
    # Core
    "ContractPreprocessor",

    # Privacy
    "PIIMasker",
    "mask_pii",

    # Search Enhancement
    "HyDEGenerator",
    "create_hyde_generator",
    "RAPTORIndexer",
    "ContractRAPTOR",

    # Constitutional AI
    "ConstitutionalAI",
    "get_labor_law_constitution",

    # CRAG
    "GraphGuidedCRAG",
    "CRAGWorkflow",

    # Neuro-Symbolic
    "LegalStressTest",
    "run_stress_test",

    # Judge
    "LLMJudge",
    "create_judge",

    # Vision
    "VisionParser",
    "ContractVisionAnalyzer",

    # Redlining
    "GenerativeRedlining",
    "redline_contract",

    # XAI
    "ReasoningTracer",
    "TraceVisualizer",

    # Self-Evolving
    "DSPyOptimizer",
    "SelfEvolvingPipeline",

    # Integrated Pipeline
    "AdvancedAIPipeline",
    "PipelineConfig",
    "PipelineResult",
    "create_pipeline",
    "quick_analyze",
]
