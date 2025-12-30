"""
Retrieval Wrapper for Evaluation

Wraps HyDE and CRAG modules to capture all queries and results
for academic evaluation without modifying core modules.

This approach ensures:
1. Clean separation between production and evaluation code
2. No performance impact when evaluation is disabled
3. Comprehensive logging of all intermediate steps
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

# Add backend to path
backend_path = Path(__file__).parent.parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Import evaluation logger
from evaluation.core.eval_logger import EvaluationLogger, get_eval_logger


@dataclass
class EvalConfig:
    """Evaluation configuration"""
    enabled: bool = True
    log_hyde_queries: bool = True
    log_crag_results: bool = True
    log_intermediate_steps: bool = True
    experiment_name: str = "retrieval_eval"


class HyDEEvalWrapper:
    """
    Wrapper for HyDE module with evaluation logging

    Usage:
        from app.ai.hyde import HyDEGenerator

        hyde = HyDEGenerator()
        eval_hyde = HyDEEvalWrapper(hyde, eval_config)

        result = eval_hyde.generate("포괄임금제")  # Automatically logged
    """

    def __init__(
        self,
        hyde_generator,
        eval_config: EvalConfig = None,
        logger: EvaluationLogger = None
    ):
        self.hyde = hyde_generator
        self.config = eval_config or EvalConfig()
        self.logger = logger or get_eval_logger(experiment_name=self.config.experiment_name)

    def generate(
        self,
        query: str,
        prompt_type: str = "labor_law",
        generate_embedding: bool = True,
        force_refresh: bool = False
    ):
        """Generate with logging"""
        if not self.config.enabled:
            return self.hyde.generate(query, prompt_type, generate_embedding, force_refresh)

        start_time = time.time()

        # Call original method
        result = self.hyde.generate(query, prompt_type, generate_embedding, force_refresh)

        # Log to evaluation logger
        if self.config.log_hyde_queries:
            with self.logger.log_search(query, retrieval_method="hyde") as log:
                log.set_hyde_query(result.primary_document)
                log.set_query_complexity(result.query_complexity.value)
                log.set_metadata("strategy", result.strategy_used.value)
                log.set_metadata("num_documents", len(result.hypothetical_documents))
                log.set_metadata("cache_hit", result.cache_hit)
                log.set_metadata("prompt_type", prompt_type)

                # Log all hypothetical documents
                for i, doc in enumerate(result.hypothetical_documents):
                    log.set_metadata(f"hyde_doc_{i}", doc[:500])

        return result

    def enhance_query(self, query: str, include_all_documents: bool = False) -> str:
        """Enhance query with logging"""
        if not self.config.enabled:
            return self.hyde.enhance_query(query, include_all_documents)

        with self.logger.log_search(query, retrieval_method="hyde_enhance") as log:
            enhanced = self.hyde.enhance_query(query, include_all_documents)
            log.set_hyde_query(enhanced)
            log.set_metadata("include_all_documents", include_all_documents)

        return enhanced


class CRAGEvalWrapper:
    """
    Wrapper for CRAG module with evaluation logging

    Usage:
        from app.ai.crag import GraphGuidedCRAG

        crag = GraphGuidedCRAG()
        eval_crag = CRAGEvalWrapper(crag, eval_config)

        result = eval_crag.retrieve_and_correct_sync(query, initial_docs)
    """

    def __init__(
        self,
        crag,
        eval_config: EvalConfig = None,
        logger: EvaluationLogger = None
    ):
        self.crag = crag
        self.config = eval_config or EvalConfig()
        self.logger = logger or get_eval_logger(experiment_name=self.config.experiment_name)

    def retrieve_and_correct_sync(
        self,
        query: str,
        initial_docs,
        max_graph_hops: int = 2
    ):
        """Retrieve and correct with logging"""
        if not self.config.enabled:
            return self.crag.retrieve_and_correct_sync(query, initial_docs, max_graph_hops)

        start_time = time.time()

        # Call original method
        result = self.crag.retrieve_and_correct_sync(query, initial_docs, max_graph_hops)

        # Log to evaluation logger
        if self.config.log_crag_results:
            with self.logger.log_search(query, retrieval_method="crag") as log:
                # Log rewritten queries
                log.set_rewritten_queries(result.rewritten_queries)
                log.set_crag_iterations(result.correction_iterations)
                log.set_quality_score(result.confidence_score)

                # Log initial docs
                log.set_metadata("initial_doc_count", len(result.initial_docs))

                # Log retrieved docs
                for doc in result.all_docs:
                    log.add_doc(
                        doc_id=doc.id,
                        text=doc.text,
                        source=doc.source,
                        score=doc.score,
                        metadata={
                            "relevance": doc.relevance.value,
                            "confidence": doc.confidence,
                            "extracted_info": doc.extracted_info
                        }
                    )

                # Log quality evaluation
                if result.quality_evaluation:
                    log.set_metadata("quality", result.quality.value)
                    log.set_metadata("quality_confidence", result.quality_evaluation.confidence)
                    log.set_metadata("correction_strategy", result.quality_evaluation.correction_strategy.value)
                    log.set_metadata("missing_info", result.quality_evaluation.missing_info)

                # Log reasoning trace
                log.set_metadata("reasoning_trace", result.reasoning_trace)

        return result


class PipelineEvalWrapper:
    """
    Wrapper for the entire analysis pipeline with evaluation logging

    Captures both search operations and final analysis results
    for comprehensive evaluation.
    """

    def __init__(
        self,
        pipeline,
        eval_config: EvalConfig = None,
        logger: EvaluationLogger = None
    ):
        self.pipeline = pipeline
        self.config = eval_config or EvalConfig()
        self.logger = logger or get_eval_logger(experiment_name=self.config.experiment_name)

        # Wrap internal components
        if self.config.enabled:
            self.pipeline.hyde = HyDEEvalWrapper(
                self.pipeline.hyde,
                self.config,
                self.logger
            )
            self.pipeline.crag = CRAGEvalWrapper(
                self.pipeline.crag,
                self.config,
                self.logger
            )

    def analyze(
        self,
        contract_text: str,
        contract_id: str = None,
        file_path: str = None
    ):
        """Analyze with logging"""
        start_time = time.time()

        # Call original analysis
        result = self.pipeline.analyze(contract_text, contract_id, file_path)

        # Log analysis result
        if self.config.enabled:
            # Extract legal citations from analysis
            legal_citations = self._extract_legal_citations(result)

            self.logger.log_analysis(
                contract_id=result.contract_id,
                contract_text=contract_text,
                analysis_output=result.to_dict(),
                legal_citations=legal_citations
            )

        return result

    def _extract_legal_citations(self, result) -> List[Dict[str, Any]]:
        """Extract legal citations from analysis result for verification"""
        citations = []

        # Extract from violations
        if result.clause_analysis and result.clause_analysis.violations:
            for v in result.clause_analysis.violations:
                if v.legal_basis:
                    citations.append({
                        "law_name": self._parse_law_name(v.legal_basis),
                        "article": self._parse_article_number(v.legal_basis),
                        "text": v.legal_basis,
                        "context": v.description
                    })

        # Extract from stress test
        if result.stress_test and result.stress_test.violations:
            for v in result.stress_test.violations:
                if v.get("legal_basis"):
                    citations.append({
                        "law_name": self._parse_law_name(v["legal_basis"]),
                        "article": self._parse_article_number(v["legal_basis"]),
                        "text": v["legal_basis"],
                        "context": v.get("description", "")
                    })

        return citations

    def _parse_law_name(self, citation: str) -> str:
        """Parse law name from citation"""
        import re

        law_patterns = [
            r'(근로기준법)',
            r'(최저임금법)',
            r'(근로자퇴직급여보장법)',
            r'(남녀고용평등법)',
            r'(산업안전보건법)'
        ]

        for pattern in law_patterns:
            match = re.search(pattern, citation)
            if match:
                return match.group(1)

        return "Unknown"

    def _parse_article_number(self, citation: str) -> str:
        """Parse article number from citation"""
        import re

        match = re.search(r'제\s*(\d+)\s*조', citation)
        if match:
            return f"제{match.group(1)}조"

        return ""


def create_eval_pipeline(
    experiment_name: str = "default_eval",
    enable_logging: bool = True,
    **pipeline_kwargs
) -> PipelineEvalWrapper:
    """
    Factory function to create an evaluation-enabled pipeline

    Usage:
        pipeline = create_eval_pipeline("retrieval_eval_v1")
        result = pipeline.analyze(contract_text)
    """
    from app.ai.pipeline import AdvancedAIPipeline, PipelineConfig

    # Create base pipeline
    config = pipeline_kwargs.get("config") or PipelineConfig()
    base_pipeline = AdvancedAIPipeline(config=config)

    # Create evaluation config
    eval_config = EvalConfig(
        enabled=enable_logging,
        experiment_name=experiment_name
    )

    # Wrap with evaluation
    return PipelineEvalWrapper(base_pipeline, eval_config)
