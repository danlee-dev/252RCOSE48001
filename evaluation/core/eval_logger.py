"""
Evaluation Logger for DocScanner AI

Captures and logs all intermediate results for academic evaluation:
- Search queries (original, HyDE-enhanced)
- Retrieved documents with relevance scores
- LLM responses for hallucination detection
- Legal basis citations for verification

This module is designed for rigorous academic evaluation with:
- Structured JSON logging
- Statistical analysis support
- Reproducibility guarantees
"""

import os
import json
import time
import hashlib
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from threading import Lock
from contextlib import contextmanager
import uuid


@dataclass
class SearchQuery:
    """Represents a single search query and its transformations"""
    query_id: str
    original_query: str
    hyde_query: Optional[str] = None
    rewritten_queries: List[str] = field(default_factory=list)
    query_complexity: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedDoc:
    """Represents a single retrieved document"""
    doc_id: str
    text: str
    source: str
    score: float
    relevance_label: Optional[int] = None  # Ground truth label (if available)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Complete search result for evaluation"""
    result_id: str
    query: SearchQuery
    retrieved_docs: List[RetrievedDoc] = field(default_factory=list)
    retrieval_method: str = "unknown"
    processing_time_ms: float = 0.0
    crag_iterations: int = 0
    quality_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LegalCitation:
    """Legal basis citation for verification"""
    citation_id: str
    law_name: str  # e.g., "근로기준법"
    article_number: str  # e.g., "제56조"
    paragraph: Optional[str] = None
    cited_text: str = ""
    context: str = ""  # Surrounding text where citation appears
    verified: Optional[bool] = None
    verification_source: Optional[str] = None


@dataclass
class AnalysisResult:
    """Complete analysis result for hallucination detection"""
    analysis_id: str
    contract_id: str
    contract_text: str

    # Analysis outputs
    risk_level: str = "Unknown"
    risk_score: float = 0.0
    analysis_summary: str = ""

    # Violations detected
    violations: List[Dict[str, Any]] = field(default_factory=list)

    # Legal citations for verification
    legal_citations: List[LegalCitation] = field(default_factory=list)

    # Metadata
    processing_time_s: float = 0.0
    pipeline_version: str = "2.0.0"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # For comparison
    baseline_comparison: Optional[Dict[str, Any]] = None


class EvaluationLogger:
    """
    Thread-safe logger for capturing evaluation data

    Usage:
        logger = EvaluationLogger(experiment_name="retrieval_eval_v1")

        with logger.log_search(query="포괄임금제") as search_log:
            # Perform search
            search_log.set_hyde_query(hyde_result)
            search_log.add_retrieved_docs(docs)

        # Save all logs
        logger.save()
    """

    _instance: Optional['EvaluationLogger'] = None
    _lock = Lock()

    def __init__(
        self,
        experiment_name: str = "default",
        output_dir: str = None,
        auto_save: bool = True,
        save_interval: int = 100
    ):
        self.experiment_name = experiment_name
        self.experiment_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Output directory
        if output_dir is None:
            base_dir = Path(__file__).parent.parent / "logs"
        else:
            base_dir = Path(output_dir)

        self.output_dir = base_dir / self.experiment_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data storage
        self.search_results: List[SearchResult] = []
        self.analysis_results: List[AnalysisResult] = []
        self.baseline_comparisons: List[Dict[str, Any]] = []

        # Configuration
        self.auto_save = auto_save
        self.save_interval = save_interval
        self._write_lock = Lock()

        # Counters
        self._search_count = 0
        self._analysis_count = 0

    @classmethod
    def get_instance(cls, **kwargs) -> 'EvaluationLogger':
        """Get singleton instance"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(**kwargs)
            return cls._instance

    @classmethod
    def set_instance(cls, logger: 'EvaluationLogger'):
        """Set singleton instance"""
        with cls._lock:
            cls._instance = logger

    @classmethod
    def clear_instance(cls):
        """Clear singleton instance"""
        with cls._lock:
            if cls._instance:
                cls._instance.save()
            cls._instance = None

    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID"""
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    @contextmanager
    def log_search(
        self,
        query: str,
        retrieval_method: str = "default"
    ):
        """
        Context manager for logging search operations

        Usage:
            with logger.log_search("포괄임금제") as log:
                log.set_hyde_query(hyde_result)
                log.add_docs(retrieved_docs)
        """
        search_log = SearchLogContext(
            query_id=self._generate_id("q"),
            original_query=query,
            retrieval_method=retrieval_method,
            logger=self
        )

        start_time = time.time()
        try:
            yield search_log
        finally:
            search_log.processing_time_ms = (time.time() - start_time) * 1000
            self._add_search_result(search_log.to_search_result())

    def _add_search_result(self, result: SearchResult):
        """Add search result (thread-safe)"""
        with self._write_lock:
            self.search_results.append(result)
            self._search_count += 1

            if self.auto_save and self._search_count % self.save_interval == 0:
                self._save_search_results()

    def log_analysis(
        self,
        contract_id: str,
        contract_text: str,
        analysis_output: Dict[str, Any],
        legal_citations: List[Dict[str, Any]] = None,
        baseline_output: Dict[str, Any] = None
    ) -> str:
        """
        Log analysis result for hallucination detection

        Args:
            contract_id: Contract identifier
            contract_text: Original contract text
            analysis_output: DocScanner pipeline output
            legal_citations: Extracted legal citations
            baseline_output: LLM baseline output (for comparison)

        Returns:
            analysis_id
        """
        analysis_id = self._generate_id("a")

        # Parse legal citations
        citations = []
        if legal_citations:
            for c in legal_citations:
                citations.append(LegalCitation(
                    citation_id=self._generate_id("c"),
                    law_name=c.get("law_name", ""),
                    article_number=c.get("article", ""),
                    paragraph=c.get("paragraph"),
                    cited_text=c.get("text", ""),
                    context=c.get("context", "")
                ))

        # Create analysis result
        result = AnalysisResult(
            analysis_id=analysis_id,
            contract_id=contract_id,
            contract_text=contract_text,
            risk_level=analysis_output.get("risk_level", "Unknown"),
            risk_score=analysis_output.get("risk_score", 0.0),
            analysis_summary=analysis_output.get("analysis_summary", ""),
            violations=analysis_output.get("stress_test", {}).get("violations", []),
            legal_citations=citations,
            processing_time_s=analysis_output.get("processing_time", 0.0),
            pipeline_version=analysis_output.get("pipeline_version", "2.0.0"),
            baseline_comparison=baseline_output
        )

        with self._write_lock:
            self.analysis_results.append(result)
            self._analysis_count += 1

            if self.auto_save and self._analysis_count % self.save_interval == 0:
                self._save_analysis_results()

        return analysis_id

    def log_baseline_comparison(
        self,
        contract_id: str,
        docscanner_output: Dict[str, Any],
        baseline_output: Dict[str, Any],
        ground_truth: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ):
        """
        Log comparison between DocScanner and baseline LLM

        Args:
            contract_id: Contract identifier
            docscanner_output: Full DocScanner pipeline output
            baseline_output: Baseline LLM output (zero-shot)
            ground_truth: Expert-labeled ground truth (if available)
            metadata: Additional metadata
        """
        comparison = {
            "comparison_id": self._generate_id("cmp"),
            "contract_id": contract_id,
            "timestamp": datetime.now().isoformat(),
            "docscanner": {
                "risk_level": docscanner_output.get("risk_level"),
                "risk_score": docscanner_output.get("risk_score"),
                "violations": docscanner_output.get("stress_test", {}).get("violations", []),
                "underpayment": docscanner_output.get("stress_test", {}).get("annual_underpayment", 0),
                "processing_time": docscanner_output.get("processing_time", 0)
            },
            "baseline": {
                "risk_level": baseline_output.get("risk_level"),
                "risk_score": baseline_output.get("risk_score"),
                "violations": baseline_output.get("violations", []),
                "underpayment": baseline_output.get("annual_underpayment", 0),
                "processing_time": baseline_output.get("processing_time", 0)
            },
            "ground_truth": ground_truth,
            "metadata": metadata or {}
        }

        with self._write_lock:
            self.baseline_comparisons.append(comparison)

    def _save_search_results(self):
        """Save search results to file"""
        output_file = self.output_dir / "search_results.jsonl"
        with open(output_file, "a", encoding="utf-8") as f:
            for result in self.search_results[-self.save_interval:]:
                f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    def _save_analysis_results(self):
        """Save analysis results to file"""
        output_file = self.output_dir / "analysis_results.jsonl"
        with open(output_file, "a", encoding="utf-8") as f:
            for result in self.analysis_results[-self.save_interval:]:
                f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    def save(self):
        """Save all results to files"""
        with self._write_lock:
            # Search results
            search_file = self.output_dir / "search_results.json"
            with open(search_file, "w", encoding="utf-8") as f:
                json.dump(
                    [asdict(r) for r in self.search_results],
                    f,
                    ensure_ascii=False,
                    indent=2
                )

            # Analysis results
            analysis_file = self.output_dir / "analysis_results.json"
            with open(analysis_file, "w", encoding="utf-8") as f:
                json.dump(
                    [asdict(r) for r in self.analysis_results],
                    f,
                    ensure_ascii=False,
                    indent=2
                )

            # Baseline comparisons
            comparison_file = self.output_dir / "baseline_comparisons.json"
            with open(comparison_file, "w", encoding="utf-8") as f:
                json.dump(
                    self.baseline_comparisons,
                    f,
                    ensure_ascii=False,
                    indent=2
                )

            # Experiment metadata
            metadata_file = self.output_dir / "experiment_metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump({
                    "experiment_name": self.experiment_name,
                    "experiment_id": self.experiment_id,
                    "total_searches": len(self.search_results),
                    "total_analyses": len(self.analysis_results),
                    "total_comparisons": len(self.baseline_comparisons),
                    "saved_at": datetime.now().isoformat()
                }, f, indent=2)

            print(f"[EvalLogger] Saved to {self.output_dir}")
            print(f"  - Search results: {len(self.search_results)}")
            print(f"  - Analysis results: {len(self.analysis_results)}")
            print(f"  - Baseline comparisons: {len(self.baseline_comparisons)}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get evaluation statistics"""
        return {
            "total_searches": len(self.search_results),
            "total_analyses": len(self.analysis_results),
            "total_comparisons": len(self.baseline_comparisons),
            "avg_search_time_ms": sum(r.processing_time_ms for r in self.search_results) / max(1, len(self.search_results)),
            "avg_docs_retrieved": sum(len(r.retrieved_docs) for r in self.search_results) / max(1, len(self.search_results))
        }


class SearchLogContext:
    """Context object for logging search operations"""

    def __init__(
        self,
        query_id: str,
        original_query: str,
        retrieval_method: str,
        logger: EvaluationLogger
    ):
        self.query_id = query_id
        self.original_query = original_query
        self.retrieval_method = retrieval_method
        self.logger = logger

        self.hyde_query: Optional[str] = None
        self.rewritten_queries: List[str] = []
        self.query_complexity: Optional[str] = None
        self.retrieved_docs: List[RetrievedDoc] = []
        self.processing_time_ms: float = 0.0
        self.crag_iterations: int = 0
        self.quality_score: Optional[float] = None
        self.metadata: Dict[str, Any] = {}

    def set_hyde_query(self, hyde_query: str):
        """Set HyDE-enhanced query"""
        self.hyde_query = hyde_query

    def set_rewritten_queries(self, queries: List[str]):
        """Set CRAG rewritten queries"""
        self.rewritten_queries = queries

    def set_query_complexity(self, complexity: str):
        """Set analyzed query complexity"""
        self.query_complexity = complexity

    def add_doc(
        self,
        doc_id: str,
        text: str,
        source: str,
        score: float,
        relevance_label: int = None,
        metadata: Dict[str, Any] = None
    ):
        """Add a single retrieved document"""
        self.retrieved_docs.append(RetrievedDoc(
            doc_id=doc_id,
            text=text,
            source=source,
            score=score,
            relevance_label=relevance_label,
            metadata=metadata or {}
        ))

    def add_docs(self, docs: List[Dict[str, Any]]):
        """Add multiple retrieved documents"""
        for doc in docs:
            self.add_doc(
                doc_id=doc.get("id", str(uuid.uuid4())[:8]),
                text=doc.get("text", ""),
                source=doc.get("source", ""),
                score=doc.get("score", 0.0),
                relevance_label=doc.get("relevance_label"),
                metadata=doc.get("metadata", {})
            )

    def set_crag_iterations(self, iterations: int):
        """Set CRAG correction iterations"""
        self.crag_iterations = iterations

    def set_quality_score(self, score: float):
        """Set retrieval quality score"""
        self.quality_score = score

    def set_metadata(self, key: str, value: Any):
        """Set metadata"""
        self.metadata[key] = value

    def to_search_result(self) -> SearchResult:
        """Convert to SearchResult"""
        return SearchResult(
            result_id=self.query_id,
            query=SearchQuery(
                query_id=self.query_id,
                original_query=self.original_query,
                hyde_query=self.hyde_query,
                rewritten_queries=self.rewritten_queries,
                query_complexity=self.query_complexity
            ),
            retrieved_docs=self.retrieved_docs,
            retrieval_method=self.retrieval_method,
            processing_time_ms=self.processing_time_ms,
            crag_iterations=self.crag_iterations,
            quality_score=self.quality_score,
            metadata=self.metadata
        )


# Convenience functions
def get_eval_logger(**kwargs) -> EvaluationLogger:
    """Get the global evaluation logger instance"""
    return EvaluationLogger.get_instance(**kwargs)


def init_eval_logger(experiment_name: str, **kwargs) -> EvaluationLogger:
    """Initialize a new evaluation logger"""
    logger = EvaluationLogger(experiment_name=experiment_name, **kwargs)
    EvaluationLogger.set_instance(logger)
    return logger


def close_eval_logger():
    """Close and save the evaluation logger"""
    EvaluationLogger.clear_instance()
