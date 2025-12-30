"""
DocScanner AI Evaluation Framework

A comprehensive evaluation framework for academic assessment of the
DocScanner AI contract analysis system.

## Evaluation Components

1. **Retrieval Evaluation** (`scripts/eval_retrieval.py`)
   - Standard IR metrics: nDCG, MRR, MAP, Recall@K, Precision@K
   - System comparison with statistical significance tests

2. **Hallucination Detection** (`scripts/eval_hallucination.py`)
   - Legal citation verification against law database
   - Factual accuracy assessment

3. **Baseline Comparison** (`scripts/eval_baseline_comparison.py`)
   - DocScanner vs pure LLM prompting
   - Risk detection F1, underpayment MAE
   - Statistical significance with paired tests

## Quick Start

```python
# Run all evaluations
from evaluation import run_all_evaluations
run_all_evaluations.main()

# Or individual evaluators
from evaluation.scripts import RetrievalEvaluator, HallucinationEvaluator

evaluator = RetrievalEvaluator()
metrics = evaluator.evaluate_system("hyde", results, queries)
```

## Usage from Command Line

```bash
# Run all evaluations
python -m evaluation.run_all_evaluations --all

# Run specific evaluation
python -m evaluation.scripts.eval_retrieval --data queries.json
python -m evaluation.scripts.eval_hallucination --data results.json
python -m evaluation.scripts.eval_baseline_comparison --data contracts.json
```
"""

__version__ = "1.0.0"
__author__ = "DocScanner AI Team"
