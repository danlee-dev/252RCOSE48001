#!/usr/bin/env python3
"""
DocScanner AI Full Evaluation Runner
전체 평가 실행 스크립트
"""

import os
import sys
import json
import yaml
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from tqdm import tqdm

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from evaluation.metrics.retrieval_metrics import (
    calculate_all_retrieval_metrics,
    compare_systems,
    RetrievalResult
)
from evaluation.metrics.risk_detection_metrics import (
    generate_evaluation_report,
    RiskDetectionResult,
    RiskClause
)


@dataclass
class EvaluationConfig:
    """평가 설정"""
    experiment_name: str
    output_dir: Path
    seed: int
    k_values: List[int]
    run_retrieval: bool = True
    run_risk_detection: bool = True
    run_ablation: bool = True
    run_human_eval: bool = False


class DocScannerEvaluator:
    """DocScanner AI 평가기"""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.results = {}
        self.logger = self._setup_logging()

        # 출력 디렉토리 생성
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # 랜덤 시드 설정
        np.random.seed(self.config.seed)

    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger("DocScannerEval")
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 파일 핸들러
        log_file = self.config.output_dir / "evaluation.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def run_full_evaluation(self):
        """전체 평가 실행"""
        self.logger.info(f"Starting evaluation: {self.config.experiment_name}")
        start_time = datetime.now()

        try:
            if self.config.run_retrieval:
                self.logger.info("Running retrieval evaluation...")
                self.results['retrieval'] = self._evaluate_retrieval()

            if self.config.run_risk_detection:
                self.logger.info("Running risk detection evaluation...")
                self.results['risk_detection'] = self._evaluate_risk_detection()

            if self.config.run_ablation:
                self.logger.info("Running ablation study...")
                self.results['ablation'] = self._run_ablation_study()

            # 결과 저장
            self._save_results()

            # 보고서 생성
            self._generate_report()

        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise

        elapsed = datetime.now() - start_time
        self.logger.info(f"Evaluation completed in {elapsed}")

        return self.results

    def _evaluate_retrieval(self) -> Dict[str, Any]:
        """검색 성능 평가"""
        from backend.app.ai.hyde import HyDEGenerator
        from backend.app.ai.crag import GraphGuidedCRAG

        results = {}

        # 데이터셋 로드
        queries, ground_truth = self._load_retrieval_dataset()

        # 베이스라인 시스템 평가
        baselines = ['bm25', 'dense', 'muvera', 'muvera_hyde', 'full_pipeline']

        for baseline in baselines:
            self.logger.info(f"  Evaluating {baseline}...")

            # 각 베이스라인으로 검색 수행
            search_results = self._run_retrieval_baseline(baseline, queries)

            # 메트릭 계산
            metrics = calculate_all_retrieval_metrics(
                search_results,
                ground_truth,
                k_values=self.config.k_values
            )

            results[baseline] = {
                'recall_at_k': metrics.recall_at_k,
                'precision_at_k': metrics.precision_at_k,
                'mrr': metrics.mrr,
                'ndcg_at_k': metrics.ndcg_at_k,
                'map': metrics.map_score
            }

        # 시스템 간 비교 (통계적 유의성)
        if 'muvera' in results and 'full_pipeline' in results:
            baseline_results = self._run_retrieval_baseline('muvera', queries)
            proposed_results = self._run_retrieval_baseline('full_pipeline', queries)
            results['comparison'] = compare_systems(
                baseline_results, proposed_results, ground_truth, k=5
            )

        return results

    def _evaluate_risk_detection(self) -> Dict[str, Any]:
        """위험 탐지 성능 평가"""
        from backend.app.ai.pipeline import AdvancedAIPipeline, PipelineConfig

        results = {}

        # 데이터셋 로드
        contracts, ground_truth = self._load_risk_detection_dataset()

        # 베이스라인 시스템 평가
        baselines = ['rule_based', 'gpt4o_zeroshot', 'gpt4o_rag', 'full_pipeline']

        for baseline in baselines:
            self.logger.info(f"  Evaluating {baseline}...")

            # 각 베이스라인으로 분석 수행
            predictions = self._run_risk_detection_baseline(baseline, contracts)

            # 메트릭 계산
            report = generate_evaluation_report(predictions, ground_truth)
            results[baseline] = report

        return results

    def _run_ablation_study(self) -> Dict[str, Any]:
        """Ablation Study 실행"""
        from backend.app.ai.pipeline import AdvancedAIPipeline, PipelineConfig

        ablation_results = {}

        # 제거할 컴포넌트 리스트
        components = [
            ('hyde', 'enable_hyde'),
            ('raptor', 'enable_raptor'),
            ('crag', 'enable_crag'),
            ('constitutional_ai', 'enable_constitutional_ai'),
            ('stress_test', 'enable_stress_test'),
            ('judge', 'enable_judge'),
            ('reasoning_trace', 'enable_reasoning_trace'),
            ('pii_masking', 'enable_pii_masking'),
            ('dspy', 'enable_dspy')
        ]

        # 전체 파이프라인 (베이스라인)
        full_config = PipelineConfig()
        contracts, ground_truth = self._load_risk_detection_dataset()

        self.logger.info("  Running full pipeline baseline...")
        full_predictions = self._run_pipeline_with_config(full_config, contracts)
        full_report = generate_evaluation_report(full_predictions, ground_truth)
        ablation_results['full_pipeline'] = full_report

        # 각 컴포넌트 제거 실험
        for component_name, config_field in components:
            self.logger.info(f"  Ablating {component_name}...")

            # 해당 컴포넌트 비활성화
            config = PipelineConfig()
            setattr(config, config_field, False)

            predictions = self._run_pipeline_with_config(config, contracts)
            report = generate_evaluation_report(predictions, ground_truth)

            # 성능 변화 계산
            delta_f1 = full_report['clause_level']['f1'] - report['clause_level']['f1']

            ablation_results[f'without_{component_name}'] = {
                **report,
                'delta_f1': delta_f1,
                'contribution': delta_f1 / full_report['clause_level']['f1'] * 100
                    if full_report['clause_level']['f1'] > 0 else 0
            }

        return ablation_results

    def _load_retrieval_dataset(self):
        """검색 평가 데이터셋 로드"""
        # TODO: 실제 데이터셋 로드 구현
        # 현재는 더미 데이터 반환
        queries = [
            {"id": f"q{i}", "text": f"Sample query {i}"}
            for i in range(100)
        ]
        ground_truth = {
            f"q{i}": {f"d{i}", f"d{i+1}"}
            for i in range(100)
        }
        return queries, ground_truth

    def _load_risk_detection_dataset(self):
        """위험 탐지 평가 데이터셋 로드"""
        # TODO: 실제 데이터셋 로드 구현
        # 현재는 더미 데이터 반환
        contracts = []
        ground_truth = []
        return contracts, ground_truth

    def _run_retrieval_baseline(self, baseline: str, queries: List[Dict]) -> List[RetrievalResult]:
        """검색 베이스라인 실행"""
        # TODO: 실제 베이스라인 구현
        results = []
        for query in queries:
            results.append(RetrievalResult(
                query_id=query['id'],
                retrieved_doc_ids=[f"d{i}" for i in range(10)],
                scores=[1.0 / (i + 1) for i in range(10)]
            ))
        return results

    def _run_risk_detection_baseline(self, baseline: str, contracts: List) -> List[RiskDetectionResult]:
        """위험 탐지 베이스라인 실행"""
        # TODO: 실제 베이스라인 구현
        return []

    def _run_pipeline_with_config(self, config, contracts: List) -> List[RiskDetectionResult]:
        """특정 설정으로 파이프라인 실행"""
        # TODO: 실제 파이프라인 실행 구현
        return []

    def _save_results(self):
        """결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON 저장
        results_file = self.config.output_dir / f"results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)

        self.logger.info(f"Results saved to {results_file}")

    def _generate_report(self):
        """마크다운 보고서 생성"""
        report_lines = [
            f"# DocScanner AI Evaluation Report",
            f"",
            f"**Experiment:** {self.config.experiment_name}",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
        ]

        # 검색 성능 결과
        if 'retrieval' in self.results:
            report_lines.extend([
                "## 1. Retrieval Performance",
                "",
                "| Method | Recall@5 | Precision@5 | MRR | nDCG@10 | MAP |",
                "|--------|----------|-------------|-----|---------|-----|",
            ])

            for method, metrics in self.results['retrieval'].items():
                if method != 'comparison':
                    report_lines.append(
                        f"| {method} | "
                        f"{metrics['recall_at_k'].get(5, 0):.4f} | "
                        f"{metrics['precision_at_k'].get(5, 0):.4f} | "
                        f"{metrics['mrr']:.4f} | "
                        f"{metrics['ndcg_at_k'].get(10, 0):.4f} | "
                        f"{metrics['map']:.4f} |"
                    )
            report_lines.append("")

        # 위험 탐지 결과
        if 'risk_detection' in self.results:
            report_lines.extend([
                "## 2. Risk Detection Performance",
                "",
                "| Method | Clause F1 | Severity F1 | MAE (KRW) |",
                "|--------|-----------|-------------|-----------|",
            ])

            for method, metrics in self.results['risk_detection'].items():
                report_lines.append(
                    f"| {method} | "
                    f"{metrics['clause_level']['f1']:.4f} | "
                    f"{metrics['severity_classification']['macro_f1']:.4f} | "
                    f"{metrics['financial_prediction']['mae']:,.0f} |"
                )
            report_lines.append("")

        # Ablation Study 결과
        if 'ablation' in self.results:
            report_lines.extend([
                "## 3. Ablation Study",
                "",
                "| Configuration | Clause F1 | Delta F1 | Contribution (%) |",
                "|---------------|-----------|----------|------------------|",
            ])

            for config, metrics in self.results['ablation'].items():
                delta = metrics.get('delta_f1', 0)
                contrib = metrics.get('contribution', 0)
                report_lines.append(
                    f"| {config} | "
                    f"{metrics['clause_level']['f1']:.4f} | "
                    f"{delta:+.4f} | "
                    f"{contrib:.1f}% |"
                )
            report_lines.append("")

        # 보고서 저장
        report_file = self.config.output_dir / "evaluation_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        self.logger.info(f"Report saved to {report_file}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="DocScanner AI Evaluation")
    parser.add_argument('--config', type=str, default='config/eval_config.yaml',
                        help='Path to evaluation config file')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--name', type=str, default='full_evaluation',
                        help='Experiment name')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no-retrieval', action='store_true',
                        help='Skip retrieval evaluation')
    parser.add_argument('--no-risk', action='store_true',
                        help='Skip risk detection evaluation')
    parser.add_argument('--no-ablation', action='store_true',
                        help='Skip ablation study')

    args = parser.parse_args()

    # 설정 로드
    config = EvaluationConfig(
        experiment_name=args.name,
        output_dir=Path(args.output),
        seed=args.seed,
        k_values=[1, 3, 5, 10, 20],
        run_retrieval=not args.no_retrieval,
        run_risk_detection=not args.no_risk,
        run_ablation=not args.no_ablation
    )

    # 평가 실행
    evaluator = DocScannerEvaluator(config)
    results = evaluator.run_full_evaluation()

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"Results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
