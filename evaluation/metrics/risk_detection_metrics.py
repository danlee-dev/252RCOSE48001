"""
Risk Detection Evaluation Metrics
위험 조항 탐지 성능 평가 지표
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, classification_report
)
from collections import Counter


@dataclass
class RiskClause:
    """위험 조항"""
    clause_id: str
    text: str
    risk_type: str
    severity: str  # High, Medium, Low
    annual_underpayment: float = 0.0


@dataclass
class RiskDetectionResult:
    """위험 탐지 결과"""
    contract_id: str
    detected_clauses: List[RiskClause]
    overall_risk_level: str
    total_underpayment: float


@dataclass
class ClauseLevelMetrics:
    """조항 수준 평가 결과"""
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass
class SeverityMetrics:
    """위험도 분류 평가 결과"""
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_f1: Dict[str, float]
    confusion_matrix: np.ndarray


@dataclass
class FinancialMetrics:
    """재무적 평가 결과 (체불액 예측)"""
    mae: float  # Mean Absolute Error
    mape: float  # Mean Absolute Percentage Error
    rmse: float  # Root Mean Square Error
    within_10_percent: float  # 10% 오차 이내 비율
    within_50k_won: float  # 5만원 오차 이내 비율


def calculate_clause_level_metrics(
    predictions: List[RiskDetectionResult],
    ground_truth: List[RiskDetectionResult],
    iou_threshold: float = 0.5
) -> ClauseLevelMetrics:
    """
    조항 수준 탐지 성능 평가

    Args:
        predictions: 예측 결과 리스트
        ground_truth: 정답 리스트
        iou_threshold: 텍스트 일치도 임계값

    Returns:
        ClauseLevelMetrics 객체
    """
    tp, fp, fn = 0, 0, 0

    for pred, gt in zip(predictions, ground_truth):
        pred_clauses = {c.clause_id: c for c in pred.detected_clauses}
        gt_clauses = {c.clause_id: c for c in gt.detected_clauses}

        # True Positives: 예측과 정답 모두에 있는 경우
        matched_gt = set()
        for pred_id, pred_clause in pred_clauses.items():
            best_match = None
            best_iou = 0

            for gt_id, gt_clause in gt_clauses.items():
                if gt_id in matched_gt:
                    continue
                iou = text_iou(pred_clause.text, gt_clause.text)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_match = gt_id

            if best_match:
                tp += 1
                matched_gt.add(best_match)
            else:
                fp += 1

        # False Negatives: 정답에는 있지만 예측에서 누락된 경우
        fn += len(gt_clauses) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return ClauseLevelMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn
    )


def text_iou(text1: str, text2: str) -> float:
    """
    텍스트 간 Jaccard 유사도 (단어 기반 IoU)

    Args:
        text1: 첫 번째 텍스트
        text2: 두 번째 텍스트

    Returns:
        IoU 값 (0.0 ~ 1.0)
    """
    words1 = set(text1.split())
    words2 = set(text2.split())

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


def calculate_severity_metrics(
    predictions: List[RiskDetectionResult],
    ground_truth: List[RiskDetectionResult]
) -> SeverityMetrics:
    """
    위험도 분류 성능 평가

    Args:
        predictions: 예측 결과 리스트
        ground_truth: 정답 리스트

    Returns:
        SeverityMetrics 객체
    """
    y_pred = []
    y_true = []

    severity_labels = ["High", "Medium", "Low"]

    for pred, gt in zip(predictions, ground_truth):
        y_pred.append(pred.overall_risk_level)
        y_true.append(gt.overall_risk_level)

    # 라벨 인코딩
    label_to_idx = {label: i for i, label in enumerate(severity_labels)}
    y_pred_idx = [label_to_idx.get(y, 0) for y in y_pred]
    y_true_idx = [label_to_idx.get(y, 0) for y in y_true]

    # 메트릭 계산
    accuracy = accuracy_score(y_true_idx, y_pred_idx)
    macro_f1 = f1_score(y_true_idx, y_pred_idx, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true_idx, y_pred_idx, average='weighted', zero_division=0)

    # 클래스별 F1
    per_class = f1_score(y_true_idx, y_pred_idx, average=None, zero_division=0)
    per_class_f1 = {label: score for label, score in zip(severity_labels, per_class)}

    # Confusion Matrix
    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=list(range(len(severity_labels))))

    return SeverityMetrics(
        accuracy=accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        per_class_f1=per_class_f1,
        confusion_matrix=cm
    )


def calculate_financial_metrics(
    predictions: List[RiskDetectionResult],
    ground_truth: List[RiskDetectionResult]
) -> FinancialMetrics:
    """
    체불액 예측 성능 평가

    Args:
        predictions: 예측 결과 리스트
        ground_truth: 정답 리스트

    Returns:
        FinancialMetrics 객체
    """
    pred_amounts = []
    true_amounts = []

    for pred, gt in zip(predictions, ground_truth):
        pred_amounts.append(pred.total_underpayment)
        true_amounts.append(gt.total_underpayment)

    pred_amounts = np.array(pred_amounts)
    true_amounts = np.array(true_amounts)

    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(pred_amounts - true_amounts))

    # MAPE (Mean Absolute Percentage Error)
    # 0으로 나누기 방지
    non_zero_mask = true_amounts != 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((true_amounts[non_zero_mask] - pred_amounts[non_zero_mask]) /
                              true_amounts[non_zero_mask])) * 100
    else:
        mape = 0.0

    # RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean((pred_amounts - true_amounts) ** 2))

    # 오차 범위 내 비율
    abs_error = np.abs(pred_amounts - true_amounts)
    relative_error = np.where(true_amounts != 0,
                              abs_error / true_amounts,
                              np.where(abs_error == 0, 0, float('inf')))

    within_10_percent = np.mean(relative_error <= 0.10)
    within_50k_won = np.mean(abs_error <= 50000)

    return FinancialMetrics(
        mae=mae,
        mape=mape,
        rmse=rmse,
        within_10_percent=within_10_percent,
        within_50k_won=within_50k_won
    )


def calculate_risk_type_accuracy(
    predictions: List[RiskDetectionResult],
    ground_truth: List[RiskDetectionResult]
) -> Dict[str, float]:
    """
    위험 유형별 분류 정확도

    Args:
        predictions: 예측 결과 리스트
        ground_truth: 정답 리스트

    Returns:
        위험 유형별 정확도 딕셔너리
    """
    type_correct = Counter()
    type_total = Counter()

    for pred, gt in zip(predictions, ground_truth):
        gt_types = {c.clause_id: c.risk_type for c in gt.detected_clauses}
        pred_types = {c.clause_id: c.risk_type for c in pred.detected_clauses}

        for clause_id, true_type in gt_types.items():
            type_total[true_type] += 1
            if clause_id in pred_types and pred_types[clause_id] == true_type:
                type_correct[true_type] += 1

    accuracy_by_type = {}
    for risk_type in type_total:
        if type_total[risk_type] > 0:
            accuracy_by_type[risk_type] = type_correct[risk_type] / type_total[risk_type]
        else:
            accuracy_by_type[risk_type] = 0.0

    return accuracy_by_type


def evaluate_redlining_quality(
    predicted_revisions: List[str],
    ground_truth_revisions: List[str],
    expert_scores: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    수정안(Redlining) 품질 평가

    Args:
        predicted_revisions: 예측된 수정안 리스트
        ground_truth_revisions: 정답 수정안 리스트
        expert_scores: 전문가 평가 점수 (선택적)

    Returns:
        수정안 품질 메트릭
    """
    from sacrebleu.metrics import BLEU
    from rouge_score import rouge_scorer

    # BLEU Score
    bleu = BLEU()
    bleu_score = bleu.corpus_score(predicted_revisions,
                                   [[ref] for ref in ground_truth_revisions])

    # ROUGE Scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    for pred, ref in zip(predicted_revisions, ground_truth_revisions):
        scores = scorer.score(ref, pred)
        for key in rouge_scores:
            rouge_scores[key].append(scores[key].fmeasure)

    results = {
        'bleu': bleu_score.score,
        'rouge1': np.mean(rouge_scores['rouge1']),
        'rouge2': np.mean(rouge_scores['rouge2']),
        'rougeL': np.mean(rouge_scores['rougeL'])
    }

    # 전문가 평가 점수가 있는 경우
    if expert_scores:
        results['expert_score_mean'] = np.mean(expert_scores)
        results['expert_score_std'] = np.std(expert_scores)

    return results


def generate_evaluation_report(
    predictions: List[RiskDetectionResult],
    ground_truth: List[RiskDetectionResult]
) -> Dict[str, Any]:
    """
    전체 위험 탐지 평가 보고서 생성

    Args:
        predictions: 예측 결과 리스트
        ground_truth: 정답 리스트

    Returns:
        평가 보고서 딕셔너리
    """
    clause_metrics = calculate_clause_level_metrics(predictions, ground_truth)
    severity_metrics = calculate_severity_metrics(predictions, ground_truth)
    financial_metrics = calculate_financial_metrics(predictions, ground_truth)
    type_accuracy = calculate_risk_type_accuracy(predictions, ground_truth)

    return {
        "clause_level": {
            "precision": clause_metrics.precision,
            "recall": clause_metrics.recall,
            "f1": clause_metrics.f1,
            "tp": clause_metrics.true_positives,
            "fp": clause_metrics.false_positives,
            "fn": clause_metrics.false_negatives
        },
        "severity_classification": {
            "accuracy": severity_metrics.accuracy,
            "macro_f1": severity_metrics.macro_f1,
            "weighted_f1": severity_metrics.weighted_f1,
            "per_class_f1": severity_metrics.per_class_f1,
            "confusion_matrix": severity_metrics.confusion_matrix.tolist()
        },
        "financial_prediction": {
            "mae": financial_metrics.mae,
            "mape": financial_metrics.mape,
            "rmse": financial_metrics.rmse,
            "within_10_percent": financial_metrics.within_10_percent,
            "within_50k_won": financial_metrics.within_50k_won
        },
        "risk_type_accuracy": type_accuracy,
        "summary": {
            "overall_f1": clause_metrics.f1,
            "severity_accuracy": severity_metrics.accuracy,
            "financial_mae_krw": f"{financial_metrics.mae:,.0f}원"
        }
    }


# 테스트 코드
if __name__ == "__main__":
    # 테스트 데이터 생성
    predictions = [
        RiskDetectionResult(
            contract_id="C001",
            detected_clauses=[
                RiskClause("1", "연장근로수당 포함 지급", "포괄임금제", "High", 2400000),
                RiskClause("2", "즉시 해고 가능", "부당해고", "High", 0)
            ],
            overall_risk_level="High",
            total_underpayment=2400000
        ),
        RiskDetectionResult(
            contract_id="C002",
            detected_clauses=[
                RiskClause("1", "경쟁사 이직 금지 3년", "경업금지", "Medium", 0)
            ],
            overall_risk_level="Medium",
            total_underpayment=0
        )
    ]

    ground_truth = [
        RiskDetectionResult(
            contract_id="C001",
            detected_clauses=[
                RiskClause("1", "연장근로수당을 포함하여 지급", "포괄임금제", "High", 2880000),
                RiskClause("2", "즉시 해고 가능", "부당해고", "High", 0),
                RiskClause("3", "연차 미사용 시 소멸", "연차소멸", "Medium", 500000)
            ],
            overall_risk_level="High",
            total_underpayment=3380000
        ),
        RiskDetectionResult(
            contract_id="C002",
            detected_clauses=[
                RiskClause("1", "경쟁사 이직 금지 3년", "경업금지", "Medium", 0)
            ],
            overall_risk_level="Medium",
            total_underpayment=0
        )
    ]

    report = generate_evaluation_report(predictions, ground_truth)

    print("Risk Detection Evaluation Report:")
    print(f"  Clause-level F1: {report['clause_level']['f1']:.4f}")
    print(f"  Severity Accuracy: {report['severity_classification']['accuracy']:.4f}")
    print(f"  Financial MAE: {report['financial_prediction']['mae']:,.0f}원")
