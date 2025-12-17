# DocScanner AI 평가 계획서

## 1. 연구 질문 (Research Questions)

본 연구는 다음의 연구 질문에 답하고자 한다:

### RQ1: 검색 성능 (Retrieval Performance)
**HyDE와 Graph-Guided CRAG가 법률 문서 검색의 정확도를 얼마나 향상시키는가?**

- RQ1.1: HyDE가 구어체 질문에서 법률 전문 용어로의 의미적 간극(Semantic Gap)을 해소하는가?
- RQ1.2: Graph-Guided CRAG가 단순 벡터 검색 대비 관련 문서 검색률을 개선하는가?
- RQ1.3: RAPTOR 계층적 인덱싱이 다양한 추상화 수준의 질문에 효과적으로 대응하는가?

### RQ2: 위험 탐지 정확도 (Risk Detection Accuracy)
**시스템이 근로계약서의 법적 위험 조항을 얼마나 정확하게 탐지하는가?**

- RQ2.1: Legal Stress Test가 최저임금/연장근로수당 위반을 정확히 계산하는가?
- RQ2.2: Generative Redlining이 위험 조항을 정확히 식별하고 적절한 수정안을 제시하는가?
- RQ2.3: Constitutional AI가 노동자 불리 조항을 효과적으로 필터링하는가?

### RQ3: 신뢰도 및 설명가능성 (Reliability & Explainability)
**시스템의 분석 결과가 신뢰할 수 있고 설명 가능한가?**

- RQ3.1: LLM-as-a-Judge가 분석 결과의 신뢰도를 정확히 평가하는가?
- RQ3.2: Reasoning Trace가 사용자의 이해도를 높이는가?
- RQ3.3: 전문가 검토 대비 시스템 분석의 일치율은 어느 정도인가?

### RQ4: 개인정보 보호 (Privacy Protection)
**PII Masking이 개인정보를 효과적으로 보호하면서 분석 정확도를 유지하는가?**

- RQ4.1: PII 탐지율(Recall)과 정밀도(Precision)는 어느 수준인가?
- RQ4.2: 마스킹 후에도 분석 성능이 유지되는가?

### RQ5: 시스템 효율성 (System Efficiency)
**전체 파이프라인의 처리 시간과 확장성은 실용적인가?**

- RQ5.1: 평균 계약서 분석 시간은 얼마인가?
- RQ5.2: 동시 처리 시 시스템 확장성은 어떠한가?

---

## 2. 데이터셋 (Datasets)

### 2.1 평가 데이터셋 구성

| 데이터셋 | 규모 | 설명 | 용도 |
|---------|------|------|------|
| LegalBench-KR | 500건 | 한국 노동법 Q&A 데이터셋 | 검색 성능 평가 |
| ContractRisk-100 | 100건 | 전문가 라벨링된 위험 계약서 | 위험 탐지 정확도 |
| SyntheticPII-1K | 1,000건 | 합성 개인정보 포함 문서 | PII 탐지 성능 |
| RealContract-50 | 50건 | 실제 근로계약서 (익명화) | End-to-End 평가 |
| ExpertAnnotated-30 | 30건 | 노동법 전문가 상세 분석 | Human Evaluation |

### 2.2 LegalBench-KR 구성

```
LegalBench-KR/
├── queries/
│   ├── colloquial/     # 구어체 질문 (250건)
│   │   ├── wage_related.json
│   │   ├── overtime_related.json
│   │   ├── termination_related.json
│   │   └── ...
│   └── formal/         # 법률 용어 질문 (250건)
│       ├── labor_law_articles.json
│       └── precedent_queries.json
├── ground_truth/
│   ├── relevant_docs.json
│   └── answer_spans.json
└── metadata/
    └── difficulty_levels.json  # Easy/Medium/Hard
```

### 2.3 ContractRisk-100 라벨링 스키마

```json
{
  "contract_id": "CR-001",
  "risk_clauses": [
    {
      "clause_id": 1,
      "text": "연장근로수당을 포함하여 월 200만원을 지급한다.",
      "risk_type": "포괄임금제",
      "severity": "High",
      "legal_basis": "근로기준법 제56조",
      "expected_revision": "기본급 180만원, 연장근로수당 별도 산정 지급",
      "annual_underpayment": 2400000
    }
  ],
  "overall_risk_level": "High",
  "expert_id": "EXP-003",
  "annotation_date": "2024-01-15"
}
```

### 2.4 데이터 수집 및 라벨링 프로토콜

1. **수집 출처**
   - 고용노동부 표준근로계약서 양식
   - 법률구조공단 상담 사례 (익명화)
   - 노동위원회 판정례
   - 대법원 판례 (근로기준법 관련)

2. **라벨링 인력**
   - 노동법 전문 변호사 2인
   - 공인노무사 3인
   - 라벨링 일치도(Inter-annotator Agreement) 측정: Cohen's Kappa >= 0.8 목표

3. **라벨링 가이드라인**
   - 위험 수준 정의: High (법 위반), Medium (불명확/해석 여지), Low (권장사항)
   - 체불액 산정: 2025년 최저임금(10,030원) 기준

---

## 3. 평가 지표 (Evaluation Metrics)

### 3.1 검색 성능 (Information Retrieval)

| 지표 | 수식 | 설명 |
|-----|------|------|
| **Recall@k** | TP / (TP + FN) | 상위 k개 결과 중 관련 문서 포함 비율 |
| **Precision@k** | TP / (TP + FP) | 상위 k개 결과의 정밀도 |
| **MRR (Mean Reciprocal Rank)** | 1/\|Q\| * sum(1/rank_i) | 첫 관련 문서 순위의 역수 평균 |
| **nDCG@k** | DCG@k / IDCG@k | 순위 가중 관련성 점수 |
| **MAP (Mean Average Precision)** | 1/\|Q\| * sum(AP_q) | 평균 정밀도의 평균 |

```python
# 측정 코드 예시
def calculate_retrieval_metrics(results, ground_truth, k=5):
    metrics = {
        "recall@k": recall_at_k(results, ground_truth, k),
        "precision@k": precision_at_k(results, ground_truth, k),
        "mrr": mean_reciprocal_rank(results, ground_truth),
        "ndcg@k": ndcg_at_k(results, ground_truth, k),
        "map": mean_average_precision(results, ground_truth)
    }
    return metrics
```

### 3.2 위험 탐지 성능 (Risk Detection)

| 지표 | 설명 |
|-----|------|
| **Clause-level F1** | 위험 조항 탐지의 F1 점수 |
| **Risk Type Accuracy** | 위험 유형 분류 정확도 |
| **Severity Classification F1** | High/Medium/Low 분류 F1 |
| **Underpayment MAE** | 체불액 예측 평균 절대 오차 |
| **Revision BLEU** | 수정안 생성 품질 (BLEU-4) |
| **Revision Legal Validity** | 수정안의 법적 타당성 (전문가 평가) |

```python
# 위험 탐지 평가 예시
def evaluate_risk_detection(predictions, ground_truth):
    clause_metrics = {
        "precision": clause_precision(predictions, ground_truth),
        "recall": clause_recall(predictions, ground_truth),
        "f1": clause_f1(predictions, ground_truth)
    }

    severity_metrics = {
        "accuracy": severity_accuracy(predictions, ground_truth),
        "macro_f1": severity_macro_f1(predictions, ground_truth),
        "confusion_matrix": severity_confusion_matrix(predictions, ground_truth)
    }

    financial_metrics = {
        "underpayment_mae": mean_absolute_error(
            [p["annual_underpayment"] for p in predictions],
            [g["annual_underpayment"] for g in ground_truth]
        ),
        "underpayment_mape": mean_absolute_percentage_error(...)
    }

    return clause_metrics, severity_metrics, financial_metrics
```

### 3.3 신뢰도 평가 (Reliability)

| 지표 | 설명 |
|-----|------|
| **Expert Agreement Rate** | 전문가 분석과의 일치율 |
| **Judge Calibration** | LLM Judge 점수와 실제 품질의 상관관계 |
| **Factual Accuracy** | 인용된 법조문/판례의 정확성 |
| **Hallucination Rate** | 존재하지 않는 법조문 인용 비율 |

### 3.4 개인정보 보호 (Privacy)

| 지표 | 수식 | 목표 |
|-----|------|------|
| **PII Recall** | Detected PII / Total PII | >= 0.95 |
| **PII Precision** | True PII / Detected | >= 0.90 |
| **Analysis Preservation** | F1(masked) / F1(original) | >= 0.95 |

### 3.5 시스템 효율성 (Efficiency)

| 지표 | 측정 방법 |
|-----|----------|
| **Latency (P50, P95, P99)** | 요청 처리 시간 백분위수 |
| **Throughput** | 초당 처리 계약서 수 |
| **Memory Usage** | 피크 메모리 사용량 |
| **GPU Utilization** | GPU 사용률 (임베딩 생성 시) |

---

## 4. 베이스라인 시스템 (Baseline Systems)

### 4.1 검색 베이스라인

| 시스템 | 설명 |
|-------|------|
| **BM25** | 전통적 키워드 기반 검색 |
| **Dense Retrieval (KURE-v1)** | 단일 벡터 밀집 검색 |
| **ColBERT-KR** | Late Interaction 다중 벡터 검색 |
| **MUVERA (Ours - Base)** | FDE 기반 다중 벡터 검색 |
| **MUVERA + HyDE** | HyDE 쿼리 확장 적용 |
| **MUVERA + CRAG** | CRAG 자기 보정 적용 |
| **Full Pipeline** | HyDE + CRAG + RAPTOR 통합 |

### 4.2 위험 탐지 베이스라인

| 시스템 | 설명 |
|-------|------|
| **Rule-based** | 키워드 매칭 기반 규칙 시스템 |
| **GPT-4o Zero-shot** | 프롬프트만으로 위험 탐지 |
| **GPT-4o + RAG** | 검색 증강 생성 |
| **GPT-4o + Constitutional AI** | 헌법적 AI 적용 |
| **Full Pipeline (Ours)** | 모든 구성요소 통합 |

### 4.3 상용 서비스 비교 (가능한 경우)

| 서비스 | 설명 |
|-------|------|
| **LawBot.kr** | 국내 법률 AI 챗봇 |
| **LegalZoom AI** | 해외 계약서 분석 서비스 |
| **Human Expert** | 노동법 전문가 분석 (Upper Bound) |

---

## 5. 실험 설계 (Experimental Setup)

### 5.1 실험 환경

```yaml
Hardware:
  GPU: NVIDIA A100 80GB x 1
  CPU: AMD EPYC 7742 64-Core
  RAM: 512GB DDR4
  Storage: NVMe SSD 2TB

Software:
  OS: Ubuntu 22.04 LTS
  Python: 3.11
  PyTorch: 2.1.0
  CUDA: 12.1

Infrastructure:
  Elasticsearch: 8.11.0 (3-node cluster)
  Neo4j: 5.14.1 (Enterprise)
  Redis: 7.2.0
  PostgreSQL: 16.0
```

### 5.2 하이퍼파라미터

```yaml
Embedding:
  model: nlpai-lab/KURE-v1
  dimension: 1024
  max_seq_length: 512

MUVERA FDE:
  num_repetitions: 1
  num_simhash_projections: 3
  encoding_type: AVERAGE
  final_projection_dimension: 1024

RAPTOR:
  max_tree_depth: 4
  cluster_method: GMM
  summary_model: gpt-4o-mini

CRAG:
  quality_threshold: 0.7
  max_graph_hops: 2
  expansion_limit: 10

LLM:
  model: gpt-4o
  temperature: 0.2 (analysis), 0.7 (generation)
  max_tokens: 4096
```

### 5.3 교차 검증 (Cross-Validation)

- **K-Fold**: 5-fold cross-validation
- **Stratification**: 위험 수준(High/Medium/Low) 기준 층화 추출
- **Random Seed**: 42, 123, 456 (3회 반복 평균)

### 5.4 통계적 유의성 검정

```python
# 통계 검정 방법
from scipy import stats

def statistical_significance_test(baseline_scores, proposed_scores, alpha=0.05):
    """
    Paired t-test 또는 Wilcoxon signed-rank test 수행
    """
    # 정규성 검정
    _, p_normal = stats.shapiro(proposed_scores - baseline_scores)

    if p_normal > 0.05:
        # 정규 분포: Paired t-test
        t_stat, p_value = stats.ttest_rel(proposed_scores, baseline_scores)
        test_name = "Paired t-test"
    else:
        # 비정규 분포: Wilcoxon signed-rank test
        t_stat, p_value = stats.wilcoxon(proposed_scores, baseline_scores)
        test_name = "Wilcoxon signed-rank"

    # 효과 크기 (Cohen's d)
    effect_size = (proposed_scores.mean() - baseline_scores.mean()) / \
                  np.sqrt((proposed_scores.std()**2 + baseline_scores.std()**2) / 2)

    return {
        "test": test_name,
        "statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < alpha,
        "effect_size": effect_size,
        "effect_interpretation": interpret_cohens_d(effect_size)
    }
```

---

## 6. Ablation Study

### 6.1 구성요소별 기여도 분석

각 모듈을 순차적으로 제거하여 기여도 측정:

| 실험 | 제거 모듈 | 측정 지표 |
|-----|----------|----------|
| A1 | - HyDE | Recall@5, MRR |
| A2 | - RAPTOR | nDCG@10 (multi-level) |
| A3 | - CRAG | Precision@5, Factual Accuracy |
| A4 | - Constitutional AI | Risk Detection F1, Bias Rate |
| A5 | - Stress Test | Underpayment MAE |
| A6 | - LLM Judge | User Trust Score |
| A7 | - Reasoning Trace | User Understanding Score |
| A8 | - PII Masking | Analysis Preservation Rate |
| A9 | - Vision RAG | Table Extraction Accuracy |
| A10 | - DSPy | Performance over Time |

### 6.2 하이퍼파라미터 민감도 분석

```python
# 민감도 분석 파라미터
sensitivity_params = {
    "crag_quality_threshold": [0.5, 0.6, 0.7, 0.8, 0.9],
    "crag_max_hops": [1, 2, 3, 4],
    "raptor_tree_depth": [2, 3, 4, 5],
    "hyde_temperature": [0.1, 0.3, 0.5, 0.7],
    "stress_test_tolerance": [0.01, 0.05, 0.10]
}
```

### 6.3 에러 분석 (Error Analysis)

1. **False Negative 분석**: 탐지 실패한 위험 조항 유형 분류
2. **False Positive 분석**: 오탐지 원인 분석
3. **체불액 오차 분석**: 과대/과소 추정 패턴 분석
4. **수정안 품질 분석**: 부적절한 수정안 유형 분류

---

## 7. Human Evaluation

### 7.1 평가자 구성

| 역할 | 인원 | 자격 요건 |
|-----|------|----------|
| 노동법 전문가 | 3명 | 노동법 전문 변호사 또는 공인노무사 5년 이상 |
| 일반 사용자 | 30명 | 근로계약 경험이 있는 성인 |
| 법학 대학원생 | 5명 | 노동법 전공 석박사 과정 |

### 7.2 평가 항목

#### 전문가 평가 (Expert Evaluation)

```
1. 법적 정확성 (Legal Accuracy) [1-5점]
   - 인용된 법조문이 정확한가?
   - 법적 해석이 올바른가?
   - 판례 인용이 적절한가?

2. 위험 탐지 완전성 (Completeness) [1-5점]
   - 모든 위험 조항이 탐지되었는가?
   - 누락된 중요 위험이 있는가?

3. 수정안 적절성 (Revision Quality) [1-5점]
   - 수정안이 법적으로 유효한가?
   - 수정안이 실무적으로 적용 가능한가?
   - 수정안이 당사자 이익 균형을 고려하는가?

4. 체불액 정확성 (Financial Accuracy)
   - 예상 체불액이 정확한가? (오차 범위 내)

5. 전반적 신뢰도 (Overall Reliability) [1-5점]
   - 이 분석 결과를 법률 자문에 활용할 수 있는가?
```

#### 사용자 평가 (User Study)

```
1. 이해도 (Understandability) [1-5점]
   - 분석 결과를 쉽게 이해할 수 있는가?
   - Reasoning Trace가 이해에 도움이 되는가?

2. 유용성 (Usefulness) [1-5점]
   - 이 분석이 계약서 검토에 도움이 되는가?
   - 수정 제안이 실제로 적용할 만한가?

3. 신뢰성 (Trustworthiness) [1-5점]
   - AI 분석 결과를 신뢰할 수 있는가?
   - 신뢰도 배지가 판단에 도움이 되는가?

4. 행동 의도 (Behavioral Intention)
   - 이 서비스를 실제로 사용할 의향이 있는가? [1-7점]
   - 타인에게 추천할 의향이 있는가? [1-7점]
```

### 7.3 평가 프로토콜

```
1. 사전 교육 (15분)
   - 시스템 사용법 안내
   - 평가 기준 설명

2. 실습 평가 (10분)
   - 연습용 계약서 1건 분석 체험
   - 평가 방법 숙지

3. 본 평가 (60분)
   - 무작위 배정된 5건의 계약서 분석 결과 평가
   - 각 계약서당 A/B 테스트 (베이스라인 vs 제안 시스템)

4. 사후 설문 (15분)
   - 전반적 인상 설문
   - 개선점 자유 기술

5. 보상
   - 전문가: 20만원/인
   - 일반 사용자: 3만원/인
```

### 7.4 평가자 간 신뢰도 (Inter-rater Reliability)

```python
# Krippendorff's Alpha 계산
from krippendorff import alpha

def calculate_inter_rater_reliability(ratings):
    """
    ratings: shape (n_raters, n_items)
    """
    reliability = {
        "krippendorff_alpha": alpha(ratings, level_of_measurement="ordinal"),
        "fleiss_kappa": fleiss_kappa(ratings),
        "icc": intraclass_correlation(ratings)  # ICC(2,k)
    }

    # 해석 기준
    # Alpha >= 0.8: 높은 신뢰도
    # 0.67 <= Alpha < 0.8: 허용 가능
    # Alpha < 0.67: 재검토 필요

    return reliability
```

---

## 8. 결과 보고 형식

### 8.1 주요 결과 테이블

```
Table 1: Retrieval Performance Comparison

| Method              | Recall@5 | Precision@5 | MRR   | nDCG@10 |
|---------------------|----------|-------------|-------|---------|
| BM25                | 0.412    | 0.328       | 0.451 | 0.389   |
| Dense (KURE-v1)     | 0.567    | 0.445       | 0.612 | 0.534   |
| MUVERA              | 0.623    | 0.489       | 0.657 | 0.578   |
| MUVERA + HyDE       | 0.701*   | 0.534*      | 0.723* | 0.645* |
| Full Pipeline       | 0.756**  | 0.589**     | 0.789** | 0.712** |

* p < 0.05, ** p < 0.01 (vs. MUVERA baseline)
```

```
Table 2: Risk Detection Performance

| Method              | Clause F1 | Severity F1 | Underpayment MAE |
|---------------------|-----------|-------------|------------------|
| Rule-based          | 0.412     | 0.356       | 892,000 KRW      |
| GPT-4o Zero-shot    | 0.567     | 0.489       | 456,000 KRW      |
| GPT-4o + RAG        | 0.645     | 0.534       | 312,000 KRW      |
| Full Pipeline       | 0.823**   | 0.756**     | 124,000 KRW**    |

** p < 0.01
```

### 8.2 시각화

1. **Ablation Study 막대 그래프**: 각 모듈 제거 시 성능 변화
2. **Confusion Matrix**: 위험 수준 분류 결과
3. **ROC/PR Curve**: 이진 분류 성능
4. **처리 시간 Box Plot**: 계약서 길이별 처리 시간 분포
5. **Reasoning Trace 시각화**: 샘플 분석의 추론 과정

---

## 9. 실험 일정

| 단계 | 기간 | 세부 내용 |
|-----|------|----------|
| 데이터 수집 | 2주 | 계약서 수집, 익명화, 전처리 |
| 전문가 라벨링 | 3주 | Gold Standard 생성, IAA 검증 |
| 자동 평가 | 1주 | 검색/탐지 성능 측정 |
| Human Evaluation | 2주 | 전문가/사용자 평가 수행 |
| 통계 분석 | 1주 | 유의성 검정, 시각화 |
| 논문 작성 | 2주 | 결과 해석 및 논문 초고 |

**총 소요 기간: 약 11주**

---

## 10. 윤리적 고려사항

### 10.1 데이터 프라이버시

- 모든 실제 계약서는 IRB 승인 후 사용
- 개인정보 완전 익명화 처리
- 연구 종료 후 데이터 폐기

### 10.2 평가자 보호

- 평가 참여 동의서 수령
- 평가자 익명성 보장
- 적정 보상 지급

### 10.3 AI 윤리

- 시스템의 한계 명시 (법률 자문 대체 불가)
- 신뢰도 수준 투명하게 표시
- 편향성 모니터링 및 보고

---

## 11. 예상 결과 및 기여점

### 11.1 예상 결과

1. HyDE + CRAG 조합이 단순 밀집 검색 대비 Recall@5 20-30% 향상
2. Legal Stress Test가 체불액 예측 오차를 50% 이상 감소
3. Constitutional AI가 노동자 불리 조항 필터링 정확도 90% 이상 달성
4. LLM-as-a-Judge 신뢰도 점수와 전문가 평가 간 상관관계 r >= 0.75

### 11.2 학술적 기여

1. **한국어 법률 도메인 RAG 벤치마크** 최초 구축
2. **Neuro-Symbolic AI**의 법률 문서 분석 적용 사례
3. **Constitutional AI**의 도메인 특화 적용 방법론
4. **멀티모달 계약서 분석** 파이프라인 설계

### 11.3 실용적 기여

1. 노동자의 계약서 검토 접근성 향상
2. 법률 상담 비용 절감 (사전 검토 자동화)
3. 위험 조항 조기 발견을 통한 노동 분쟁 예방

---

## 12. 참고문헌 형식

```bibtex
@inproceedings{docscanner2024,
  title={DocScanner: A Neuro-Symbolic AI System for Korean Labor Contract Analysis},
  author={Lee, Seongmin and ...},
  booktitle={Proceedings of the 2024 Conference on ...},
  year={2024},
  pages={...}
}
```

---

## Appendix A: 평가 도구 설치

```bash
# 평가 환경 설정
pip install scikit-learn scipy pandas matplotlib seaborn
pip install krippendorff  # Inter-rater reliability
pip install sacrebleu rouge-score  # Text generation metrics

# 실행
python evaluation/run_retrieval_eval.py --dataset legalbench-kr
python evaluation/run_risk_detection_eval.py --dataset contractrisk-100
python evaluation/run_ablation_study.py
```

## Appendix B: 평가 스크립트 구조

```
evaluation/
├── config/
│   ├── eval_config.yaml
│   └── baseline_config.yaml
├── datasets/
│   ├── legalbench_kr/
│   ├── contractrisk_100/
│   └── synthetic_pii/
├── metrics/
│   ├── retrieval_metrics.py
│   ├── risk_detection_metrics.py
│   └── reliability_metrics.py
├── baselines/
│   ├── bm25_baseline.py
│   ├── dense_baseline.py
│   └── gpt_baseline.py
├── scripts/
│   ├── run_retrieval_eval.py
│   ├── run_risk_detection_eval.py
│   ├── run_ablation_study.py
│   └── generate_tables.py
├── human_eval/
│   ├── expert_evaluation_form.pdf
│   ├── user_study_protocol.md
│   └── analyze_human_eval.py
└── results/
    ├── tables/
    └── figures/
```
