# LLM Configuration

DocScanner AI에서 사용하는 LLM 모델 구성 문서.

## 환경 변수

```bash
# .env
LLM_RETRIEVAL_MODEL=gemini-2.5-flash-lite  # Vision/요약
LLM_REASONING_MODEL=gpt-5-mini             # 법률 추론
LLM_HYDE_MODEL=gpt-4o                      # HyDE 생성
```

## 모델 용도별 분류

### 1. Retrieval Model (Gemini)

빠른 응답과 비용 효율이 필요한 작업에 사용.

| 모듈 | 모델 | 용도 |
|------|------|------|
| Vision Parser | `gemini-2.5-flash-lite` | 이미지/PDF OCR, 문서 구조화 |
| Summarization | `gemini-2.5-flash-lite` | 문서 요약 |

### 2. HyDE Model (OpenAI GPT-4o)

가상 문서 생성에 특화. Temperature 파라미터 지원 필요.

| 모듈 | 모델 | 용도 |
|------|------|------|
| HyDE | `gpt-4o` | 가상 문서 생성 (Hypothetical Document Embedding) |

### 3. Reasoning Model (OpenAI GPT-5-mini)

복잡한 추론과 분석이 필요한 작업에 사용. (주의: temperature 미지원)

| 모듈 | 모델 | 용도 |
|------|------|------|
| ClauseAnalyzer | `gpt-5-mini` | 조항 추출 및 위반 분석 |
| RAPTOR | `gpt-5-mini` | 계층적 요약 (Recursive Abstractive Processing) |
| CRAG | `gpt-5-mini` | 검색 품질 평가 (Corrective RAG) |
| Constitutional AI | `gpt-5-mini` | 노동법 원칙 기반 검증 |
| DSPy | `gpt-5-mini` | 프롬프트 자동 최적화 |
| Redlining | `gpt-5-mini` | 위험 조항 수정 제안 |
| Judge | `gpt-5-mini` | 분석 결과 검증 (LLM-as-a-Judge) |

### 4. Embedding Model (Local)

벡터 임베딩 생성에 사용. 로컬에서 실행되어 API 비용 없음.

| 모듈 | 모델 | 용도 |
|------|------|------|
| Embedding | `nlpai-lab/KURE-v1` | 한국어 특화 임베딩 (sentence-transformers) |

## 모델 선택 기준

### Gemini (Retrieval)
- 빠른 응답 속도
- 저렴한 비용
- Vision 기능 지원 (이미지/PDF 처리)
- 간단한 텍스트 추출 작업

### OpenAI GPT (Reasoning)
- 복잡한 법률 분석
- 다단계 추론
- 구조화된 출력 생성
- 판례/법령 해석

### Local Embedding
- API 비용 없음
- 한국어 특화
- 오프라인 사용 가능

## 파이프라인 흐름

```
[문서 업로드]
     |
     v
[Vision Parser] -----> gemini-2.5-flash-lite (OCR/구조화)
     |
     v
[Text Chunking]
     |
     v
[Embedding] ---------> nlpai-lab/KURE-v1 (벡터화)
     |
     v
[HyDE] --------------> gpt-4o (가상 문서 생성)
     |
     v
[RAPTOR] ------------> gpt-5-mini (계층적 요약)
     |
     v
[CRAG] --------------> gpt-5-mini (검색 품질 평가)
     |
     v
[Constitutional AI] -> gpt-5-mini (법적 원칙 검증)
     |
     v
[Stress Test] -------> 규칙 기반 (LLM 미사용)
     |
     v
[Redlining] ---------> gpt-5-mini (수정 제안)
     |
     v
[Judge] -------------> gpt-5-mini (결과 검증)
     |
     v
[최종 분석 결과]
```

## 설정 파일 위치

- 환경 변수: `/.env`
- 설정 클래스: `/backend/app/core/config.py`
- LLM 클라이언트: `/backend/app/core/llm_client.py`
- 파이프라인 설정: `/backend/app/ai/pipeline.py`

## Reasoning Model 지원

gpt-5-mini, o1, o3 등 reasoning 모델은 temperature 파라미터를 지원하지 않음.
코드에서 `_is_reasoning_model()` 헬퍼 함수로 자동 감지하여 처리.

```python
def _is_reasoning_model(self) -> bool:
    """reasoning 모델 여부 확인 (temperature 미지원)"""
    reasoning_keywords = ["o1", "o3", "gpt-5"]
    return any(kw in self.model.lower() for kw in reasoning_keywords)
```

적용된 모듈:
- HyDE (`hyde.py`)
- RAPTOR (`raptor.py`)
- CRAG (`crag.py`)

## 모델 변경 시 주의사항

1. `.env` 파일의 환경 변수 수정
2. Celery worker 재시작 필요
3. 모델별 API 키 확인 (OPENAI_API_KEY, GEMINI_API_KEY)
4. 토큰 제한 및 비용 고려
5. Reasoning 모델은 temperature 미지원 (자동 처리됨)
