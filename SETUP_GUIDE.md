# 프로젝트 세팅 가이드

## 프로젝트 개요

근로계약서 분석 AI 시스템 - 데이터 수집부터 임베딩까지 완료

**현재 진행 상황:**
- 데이터 수집 완료
- 청킹 완료 (총 15,223개)
- 임베딩 생성 완료 (KURE-v1 모델)
- 다음 단계: Elasticsearch 인덱싱

---

## 데이터 구성

### PDF 문서 (674개 청크)
- 표준 근로계약서, 취업규칙, 채용절차 매뉴얼 등
- 최대 500자 단위 청킹

### 법률 데이터 (14,549개 청크)

| 데이터 타입 | 청크 수 | 출처 |
|------------|---------|------|
| 법령해석례 | 589개 | 국가법령정보센터 API |
| 판례 | 10,576개 | 국가법령정보센터 API |
| 고용노동부 법령해설 | 3,384개 | 국가법령정보센터 API |

**총 15,223개 청크** (PDF 674 + 법률 14,549)

---

## 시작하기

### 1. 저장소 클론

```bash
git clone https://github.com/danlee-dev/doc-scanner-ai.git
cd doc-scanner-ai
git checkout develop
```

### 2. 데이터 및 환경변수 다운로드

**구글 드라이브:** https://drive.google.com/drive/folders/1wtAp8VxmnYoLosFQOV2M51qeA5JWjzZ_?usp=sharing

다운로드 항목:
- `data/` 폴더 전체 (~560MB)
- `.env` 파일

### 3. 파일 배치

```
doc-scanner-ai/
├── ai/
│   ├── data/              # 다운받은 data 폴더 여기에 넣기
│   │   ├── raw/
│   │   └── processed/
│   └── preprocessing/
└── .env                   # 다운받은 .env 루트에 넣기
```

### 4. 가상환경 및 의존성 설치

**방법 1: venv 사용**

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**방법 2: conda 사용**

```bash
conda create -n docscanner python=3.9
conda activate docscanner
pip install -r requirements.txt
```

venv와 conda 중 편한 것 사용하면 됩니다.

### 5. 검색 테스트

```bash
cd ai/preprocessing/test
python search.py i
```

테스트 쿼리 예시:
- `수습기간 3개월`
- `최저임금 @legal`
- `연차 휴가 #10`

---

## 주요 문서

프로젝트 이해를 위해 다음 문서를 확인하세요:

**docs/** 폴더 구조:
- `guides/embedding-search.md` - 임베딩 검색 테스트 가이드
- `data-pipeline/legal-data-pipeline.md` - 법률 데이터 전체 파이프라인
- `data-pipeline/pdf-processing.md` - PDF 처리 파이프라인
- `project/project-structure.md` - 프로젝트 폴더 구조

**추천 읽기 순서:**
1. `project/project-structure.md` - 전체 구조 파악
2. `guides/embedding-search.md` - 검색 테스트 방법
3. `data-pipeline/legal-data-pipeline.md` - 데이터 파이프라인 이해

---

## Git 브랜치 전략

### 브랜치 구조
- `main` - 프로덕션 (배포용)
- `develop` - 개발 메인 브랜치
- `feature/*` - 기능 개발 브랜치

### 작업 방식

**방법 1: develop에서 직접 작업**

```bash
git checkout develop
git pull origin develop
# 작업...
git add .
git commit -m "feat: 기능 설명"
git push origin develop
```

**방법 2: feature 브랜치 생성 (권장)**

```bash
git checkout develop
git checkout -b feature/기능명

# 작업...
git add .
git commit -m "feat: 기능 설명"
git push origin feature/기능명

# GitHub에서 Pull Request 생성 → develop에 머지
```

편한 방식으로 작업하면 됩니다. feature 브랜치가 더 안전합니다.

---

## 주요 스크립트

### 법률 데이터 파이프라인

```bash
cd ai/preprocessing/legal

# 1. 데이터 수집
python 1_collect.py

# 2. 청킹
python 2_chunk.py

# 3. 임베딩 생성
python 3_embed.py
```

### PDF 데이터 파이프라인

```bash
cd ai/preprocessing/pdf

# 1. 텍스트 추출
python 1_extract.py

# 2. 청킹
python 2_chunk.py

# 3. 임베딩 생성
python 3_embed.py
```

### 검색 테스트

```bash
cd ai/preprocessing/test

# 대화형 모드
python search.py i

# 직접 쿼리
python search.py "수습기간 3개월"

# 사전 정의 테스트
python search.py t
```

---

## 다음 단계

### 진행 예정 작업

1. **Elasticsearch 인덱싱**
   - 하이브리드 검색 (BM25 + Vector) 구현
   - FastAPI 검색 엔드포인트 구축

2. **검색 품질 향상**
   - Re-ranking 모델 추가
   - 추가 키워드로 법률 데이터 확장

3. **프론트엔드 연동**
   - 계약서 업로드 → 분석 → 결과 표시

---

## 문제 해결

### 데이터 파일을 찾을 수 없다는 오류

구글 드라이브에서 `data/` 폴더를 다운받아 `ai/` 아래에 넣었는지 확인

### 모델 로딩 오류

인터넷 연결 확인 (KURE-v1 모델 자동 다운로드)

### 메모리 부족 오류

최소 2GB RAM 필요, `test_embeddings.py`만 실행하면 약 500MB 사용

---

## 연락처

문의사항이 있으면 팀 채널에 남겨주세요.

- **GitHub 저장소:** https://github.com/danlee-dev/doc-scanner-ai
- **구글 드라이브:** https://drive.google.com/drive/folders/1wtAp8VxmnYoLosFQOV2M51qeA5JWjzZ_?usp=sharing
- **작업 브랜치:** `develop`
