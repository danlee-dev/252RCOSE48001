# 프로젝트 폴더 구조

## 개요

본 문서는 근로계약서 분석 AI 시스템의 전체 폴더 구조와 각 디렉토리의 역할을 설명합니다.

## 전체 구조

```
docscanner-ai/
├── ai/                    # AI/ML 관련 모든 코드
├── backend/              # 백엔드 API 서버 (준비 중)
├── frontend/             # Next.js 프론트엔드
└── docs/                 # 프로젝트 문서
```

## 1. ai/ - AI/ML 디렉토리

AI 모델, 데이터 처리, 임베딩 생성 등 AI 관련 모든 코드를 포함합니다.

```
ai/
├── data/
│   ├── raw/
│   │   └── documents/
│   │       ├── guides/
│   │       └── standard_contracts/
│   └── processed/
│       ├── chunks/
│       ├── documents/
│       ├── embeddings/
│       └── required_contract_fields.json
├── preprocessing/
│   ├── legal/              # 법률 데이터 파이프라인
│   │   ├── 1_collect.py
│   │   ├── 2_chunk.py
│   │   └── 3_embed.py
│   ├── pdf/                # PDF 파이프라인
│   │   ├── 1_extract.py
│   │   ├── 2_chunk.py
│   │   └── 3_embed.py
│   ├── contract/           # 계약서 분석
│   │   └── extract_fields.py
│   └── test/               # 테스트
│       └── search.py
└── requirements.txt
```

### 1.1 data/ - 데이터 디렉토리

#### data/raw/ - 원본 데이터

**documents/standard_contracts/**
- 원본 PDF 파일 저장 (5개)
- 개정 표준근로계약서(2025년, 배포).pdf
- '25년 채용절차의 공정화에 관한 법률 업무 매뉴얼.pdf
- 개정 표준취업규칙(2025년, 배포).pdf
- 2025년 적용 최저임금 안내.pdf
- 채용절차의 공정화에 관한 법률 리플릿.pdf

**documents/guides/**
- 추가 안내문서 저장소

#### data/processed/ - 처리된 데이터

**documents/standard_contracts/**
- PDF에서 추출된 JSON 형식 텍스트
- 각 파일은 페이지 구분자와 메타데이터 포함

**chunks/**
- `all_chunks.json`: 전체 청크 데이터 (674개)
- `metadata.json`: 청크 통계 정보
  - 문서 유형별 개수
  - 카테고리별 분포
  - 소스별 청크 수

**embeddings/**
- `chunks_with_embeddings.json`: 임베딩 벡터가 포함된 청크
- `embeddings.npy`: NumPy 배열 형태의 임베딩 (빠른 로딩)
- `embedding_metadata.json`: 임베딩 설정 정보
  - 모델명: nlpai-lab/KURE-v1
  - 차원: 1024
  - 배치 크기: 8

**required_contract_fields.json**
- 계약서 유형별 필수 필드 체크리스트
- 5가지 계약 유형 (정규직, 기간제, 연소근로자, 건설일용, 단시간근로자)
- 각 필드의 template, description, regulation 포함

### 1.2 preprocessing/ - 전처리 스크립트

**legal/** - 법률 데이터 파이프라인
- `1_collect.py`: 국가법령정보 API로 법률 데이터 수집
- `2_chunk.py`: 법률 문서 구조 기반 청킹
- `3_embed.py`: KURE-v1 모델로 법률 데이터 임베딩

**pdf/** - PDF 문서 파이프라인
- `1_extract.py`: pdfplumber로 PDF 텍스트 추출
- `2_chunk.py`: 문서 유형별 청킹 전략 구현
- `3_embed.py`: KURE-v1 모델로 PDF 청크 임베딩

**contract/** - 계약서 분석
- `extract_fields.py`: 표준근로계약서 필수 필드 추출 및 체크리스트 생성

**test/** - 테스트 도구
- `search.py`: 통합 임베딩 검색 테스트 (PDF + 법률 데이터)

### 1.3 requirements.txt

Python 의존성 패키지:
- pdfplumber: PDF 텍스트 추출
- sentence-transformers: 임베딩 모델
- torch: 딥러닝 프레임워크
- numpy: 수치 연산
- tqdm: 진행률 표시

## 2. backend/ - 백엔드 디렉토리

현재 비어있음. 향후 다음 기능 구현 예정:

**계획된 구조**:
```
backend/
├── api/              # API 엔드포인트
├── models/           # 데이터 모델
├── services/         # 비즈니스 로직
│   ├── contract_analyzer.py
│   ├── rag_engine.py
│   └── elasticsearch_client.py
├── config/           # 설정 파일
└── requirements.txt
```

**주요 기능**:
- 계약서 업로드 및 파싱
- RAG 기반 질의응답
- Elasticsearch 연동
- 계약서 분석 결과 반환

## 3. frontend/ - 프론트엔드 디렉토리

Next.js 기반 웹 애플리케이션

```
frontend/
├── public/           # 정적 파일
├── src/
│   ├── app/          # Next.js App Router
│   │   ├── analysis/
│   │   ├── compare/
│   │   ├── contracts/
│   │   ├── upload/
│   │   ├── layout.tsx
│   │   └── page.tsx
│   ├── components/   # React 컴포넌트
│   │   ├── analysis/
│   │   ├── compare/
│   │   ├── contracts/
│   │   ├── dashboard/
│   │   ├── layout/
│   │   ├── ui/
│   │   └── upload/
│   └── lib/          # 유틸리티
│       └── utils.ts
├── package.json
├── tsconfig.json
├── tailwind.config.js
└── next.config.ts
```

### 3.1 app/ - 페이지 라우팅

**upload/**
- 계약서 업로드 페이지
- 파일 선택 및 업로드 인터페이스

**analysis/**
- 계약서 분석 결과 페이지
- 위험 조항, 누락 항목, 불공정 조항 표시

**compare/**
- 계약서 비교 페이지
- 여러 계약서 간 차이점 비교

**contracts/**
- 저장된 계약서 목록
- 계약서 관리 인터페이스

### 3.2 components/ - 컴포넌트

**ui/**
- 재사용 가능한 UI 컴포넌트
- 버튼, 카드, 모달 등

**layout/**
- 레이아웃 관련 컴포넌트
- 헤더, 사이드바, 푸터

**upload/**
- 파일 업로드 관련 컴포넌트
- 드래그 앤 드롭, 진행률 표시

**analysis/**
- 분석 결과 표시 컴포넌트
- 하이라이트, 툴팁, 설명 패널

**compare/**
- 비교 결과 표시 컴포넌트
- 차이점 표시, 병렬 비교

**contracts/**
- 계약서 목록 및 관리 컴포넌트
- 카드, 필터, 정렬

## 4. docs/ - 문서 디렉토리

프로젝트 관련 문서 모음

```
docs/
├── data-processing-pipeline.md
└── project-structure.md
```

**data-processing-pipeline.md**
- 데이터 수집부터 임베딩까지 전체 파이프라인 설명
- 청킹 전략, 임베딩 모델, 성능 테스트 결과

**project-structure.md**
- 현재 문서
- 프로젝트 폴더 구조 및 각 디렉토리 역할 설명

## 5. 데이터 흐름

**PDF 파이프라인:**
```
[PDF 파일]
    ↓ pdf/1_extract.py
[추출된 JSON]
    ↓ pdf/2_chunk.py
[PDF 청크 (674개)]
    ↓ pdf/3_embed.py
[PDF 임베딩 (1024차원)]
```

**법률 데이터 파이프라인:**
```
[국가법령정보 API]
    ↓ legal/1_collect.py
[법률 JSON (2,931건)]
    ↓ legal/2_chunk.py
[법률 청크 (14,549개)]
    ↓ legal/3_embed.py
[법률 임베딩 (1024차원)]
```

**통합 검색:**
```
[PDF 임베딩 + 법률 임베딩]
    ↓ test/search.py
[통합 검색 (15,223개 청크)]
    ↓
[Elasticsearch 인덱스] (예정)
    ↓
[RAG 시스템] (예정)
    ↓
[프론트엔드 UI]
```

## 6. 개발 환경

### 6.1 AI/ML
- Python: 3.10
- Conda 환경: docscanner-py3.10
- GPU: Apple M3 Pro (MPS 지원)

### 6.2 프론트엔드
- Node.js
- Next.js (App Router)
- TypeScript
- Tailwind CSS

### 6.3 백엔드 (예정)
- Python FastAPI 또는 Node.js Express
- Elasticsearch
- PostgreSQL (선택적)

## 7. 실행 방법

### 7.1 데이터 전처리

**PDF 파이프라인:**
```bash
cd ai/preprocessing/pdf

# PDF 추출
python 1_extract.py

# 청킹
python 2_chunk.py

# 임베딩 생성
python 3_embed.py
```

**법률 데이터 파이프라인:**
```bash
cd ai/preprocessing/legal

# 데이터 수집
python 1_collect.py

# 청킹
python 2_chunk.py

# 임베딩 생성
python 3_embed.py
```

**검색 테스트:**
```bash
cd ai/preprocessing/test

# 대화형 모드
python search.py i

# 사전 정의 테스트
python search.py t

# 직접 검색
python search.py "쿼리"
```

### 7.2 프론트엔드

```bash
cd frontend
npm install
npm run dev
```

브라우저에서 http://localhost:3000 접속

## 8. 향후 추가 예정

### 8.1 백엔드 구현
- FastAPI 기반 REST API
- Elasticsearch 연동
- RAG 엔진 구현
- 계약서 분석 로직

### 8.2 법률 정보 API 연동
- open.law.go.kr API 승인 대기 중
- 법령, 판례, 행정규칙 데이터 추가
- 증분 인덱싱

### 8.3 추가 기능
- 사용자 인증 및 권한 관리
- 계약서 히스토리 관리
- 대시보드 및 통계
- PDF 다운로드 및 보고서 생성

## 9. 참고 사항

### 9.1 .gitignore 설정

다음 항목은 Git에서 제외:
- `ai/data/` (데이터 파일)
- `__pycache__/`, `*.pyc` (Python 캐시)
- `node_modules/` (Node 패키지)
- `.next/` (Next.js 빌드)
- `.env*` (환경 변수)

### 9.2 용량 관리

대용량 파일:
- `embeddings.npy`: 약 2.7MB
- `chunks_with_embeddings.json`: 약 4.5MB
- PDF 파일: 총 약 9MB

## 10. 변경 이력

### 2025-10-27
- 초기 폴더 구조 생성
- 불필요한 빈 폴더 제거 (8개)
- AI 전처리 파이프라인 구현 완료
- 프로젝트 구조 문서 작성
