# 문서 구조

근로계약서 분석 AI 시스템의 전체 문서 모음입니다.

## 디렉토리 구조

```
docs/
├── README.md                    # 이 파일
├── project/                     # 프로젝트 관리
│   ├── collaboration-guide.md   # 협업 가이드
│   └── project-structure.md     # 프로젝트 폴더 구조
├── data-pipeline/               # 데이터 처리 파이프라인
│   ├── pdf-processing.md        # PDF 문서 처리 파이프라인
│   ├── legal-data-pipeline.md   # 법률 데이터 전체 파이프라인
│   ├── legal-data-collection.md # 법률 데이터 수집 가이드
│   └── legal-data-types.md      # 법률 데이터 타입 설명
└── guides/                      # 사용 가이드
    ├── development-setup.md     # 개발 환경 설정 가이드
    └── embedding-search.md      # 임베딩 검색 테스트 가이드
```

## 문서 소개

### 프로젝트 관리 (project/)

**[협업 가이드](project/collaboration-guide.md)**
- 팀 협업 규칙
- Git 사용 방법
- 코드 리뷰 프로세스

**[프로젝트 구조](project/project-structure.md)**
- 전체 폴더 구조
- 각 디렉토리 역할
- 파일 네이밍 규칙

### 데이터 처리 (data-pipeline/)

**[PDF 데이터 처리](data-pipeline/pdf-processing.md)**
- PDF 추출 및 청킹
- 카테고리 분류
- 임베딩 생성

**[법률 데이터 파이프라인](data-pipeline/legal-data-pipeline.md)**
- 법률 데이터 수집부터 임베딩까지 전체 프로세스
- 데이터 통계 및 결과물
- 다음 단계 (Elasticsearch)

**[법률 데이터 수집](data-pipeline/legal-data-collection.md)**
- API 사용 방법
- 데이터 소스 설명
- 수집 스크립트 실행

**[법률 데이터 타입](data-pipeline/legal-data-types.md)**
- 법령해석례, 판례, 고용노동부 해설 설명
- 각 타입별 데이터 구조
- 활용 방법

### 사용 가이드 (guides/)

**[개발 환경 설정 가이드](guides/development-setup.md)**

- Docker, Backend, Celery, Frontend 실행 방법
- Mac / Windows 환경별 설정
- 문제 해결 가이드

**[임베딩 검색 가이드](guides/embedding-search.md)**
- 통합 임베딩 검색 시스템 사용법
- 현재 데이터 구성 및 통계
- 청킹 기준 및 검색 테스트 방법
- **팀원이 바로 테스트할 수 있는 완전한 가이드**

## 빠른 시작

### 새로운 팀원이 처음 시작할 때

1. **개발 환경 설정**:
   - [개발 환경 설정 가이드](guides/development-setup.md) 따라 환경 구축
   - Docker, Backend, Celery 실행

2. **프로젝트 이해**:
   - [프로젝트 구조](project/project-structure.md) 읽기
   - [협업 가이드](project/collaboration-guide.md) 확인

3. **데이터 파이프라인 이해**:
   - [PDF 데이터 처리](data-pipeline/pdf-processing.md) 읽기
   - [법률 데이터 파이프라인](data-pipeline/legal-data-pipeline.md) 읽기

4. **검색 테스트 실행**:
   - [임베딩 검색 가이드](guides/embedding-search.md) 참고
   - 테스트 스크립트 실행

### 특정 작업별 참고 문서

| 작업 | 참고 문서 |
|------|----------|
| 개발 환경 설정 | [개발 환경 설정 가이드](guides/development-setup.md) |
| 법률 데이터 추가 수집 | [법률 데이터 수집](data-pipeline/legal-data-collection.md) |
| PDF 문서 추가 | [PDF 데이터 처리](data-pipeline/pdf-processing.md) |
| 검색 품질 테스트 | [임베딩 검색 가이드](guides/embedding-search.md) |
| 데이터 타입 이해 | [법률 데이터 타입](data-pipeline/legal-data-types.md) |
| 프로젝트 구조 파악 | [프로젝트 구조](project/project-structure.md) |

## 현재 시스템 상태

**데이터**:
- PDF 문서: 674개 청크
- 법률 데이터: 14,549개 청크
- 총 15,223개 청크

**모델**:
- KURE-v1 (한국어 법률 특화)
- 임베딩 차원: 1024

**검색**:
- 통합 검색 시스템 구축 완료
- 필터링 기능 지원 (PDF/법률/문서타입별)
- 테스트 스크립트 제공

## 다음 단계

1. **Elasticsearch 인덱싱**
   - 하이브리드 검색 (BM25 + Vector)
   - FastAPI 검색 엔드포인트 구현

2. **검색 품질 향상**
   - Re-ranking 모델 추가
   - 추가 키워드로 법률 데이터 확장

3. **프론트엔드 연동**
   - 계약서 업로드 → 분석 → 결과 표시
   - 유사 판례 검색 기능

## 문서 작성 규칙

새로운 문서를 추가할 때는 다음 규칙을 따라주세요:

1. **파일명**: kebab-case 사용 (예: `embedding-search.md`)
2. **제목 형식**:
   ```markdown
   # 문서 제목

   ## 개요

   간단한 설명...
   ```
3. **이모지**: 사용하지 않음
4. **적절한 폴더**: 프로젝트 관리는 `project/`, 데이터는 `data-pipeline/`, 사용법은 `guides/`

## 문의

문서 관련 문의사항이나 개선 제안은 팀 미팅에서 논의하거나 GitHub Issue로 등록해주세요.
