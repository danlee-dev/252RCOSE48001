# DocScanner.ai 문서

근로계약서 분석 AI 시스템의 전체 문서 모음입니다.

---

## 디렉토리 구조

```
docs/
├── README.md                     # 이 파일
├── architecture/                 # 시스템 아키텍처
│   ├── tech-specification.md     # 기술 명세서 (전체 개요)
│   ├── v1/                       # V1 아키텍처 (Archive)
│   │   ├── system-architecture-v1.md
│   │   ├── pipeline-architecture.md
│   │   └── v1-development-status.md
│   └── v2/                       # V2 아키텍처 (Current)
│       ├── contract-analysis-pipeline.md   # 계약서 분석 파이프라인
│       ├── chat-agent-architecture-v2.md   # 채팅 에이전트 아키텍처
│       ├── feature-list-v2.md              # 기능 리스트
│       └── v2-implementation-summary.md    # V2 구현 요약
├── api/                          # API 문서
│   └── api-specification.md
├── data-pipeline/                # 데이터 처리 파이프라인
│   ├── legal-data-collection.md
│   ├── legal-data-pipeline.md
│   ├── legal-data-types.md
│   └── pdf-processing.md
├── database/                     # 데이터베이스
│   ├── data-architecture.md
│   └── erd.dbml
├── evaluation/                   # 평가 계획
│   └── evaluation-plan.md
├── frontend/                     # 프론트엔드
│   ├── design-guide.md
│   └── pages-and-features.md
├── guides/                       # 가이드 문서
│   ├── development-setup.md
│   ├── embedding-search.md
│   └── v1-upgrade-guide.md
├── presentation/                 # 발표 자료
│   ├── script.md
│   └── qa.md
├── project/                      # 프로젝트 관리
│   ├── collaboration-guide.md
│   └── project-structure.md
├── reference/                    # 참고 자료
│   └── llm-configuration.md
└── troubleshooting/              # 트러블슈팅
    ├── frontend.md
    ├── backend.md
    └── ai.md
```

---

## 문서 카테고리

### Architecture (시스템 아키텍처)

현재 시스템의 아키텍처 설계 문서입니다.

| 문서 | 설명 |
|------|------|
| [기술 명세서](architecture/tech-specification.md) | 전체 시스템 기술 개요 및 설계 철학 |
| [V2 분석 파이프라인](architecture/v2/contract-analysis-pipeline.md) | 계약서 분석 AI 파이프라인 상세 |
| [V2 채팅 에이전트](architecture/v2/chat-agent-architecture-v2.md) | LangGraph 기반 채팅 에이전트 |
| [V2 기능 리스트](architecture/v2/feature-list-v2.md) | 전체 기능 목록 및 구현 현황 |

### Guides (가이드)

개발 환경 설정 및 사용 가이드입니다.

| 문서 | 설명 |
|------|------|
| [개발 환경 설정](guides/development-setup.md) | Docker, Backend, Frontend 실행 |
| [임베딩 검색 가이드](guides/embedding-search.md) | 통합 검색 시스템 테스트 |

### Data Pipeline (데이터 처리)

법률 데이터 수집 및 처리 파이프라인입니다.

| 문서 | 설명 |
|------|------|
| [법률 데이터 파이프라인](data-pipeline/legal-data-pipeline.md) | 전체 데이터 처리 흐름 |
| [법률 데이터 수집](data-pipeline/legal-data-collection.md) | API 활용 데이터 수집 |
| [법률 데이터 타입](data-pipeline/legal-data-types.md) | 법령, 판례, 해석례 설명 |
| [PDF 처리](data-pipeline/pdf-processing.md) | PDF 추출 및 청킹 |

### Troubleshooting (문제 해결)

개발 중 발생한 문제와 해결 방법입니다.

| 문서 | 영역 |
|------|------|
| [Frontend](troubleshooting/frontend.md) | 프론트엔드 이슈 |
| [Backend](troubleshooting/backend.md) | 백엔드 이슈 |
| [AI/ML](troubleshooting/ai.md) | AI/ML 관련 이슈 |

### Frontend (프론트엔드)

UI/UX 및 프론트엔드 관련 문서입니다.

| 문서 | 설명 |
|------|------|
| [디자인 가이드](frontend/design-guide.md) | 색상, 타이포그래피, 컴포넌트 스타일 |
| [페이지 및 기능](frontend/pages-and-features.md) | 페이지별 기능 명세 |

---

## 빠른 시작

### 신규 팀원

1. [개발 환경 설정](guides/development-setup.md) - 환경 구축
2. [프로젝트 구조](project/project-structure.md) - 폴더 구조 파악
3. [협업 가이드](project/collaboration-guide.md) - Git 워크플로우

### 시스템 이해

1. [기술 명세서](architecture/tech-specification.md) - 전체 개요
2. [V2 분석 파이프라인](architecture/v2/contract-analysis-pipeline.md) - AI 파이프라인
3. [V2 채팅 에이전트](architecture/v2/chat-agent-architecture-v2.md) - 채팅 시스템

### 작업별 참고 문서

| 작업 | 참고 문서 |
|------|----------|
| 개발 환경 설정 | [development-setup.md](guides/development-setup.md) |
| 법률 데이터 추가 | [legal-data-collection.md](data-pipeline/legal-data-collection.md) |
| 검색 테스트 | [embedding-search.md](guides/embedding-search.md) |
| API 확인 | [api-specification.md](api/api-specification.md) |
| 문제 해결 | [troubleshooting/](troubleshooting/) |

---

## 현재 시스템 상태

**버전**: V2 (Current)

**데이터**:
- PDF 문서: 674개 청크
- 법률 데이터: 14,549개 청크
- 총 15,223개 청크

**모델**:
- Embedding: KURE-v1 (한국어 법률 특화, 1024차원)
- LLM: GPT-4o, Gemini 2.5 Flash (하이브리드)

**검색**:
- Elasticsearch (Vector + BM25 Hybrid)
- Neo4j (Knowledge Graph)

---

## 문서 작성 규칙

1. **파일명**: kebab-case 사용 (예: `chat-agent-architecture.md`)
2. **이모지**: 사용하지 않음
3. **폴더 배치**:
   - 아키텍처: `architecture/`
   - 가이드: `guides/`
   - 데이터 관련: `data-pipeline/`
   - 문제 해결: `troubleshooting/`
   - 발표 자료: `presentation/`

---

*마지막 업데이트: 2025-12-17*
