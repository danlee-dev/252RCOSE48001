# DocScanner AI API Specification v1

이 문서는 DocScanner AI 백엔드의 모든 REST API 엔드포인트를 상세히 정의합니다.

---

## 목차

1. [Base URL](#base-url)
2. [인증](#인증)
3. [Auth API](#1-auth-api)
4. [Users API](#2-users-api)
5. [Contracts API](#3-contracts-api)
6. [Agent Chat API](#4-agent-chat-api)
7. [Chat API (Dify)](#5-chat-api-dify)
8. [Analysis API](#6-analysis-api)
9. [Search API](#7-search-api)
10. [Scan API](#8-scan-api)
11. [Checklist API](#9-checklist-api)
12. [Tool API (Internal)](#10-tool-api-internal)
13. [공통 에러 코드](#공통-에러-코드)

---

## Base URL

```
개발: http://localhost:8000/api/v1
프로덕션: https://api.docscanner.ai/api/v1
```

---

## 인증

대부분의 API는 JWT Bearer 토큰 인증이 필요합니다.

### 요청 헤더

```http
Authorization: Bearer <access_token>
```

### 토큰 구조

| 토큰 유형 | 만료 시간 | 용도 |
|----------|----------|------|
| Access Token | 30분 | API 요청 인증 |
| Refresh Token | 7일 | Access Token 갱신 |

---

## 1. Auth API

인증 관련 엔드포인트입니다.

### 1.1 회원가입

```http
POST /api/v1/auth/signup
```

새로운 사용자 계정을 생성합니다.

**Request Body**

```json
{
  "email": "user@example.com",
  "username": "홍길동",
  "password": "securePassword123"
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| email | string (email) | O | 이메일 (unique) |
| username | string | O | 사용자명 |
| password | string | O | 비밀번호 |

**Response (201 Created)**

```json
{
  "id": 1,
  "email": "user@example.com",
  "username": "홍길동",
  "created_at": "2025-01-01T00:00:00Z"
}
```

**Error Responses**

| 상태 코드 | 설명 |
|----------|------|
| 400 | 이미 가입된 이메일 |

---

### 1.2 로그인

```http
POST /api/v1/auth/login
```

이메일과 비밀번호로 로그인하여 JWT 토큰을 발급받습니다.

**Request Body**

```json
{
  "email": "user@example.com",
  "password": "securePassword123"
}
```

**Response (200 OK)**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

**Error Responses**

| 상태 코드 | 설명 |
|----------|------|
| 401 | 이메일 또는 비밀번호 불일치 |

---

### 1.3 토큰 갱신

```http
POST /api/v1/auth/refresh
```

Refresh Token으로 새로운 Access Token을 발급받습니다.

**Request Body**

```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response (200 OK)**

```json
{
  "access_token": "new_access_token",
  "refresh_token": "new_refresh_token",
  "token_type": "bearer"
}
```

---

### 1.4 로그아웃

```http
POST /api/v1/auth/logout
```

현재 세션을 종료하고 Refresh Token을 무효화합니다.

**Headers**: `Authorization: Bearer <access_token>` (필수)

**Response (204 No Content)**

---

## 2. Users API

사용자 정보 관리 엔드포인트입니다.

### 2.1 내 정보 조회

```http
GET /api/v1/users/me
```

현재 로그인한 사용자의 정보를 조회합니다.

**Headers**: `Authorization: Bearer <access_token>` (필수)

**Response (200 OK)**

```json
{
  "id": 1,
  "email": "user@example.com",
  "username": "홍길동",
  "created_at": "2025-01-01T00:00:00Z"
}
```

---

### 2.2 비밀번호 변경

```http
PATCH /api/v1/users/me/password
```

현재 비밀번호를 확인하고 새로운 비밀번호로 변경합니다.

**Headers**: `Authorization: Bearer <access_token>` (필수)

**Request Body**

```json
{
  "current_password": "oldPassword123",
  "new_password": "newSecurePassword456"
}
```

**Response (200 OK)**

```json
{
  "message": "비밀번호가 성공적으로 변경되었습니다."
}
```

**Error Responses**

| 상태 코드 | 설명 |
|----------|------|
| 400 | 현재 비밀번호 불일치 |
| 400 | 새 비밀번호가 현재 비밀번호와 동일 |

---

### 2.3 회원 탈퇴

```http
DELETE /api/v1/users/me
```

현재 로그인한 사용자의 계정을 영구 삭제합니다.

**Headers**: `Authorization: Bearer <access_token>` (필수)

**Response (204 No Content)**

---

## 3. Contracts API

계약서 업로드 및 관리 엔드포인트입니다.

### 3.1 계약서 업로드

```http
POST /api/v1/contracts/
```

PDF 계약서 파일을 업로드하고 AI 분석 작업을 시작합니다.

**Headers**: `Authorization: Bearer <access_token>` (필수)

**Request**: `multipart/form-data`

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| file | File (PDF) | O | PDF 계약서 파일 (10MB 이하 권장) |

**Response (202 Accepted)**

```json
{
  "message": "Accepted",
  "contract_id": 42,
  "status": "PENDING"
}
```

분석은 백그라운드(Celery)에서 비동기 처리됩니다.

**Error Responses**

| 상태 코드 | 설명 |
|----------|------|
| 400 | PDF가 아닌 파일 형식 |
| 500 | 파일 저장 오류 |

---

### 3.2 계약서 목록 조회

```http
GET /api/v1/contracts/
```

현재 사용자가 업로드한 모든 계약서 목록을 조회합니다.

**Headers**: `Authorization: Bearer <access_token>` (필수)

**Query Parameters**

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|-------|------|
| skip | integer | 0 | 페이지네이션 오프셋 |
| limit | integer | 10 | 페이지 크기 |
| search | string | - | 제목 검색 키워드 |

**Response (200 OK)**

```json
[
  {
    "id": 42,
    "title": "2025년 표준근로계약서.pdf",
    "status": "COMPLETED",
    "risk_level": "HIGH",
    "created_at": "2025-01-01T12:00:00Z"
  },
  {
    "id": 41,
    "title": "아르바이트 계약서.pdf",
    "status": "ANALYZING",
    "risk_level": null,
    "created_at": "2025-01-01T11:00:00Z"
  }
]
```

**Status 값**

| 값 | 설명 |
|----|------|
| PENDING | 대기 중 |
| ANALYZING | 분석 중 |
| COMPLETED | 분석 완료 |
| FAILED | 분석 실패 |

---

### 3.3 계약서 상세 조회

```http
GET /api/v1/contracts/{contract_id}
```

특정 계약서의 상세 정보와 AI 분석 결과를 조회합니다.

**Headers**: `Authorization: Bearer <access_token>` (필수)

**Path Parameters**

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| contract_id | integer | 계약서 ID |

**Response (200 OK)**

```json
{
  "id": 42,
  "title": "2025년 표준근로계약서.pdf",
  "status": "COMPLETED",
  "risk_level": "HIGH",
  "file_url": "/uploads/user_1/contracts/abc123.pdf",
  "extracted_text": "표준근로계약서\n\n제1조 (계약기간)...",
  "analysis_result": {
    "risk_level": "HIGH",
    "summary": "이 계약서에서 3건의 고위험 조항이 발견되었습니다.",
    "stress_test": {
      "total_underpayment": 150000,
      "annual_underpayment": 1800000,
      "violations": [
        {
          "clause_number": "제5조",
          "type": "최저임금 미달",
          "severity": "HIGH",
          "description": "시급이 2025년 최저임금(10,030원) 미만입니다.",
          "legal_basis": "최저임금법 제6조",
          "suggestion": "시급을 10,030원 이상으로 조정하세요."
        }
      ]
    },
    "judge_score": {
      "overall": 0.85,
      "accuracy": 0.9,
      "legal_basis": 0.85
    }
  },
  "created_at": "2025-01-01T12:00:00Z"
}
```

**Error Responses**

| 상태 코드 | 설명 |
|----------|------|
| 404 | 계약서를 찾을 수 없음 |

---

### 3.4 계약서 삭제

```http
DELETE /api/v1/contracts/{contract_id}
```

특정 계약서와 관련 파일을 삭제합니다.

**Headers**: `Authorization: Bearer <access_token>` (필수)

**Path Parameters**

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| contract_id | integer | 계약서 ID |

**Response (204 No Content)**

---

## 4. Agent Chat API

LangGraph 기반 AI 채팅 에이전트입니다. SSE(Server-Sent Events) 스트리밍을 지원합니다.

### 4.1 스트리밍 채팅

```http
POST /api/v1/agent/{contract_id}/stream
```

계약서 컨텍스트를 포함하여 AI와 채팅합니다. 응답은 SSE로 스트리밍됩니다.

**Headers**: `Authorization: Bearer <access_token>` (필수)

**Path Parameters**

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| contract_id | integer | 계약서 ID |

**Request Body**

```json
{
  "message": "이 계약서에서 가장 위험한 조항은 뭐야?",
  "conversation_id": null,
  "include_contract": true
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| message | string | O | 사용자 메시지 |
| conversation_id | string | X | 대화 ID (이어서 대화 시) |
| include_contract | boolean | X | 계약서 원문 포함 여부 (기본: true) |

**Response**: `text/event-stream`

```
data: {"type": "step", "step": "analyzing", "message": "질문 분석 중..."}

data: {"type": "tool", "tool": "search_vector_db", "status": "searching", "message": "법령/판례 검색 중..."}

data: {"type": "tool", "tool": "search_vector_db", "status": "complete", "message": "법령/판례 검색 완료"}

data: {"type": "step", "step": "generating", "message": "답변 생성 중..."}

data: {"type": "token", "content": "이 "}

data: {"type": "token", "content": "계약서의 "}

data: {"type": "token", "content": "가장 위험한 조항은..."}

data: {"type": "done", "full_response": "이 계약서의 가장 위험한 조항은..."}
```

**SSE Event Types**

| type | 설명 | 데이터 |
|------|------|--------|
| step | 처리 단계 변경 | step, message |
| tool | 도구 실행 상태 | tool, status, message |
| token | 응답 토큰 (스트리밍) | content |
| done | 완료 | full_response |
| error | 오류 발생 | message |

---

### 4.2 대화 히스토리 포함 스트리밍 채팅

```http
POST /api/v1/agent/{contract_id}/stream/history
```

이전 대화 내용을 포함하여 채팅합니다.

**Request Body**

```json
{
  "message": "그럼 어떻게 신고해?",
  "history": [
    {"role": "user", "content": "최저임금 위반이 있어?"},
    {"role": "assistant", "content": "네, 제5조에서 최저임금 위반이 발견되었습니다..."}
  ],
  "include_contract": true
}
```

---

### 4.3 동기식 채팅 (Fallback)

```http
POST /api/v1/agent/{contract_id}
```

SSE를 지원하지 않는 클라이언트를 위한 동기식 채팅입니다.

**Response (200 OK)**

```json
{
  "answer": "이 계약서에서 가장 위험한 조항은 제5조입니다...",
  "tools_used": ["search_vector_db", "web_search"],
  "contract_id": 42
}
```

---

### 4.4 에이전트 상태 확인

```http
GET /api/v1/agent/health
```

에이전트 서비스 상태를 확인합니다.

**Response (200 OK)**

```json
{
  "status": "healthy",
  "agent": "ready"
}
```

---

## 5. Chat API (Dify)

Dify Agent와 연동된 채팅 API입니다.

### 5.1 Dify 채팅

```http
POST /api/v1/chat/{contract_id}
```

Dify Agent를 통해 계약서 기반 채팅을 수행합니다.

**Request Body**

```json
{
  "message": "포괄임금제가 왜 위험한가요?",
  "conversation_id": null
}
```

**Response (200 OK)**

```json
{
  "answer": "포괄임금제는 다음과 같은 이유로 위험합니다...",
  "conversation_id": "conv_abc123",
  "message_id": "msg_xyz789",
  "sources": [
    {
      "content": "근로기준법 제56조...",
      "source": "법령해석례"
    }
  ]
}
```

---

### 5.2 대화 목록 조회

```http
GET /api/v1/chat/{contract_id}/conversations
```

해당 계약서와 관련된 Dify 대화 목록을 조회합니다.

---

### 5.3 대화 내역 조회

```http
GET /api/v1/chat/{contract_id}/conversations/{conversation_id}/messages
```

특정 대화의 메시지 내역을 조회합니다.

---

## 6. Analysis API

고급 AI 분석 기능을 개별적으로 호출할 수 있는 API입니다.

### 6.1 Legal Stress Test

```http
POST /api/v1/analysis/stress-test
```

계약서의 임금/근로시간 조건을 수치 시뮬레이션합니다.

**Headers**: `Authorization: Bearer <access_token>` (필수)

**Request Body**

```json
{
  "contract_text": "제3조 (임금) 월 급여는 200만원으로 한다...",
  "monthly_hours": 209,
  "include_details": true
}
```

**Response (200 OK)**

```json
{
  "violations": [
    {
      "type": "최저임금 미달",
      "severity": "HIGH",
      "amount": 96270,
      "legal_basis": "최저임금법 제6조"
    }
  ],
  "total_underpayment": 96270,
  "annual_underpayment": 1155240,
  "risk_score": 0.75,
  "summary": "최저임금 미달로 월 96,270원의 체불이 예상됩니다."
}
```

---

### 6.2 Generative Redlining

```http
POST /api/v1/analysis/redlining
```

위험 조항을 탐지하고 수정 제안을 생성합니다.

**Request Body**

```json
{
  "contract_text": "제5조 계약 해지 시 위약금 300%를 배상한다...",
  "output_format": "json"
}
```

| output_format | 설명 |
|---------------|------|
| json | 구조화된 JSON |
| diff | Git-style unified diff |
| html | HTML 시각화 |

**Response (200 OK)**

```json
{
  "changes": [
    {
      "type": "DELETE",
      "original": "위약금 300%를 배상한다",
      "revised": null,
      "reason": "근로기준법 제20조 위반 (위약 예정 금지)",
      "legal_basis": "근로기준법 제20조",
      "severity": "HIGH"
    }
  ],
  "change_count": 1,
  "high_risk_count": 1,
  "diff_view": null
}
```

---

### 6.3 LLM-as-a-Judge

```http
POST /api/v1/analysis/judge
```

AI 분석 결과의 신뢰도를 평가합니다.

**Request Body**

```json
{
  "analysis": "이 계약서는 최저임금 위반 소지가 있습니다...",
  "context": "원본 계약서 텍스트..."
}
```

**Response (200 OK)**

```json
{
  "overall_score": 0.85,
  "confidence_level": "HIGH",
  "is_reliable": true,
  "verdict": "이 분석은 신뢰할 수 있습니다.",
  "scores": {
    "accuracy": 0.9,
    "legal_basis": 0.85,
    "consistency": 0.8,
    "completeness": 0.85,
    "relevance": 0.9
  },
  "recommendations": []
}
```

---

### 6.4 PII Masking

```http
POST /api/v1/analysis/pii-mask
```

텍스트에서 개인정보를 탐지하고 마스킹합니다.

**Request Body**

```json
{
  "text": "홍길동 (010-1234-5678) 서울시 강남구 테헤란로 123",
  "format_preserving": true
}
```

**Response (200 OK)**

```json
{
  "masked_text": "[이름_1] (010-****-5678) 서울시 [주소_1]",
  "pii_count": 3,
  "pii_types": {
    "NAME": 1,
    "PHONE": 1,
    "ADDRESS": 1
  }
}
```

---

### 6.5 HyDE (검색 쿼리 강화)

```http
POST /api/v1/analysis/hyde
```

구어체 질문을 법률 전문 용어로 변환합니다.

**Request Body**

```json
{
  "query": "야근 수당 안 줘도 되나요?",
  "prompt_type": "labor_law"
}
```

**Response (200 OK)**

```json
{
  "original_query": "야근 수당 안 줘도 되나요?",
  "primary_document": "근로기준법 제56조에 따르면 연장근로에 대해서는...",
  "enhanced_query": "연장근로수당 지급의무 근로기준법 제56조"
}
```

---

### 6.6 Constitutional AI Review

```http
POST /api/v1/analysis/constitutional-review
```

AI 응답이 노동법 원칙에 부합하는지 검토하고 수정합니다.

**Request Body**

```json
{
  "response": "이 위약금 조항은 계약 위반 시 손해배상을 규정하고 있습니다.",
  "context": "원본 계약서..."
}
```

**Response (200 OK)**

```json
{
  "original_response": "이 위약금 조항은 계약 위반 시 손해배상을 규정하고 있습니다.",
  "revised_response": "이 위약금 조항은 근로기준법 제20조(위약 예정의 금지)를 위반할 수 있습니다...",
  "was_revised": true,
  "critique": "제3조(최저기준 보장) 위반 - 위약 예정 금지 원칙 미언급",
  "principles_violated": ["MINIMUM_STANDARD"]
}
```

---

### 6.7 Reasoning Trace

```http
GET /api/v1/analysis/contract/{contract_id}/reasoning-trace
```

계약서 분석의 추론 과정을 시각화합니다.

**Query Parameters**

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|-------|------|
| format | string | mermaid | 출력 형식: mermaid, d3, cytoscape |

**Response (200 OK)**

```json
{
  "format": "mermaid",
  "diagram": "graph TD\n    A[계약서 조항] --> B[법령 검색]..."
}
```

---

### 6.8 상세 분석 결과 조회

```http
GET /api/v1/analysis/contract/{contract_id}/analysis-detail
```

계약서의 전체 상세 분석 결과를 조회합니다.

---

### 6.9 파이프라인 정보 조회

```http
GET /api/v1/analysis/pipeline-info
```

Advanced AI Pipeline의 구성 요소 정보를 조회합니다. 인증 불필요.

---

## 7. Search API

법률 지식 검색 API입니다. Dify Agent의 Custom Tool로 사용됩니다.

### 7.1 법률 지식 검색

```http
POST /api/v1/search/legal
```

법령해석례, 판례, 고용노동부 해설을 검색합니다.

**Request Body**

```json
{
  "query": "포괄임금제가 위법한가요?",
  "contract_context": "제5조 급여는 월 250만원으로 하며, 연장근로수당을 포함한다.",
  "top_k": 5
}
```

**Response (200 OK)**

```json
{
  "documents": [
    {
      "content": "포괄임금제는 근로기준법 제56조에 따라...",
      "source": "대법원 판례 2019다12345",
      "source_type": "precedent",
      "relevance": 0.95
    }
  ],
  "quality": "correct",
  "total_found": 5
}
```

---

### 7.2 검색 서비스 상태 확인

```http
GET /api/v1/search/health
```

**Response (200 OK)**

```json
{
  "status": "healthy",
  "service": "DocScanner Legal Search API",
  "version": "1.0.0"
}
```

---

## 8. Scan API

실시간 계약서 위험 탐지 API입니다. 카메라로 촬영한 이미지를 빠르게 분석합니다.

### 8.1 Quick Scan

```http
POST /api/v1/scan/quick
```

계약서 이미지를 OCR + 키워드 매칭으로 빠르게 분석합니다 (목표: 3초 이내).

**Headers**: `Authorization: Bearer <access_token>` (필수)

**Request**: `multipart/form-data`

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| image | File (image/*) | O | 계약서 이미지 |

**Response (200 OK)**

```json
{
  "risk_level": "HIGH",
  "detected_clauses": [
    {
      "text": "계약 해지 시 위약금 300% 청구",
      "risk_level": "HIGH",
      "keyword": "위약금",
      "bbox": null
    },
    {
      "text": "초과 근무 수당은 별도 협의",
      "risk_level": "HIGH",
      "keyword": "초과 근무",
      "bbox": null
    }
  ],
  "summary": "2건의 주의가 필요한 조항이 발견되었습니다.",
  "scan_time_ms": 1523
}
```

**Risk Level 값**

| 값 | 설명 |
|----|------|
| HIGH | 고위험 조항 발견 |
| MEDIUM | 주의 필요 조항 발견 |
| LOW | 확인 필요 조항 발견 |
| SAFE | 위험 조항 없음 |

---

### 8.2 위험 키워드 목록 조회

```http
GET /api/v1/scan/keywords
```

Quick Scan에서 사용하는 위험 키워드 목록을 조회합니다.

**Headers**: `Authorization: Bearer <access_token>` (필수)

**Response (200 OK)**

```json
{
  "high_risk": ["위약금", "벌금", "손해배상", "최저임금", "무급", "즉시 해고"],
  "medium_risk": ["연차", "휴가", "야간 근무", "4대보험", "퇴직금", "수습"],
  "low_risk": ["업무 범위", "근무 장소", "복리후생"]
}
```

---

## 9. Checklist API

2025년 고용계약 체크리스트 데이터를 제공합니다.

### 9.1 체크리스트 조회

```http
GET /api/v1/checklist/
```

2025년 근로계약 체크리스트를 조회합니다.

**Response (200 OK)**

```json
{
  "title": "2025년 근로계약 체크리스트",
  "categories": [
    {
      "name": "임금",
      "items": [
        {
          "id": 1,
          "question": "시급이 10,030원 이상인가요?",
          "legal_basis": "최저임금법 제6조",
          "risk_level": "HIGH"
        }
      ]
    }
  ]
}
```

---

## 10. Tool API (Internal)

Dify 등 내부 서비스가 호출하는 Tool API입니다. `X-Internal-API-Key` 헤더 인증이 필요합니다.

### 10.1 MUVERA 벡터 검색

```http
GET /api/v1/contracts/v1/search-muvera
```

MUVERA 멀티벡터 임베딩으로 유사 조항을 검색합니다.

**Headers**

```http
X-Internal-API-Key: <INTERNAL_API_KEY>
```

**Query Parameters**

| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| query_text | string | O | 검색할 조항 텍스트 |

**Response (200 OK)**

```json
{
  "context": [
    {
      "source": "근로기준법/law",
      "text": "제17조(근로조건의 명시) 사용자는 근로계약을 체결할 때에..."
    }
  ]
}
```

---

### 10.2 GraphDB 위험 패턴 검색

```http
GET /api/v1/contracts/v1/search-risk-pattern
```

Neo4j 지식 그래프에서 위험 패턴을 검색합니다.

**Headers**

```http
X-Internal-API-Key: <INTERNAL_API_KEY>
```

**Query Parameters**

| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| query_text | string | O | 검색할 조항 텍스트 |

**Response (200 OK)**

```json
{
  "context": [
    {
      "rule_name": "포괄임금제",
      "text": "위험 패턴 '포괄임금제' (임금 조항, 위험도: High): 연장근로수당을 포함하여..."
    }
  ]
}
```

---

## 공통 에러 코드

| 상태 코드 | 설명 |
|----------|------|
| 400 | Bad Request - 잘못된 요청 |
| 401 | Unauthorized - 인증 실패 |
| 403 | Forbidden - 권한 없음 |
| 404 | Not Found - 리소스 없음 |
| 500 | Internal Server Error - 서버 오류 |
| 502 | Bad Gateway - 외부 서비스 호출 실패 (Dify 등) |
| 503 | Service Unavailable - DB 연결 불가 |
| 504 | Gateway Timeout - 외부 서비스 타임아웃 |

**에러 응답 형식**

```json
{
  "detail": "에러 메시지"
}
```

---

## 환경 변수

API 서버 실행에 필요한 환경 변수:

| 변수명 | 필수 | 설명 |
|--------|------|------|
| DATABASE_URL | O | PostgreSQL 연결 URL |
| SECRET_KEY | O | JWT 서명 키 |
| ES_URL | O | Elasticsearch URL |
| NEO4J_URI | O | Neo4j 연결 URI |
| NEO4J_USER | O | Neo4j 사용자명 |
| NEO4J_PASSWORD | O | Neo4j 비밀번호 |
| REDIS_HOST | O | Redis 호스트 |
| REDIS_PORT | X | Redis 포트 (기본: 6379) |
| GEMINI_API_KEY | O | Google Gemini API 키 |
| OPENAI_API_KEY | O | OpenAI API 키 |
| INTERNAL_API_KEY | O | 내부 Tool API 인증키 |
| DIFY_API_KEY | X | Dify API 키 |
| TAVILY_API_KEY | X | Tavily 웹 검색 API 키 |

---

*최종 업데이트: 2025-12*
