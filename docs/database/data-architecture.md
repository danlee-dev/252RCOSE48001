# DocScanner AI Data Architecture

이 문서는 DocScanner AI의 전체 데이터 저장소 구조를 정의합니다.

---

## 개요

```
+------------------+     +---------------------+     +------------------+
|   PostgreSQL     |     |   Elasticsearch     |     |     Neo4j        |
|   (Primary DB)   |     |   (Vector Search)   |     |   (Graph DB)     |
+------------------+     +---------------------+     +------------------+
        |                         |                         |
   사용자/계약서            법률 문서 검색             지식 그래프
   메타데이터               (15,223 청크)              (관계 탐색)
```

---

## 1. PostgreSQL (Primary Database)

사용자 정보와 계약서 메타데이터를 저장합니다.

### ERD

> dbdiagram.io에서 시각화: [erd.dbml](./erd.dbml)

```
+----------------+          +------------------+
|     users      |          |    contracts     |
+----------------+          +------------------+
| id (PK)        |<-------->| id (PK)          |
| email          |    1:N   | user_id (FK)     |
| password       |          | title            |
| username       |          | file_url         |
| hashed_refresh |          | status           |
| created_at     |          | extracted_text   |
+----------------+          | analysis_result  |
                            | risk_level       |
                            | created_at       |
                            +------------------+
```

### 테이블 상세

#### users
| 컬럼 | 타입 | 제약조건 | 설명 |
|------|------|---------|------|
| id | INTEGER | PK, AUTO | 사용자 ID |
| email | VARCHAR | UNIQUE, NOT NULL | 이메일 |
| password | VARCHAR | NOT NULL | 해시된 비밀번호 |
| username | VARCHAR | NOT NULL | 사용자명 |
| hashed_refresh_token | VARCHAR | - | 리프레시 토큰 해시 |
| created_at | TIMESTAMP | DEFAULT NOW() | 생성일시 |

#### contracts
| 컬럼 | 타입 | 제약조건 | 설명 |
|------|------|---------|------|
| id | INTEGER | PK, AUTO | 계약서 ID |
| user_id | INTEGER | FK(users.id), NOT NULL | 소유자 |
| title | VARCHAR | DEFAULT '분석중...' | 파일명 |
| file_url | VARCHAR | NOT NULL | 저장 경로 |
| status | VARCHAR | DEFAULT 'PENDING' | PENDING/ANALYZING/COMPLETED/FAILED |
| extracted_text | TEXT | - | OCR 추출 텍스트 |
| analysis_result | JSONB | - | AI 분석 결과 |
| risk_level | VARCHAR | - | HIGH/MEDIUM/LOW |
| created_at | TIMESTAMP | DEFAULT NOW() | 생성일시 |

#### analysis_result JSONB 구조

```json
{
  "risk_level": "HIGH",
  "summary": "분석 요약",
  "stress_test": {
    "total_underpayment": 150000,
    "annual_underpayment": 1800000,
    "violations": [
      {
        "clause_number": "제5조",
        "type": "최저임금 미달",
        "severity": "HIGH",
        "description": "설명",
        "legal_basis": "근로기준법 제6조",
        "suggestion": "수정 제안"
      }
    ]
  },
  "judge_score": {
    "overall": 0.85,
    "accuracy": 0.9,
    "legal_basis": 0.85
  },
  "reasoning_trace": {
    "nodes": [],
    "edges": []
  }
}
```

---

## 2. Elasticsearch (Vector Search)

법률 문서의 벡터 검색을 위한 인덱스입니다.

### 인덱스: `docscanner_chunks`

**총 15,223개 문서** (2025년 11월 기준)

### 매핑 (Mapping)

```json
{
  "mappings": {
    "properties": {
      "text": {
        "type": "text",
        "analyzer": "korean"
      },
      "source": {
        "type": "keyword"
      },
      "doc_type": {
        "type": "keyword"
      },
      "title": {
        "type": "text"
      },
      "keywords": {
        "type": "keyword"
      },
      "embedding": {
        "type": "dense_vector",
        "dims": 1024,
        "index": true,
        "similarity": "cosine"
      }
    }
  }
}
```

### 필드 설명

| 필드 | 타입 | 설명 |
|------|------|------|
| text | text | 문서 본문 (최대 1,000자) |
| source | keyword | 출처 (법령명, 판례번호 등) |
| doc_type | keyword | 문서 유형 |
| title | text | 제목 |
| keywords | keyword[] | 추출된 키워드 |
| embedding | dense_vector | MUVERA FDE 벡터 (1024차원) |

### doc_type 값

| 값 | 설명 | 문서 수 |
|----|------|--------|
| precedent | 판례 | 10,576 |
| labor_ministry | 고용노동부 해설 | 3,384 |
| interpretation | 법령해석례 | 589 |
| manual | 업무 매뉴얼 | 296 |
| employment_rules | 표준취업규칙 | 367 |
| guide | 가이드 | 4 |
| leaflet | 리플릿 | 7 |

### 임베딩: MUVERA (Multi-Vector Retrieval)

```
문장 분할 -> 개별 임베딩 (KURE-v1) -> FDE 압축 -> 1024차원 단일 벡터
```

- 모델: `nlpai-lab/KURE-v1` (한국어 법률 특화)
- FDE 설정: repetitions=1, simhash_projections=3, partitions=8

---

## 3. Neo4j (Knowledge Graph)

법률 문서 간의 관계를 그래프로 구조화합니다.

### 노드 (Nodes)

```
+---------------+     +---------------+     +---------------+
|   Document    |     |   Category    |     |  ClauseType   |
| (15,223)      |     | (6)           |     | (6)           |
+---------------+     +---------------+     +---------------+
        |                                           |
        v                                           v
+---------------+     +---------------+     +---------------+
|   Precedent   |     | Interpretation|     |  RiskPattern  |
| (10,576)      |     | (3,973)       |     | (4)           |
+---------------+     +---------------+     +---------------+
        |
        v
+---------------+
|     Law       |
| (추출된 법령) |
+---------------+
```

### 노드 상세

#### Document (기본 노드)
| 속성 | 타입 | 설명 |
|------|------|------|
| id | STRING | 문서 ID |
| content | STRING | 문서 본문 |
| source | STRING | 출처 |
| doc_type | STRING | 문서 유형 |

#### Precedent (판례)
Document를 상속하며 판례 고유 속성 추가

#### Interpretation (법령해석례 + 고용노동부 해설)
Document를 상속

#### ClauseType (조항 유형)
| 속성 | 타입 | 설명 |
|------|------|------|
| name | STRING | 조항명 |
| required | BOOLEAN | 필수 여부 |
| related_law | STRING | 관련 법령 |

**ClauseType 값**:
- 임금
- 근로시간
- 휴일_휴가
- 계약기간
- 해고_퇴직
- 손해배상

#### RiskPattern (위험 패턴)
| 속성 | 타입 | 설명 |
|------|------|------|
| name | STRING | 패턴명 |
| explanation | STRING | 설명 |
| riskLevel | STRING | HIGH/MEDIUM |
| triggers | STRING[] | 트리거 키워드 |

**RiskPattern 값**:
| 패턴 | 위험도 | 트리거 키워드 |
|------|--------|--------------|
| 포괄임금제 | HIGH | "포괄하여", "제수당 포함" |
| 과도한_위약금 | HIGH | "배상하여야", "위약금" |
| 최저임금_미달 | HIGH | "최저임금", "수습기간 90%" |
| 부당_해고_조항 | MEDIUM | "즉시 해고", "갑의 판단" |

### 관계 (Relationships)

```
(:Document)-[:CATEGORIZED_AS]->(:Category)
(:Document)-[:SOURCE_IS]->(:Source)
(:RiskPattern)-[:IS_A_TYPE_OF]->(:ClauseType)
(:RiskPattern)-[:HAS_CASE]->(:Precedent)
(:RiskPattern)-[:HAS_INTERPRETATION]->(:Interpretation)
(:Precedent)-[:CITES]->(:Law)
(:Interpretation)-[:CITES]->(:Law)
```

| 관계 | 시작 노드 | 끝 노드 | 설명 |
|------|----------|--------|------|
| CATEGORIZED_AS | Document | Category | 카테고리 분류 |
| SOURCE_IS | Document | Source | 출처 연결 |
| IS_A_TYPE_OF | RiskPattern | ClauseType | 위험패턴-조항유형 |
| HAS_CASE | RiskPattern | Precedent | 관련 판례 |
| HAS_INTERPRETATION | RiskPattern | Interpretation | 관련 해석례 |
| CITES | Precedent/Interpretation | Law | 법령 인용 |

### 예시 Cypher 쿼리

```cypher
// 위험 패턴에서 관련 판례 찾기
MATCH (r:RiskPattern {name: '포괄임금제'})-[:HAS_CASE]->(p:Precedent)
RETURN p.content, p.source
LIMIT 5

// 특정 법령을 인용하는 모든 문서 찾기
MATCH (d)-[:CITES]->(l:Law {name: '근로기준법 제56조'})
RETURN d.content, labels(d)
```

---

## 4. Redis (Cache & Queue)

세션 캐시와 Celery 작업 큐에 사용됩니다.

### 용도

| 키 패턴 | 용도 | TTL |
|---------|------|-----|
| `celery:*` | Celery 작업 브로커 | - |
| `session:*` | 사용자 세션 | 24h |
| `cache:*` | API 응답 캐시 | 5m |

---

## 데이터 흐름

```
[계약서 업로드]
      |
      v
+------------------+
|   PostgreSQL     |  <- 메타데이터 저장
|   (contracts)    |
+------------------+
      |
      v
+------------------+
|   Celery/Redis   |  <- 분석 작업 큐
+------------------+
      |
      v
+------------------+     +------------------+
|  Elasticsearch   |<--->|     Neo4j        |
| (법령/판례 검색)  |     | (관계 확장 검색)  |
+------------------+     +------------------+
      |
      v
+------------------+
|   PostgreSQL     |  <- analysis_result 저장
|   (contracts)    |
+------------------+
```

---

*최종 업데이트: 2025-12*
