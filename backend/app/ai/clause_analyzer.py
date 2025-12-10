"""
LLM-based Clause Analyzer with CRAG Integration
- LLM 기반 계약서 조항 분할
- 조항별 CRAG 검색으로 법률 컨텍스트 확보
- LLM이 법률 컨텍스트 기반으로 위반 여부 판단

Flow:
1. LLM이 계약서를 조항 단위로 분할 + 값 추출
2. 각 조항에 대해 CRAG 검색 (Graph + Vector DB)
3. LLM이 CRAG 결과를 컨텍스트로 위반 분석
4. 결과 통합
"""

import os
import json
import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from app.core.config import settings


class ClauseType(Enum):
    """조항 유형"""
    WORK_START_DATE = "근로개시일"
    WORKPLACE = "근무장소"
    JOB_DESCRIPTION = "업무내용"
    WORK_HOURS = "근로시간"
    BREAK_TIME = "휴게시간"
    WORK_DAYS = "근무일"
    HOLIDAYS = "휴일"
    SALARY = "임금"
    BONUS = "상여금"
    ALLOWANCES = "수당"
    PAYMENT_DATE = "임금지급일"
    ANNUAL_LEAVE = "연차휴가"
    SOCIAL_INSURANCE = "사회보험"
    CONTRACT_DELIVERY = "계약서교부"
    PENALTY = "위약금"
    TERMINATION = "해지"
    OTHER = "기타"


class ViolationSeverity(Enum):
    """위반 심각도"""
    CRITICAL = "CRITICAL"  # 즉시 시정 필요
    HIGH = "HIGH"          # 심각한 위반
    MEDIUM = "MEDIUM"      # 주의 필요
    LOW = "LOW"            # 경미한 문제
    INFO = "INFO"          # 정보 제공


@dataclass
class ExtractedClause:
    """추출된 조항"""
    clause_number: str              # 조항 번호 (예: "4", "6-1")
    clause_type: ClauseType         # 조항 유형
    title: str                      # 조항 제목
    original_text: str              # 원문 텍스트
    extracted_values: Dict[str, Any] = field(default_factory=dict)  # 추출된 값들
    position: Dict[str, int] = field(default_factory=dict)  # 시작/끝 위치


@dataclass
class ClauseViolation:
    """조항 위반 정보"""
    clause: ExtractedClause
    violation_type: str             # 위반 유형
    severity: ViolationSeverity
    description: str                # 위반 설명
    legal_basis: str                # 법적 근거
    current_value: Any              # 현재 계약서 값
    legal_standard: Any             # 법적 기준 값
    suggestion: str                 # 수정 제안
    crag_sources: List[str] = field(default_factory=list)  # 참조한 법률 출처
    confidence: float = 0.0         # 판단 신뢰도

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.violation_type,
            "severity": self.severity.value,
            "description": self.description,
            "legal_basis": self.legal_basis,
            "current_value": self.current_value,
            "legal_standard": self.legal_standard,
            "suggestion": self.suggestion,
            "clause_number": self.clause.clause_number,
            "clause_title": self.clause.title,
            "original_text": self.clause.original_text[:200],
            "sources": self.crag_sources,
            "confidence": self.confidence
        }


@dataclass
class ClauseAnalysisResult:
    """조항 분석 결과"""
    clauses: List[ExtractedClause] = field(default_factory=list)
    violations: List[ClauseViolation] = field(default_factory=list)
    total_underpayment: int = 0
    annual_underpayment: int = 0
    processing_time: float = 0.0

    @property
    def violation_count(self) -> int:
        return len(self.violations)

    @property
    def high_severity_count(self) -> int:
        return sum(1 for v in self.violations
                   if v.severity in [ViolationSeverity.CRITICAL, ViolationSeverity.HIGH])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "violations": [v.to_dict() for v in self.violations],
            "total_underpayment": self.total_underpayment,
            "annual_underpayment": self.annual_underpayment,
            "clause_count": len(self.clauses),
            "violation_count": self.violation_count,
            "high_severity_count": self.high_severity_count,
            "processing_time": self.processing_time
        }


class LLMClauseAnalyzer:
    """
    LLM 기반 조항 분석기

    사용법:
        analyzer = LLMClauseAnalyzer(crag=crag_instance)
        result = analyzer.analyze(contract_text)
    """

    # 조항 분할 및 값 추출 프롬프트
    CLAUSE_EXTRACTION_PROMPT = """당신은 한국 근로계약서 분석 전문가입니다.
다음 근로계약서를 분석하여 각 조항을 구조화된 형식으로 추출하세요.

[근로계약서]
{contract_text}

[지시사항]
1. 각 조항을 개별적으로 식별
2. 조항별 핵심 값 추출 (숫자, 날짜, 시간 등)
3. 조항 유형 분류

[응답 형식 - JSON]
{{
    "clauses": [
        {{
            "clause_number": "1",
            "clause_type": "근로개시일/근무장소/업무내용/근로시간/휴게시간/근무일/휴일/임금/상여금/수당/임금지급일/연차휴가/사회보험/계약서교부/위약금/해지/기타",
            "title": "조항 제목",
            "original_text": "원문 텍스트",
            "extracted_values": {{
                "key": "value"
            }}
        }}
    ],
    "contract_metadata": {{
        "employer": "사업주명",
        "employee": "근로자명",
        "contract_date": "계약일",
        "contract_type": "정규직/계약직/기간제/일용직"
    }}
}}

[추출 예시]
- 근로시간: {{"start_time": "09:00", "end_time": "18:00", "daily_hours": 8}}
- 휴게시간: {{"break_minutes": 60, "break_start": "12:00", "break_end": "13:00"}}
- 임금: {{"base_salary": 3200000, "salary_type": "월급"}}
- 수당: {{"meal_allowance": 200000, "position_allowance": 300000}}

모든 숫자는 정수로 추출하세요."""

    # 조항별 위반 분석 프롬프트 (법적 기준은 CRAG 검색 결과에서 가져옴)
    VIOLATION_ANALYSIS_PROMPT = """당신은 한국 노동법 전문 변호사입니다.
다음 계약서 조항을 분석하고 법적 위반 여부를 판단하세요.

[분석 대상 조항]
조항 번호: {clause_number}
조항 유형: {clause_type}
원문: {original_text}
추출된 값: {extracted_values}

[관련 법령 - 검색 결과]
{law_context}

[관련 판례 - 검색 결과]
{precedent_context}

[관련 해석례/지침 - 검색 결과]
{interpretation_context}

[위험 패턴 - Graph DB 검색 결과]
{pattern_context}

[분석 지시]
1. 위 법령, 판례, 해석례를 바탕으로 위반 여부 판단
2. 반드시 구체적 법적 근거 (조항 번호)와 함께 설명
3. 관련 판례가 있으면 판례 요지 인용
4. 수정 제안 제시

주요 분석 포인트:
- 근로기준법상 강행규정 위반 여부
- 위약금 예정 금지 (제20조) 위반 여부
- 임금 전액 지급 원칙 (제43조) 위반 여부
- 근로계약서 교부 의무 (제17조) 위반 여부
- 근로기준법에 미달하는 조건의 무효 (제15조)

[응답 형식 - JSON]
{{
    "has_violation": true/false,
    "violations": [
        {{
            "violation_type": "위반 유형 (예: 최저임금_미달, 연장근로_초과, 휴게시간_미부여, 위약금_예정, 전액지급_위반)",
            "severity": "CRITICAL/HIGH/MEDIUM/LOW/INFO",
            "description": "구체적 위반 내용 설명",
            "legal_basis": "법조문만 기재 (예: 근로기준법 제50조)",
            "current_value": "현재 계약서 값",
            "legal_standard": "법적 기준 값 (검색된 법령에서 추출)",
            "suggestion": "구체적인 계약서 수정 방법을 문장으로 작성. 절대 법조문만 적지 마세요!",
            "confidence": 0.0-1.0
        }}
    ],
    "underpayment": {{
        "monthly": 0,
        "annual": 0,
        "calculation": "계산 과정 설명"
    }}
}}

[중요]
- legal_basis와 suggestion은 반드시 다른 내용이어야 합니다!
- legal_basis: 법률명과 조항 번호만 (예: "근로기준법 제20조")
- suggestion: 계약서를 어떻게 수정해야 하는지 구체적 방법 (예: "위약금 조항을 삭제하고, 실제 발생한 손해에 한해 배상 청구 가능하도록 수정")

위반이 없으면 violations를 빈 배열로 반환하세요."""

    # 종합 분석 프롬프트 (Cross-clause analysis)
    HOLISTIC_ANALYSIS_PROMPT = """당신은 한국 노동법 전문 변호사입니다.
전체 근로계약서를 종합 분석하여 조항간 연관 분석이 필요한 위반 사항을 찾으세요.

[계약서 요약]
{contract_summary}

[전체 조항 정보]
{all_clauses}

[관련 법령]
{law_context}

[분석 포인트]
1. 최저임금 위반 여부 (월급여 / 총 근로시간으로 시급 계산)
   - 기본급 + 고정수당 / (월 소정근로시간 + 연장/야간/휴일 근로시간)
   - 2025년 최저시급 기준 적용

2. 주 52시간 초과 여부 (모든 근로시간 합산)

3. 휴일 부족 여부 (주휴일 + 공휴일)

4. 연장근로 가산수당 미지급 여부

5. 포괄임금제 적법성 여부

[응답 형식 - JSON]
{{
    "holistic_violations": [
        {{
            "violation_type": "위반 유형",
            "severity": "CRITICAL/HIGH/MEDIUM/LOW",
            "description": "구체적 설명 (계산 과정 포함)",
            "legal_basis": "법조문만 기재 (예: 근로기준법 제50조)",
            "current_value": "현재 값 (계산 결과)",
            "legal_standard": "법적 기준",
            "suggestion": "구체적인 계약서 수정 방법을 문장으로 작성 (예: '기본급을 3,500,000원 이상으로 인상하고 연장근로수당을 별도 지급하도록 수정'). 절대 법조문만 적지 마세요.",
            "related_clauses": ["관련 조항 번호들"],
            "calculation": "상세 계산 과정"
        }}
    ],
    "minimum_wage_analysis": {{
        "hourly_wage": 0,
        "legal_minimum": 10030,
        "is_violation": true/false,
        "monthly_underpayment": 0,
        "calculation_detail": "계산 과정"
    }}
}}

[중요 주의사항]
- legal_basis: 법률명과 조항 번호만 (예: "근로기준법 제50조")
- suggestion: 계약서를 어떻게 수정해야 하는지 구체적인 방법을 문장으로 작성. 법조문을 복사하면 안 됨!
  예시: "소정근로시간을 주 40시간으로 단축하고, 연장근로 시 가산임금(통상임금의 50% 추가)을 별도 지급하도록 계약서 수정"
"""

    def __init__(
        self,
        crag: Optional[Any] = None,
        llm_client: Optional[Any] = None,
        model: str = None,
        enable_crag: bool = True
    ):
        """
        Args:
            crag: GraphGuidedCRAG 인스턴스
            llm_client: OpenAI 클라이언트
            model: 사용할 LLM 모델 (기본값: settings.LLM_REASONING_MODEL)
            enable_crag: CRAG 검색 활성화 여부
        """
        self.crag = crag
        # 모델 기본값: settings에서 가져옴
        self.model = model if model else settings.LLM_REASONING_MODEL  # gpt-5-mini
        self.enable_crag = enable_crag

        if llm_client is None:
            try:
                from openai import OpenAI
                self.llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                self.llm_client = None
        else:
            self.llm_client = llm_client

        # Elasticsearch client (for Vector DB search)
        self.es_client = None
        try:
            from elasticsearch import Elasticsearch
            es_host = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
            self.es_client = Elasticsearch([es_host])
        except Exception:
            pass

        # Neo4j driver (for Graph DB search) - CRAG에서 가져옴
        self.neo4j_driver = None
        if self.crag and hasattr(self.crag, 'neo4j_driver'):
            self.neo4j_driver = self.crag.neo4j_driver

    def _is_reasoning_model(self) -> bool:
        """reasoning 모델 여부 확인 (temperature 미지원)"""
        reasoning_keywords = ["o1", "o3", "gpt-5"]
        return any(kw in self.model.lower() for kw in reasoning_keywords)

    def _search_by_category(
        self,
        query: str,
        doc_type: str,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        카테고리별 문서 검색 (Vector DB)

        Args:
            query: 검색 쿼리
            doc_type: 문서 타입 (law, precedent, interpretation, manual)
            limit: 검색 결과 수

        Returns:
            검색 결과 리스트
        """
        results = []

        if self.es_client is None:
            return results

        try:
            # Elasticsearch 쿼리 (doc_type 필터링)
            search_body = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["text^2", "title", "keywords"],
                                    "type": "best_fields"
                                }
                            }
                        ],
                        "filter": [
                            {
                                "term": {
                                    "doc_type": doc_type
                                }
                            }
                        ]
                    }
                },
                "size": limit,
                "_source": ["text", "source", "doc_type", "title"]
            }

            response = self.es_client.search(
                index="docscanner_chunks",
                body=search_body
            )

            for hit in response.get("hits", {}).get("hits", []):
                source = hit.get("_source", {})
                results.append({
                    "id": hit.get("_id", ""),
                    "text": source.get("text", ""),
                    "source": source.get("source", ""),
                    "doc_type": source.get("doc_type", doc_type),
                    "score": hit.get("_score", 0)
                })

        except Exception as e:
            print(f"Category search error ({doc_type}): {e}")

        return results

    def _get_legal_context_by_categories(
        self,
        query: str
    ) -> Tuple[str, str, str, List[str]]:
        """
        카테고리별로 법률 컨텍스트 검색

        Args:
            query: 검색 쿼리

        Returns:
            Tuple[법령 컨텍스트, 판례 컨텍스트, 해석례 컨텍스트, 출처 목록]
        """
        all_sources = []

        # 1. 법령 검색 (law)
        law_docs = self._search_by_category(query, "law", limit=3)
        if not law_docs and self.crag:
            # CRAG fallback for law
            try:
                crag_result = self.crag.retrieve_and_correct_sync(
                    f"법령 {query}", [], max_graph_hops=1
                )
                law_docs = [
                    {"text": d.text, "source": d.source}
                    for d in crag_result.all_docs[:3]
                    if "법" in d.source or "law" in d.source.lower()
                ]
            except Exception:
                pass

        law_context = self._format_context(law_docs, "법령")
        all_sources.extend([d.get("source", "") for d in law_docs])

        # 2. 판례 검색 (precedent)
        precedent_docs = self._search_by_category(query, "precedent", limit=2)
        if not precedent_docs:
            # source 이름으로 필터링 fallback
            precedent_docs = self._search_by_source_pattern(query, ["판례", "대법원", "precedent"], limit=2)

        precedent_context = self._format_context(precedent_docs, "판례")
        all_sources.extend([d.get("source", "") for d in precedent_docs])

        # 3. 해석례/지침 검색 (interpretation)
        interpretation_docs = self._search_by_category(query, "interpretation", limit=2)
        if not interpretation_docs:
            interpretation_docs = self._search_by_source_pattern(query, ["해석", "지침", "고시", "interpretation"], limit=2)

        interpretation_context = self._format_context(interpretation_docs, "해석례")
        all_sources.extend([d.get("source", "") for d in interpretation_docs])

        return law_context, precedent_context, interpretation_context, all_sources

    def _search_by_source_pattern(
        self,
        query: str,
        patterns: List[str],
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """source 필드 패턴으로 검색 (fallback)"""
        results = []

        if self.es_client is None:
            return results

        try:
            should_clauses = [
                {"wildcard": {"source": f"*{pattern}*"}}
                for pattern in patterns
            ]

            search_body = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["text^2", "title"],
                                    "type": "best_fields"
                                }
                            }
                        ],
                        "should": should_clauses,
                        "minimum_should_match": 1
                    }
                },
                "size": limit,
                "_source": ["text", "source", "doc_type"]
            }

            response = self.es_client.search(
                index="docscanner_chunks",
                body=search_body
            )

            for hit in response.get("hits", {}).get("hits", []):
                source = hit.get("_source", {})
                results.append({
                    "text": source.get("text", ""),
                    "source": source.get("source", ""),
                    "score": hit.get("_score", 0)
                })

        except Exception as e:
            print(f"Source pattern search error: {e}")

        return results

    def _format_context(
        self,
        docs: List[Dict[str, Any]],
        category: str
    ) -> str:
        """검색 결과를 컨텍스트 문자열로 포맷팅"""
        if not docs:
            return f"관련 {category} 검색 결과 없음"

        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.get("source", "N/A")
            text = doc.get("text", "")[:800]  # 길이 제한
            parts.append(f"[{category} {i}] 출처: {source}\n{text}")

        return "\n\n".join(parts)

    # ========== Graph DB 검색 메서드 (현재 스키마: Document, RiskPattern, Category) ==========

    def _search_graph_risk_patterns(
        self,
        clause_type: str,
        clause_text: str
    ) -> List[Dict[str, Any]]:
        """
        Graph DB에서 위험 패턴 검색

        현재 스키마:
        - RiskPattern (name, explanation, riskLevel, triggers)
        - RiskPattern --[RELATES_TO]--> Document
        - RiskPattern --[IS_A_TYPE_OF]--> ClauseType

        Args:
            clause_type: 조항 유형 (예: "휴게시간", "임금")
            clause_text: 조항 원문

        Returns:
            위험 패턴 및 관련 문서 리스트
        """
        results = []

        if self.neo4j_driver is None:
            return results

        # 위험 키워드 매핑 (triggers 필드와 매칭)
        risk_triggers = {
            "휴게시간": ["휴게", "휴식", "점심"],
            "근로시간": ["근로시간", "연장", "야간", "휴일근로", "52시간"],
            "임금": ["임금", "급여", "최저임금", "시급", "월급", "포괄"],
            "수당": ["수당", "포괄하여", "포함하여", "제수당"],
            "위약금": ["위약금", "손해배상", "벌금", "벌칙", "배상하여야", "반환"],
            "사회보험": ["4대보험", "국민연금", "건강보험", "고용보험", "산재"],
            "연차휴가": ["연차", "휴가", "유급휴일"],
            "해지": ["해고", "해지", "계약해지", "퇴직"],
        }

        triggers = risk_triggers.get(clause_type, [clause_type])

        # 조항 텍스트에서 추가 트리거 추출
        text_triggers = []
        trigger_keywords = ["포괄하여", "포함하여", "위약금", "손해배상", "최저임금", "수습"]
        for kw in trigger_keywords:
            if kw in clause_text:
                text_triggers.append(kw)

        all_triggers = list(set(triggers + text_triggers))

        try:
            with self.neo4j_driver.session() as session:
                # 1. RiskPattern 직접 검색 (triggers 매칭)
                # triggers는 배열 또는 문자열일 수 있음
                query = """
                MATCH (r:RiskPattern)
                WHERE any(trigger IN $triggers WHERE
                    (r.triggers IS NOT NULL AND (
                        any(t IN r.triggers WHERE t CONTAINS trigger)
                        OR toString(r.triggers) CONTAINS trigger
                    ))
                    OR r.name CONTAINS trigger
                )
                OPTIONAL MATCH (r)-[:RELATES_TO]->(d:Document)
                RETURN r.name AS pattern_name,
                       r.explanation AS explanation,
                       r.riskLevel AS risk_level,
                       collect(d.content)[0..1] AS related_docs,
                       collect(d.source)[0..1] AS doc_sources
                LIMIT 3
                """

                result = session.run(query, triggers=all_triggers)
                for record in result:
                    related_doc_text = ""
                    if record['related_docs']:
                        for doc in record['related_docs']:
                            if doc:
                                related_doc_text += doc[:300] + "\n"

                    results.append({
                        "type": "risk_pattern",
                        "text": f"[위험 패턴: {record['pattern_name']}]\n"
                               f"위험도: {record['risk_level']}\n"
                               f"설명: {record['explanation']}\n"
                               f"관련 문서: {related_doc_text[:400] if related_doc_text else '없음'}",
                        "source": f"RiskPattern/{record['pattern_name']}",
                        "score": 0.95
                    })

        except Exception as e:
            print(f"Graph risk pattern search error: {e}")

        return results

    def _search_graph_documents_by_category(
        self,
        clause_type: str,
        keywords: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Graph DB에서 카테고리별 문서 검색

        현재 스키마:
        - Document (id, source, content, category, type)
        - Document --[CATEGORIZED_AS]--> Category
        - Category (name): 근로시간, 임금, 휴일휴가 등

        Args:
            clause_type: 조항 유형
            keywords: 검색 키워드 리스트

        Returns:
            관련 법령/문서 리스트
        """
        results = []

        if self.neo4j_driver is None:
            return results

        # 조항 유형 → 카테고리 매핑
        category_map = {
            "휴게시간": ["근로시간"],
            "근로시간": ["근로시간"],
            "임금": ["임금"],
            "수당": ["임금"],
            "위약금": ["기타", "인사"],
            "연차휴가": ["휴일휴가"],
            "휴일": ["휴일휴가"],
            "사회보험": ["복리후생"],
            "해지": ["인사"],
            "계약서교부": ["채용절차"],
        }

        categories = category_map.get(clause_type, ["기타"])

        try:
            with self.neo4j_driver.session() as session:
                # 카테고리 기반 문서 검색
                query = """
                MATCH (d:Document)-[:CATEGORIZED_AS]->(c:Category)
                WHERE c.name IN $categories
                  AND d.content IS NOT NULL
                  AND (
                    any(kw IN $keywords WHERE d.content CONTAINS kw)
                    OR any(kw IN $keywords WHERE d.source CONTAINS kw)
                  )
                RETURN d.content AS content,
                       d.source AS source,
                       c.name AS category
                LIMIT 3
                """

                result = session.run(query, categories=categories, keywords=keywords)
                for record in result:
                    content = record['content'] or ""
                    results.append({
                        "type": "document",
                        "text": f"[{record['category']}] 출처: {record['source']}\n{content[:500]}",
                        "source": record['source'],
                        "score": 0.85
                    })

        except Exception as e:
            print(f"Graph document search error: {e}")

        return results

    def _search_graph_by_source(
        self,
        clause_type: str,
        keywords: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Graph DB에서 소스별 문서 검색 (법령해석례, 매뉴얼 등)

        Args:
            clause_type: 조항 유형
            keywords: 검색 키워드

        Returns:
            관련 문서 리스트
        """
        results = []

        if self.neo4j_driver is None:
            return results

        try:
            with self.neo4j_driver.session() as session:
                # 키워드 기반 문서 검색
                query = """
                MATCH (d:Document)
                WHERE d.content IS NOT NULL
                  AND any(kw IN $keywords WHERE d.content CONTAINS kw)
                RETURN d.content AS content,
                       d.source AS source,
                       d.category AS category
                ORDER BY size(d.content) DESC
                LIMIT 2
                """

                result = session.run(query, keywords=keywords)
                for record in result:
                    content = record['content'] or ""
                    results.append({
                        "type": "reference",
                        "text": f"[참조문서] 출처: {record['source']}\n{content[:500]}",
                        "source": record['source'],
                        "score": 0.8
                    })

        except Exception as e:
            print(f"Graph source search error: {e}")

        return results

    # ========== 하이브리드 검색 (Vector + Graph) ==========

    def _get_hybrid_legal_context(
        self,
        clause_type: str,
        clause_text: str
    ) -> Tuple[str, str, str, str, List[str]]:
        """
        하이브리드 검색: Vector DB + Graph DB 결합

        현재 스키마:
        - Vector DB (ES): doc_type 기반 필터링 (law, precedent, interpretation)
        - Graph DB (Neo4j): RiskPattern, Document, Category 노드

        Args:
            clause_type: 조항 유형
            clause_text: 조항 원문

        Returns:
            Tuple[법령, 판례, 해석례, 위험패턴, 출처목록]
        """
        all_sources = []
        query = f"{clause_type} {clause_text[:200]}"

        # 키워드 추출
        keywords = [clause_type]
        keyword_map = {
            "시간": ["시간", "근로시간", "휴게"],
            "임금": ["임금", "급여", "월급", "시급"],
            "휴게": ["휴게", "휴식", "점심"],
            "연장": ["연장", "야간", "휴일근로"],
            "위약": ["위약금", "손해배상", "벌금"],
            "포괄": ["포괄", "포함하여", "제수당"],
            "보험": ["4대보험", "사회보험", "국민연금"],
        }
        for key, vals in keyword_map.items():
            if key in clause_text:
                keywords.extend(vals)
        keywords = list(set(keywords))

        # ========== 1. Vector DB 검색 (의미적 유사도) ==========
        law_docs = self._search_by_category(query, "law", limit=2)
        precedent_docs = self._search_by_category(query, "precedent", limit=2)
        interpretation_docs = self._search_by_category(query, "interpretation", limit=2)

        # ========== 2. Graph DB 검색 (구조적 관계) ==========

        # 2-1. 위험 패턴 검색 (RiskPattern 노드)
        graph_risk_patterns = self._search_graph_risk_patterns(clause_type, clause_text)

        # 2-2. 카테고리별 문서 검색 (Document → Category)
        graph_docs = self._search_graph_documents_by_category(clause_type, keywords)

        # 2-3. 키워드 기반 참조 문서 검색
        graph_refs = self._search_graph_by_source(clause_type, keywords)

        # ========== 3. 결과 병합 및 랭킹 ==========

        # 법령: Vector DB + Graph DB 문서 병합
        all_laws = law_docs + graph_docs
        seen_law_sources = set()
        unique_laws = []
        for doc in all_laws:
            source = doc.get("source", "")
            if source and source not in seen_law_sources:
                seen_law_sources.add(source)
                unique_laws.append(doc)
        unique_laws = sorted(unique_laws, key=lambda x: x.get("score", 0), reverse=True)[:3]

        # 판례/참조: Vector DB + Graph 참조문서
        all_precedents = precedent_docs + graph_refs
        seen_precedent = set()
        unique_precedents = []
        for doc in all_precedents:
            source = doc.get("source", "")
            if source and source not in seen_precedent:
                seen_precedent.add(source)
                unique_precedents.append(doc)
        unique_precedents = sorted(unique_precedents, key=lambda x: x.get("score", 0), reverse=True)[:2]

        # 해석례: Vector DB 결과
        unique_interpretations = interpretation_docs[:2]

        # 위험 패턴: Graph DB RiskPattern 노드
        unique_patterns = graph_risk_patterns[:2]

        # ========== 4. 컨텍스트 생성 ==========
        law_context = self._format_context(unique_laws, "법령")
        precedent_context = self._format_context(unique_precedents, "판례/참조")
        interpretation_context = self._format_context(unique_interpretations, "해석례")
        pattern_context = self._format_context(unique_patterns, "위험패턴") if unique_patterns else ""

        # 출처 수집
        for docs in [unique_laws, unique_precedents, unique_interpretations, unique_patterns]:
            all_sources.extend([d.get("source", "") for d in docs if d.get("source")])

        return law_context, precedent_context, interpretation_context, pattern_context, all_sources

    def analyze(self, contract_text: str) -> ClauseAnalysisResult:
        """
        계약서 전체 분석

        Args:
            contract_text: 계약서 텍스트

        Returns:
            ClauseAnalysisResult: 분석 결과
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        start_time = time.time()

        result = ClauseAnalysisResult()

        # 1. LLM으로 조항 분할 및 값 추출
        clauses = self._extract_clauses(contract_text)
        result.clauses = clauses

        # 2. 각 조항별 분석 (병렬 처리)
        total_monthly_underpayment = 0

        # 분석 우선순위가 높은 조항 유형
        priority_types = [
            ClauseType.WORK_HOURS,
            ClauseType.BREAK_TIME,
            ClauseType.SALARY,
            ClauseType.ALLOWANCES,
            ClauseType.ANNUAL_LEAVE,
            ClauseType.SOCIAL_INSURANCE,
            ClauseType.PENALTY,
            ClauseType.TERMINATION,
            ClauseType.CONTRACT_DELIVERY,
            ClauseType.HOLIDAYS,
            ClauseType.WORK_DAYS
        ]

        # 분석 대상 조항 필터링
        clauses_to_analyze = [
            clause for clause in clauses
            if clause.clause_type in priority_types or
               self._contains_risk_keywords(clause.original_text)
        ]

        # 병렬 처리로 개별 조항 분석
        if clauses_to_analyze:
            with ThreadPoolExecutor(max_workers=min(len(clauses_to_analyze), 5)) as executor:
                future_to_clause = {
                    executor.submit(self._analyze_clause, clause): clause
                    for clause in clauses_to_analyze
                }

                for future in as_completed(future_to_clause):
                    try:
                        violations, underpayment = future.result()
                        result.violations.extend(violations)
                        total_monthly_underpayment += underpayment
                    except Exception as e:
                        print(f"Clause analysis error: {e}")

        # 3. 종합 분석 (Cross-clause analysis) - 순차 처리 필수
        # 최저임금 계산, 주간 근무시간 검증 등 모든 조항 정보 필요
        holistic_violations, holistic_underpayment = self._holistic_analysis(
            clauses, contract_text
        )
        result.violations.extend(holistic_violations)
        total_monthly_underpayment += holistic_underpayment

        # 4. 체불액 계산
        result.total_underpayment = total_monthly_underpayment
        result.annual_underpayment = total_monthly_underpayment * 12

        result.processing_time = time.time() - start_time

        return result

    def _extract_clauses(self, contract_text: str) -> List[ExtractedClause]:
        """LLM으로 조항 추출"""
        if self.llm_client is None:
            return self._fallback_extract_clauses(contract_text)

        try:
            prompt = self.CLAUSE_EXTRACTION_PROMPT.format(
                contract_text=contract_text[:8000]
            )

            if self._is_reasoning_model():
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
            else:
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "당신은 한국 근로계약서 분석 전문가입니다. JSON 형식으로 응답하세요."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )

            result = json.loads(response.choices[0].message.content)
            clauses = []

            for c in result.get("clauses", []):
                clause_type = self._map_clause_type(c.get("clause_type", "기타"))
                clauses.append(ExtractedClause(
                    clause_number=str(c.get("clause_number", "")),
                    clause_type=clause_type,
                    title=c.get("title", ""),
                    original_text=c.get("original_text", ""),
                    extracted_values=c.get("extracted_values", {})
                ))

            return clauses

        except Exception as e:
            print(f"Clause extraction error: {e}")
            return self._fallback_extract_clauses(contract_text)

    def _map_clause_type(self, type_str: str) -> ClauseType:
        """문자열을 ClauseType으로 변환"""
        type_map = {
            "근로개시일": ClauseType.WORK_START_DATE,
            "근무장소": ClauseType.WORKPLACE,
            "업무내용": ClauseType.JOB_DESCRIPTION,
            "근로시간": ClauseType.WORK_HOURS,
            "휴게시간": ClauseType.BREAK_TIME,
            "근무일": ClauseType.WORK_DAYS,
            "휴일": ClauseType.HOLIDAYS,
            "임금": ClauseType.SALARY,
            "상여금": ClauseType.BONUS,
            "수당": ClauseType.ALLOWANCES,
            "임금지급일": ClauseType.PAYMENT_DATE,
            "연차휴가": ClauseType.ANNUAL_LEAVE,
            "사회보험": ClauseType.SOCIAL_INSURANCE,
            "계약서교부": ClauseType.CONTRACT_DELIVERY,
            "위약금": ClauseType.PENALTY,
            "해지": ClauseType.TERMINATION,
        }
        return type_map.get(type_str, ClauseType.OTHER)

    def _fallback_extract_clauses(self, contract_text: str) -> List[ExtractedClause]:
        """폴백: 정규식 기반 조항 추출"""
        clauses = []

        # 번호 패턴으로 분할
        patterns = [
            r'(\d{1,2})\s*\.\s*([^\n:]+)\s*[:：]\s*([^\n]+(?:\n(?!\d{1,2}\s*\.)[^\n]+)*)',
            r'제\s*(\d+)\s*조\s*\(([^)]+)\)\s*([^\n]+(?:\n(?!제\s*\d+\s*조)[^\n]+)*)',
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, contract_text, re.MULTILINE)
            for match in matches:
                clause_num = match.group(1)
                title = match.group(2).strip()
                content = match.group(3).strip() if len(match.groups()) > 2 else ""

                clause_type = self._infer_clause_type(title)

                clauses.append(ExtractedClause(
                    clause_number=clause_num,
                    clause_type=clause_type,
                    title=title,
                    original_text=f"{title}: {content}",
                    extracted_values=self._extract_values_regex(content, clause_type)
                ))

            if clauses:
                break

        return clauses

    def _infer_clause_type(self, title: str) -> ClauseType:
        """제목에서 조항 유형 추론"""
        title_lower = title.lower()

        if "근로시간" in title or "소정근로" in title:
            return ClauseType.WORK_HOURS
        elif "휴게" in title:
            return ClauseType.BREAK_TIME
        elif "임금" in title or "급여" in title:
            return ClauseType.SALARY
        elif "수당" in title:
            return ClauseType.ALLOWANCES
        elif "연차" in title or "휴가" in title:
            return ClauseType.ANNUAL_LEAVE
        elif "보험" in title or "4대" in title:
            return ClauseType.SOCIAL_INSURANCE
        elif "위약" in title or "손해배상" in title:
            return ClauseType.PENALTY
        elif "해지" in title or "해고" in title:
            return ClauseType.TERMINATION
        elif "근무일" in title or "휴일" in title:
            return ClauseType.WORK_DAYS
        elif "개시" in title or "시작" in title:
            return ClauseType.WORK_START_DATE
        elif "장소" in title:
            return ClauseType.WORKPLACE
        elif "업무" in title:
            return ClauseType.JOB_DESCRIPTION
        else:
            return ClauseType.OTHER

    def _extract_values_regex(self, content: str, clause_type: ClauseType) -> Dict[str, Any]:
        """정규식으로 값 추출 (폴백)"""
        values = {}

        if clause_type == ClauseType.WORK_HOURS:
            # 시간 추출
            time_match = re.search(
                r'(\d{1,2})\s*시\s*(\d{2})?\s*분?\s*[~\-]\s*(\d{1,2})\s*시\s*(\d{2})?\s*분?',
                content
            )
            if time_match:
                start_h = int(time_match.group(1))
                end_h = int(time_match.group(3))
                values["start_time"] = f"{start_h:02d}:00"
                values["end_time"] = f"{end_h:02d}:00"
                values["daily_hours"] = end_h - start_h

        elif clause_type == ClauseType.BREAK_TIME:
            # 휴게시간 추출
            if "없음" in content or "0분" in content:
                values["break_minutes"] = 0
            else:
                break_match = re.search(r'(\d+)\s*(분|시간)', content)
                if break_match:
                    minutes = int(break_match.group(1))
                    if "시간" in break_match.group(2):
                        minutes *= 60
                    values["break_minutes"] = minutes

        elif clause_type == ClauseType.SALARY:
            # 급여 추출
            salary_match = re.search(r'(\d{1,3}(?:,\d{3})*)\s*원', content)
            if salary_match:
                values["base_salary"] = int(salary_match.group(1).replace(",", ""))

        return values

    def _analyze_clause(
        self,
        clause: ExtractedClause
    ) -> Tuple[List[ClauseViolation], int]:
        """
        개별 조항 분석

        Returns:
            Tuple[List[ClauseViolation], int]: (위반 목록, 월간 체불액)
        """
        violations = []
        monthly_underpayment = 0

        # 1. 하이브리드 검색: Vector DB + Graph DB 결합
        law_context, precedent_context, interpretation_context, pattern_context, crag_sources = \
            self._get_hybrid_legal_context(clause.clause_type.value, clause.original_text)

        # 2. LLM으로 위반 분석
        if self.llm_client is not None:
            try:
                prompt = self.VIOLATION_ANALYSIS_PROMPT.format(
                    clause_number=clause.clause_number,
                    clause_type=clause.clause_type.value,
                    original_text=clause.original_text,
                    extracted_values=json.dumps(clause.extracted_values, ensure_ascii=False),
                    law_context=law_context,
                    precedent_context=precedent_context,
                    interpretation_context=interpretation_context,
                    pattern_context=pattern_context if pattern_context else "관련 위험 패턴 없음"
                )

                if self._is_reasoning_model():
                    response = self.llm_client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"}
                    )
                else:
                    response = self.llm_client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": "당신은 한국 노동법 전문 변호사입니다. 엄격하게 법적 기준을 적용하여 분석하세요."
                            },
                            {"role": "user", "content": prompt}
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.1
                    )

                result = json.loads(response.choices[0].message.content)

                if result.get("has_violation", False):
                    for v in result.get("violations", []):
                        severity = ViolationSeverity[v.get("severity", "MEDIUM")]
                        violations.append(ClauseViolation(
                            clause=clause,
                            violation_type=v.get("violation_type", ""),
                            severity=severity,
                            description=v.get("description", ""),
                            legal_basis=v.get("legal_basis", ""),
                            current_value=v.get("current_value"),
                            legal_standard=v.get("legal_standard"),
                            suggestion=v.get("suggestion", ""),
                            crag_sources=crag_sources,
                            confidence=v.get("confidence", 0.8)
                        ))

                # 체불액 추출
                underpayment = result.get("underpayment", {})
                raw_monthly = underpayment.get("monthly") or 0
                monthly_underpayment = int(raw_monthly) if raw_monthly else 0

            except Exception as e:
                print(f"Violation analysis error: {e}")
                # 폴백: 규칙 기반 분석
                fallback_violations = self._rule_based_analysis(clause)
                violations.extend(fallback_violations)

        else:
            # LLM 없으면 규칙 기반
            violations = self._rule_based_analysis(clause)

        return violations, monthly_underpayment

    def _contains_risk_keywords(self, text: str) -> bool:
        """위험 키워드 포함 여부 확인"""
        risk_keywords = [
            "위약금", "손해배상", "벌금", "벌칙", "감봉",
            "미가입", "제외", "적용하지", "포기",
            "야간", "연장", "휴일근로", "초과근무",
            "지급하지", "공제", "차감",
            "해고", "해지", "계약해지",
            "퇴직금", "미지급"
        ]
        return any(kw in text for kw in risk_keywords)

    def _holistic_analysis(
        self,
        clauses: List[ExtractedClause],
        contract_text: str
    ) -> Tuple[List[ClauseViolation], int]:
        """
        종합 분석 (Cross-clause analysis)
        - 최저임금 계산 (임금 / 근로시간)
        - 주 52시간 초과 여부
        - 포괄임금제 적법성

        Returns:
            Tuple[List[ClauseViolation], int]: (위반 목록, 월간 체불액)
        """
        violations = []
        monthly_underpayment = 0

        if self.llm_client is None:
            return violations, monthly_underpayment

        # 1. 조항 정보 요약 생성
        clause_summaries = []
        for c in clauses:
            clause_summaries.append({
                "number": c.clause_number,
                "type": c.clause_type.value,
                "title": c.title,
                "values": c.extracted_values
            })

        all_clauses_json = json.dumps(clause_summaries, ensure_ascii=False, indent=2)

        # 2. 계약서 핵심 정보 요약
        contract_summary = self._extract_contract_summary(clauses)

        # 3. 법령 검색 (최저임금, 근로시간 관련)
        law_docs = self._search_by_category("최저임금 근로시간 주 52시간 포괄임금", "law", limit=3)
        law_context = self._format_context(law_docs, "법령")

        # 4. LLM 종합 분석
        try:
            prompt = self.HOLISTIC_ANALYSIS_PROMPT.format(
                contract_summary=contract_summary,
                all_clauses=all_clauses_json,
                law_context=law_context
            )

            if self._is_reasoning_model():
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
            else:
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "당신은 한국 노동법 전문 변호사입니다. 계약서 전체를 종합 분석하세요."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )

            result = json.loads(response.choices[0].message.content)

            # 종합 위반 사항 처리
            for v in result.get("holistic_violations", []):
                severity = ViolationSeverity[v.get("severity", "MEDIUM")]

                # 종합 분석용 더미 조항 생성
                dummy_clause = ExtractedClause(
                    clause_number="종합",
                    clause_type=ClauseType.OTHER,
                    title="종합 분석",
                    original_text=v.get("description", ""),
                    extracted_values={}
                )

                violations.append(ClauseViolation(
                    clause=dummy_clause,
                    violation_type=v.get("violation_type", ""),
                    severity=severity,
                    description=v.get("description", ""),
                    legal_basis=v.get("legal_basis", ""),
                    current_value=v.get("current_value"),
                    legal_standard=v.get("legal_standard"),
                    suggestion=v.get("suggestion", ""),
                    crag_sources=[d.get("source", "") for d in law_docs],
                    confidence=0.85
                ))

            # 최저임금 분석 결과에서 체불액 추출
            min_wage = result.get("minimum_wage_analysis", {})
            if min_wage.get("is_violation", False):
                raw_underpayment = min_wage.get("monthly_underpayment") or 0
                monthly_underpayment = int(raw_underpayment) if raw_underpayment else 0

        except Exception as e:
            print(f"Holistic analysis error: {e}")

        return violations, monthly_underpayment

    def _extract_contract_summary(self, clauses: List[ExtractedClause]) -> str:
        """조항들에서 핵심 정보 요약 추출"""
        summary_parts = []

        for c in clauses:
            if c.clause_type == ClauseType.SALARY:
                vals = c.extracted_values
                if vals:
                    summary_parts.append(f"임금: {vals}")

            elif c.clause_type == ClauseType.WORK_HOURS:
                vals = c.extracted_values
                if vals:
                    summary_parts.append(f"근로시간: {vals}")

            elif c.clause_type == ClauseType.BREAK_TIME:
                vals = c.extracted_values
                if vals:
                    summary_parts.append(f"휴게시간: {vals}")

            elif c.clause_type == ClauseType.ALLOWANCES:
                vals = c.extracted_values
                if vals:
                    summary_parts.append(f"수당: {vals}")

            elif c.clause_type == ClauseType.WORK_DAYS:
                vals = c.extracted_values
                if vals:
                    summary_parts.append(f"근무일: {vals}")

        if not summary_parts:
            return "조항별 핵심 정보 추출 실패"

        return "\n".join(summary_parts)

    def _rule_based_analysis(self, clause: ExtractedClause) -> List[ClauseViolation]:
        """규칙 기반 폴백 분석"""
        violations = []
        values = clause.extracted_values

        if clause.clause_type == ClauseType.WORK_HOURS:
            daily_hours = values.get("daily_hours", 0)
            if daily_hours > 8:
                excess = daily_hours - 8
                violations.append(ClauseViolation(
                    clause=clause,
                    violation_type="과도한_근로시간",
                    severity=ViolationSeverity.HIGH,
                    description=f"1일 {daily_hours}시간 근무는 법정 8시간을 {excess}시간 초과",
                    legal_basis="근로기준법 제50조",
                    current_value=f"{daily_hours}시간",
                    legal_standard="8시간",
                    suggestion=f"1일 근로시간을 8시간으로 조정 (연장근로는 별도 합의 필요)",
                    confidence=0.9
                ))

        elif clause.clause_type == ClauseType.BREAK_TIME:
            break_minutes = values.get("break_minutes", -1)
            daily_hours = 8  # 기본값

            if break_minutes == 0:
                violations.append(ClauseViolation(
                    clause=clause,
                    violation_type="휴게시간_미부여",
                    severity=ViolationSeverity.HIGH,
                    description="8시간 이상 근무 시 1시간 이상 휴게시간 부여 필수",
                    legal_basis="근로기준법 제54조",
                    current_value="휴게 없음",
                    legal_standard="1시간 이상",
                    suggestion="휴게시간 1시간 이상 명시 (예: 12:00~13:00)",
                    confidence=0.95
                ))

        elif clause.clause_type == ClauseType.PENALTY:
            if "위약금" in clause.original_text or "손해배상" in clause.original_text:
                violations.append(ClauseViolation(
                    clause=clause,
                    violation_type="위약금_예정_금지",
                    severity=ViolationSeverity.HIGH,
                    description="근로계약 불이행에 대한 위약금 예정은 금지됨",
                    legal_basis="근로기준법 제20조",
                    current_value=clause.original_text[:100],
                    legal_standard="위약금 예정 금지",
                    suggestion="해당 조항 삭제 (실손해 배상만 가능)",
                    confidence=0.85
                ))

        elif clause.clause_type == ClauseType.SOCIAL_INSURANCE:
            text = clause.original_text
            if "가입하지" in text or "미가입" in text or "제외" in text:
                violations.append(ClauseViolation(
                    clause=clause,
                    violation_type="사회보험_미가입",
                    severity=ViolationSeverity.HIGH,
                    description="4대 사회보험은 입사일부터 가입 의무 (수습 여부 무관)",
                    legal_basis="고용보험법, 국민연금법, 국민건강보험법, 산업재해보상보험법",
                    current_value="4대보험 미가입",
                    legal_standard="입사일부터 가입",
                    suggestion="입사일부터 4대 사회보험 가입 명시",
                    confidence=0.9
                ))

        return violations


# 편의 함수
def analyze_contract_clauses(
    contract_text: str,
    crag: Optional[Any] = None
) -> ClauseAnalysisResult:
    """계약서 조항 분석"""
    analyzer = LLMClauseAnalyzer(crag=crag)
    return analyzer.analyze(contract_text)