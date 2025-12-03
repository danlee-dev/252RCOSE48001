"""
Reasoning Trace (추론 과정 시각화) - Production Grade
- AI의 분석 추론 경로를 지식 그래프로 시각화
- 상세 메타데이터 및 신뢰도 전파 시스템
- 멀티레벨 XAI (Explainable AI) 구현
- 법적 추론 체인 및 근거 연결

Reference: Chain-of-Thought, Explainable AI, Legal Reasoning
"""

import os
import json
import hashlib
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import math


class NodeType(Enum):
    """노드 유형 (확장)"""
    CONTRACT_CLAUSE = "contract_clause"     # 계약서 조항
    LEGAL_REFERENCE = "legal_reference"     # 법령 참조
    PRECEDENT = "precedent"                 # 판례
    RISK_PATTERN = "risk_pattern"           # 위험 패턴
    REASONING_STEP = "reasoning_step"       # 추론 단계
    CONCLUSION = "conclusion"               # 결론
    EVIDENCE = "evidence"                   # 근거
    QUESTION = "question"                   # 질문/쿼리
    CONTEXT = "context"                     # 컨텍스트
    CALCULATION = "calculation"             # 계산 결과
    COMPARISON = "comparison"               # 비교 분석
    RECOMMENDATION = "recommendation"       # 권고사항


class EdgeType(Enum):
    """엣지 유형 (확장)"""
    SIMILAR_TO = "similar_to"               # 유사함
    CITES = "cites"                         # 인용
    LEADS_TO = "leads_to"                   # ~로 이어짐
    SUPPORTS = "supports"                   # 근거
    CONTRADICTS = "contradicts"             # 반박
    APPLIES = "applies"                     # 적용
    DERIVES_FROM = "derives_from"           # ~에서 도출
    COMPARES_TO = "compares_to"             # 비교
    VALIDATES = "validates"                 # 검증
    REQUIRES = "requires"                   # 필요조건
    IMPLIES = "implies"                     # 함의
    RECOMMENDS = "recommends"               # 권고


class ConfidenceLevel(Enum):
    """신뢰도 수준"""
    VERY_HIGH = "very_high"     # 0.9+
    HIGH = "high"               # 0.75-0.9
    MEDIUM = "medium"           # 0.5-0.75
    LOW = "low"                 # 0.25-0.5
    VERY_LOW = "very_low"       # 0-0.25


class ReasoningDepth(Enum):
    """추론 깊이"""
    SHALLOW = "shallow"         # 단순 매칭
    MODERATE = "moderate"       # 규칙 적용
    DEEP = "deep"               # 다단계 추론
    COMPLEX = "complex"         # 복합 추론


@dataclass
class NodeMetadata:
    """노드 메타데이터"""
    source: str = ""                        # 정보 출처
    timestamp: datetime = field(default_factory=datetime.now)
    author: str = ""                        # 작성자 (AI/사용자)
    version: int = 1                        # 버전
    legal_basis: List[str] = field(default_factory=list)  # 법적 근거
    keywords: List[str] = field(default_factory=list)      # 키워드
    category: str = ""                      # 카테고리
    importance: float = 0.5                 # 중요도 (0-1)
    uncertainty: float = 0.0                # 불확실성 (0-1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "author": self.author,
            "version": self.version,
            "legal_basis": self.legal_basis,
            "keywords": self.keywords,
            "category": self.category,
            "importance": self.importance,
            "uncertainty": self.uncertainty,
        }


@dataclass
class TraceNode:
    """추론 추적 노드 (확장)"""
    id: str
    node_type: NodeType
    label: str
    content: str
    metadata: NodeMetadata = field(default_factory=NodeMetadata)
    confidence: float = 1.0                 # 신뢰도 (0-1)
    confidence_level: ConfidenceLevel = ConfidenceLevel.HIGH
    position: Dict[str, float] = field(default_factory=dict)
    depth: int = 0                          # 그래프에서의 깊이
    is_critical: bool = False               # 핵심 노드 여부
    children_ids: List[str] = field(default_factory=list)
    parent_ids: List[str] = field(default_factory=list)
    explanation: str = ""                   # 상세 설명
    alternatives: List[str] = field(default_factory=list)  # 대안 해석

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.node_type.value,
            "label": self.label,
            "content": self.content,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level.value,
            "metadata": self.metadata.to_dict(),
            "position": self.position,
            "depth": self.depth,
            "is_critical": self.is_critical,
            "explanation": self.explanation,
        }


@dataclass
class TraceEdge:
    """추론 추적 엣지 (확장)"""
    id: str
    source: str
    target: str
    edge_type: EdgeType
    label: str = ""
    weight: float = 1.0                     # 연결 강도
    confidence: float = 1.0                 # 신뢰도
    bidirectional: bool = False             # 양방향 여부
    reasoning: str = ""                     # 연결 이유
    evidence: List[str] = field(default_factory=list)  # 근거

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "type": self.edge_type.value,
            "label": self.label,
            "weight": self.weight,
            "confidence": self.confidence,
            "bidirectional": self.bidirectional,
            "reasoning": self.reasoning,
        }


@dataclass
class ReasoningPath:
    """추론 경로"""
    nodes: List[str]                        # 노드 ID 시퀀스
    edges: List[str]                        # 엣지 ID 시퀀스
    total_confidence: float = 1.0           # 경로 전체 신뢰도
    depth: ReasoningDepth = ReasoningDepth.MODERATE
    description: str = ""                   # 경로 설명

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "total_confidence": self.total_confidence,
            "depth": self.depth.value,
            "description": self.description,
        }


@dataclass
class ReasoningTrace:
    """추론 추적 그래프 (확장)"""
    id: str = ""
    nodes: List[TraceNode] = field(default_factory=list)
    edges: List[TraceEdge] = field(default_factory=list)
    reasoning_paths: List[ReasoningPath] = field(default_factory=list)
    summary: str = ""
    conclusion: str = ""
    overall_confidence: float = 1.0
    reasoning_depth: ReasoningDepth = ReasoningDepth.MODERATE
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5(
                f"{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12]

    def add_node(self, node: TraceNode):
        self.nodes.append(node)

    def add_edge(self, edge: TraceEdge):
        self.edges.append(edge)

    def get_node(self, node_id: str) -> Optional[TraceNode]:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_node_by_type(self, node_type: NodeType) -> List[TraceNode]:
        return [n for n in self.nodes if n.node_type == node_type]

    def get_edges_from(self, node_id: str) -> List[TraceEdge]:
        return [e for e in self.edges if e.source == node_id]

    def get_edges_to(self, node_id: str) -> List[TraceEdge]:
        return [e for e in self.edges if e.target == node_id]

    def get_critical_path(self) -> List[TraceNode]:
        """핵심 노드 경로 반환"""
        return [n for n in self.nodes if n.is_critical]

    def calculate_path_confidence(self, path: List[str]) -> float:
        """경로 신뢰도 계산"""
        if len(path) < 2:
            return 1.0

        confidence = 1.0
        for i in range(len(path) - 1):
            edges = [e for e in self.edges
                     if e.source == path[i] and e.target == path[i+1]]
            if edges:
                confidence *= edges[0].confidence
            else:
                confidence *= 0.5  # 연결 없으면 패널티

        # 노드 신뢰도도 반영
        for node_id in path:
            node = self.get_node(node_id)
            if node:
                confidence *= node.confidence

        return confidence ** (1 / len(path))  # 기하평균

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (JSON 직렬화용)"""
        return {
            "id": self.id,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "reasoning_paths": [p.to_dict() for p in self.reasoning_paths],
            "summary": self.summary,
            "conclusion": self.conclusion,
            "overall_confidence": self.overall_confidence,
            "reasoning_depth": self.reasoning_depth.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """JSON 문자열로 변환"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def to_mermaid(self, detailed: bool = False) -> str:
        """Mermaid 다이어그램으로 변환"""
        lines = ["graph TD"]

        # 노드 스타일 정의
        style_map = {
            NodeType.CONTRACT_CLAUSE: "fill:#ffcccc,stroke:#cc0000",
            NodeType.LEGAL_REFERENCE: "fill:#ccffcc,stroke:#00cc00",
            NodeType.PRECEDENT: "fill:#ccccff,stroke:#0000cc",
            NodeType.RISK_PATTERN: "fill:#ffcc99,stroke:#cc6600",
            NodeType.REASONING_STEP: "fill:#ffffcc,stroke:#cccc00",
            NodeType.CONCLUSION: "fill:#99ccff,stroke:#0066cc",
            NodeType.EVIDENCE: "fill:#e6ffe6,stroke:#009900",
            NodeType.CALCULATION: "fill:#ffe6cc,stroke:#cc6600",
            NodeType.RECOMMENDATION: "fill:#e6ccff,stroke:#6600cc",
        }

        # 서브그래프로 깊이별 그룹화
        depth_groups: Dict[int, List[TraceNode]] = {}
        for node in self.nodes:
            if node.depth not in depth_groups:
                depth_groups[node.depth] = []
            depth_groups[node.depth].append(node)

        # 노드 추가
        for node in self.nodes:
            label = node.label.replace('"', "'")[:40]
            if detailed:
                conf = f" ({node.confidence:.0%})"
                label = f"{label}{conf}"

            # 핵심 노드 강조
            if node.is_critical:
                lines.append(f'    {node.id}[["*{label}*"]]')
            else:
                lines.append(f'    {node.id}["{label}"]')

        # 엣지 추가
        edge_arrows = {
            EdgeType.LEADS_TO: "-->",
            EdgeType.SUPPORTS: "-->|근거|",
            EdgeType.CONTRADICTS: "-.->|반박|",
            EdgeType.SIMILAR_TO: "-->|유사|",
            EdgeType.CITES: "-->|인용|",
            EdgeType.DERIVES_FROM: "-->|도출|",
            EdgeType.APPLIES: "-->|적용|",
            EdgeType.VALIDATES: "-->|검증|",
            EdgeType.RECOMMENDS: "-->|권고|",
        }

        for edge in self.edges:
            arrow = edge_arrows.get(edge.edge_type, "-->")
            if edge.label and "|" not in arrow:
                arrow = f"-->|{edge.label}|"
            lines.append(f'    {edge.source} {arrow} {edge.target}')

        # 스타일 적용
        for node in self.nodes:
            style = style_map.get(node.node_type, "fill:#ffffff,stroke:#000000")
            lines.append(f'    style {node.id} {style}')

        return "\n".join(lines)

    def to_graphviz(self) -> str:
        """Graphviz DOT 형식으로 변환"""
        lines = [
            "digraph ReasoningTrace {",
            '    rankdir=TB;',
            '    node [shape=box, style="rounded,filled"];',
            '    edge [fontsize=10];',
        ]

        # 노드 색상
        color_map = {
            NodeType.CONTRACT_CLAUSE: "#ffcccc",
            NodeType.LEGAL_REFERENCE: "#ccffcc",
            NodeType.PRECEDENT: "#ccccff",
            NodeType.RISK_PATTERN: "#ffcc99",
            NodeType.REASONING_STEP: "#ffffcc",
            NodeType.CONCLUSION: "#99ccff",
        }

        # 노드 추가
        for node in self.nodes:
            color = color_map.get(node.node_type, "#ffffff")
            label = node.label[:30].replace('"', '\\"')
            lines.append(f'    {node.id} [label="{label}", fillcolor="{color}"];')

        # 엣지 추가
        for edge in self.edges:
            style = "dashed" if edge.edge_type == EdgeType.CONTRADICTS else "solid"
            label = edge.label[:15] if edge.label else ""
            lines.append(
                f'    {edge.source} -> {edge.target} '
                f'[label="{label}", style="{style}"];'
            )

        lines.append("}")
        return "\n".join(lines)


class ConfidencePropagator:
    """신뢰도 전파 시스템"""

    @staticmethod
    def propagate_confidence(
        trace: ReasoningTrace,
        decay_factor: float = 0.9
    ) -> ReasoningTrace:
        """
        노드 간 신뢰도 전파

        Args:
            trace: 추론 추적
            decay_factor: 전파 시 감쇠 비율

        Returns:
            업데이트된 추론 추적
        """
        # 토폴로지 정렬 (DAG 가정)
        in_degree = {n.id: 0 for n in trace.nodes}
        for edge in trace.edges:
            in_degree[edge.target] = in_degree.get(edge.target, 0) + 1

        # 소스 노드부터 시작
        queue = [n for n in trace.nodes if in_degree[n.id] == 0]
        visited = set()

        while queue:
            node = queue.pop(0)
            if node.id in visited:
                continue
            visited.add(node.id)

            # 부모 노드들의 신뢰도 수집
            incoming_edges = trace.get_edges_to(node.id)
            if incoming_edges:
                parent_confidences = []
                for edge in incoming_edges:
                    parent = trace.get_node(edge.source)
                    if parent:
                        # 엣지 가중치와 부모 신뢰도 결합
                        propagated = parent.confidence * edge.confidence * decay_factor
                        parent_confidences.append(propagated)

                # 여러 부모가 있으면 최대값 사용 (OR 의미론)
                if parent_confidences:
                    node.confidence = min(node.confidence, max(parent_confidences))

            # 신뢰도 수준 업데이트
            node.confidence_level = ConfidencePropagator._get_confidence_level(
                node.confidence
            )

            # 자식 노드 큐에 추가
            outgoing = trace.get_edges_from(node.id)
            for edge in outgoing:
                child = trace.get_node(edge.target)
                if child and child.id not in visited:
                    queue.append(child)

        # 전체 신뢰도 업데이트
        if trace.nodes:
            conclusion_nodes = trace.get_node_by_type(NodeType.CONCLUSION)
            if conclusion_nodes:
                trace.overall_confidence = sum(
                    n.confidence for n in conclusion_nodes
                ) / len(conclusion_nodes)
            else:
                trace.overall_confidence = sum(
                    n.confidence for n in trace.nodes
                ) / len(trace.nodes)

        return trace

    @staticmethod
    def _get_confidence_level(confidence: float) -> ConfidenceLevel:
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.75:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.25:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


class ReasoningTracer:
    """
    추론 과정 추적기 (Production Grade)

    사용법:
        tracer = ReasoningTracer()
        trace = tracer.trace_analysis(contract_clause, analysis_result, context_docs)
        print(trace.to_mermaid())
    """

    def __init__(
        self,
        neo4j_driver: Optional[Any] = None,
        include_confidence: bool = True,
        propagate_confidence: bool = True,
        max_depth: int = 10
    ):
        """
        Args:
            neo4j_driver: Neo4j 드라이버 (그래프 확장용)
            include_confidence: 신뢰도 포함 여부
            propagate_confidence: 신뢰도 전파 수행 여부
            max_depth: 최대 추론 깊이
        """
        self.neo4j_driver = neo4j_driver
        self.include_confidence = include_confidence
        self.propagate_confidence = propagate_confidence
        self.max_depth = max_depth
        self._node_counter = 0
        self._edge_counter = 0

    def trace_analysis(
        self,
        contract_clause: str,
        analysis_result: Dict[str, Any],
        context_docs: List[Dict[str, Any]] = None,
        query: str = ""
    ) -> ReasoningTrace:
        """
        분석 과정 추적

        Args:
            contract_clause: 분석된 계약서 조항
            analysis_result: AI 분석 결과
            context_docs: 참조된 컨텍스트 문서들
            query: 원본 질문

        Returns:
            ReasoningTrace: 추론 추적 그래프
        """
        trace = ReasoningTrace()
        self._node_counter = 0
        self._edge_counter = 0

        # 1. 시작점: 질문 노드
        if query:
            query_node = self._create_node(
                NodeType.QUESTION,
                "사용자 질문",
                query,
                depth=0,
                is_critical=True
            )
            trace.add_node(query_node)

        # 2. 계약서 조항 노드
        clause_node = self._create_node(
            NodeType.CONTRACT_CLAUSE,
            "분석 대상 조항",
            contract_clause[:500],
            depth=1,
            is_critical=True,
            metadata=NodeMetadata(
                source="contract",
                category="clause",
                importance=1.0
            )
        )
        trace.add_node(clause_node)

        # 질문 -> 조항 연결
        if query:
            trace.add_edge(self._create_edge(
                query_node.id,
                clause_node.id,
                EdgeType.LEADS_TO,
                "분석 대상"
            ))

        # 3. 참조 문서들
        if context_docs:
            for i, doc in enumerate(context_docs[:7]):
                doc_type = doc.get("type", "reference")
                node_type = self._get_node_type(doc_type)
                score = doc.get("score", 0.8)

                doc_node = self._create_node(
                    node_type,
                    f"참조 {i+1}: {doc.get('source', 'unknown')[:20]}",
                    doc.get("text", "")[:400],
                    confidence=score,
                    depth=2,
                    metadata=NodeMetadata(
                        source=doc.get("source", ""),
                        legal_basis=doc.get("legal_references", []),
                        category=doc_type
                    )
                )
                trace.add_node(doc_node)

                # 조항 -> 참조 엣지
                trace.add_edge(self._create_edge(
                    clause_node.id,
                    doc_node.id,
                    EdgeType.SIMILAR_TO,
                    f"유사도: {score:.0%}",
                    confidence=score
                ))

        # 4. 위험 패턴
        risk_patterns = analysis_result.get("risk_patterns", [])
        for i, pattern in enumerate(risk_patterns):
            pattern_node = self._create_node(
                NodeType.RISK_PATTERN,
                f"위험: {pattern.get('name', f'패턴 {i+1}')}",
                pattern.get("explanation", ""),
                depth=2,
                is_critical=True,
                metadata=NodeMetadata(
                    category=pattern.get("severity", "Medium"),
                    legal_basis=[pattern.get("legal_basis", "")],
                    importance=0.9 if pattern.get("severity") == "High" else 0.6
                )
            )
            trace.add_node(pattern_node)

            # 조항 -> 위험 패턴
            trace.add_edge(self._create_edge(
                clause_node.id,
                pattern_node.id,
                EdgeType.APPLIES,
                "위험 탐지"
            ))

        # 5. 추론 단계들
        reasoning_steps = analysis_result.get("reasoning_steps", [])
        prev_step_id = clause_node.id

        for i, step in enumerate(reasoning_steps):
            step_confidence = step.get("confidence", 0.9)

            step_node = self._create_node(
                NodeType.REASONING_STEP,
                f"추론 {i+1}",
                step.get("description", ""),
                confidence=step_confidence,
                depth=3 + i,
                is_critical=step.get("is_critical", False),
                explanation=step.get("explanation", ""),
                metadata=NodeMetadata(
                    source="ai_reasoning",
                    keywords=step.get("keywords", [])
                )
            )
            trace.add_node(step_node)

            # 연결
            trace.add_edge(self._create_edge(
                prev_step_id,
                step_node.id,
                EdgeType.LEADS_TO,
                confidence=step_confidence
            ))

            # 근거 연결
            for evidence_id in step.get("evidence_refs", []):
                evidence_node = trace.get_node(evidence_id)
                if evidence_node:
                    trace.add_edge(self._create_edge(
                        evidence_id,
                        step_node.id,
                        EdgeType.SUPPORTS,
                        "근거 제공"
                    ))

            prev_step_id = step_node.id

        # 6. 계산 결과 (있는 경우)
        calculations = analysis_result.get("calculations", {})
        for calc_name, calc_data in list(calculations.items())[:5]:
            calc_node = self._create_node(
                NodeType.CALCULATION,
                f"계산: {calc_name}",
                json.dumps(calc_data, ensure_ascii=False)[:300],
                depth=3,
                metadata=NodeMetadata(
                    category="calculation",
                    source="symbolic_computation"
                )
            )
            trace.add_node(calc_node)

            trace.add_edge(self._create_edge(
                clause_node.id,
                calc_node.id,
                EdgeType.DERIVES_FROM,
                "수치 분석"
            ))

        # 7. 결론
        conclusion = analysis_result.get("conclusion", "")
        if conclusion:
            conclusion_node = self._create_node(
                NodeType.CONCLUSION,
                "분석 결론",
                conclusion,
                depth=self.max_depth - 1,
                is_critical=True,
                metadata=NodeMetadata(
                    source="ai_analysis",
                    importance=1.0
                )
            )
            trace.add_node(conclusion_node)

            # 마지막 추론 단계 -> 결론
            trace.add_edge(self._create_edge(
                prev_step_id,
                conclusion_node.id,
                EdgeType.LEADS_TO,
                "결론 도출"
            ))

            # 법령/판례 -> 결론 근거 연결
            for node in trace.nodes:
                if node.node_type in [NodeType.LEGAL_REFERENCE, NodeType.PRECEDENT]:
                    trace.add_edge(self._create_edge(
                        node.id,
                        conclusion_node.id,
                        EdgeType.SUPPORTS,
                        confidence=node.confidence
                    ))

        # 8. 권고사항
        recommendations = analysis_result.get("recommendations", [])
        conclusion_id = conclusion_node.id if conclusion else prev_step_id

        for i, rec in enumerate(recommendations[:3]):
            rec_node = self._create_node(
                NodeType.RECOMMENDATION,
                f"권고 {i+1}",
                rec.get("text", str(rec)),
                depth=self.max_depth,
                metadata=NodeMetadata(
                    category="recommendation",
                    importance=rec.get("priority", 0.5) if isinstance(rec, dict) else 0.5
                )
            )
            trace.add_node(rec_node)

            trace.add_edge(self._create_edge(
                conclusion_id,
                rec_node.id,
                EdgeType.RECOMMENDS,
                "시정 권고"
            ))

        # 9. 신뢰도 전파
        if self.propagate_confidence:
            trace = ConfidencePropagator.propagate_confidence(trace)

        # 10. 요약 및 메타데이터 설정
        trace.summary = self._generate_trace_summary(trace)
        trace.conclusion = conclusion
        trace.reasoning_depth = self._determine_reasoning_depth(trace)

        # 11. 주요 추론 경로 식별
        trace.reasoning_paths = self._find_reasoning_paths(trace)

        # 12. 레이아웃 계산
        self._calculate_layout(trace)

        return trace

    def _create_node(
        self,
        node_type: NodeType,
        label: str,
        content: str,
        confidence: float = 1.0,
        depth: int = 0,
        is_critical: bool = False,
        metadata: NodeMetadata = None,
        explanation: str = ""
    ) -> TraceNode:
        """노드 생성"""
        self._node_counter += 1
        node_id = f"node_{self._node_counter}"

        if metadata is None:
            metadata = NodeMetadata()

        return TraceNode(
            id=node_id,
            node_type=node_type,
            label=label,
            content=content,
            confidence=confidence,
            confidence_level=ConfidencePropagator._get_confidence_level(confidence),
            depth=depth,
            is_critical=is_critical,
            metadata=metadata,
            explanation=explanation
        )

    def _create_edge(
        self,
        source: str,
        target: str,
        edge_type: EdgeType,
        label: str = "",
        confidence: float = 1.0
    ) -> TraceEdge:
        """엣지 생성"""
        self._edge_counter += 1
        edge_id = f"edge_{self._edge_counter}"

        return TraceEdge(
            id=edge_id,
            source=source,
            target=target,
            edge_type=edge_type,
            label=label,
            confidence=confidence
        )

    def _get_node_type(self, doc_type: str) -> NodeType:
        """문서 유형에서 노드 유형 결정"""
        type_map = {
            "law": NodeType.LEGAL_REFERENCE,
            "precedent": NodeType.PRECEDENT,
            "interpretation": NodeType.LEGAL_REFERENCE,
            "risk_pattern": NodeType.RISK_PATTERN,
            "evidence": NodeType.EVIDENCE,
            "context": NodeType.CONTEXT,
        }
        return type_map.get(doc_type, NodeType.LEGAL_REFERENCE)

    def _generate_trace_summary(self, trace: ReasoningTrace) -> str:
        """추적 요약 생성"""
        node_counts = {}
        for node in trace.nodes:
            node_type = node.node_type.value
            node_counts[node_type] = node_counts.get(node_type, 0) + 1

        critical_nodes = len([n for n in trace.nodes if n.is_critical])
        avg_confidence = sum(n.confidence for n in trace.nodes) / len(trace.nodes) if trace.nodes else 0

        lines = [
            "=== 추론 경로 요약 ===",
            "",
            f"총 노드: {len(trace.nodes)}개",
            f"핵심 노드: {critical_nodes}개",
            f"연결: {len(trace.edges)}개",
            f"평균 신뢰도: {avg_confidence:.0%}",
            "",
            "[노드 유형별 분포]"
        ]

        for node_type, count in sorted(node_counts.items()):
            lines.append(f"  - {node_type}: {count}개")

        return "\n".join(lines)

    def _determine_reasoning_depth(self, trace: ReasoningTrace) -> ReasoningDepth:
        """추론 깊이 결정"""
        max_depth = max((n.depth for n in trace.nodes), default=0)
        reasoning_steps = len(trace.get_node_by_type(NodeType.REASONING_STEP))

        if max_depth <= 2 and reasoning_steps <= 1:
            return ReasoningDepth.SHALLOW
        elif max_depth <= 4 and reasoning_steps <= 3:
            return ReasoningDepth.MODERATE
        elif max_depth <= 6 and reasoning_steps <= 5:
            return ReasoningDepth.DEEP
        else:
            return ReasoningDepth.COMPLEX

    def _find_reasoning_paths(self, trace: ReasoningTrace) -> List[ReasoningPath]:
        """주요 추론 경로 식별"""
        paths = []

        # 시작 노드들 (질문 또는 조항)
        start_nodes = (
            trace.get_node_by_type(NodeType.QUESTION) or
            trace.get_node_by_type(NodeType.CONTRACT_CLAUSE)
        )

        # 종료 노드들 (결론 또는 권고)
        end_nodes = (
            trace.get_node_by_type(NodeType.CONCLUSION) +
            trace.get_node_by_type(NodeType.RECOMMENDATION)
        )

        for start in start_nodes[:2]:
            for end in end_nodes[:3]:
                path = self._find_path(trace, start.id, end.id)
                if path:
                    confidence = trace.calculate_path_confidence(path)
                    paths.append(ReasoningPath(
                        nodes=path,
                        edges=[],  # 엣지 ID는 별도 계산 필요
                        total_confidence=confidence,
                        description=f"{start.label} -> {end.label}"
                    ))

        return sorted(paths, key=lambda p: p.total_confidence, reverse=True)[:5]

    def _find_path(
        self,
        trace: ReasoningTrace,
        start_id: str,
        end_id: str
    ) -> List[str]:
        """BFS로 경로 찾기"""
        from collections import deque

        queue = deque([(start_id, [start_id])])
        visited = set()

        while queue:
            current, path = queue.popleft()

            if current == end_id:
                return path

            if current in visited:
                continue
            visited.add(current)

            for edge in trace.get_edges_from(current):
                if edge.target not in visited:
                    queue.append((edge.target, path + [edge.target]))

        return []

    def _calculate_layout(self, trace: ReasoningTrace):
        """노드 레이아웃 계산 (계층적)"""
        # 깊이별 그룹화
        depth_groups: Dict[int, List[str]] = {}
        for node in trace.nodes:
            if node.depth not in depth_groups:
                depth_groups[node.depth] = []
            depth_groups[node.depth].append(node.id)

        # 위치 할당
        for depth, node_ids in depth_groups.items():
            width = len(node_ids)
            for i, node_id in enumerate(node_ids):
                node = trace.get_node(node_id)
                if node:
                    # 균등 분포
                    x = (i + 1) * 100 / (width + 1)
                    y = depth * 80
                    node.position = {"x": x, "y": y}

    def expand_with_graph(
        self,
        trace: ReasoningTrace,
        max_hops: int = 2
    ) -> ReasoningTrace:
        """그래프 DB로 추론 추적 확장"""
        if self.neo4j_driver is None:
            return trace

        try:
            with self.neo4j_driver.session() as session:
                # 법령 참조 노드 확장
                for node in trace.get_node_by_type(NodeType.LEGAL_REFERENCE):
                    expanded = self._expand_legal_reference(
                        session, node.content, max_hops
                    )
                    for exp_node, exp_edge in expanded:
                        trace.add_node(exp_node)
                        exp_edge.source = node.id
                        trace.add_edge(exp_edge)

        except Exception as e:
            print(f"Graph expansion error: {e}")

        return trace

    def _expand_legal_reference(
        self,
        session,
        content: str,
        max_hops: int
    ) -> List[Tuple[TraceNode, TraceEdge]]:
        """법령 참조 확장"""
        expanded = []

        import re
        law_pattern = r'(근로기준법|최저임금법|근로자퇴직급여보장법)\s*제\s*(\d+)\s*조'
        match = re.search(law_pattern, content)

        if match:
            keyword = match.group(0)
            query = """
            MATCH (d:Document)-[:CITES]->(law:Law)
            WHERE d.content CONTAINS $keyword
            RETURN law.name AS law_name,
                   law.articleNumber AS article,
                   d.content AS citing_doc
            LIMIT 3
            """

            try:
                result = session.run(query, keyword=keyword)
                for record in result:
                    node = self._create_node(
                        NodeType.LEGAL_REFERENCE,
                        f"{record['law_name']} 제{record['article']}조",
                        record["citing_doc"][:200] if record["citing_doc"] else "",
                        metadata=NodeMetadata(
                            legal_basis=[f"{record['law_name']} 제{record['article']}조"],
                            source="graph_db"
                        )
                    )
                    edge = TraceEdge(
                        id=f"edge_exp_{self._edge_counter}",
                        source="",  # 나중에 설정
                        target=node.id,
                        edge_type=EdgeType.CITES
                    )
                    self._edge_counter += 1
                    expanded.append((node, edge))
            except Exception as e:
                print(f"Legal reference expansion error: {e}")

        return expanded


class TraceVisualizer:
    """추론 추적 시각화 도구 (확장)"""

    @staticmethod
    def to_cytoscape(trace: ReasoningTrace) -> Dict[str, Any]:
        """Cytoscape.js 형식으로 변환"""
        elements = []

        # 노드
        for node in trace.nodes:
            elements.append({
                "data": {
                    "id": node.id,
                    "label": node.label,
                    "type": node.node_type.value,
                    "content": node.content[:200],
                    "confidence": node.confidence,
                    "confidence_level": node.confidence_level.value,
                    "is_critical": node.is_critical,
                    "depth": node.depth,
                },
                "position": node.position,
                "classes": f"{node.node_type.value} {'critical' if node.is_critical else ''}"
            })

        # 엣지
        for edge in trace.edges:
            elements.append({
                "data": {
                    "id": edge.id,
                    "source": edge.source,
                    "target": edge.target,
                    "label": edge.label,
                    "type": edge.edge_type.value,
                    "weight": edge.weight,
                    "confidence": edge.confidence,
                },
                "classes": edge.edge_type.value
            })

        return {
            "elements": elements,
            "metadata": {
                "trace_id": trace.id,
                "overall_confidence": trace.overall_confidence,
                "reasoning_depth": trace.reasoning_depth.value,
            }
        }

    @staticmethod
    def to_d3(trace: ReasoningTrace) -> Dict[str, Any]:
        """D3.js Force Graph 형식으로 변환"""
        return {
            "nodes": [
                {
                    "id": n.id,
                    "group": n.node_type.value,
                    "label": n.label,
                    "content": n.content[:100],
                    "confidence": n.confidence,
                    "is_critical": n.is_critical,
                    "depth": n.depth,
                    "size": 20 if n.is_critical else 10,
                }
                for n in trace.nodes
            ],
            "links": [
                {
                    "source": e.source,
                    "target": e.target,
                    "value": e.weight,
                    "type": e.edge_type.value,
                    "label": e.label,
                }
                for e in trace.edges
            ],
            "metadata": {
                "overall_confidence": trace.overall_confidence,
                "reasoning_depth": trace.reasoning_depth.value,
                "paths": [p.to_dict() for p in trace.reasoning_paths],
            }
        }

    @staticmethod
    def to_react_flow(trace: ReasoningTrace) -> Dict[str, Any]:
        """React Flow 형식으로 변환"""
        nodes = []
        edges = []

        # 노드 색상 매핑
        color_map = {
            NodeType.CONTRACT_CLAUSE: "#ffcccc",
            NodeType.LEGAL_REFERENCE: "#ccffcc",
            NodeType.PRECEDENT: "#ccccff",
            NodeType.RISK_PATTERN: "#ffcc99",
            NodeType.REASONING_STEP: "#ffffcc",
            NodeType.CONCLUSION: "#99ccff",
            NodeType.RECOMMENDATION: "#e6ccff",
        }

        for node in trace.nodes:
            nodes.append({
                "id": node.id,
                "type": "custom",
                "position": {
                    "x": node.position.get("x", 0) * 3,
                    "y": node.position.get("y", 0) * 2,
                },
                "data": {
                    "label": node.label,
                    "content": node.content[:150],
                    "type": node.node_type.value,
                    "confidence": node.confidence,
                    "is_critical": node.is_critical,
                    "color": color_map.get(node.node_type, "#ffffff"),
                },
                "style": {
                    "border": "2px solid #333" if node.is_critical else "1px solid #999",
                }
            })

        for edge in trace.edges:
            edges.append({
                "id": edge.id,
                "source": edge.source,
                "target": edge.target,
                "label": edge.label,
                "type": "smoothstep",
                "animated": edge.edge_type == EdgeType.LEADS_TO,
                "style": {
                    "stroke": "#333",
                    "strokeDasharray": "5,5" if edge.edge_type == EdgeType.CONTRADICTS else None,
                }
            })

        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def to_html(trace: ReasoningTrace, detailed: bool = True) -> str:
        """HTML 시각화 생성"""
        mermaid = trace.to_mermaid(detailed=detailed)

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>AI 추론 경로 시각화</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        body {{
            font-family: 'Noto Sans KR', Arial, sans-serif;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }}
        .mermaid {{
            text-align: center;
            margin: 20px 0;
        }}
        .meta-section {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .meta-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }}
        .meta-card h3 {{
            margin-top: 0;
            color: #007bff;
        }}
        .confidence-bar {{
            height: 20px;
            background: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
        }}
        .confidence-fill {{
            height: 100%;
            background: linear-gradient(90deg, #28a745, #007bff);
            transition: width 0.3s;
        }}
        .summary {{
            background: #e8f4fd;
            padding: 15px;
            border-radius: 8px;
            white-space: pre-line;
        }}
        .conclusion {{
            background: #d4edda;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #28a745;
        }}
        .paths {{
            background: #fff3cd;
            padding: 15px;
            border-radius: 8px;
        }}
        .path-item {{
            padding: 10px;
            margin: 5px 0;
            background: white;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>AI 추론 경로 분석</h1>

        <div class="meta-section">
            <div class="meta-card">
                <h3>전체 신뢰도</h3>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {trace.overall_confidence * 100}%"></div>
                </div>
                <p>{trace.overall_confidence:.1%}</p>
            </div>
            <div class="meta-card">
                <h3>추론 깊이</h3>
                <p>{trace.reasoning_depth.value}</p>
            </div>
            <div class="meta-card">
                <h3>노드 수</h3>
                <p>{len(trace.nodes)}개</p>
            </div>
            <div class="meta-card">
                <h3>연결 수</h3>
                <p>{len(trace.edges)}개</p>
            </div>
        </div>

        <h2>추론 그래프</h2>
        <div class="mermaid">
{mermaid}
        </div>

        <div class="summary">
            <h3>요약</h3>
            <pre>{trace.summary}</pre>
        </div>

        {"<div class='conclusion'><h3>결론</h3><p>" + trace.conclusion + "</p></div>" if trace.conclusion else ""}

        {"<div class='paths'><h3>주요 추론 경로</h3>" + "".join(f"<div class='path-item'>{p.description} (신뢰도: {p.total_confidence:.0%})</div>" for p in trace.reasoning_paths[:3]) + "</div>" if trace.reasoning_paths else ""}
    </div>

    <script>
        mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    </script>
</body>
</html>
"""
        return html


# ========== 편의 함수 ==========

def trace_analysis(
    clause: str,
    result: Dict[str, Any],
    context: List[Dict[str, Any]] = None,
    query: str = ""
) -> ReasoningTrace:
    """간편 추론 추적"""
    tracer = ReasoningTracer()
    return tracer.trace_analysis(clause, result, context, query)


def visualize_trace(
    trace: ReasoningTrace,
    format: str = "mermaid",
    detailed: bool = False
) -> str:
    """추론 추적 시각화"""
    if format == "mermaid":
        return trace.to_mermaid(detailed=detailed)
    elif format == "html":
        return TraceVisualizer.to_html(trace, detailed=detailed)
    elif format == "json":
        return trace.to_json()
    elif format == "graphviz":
        return trace.to_graphviz()
    else:
        return trace.to_mermaid()


def create_tracer(
    neo4j_driver: Optional[Any] = None,
    propagate_confidence: bool = True
) -> ReasoningTracer:
    """추론 추적기 생성"""
    return ReasoningTracer(
        neo4j_driver=neo4j_driver,
        propagate_confidence=propagate_confidence
    )
