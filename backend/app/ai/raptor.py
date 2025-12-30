"""
RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) - Production-Grade
- 동적 클러스터링 (밀도 기반 + 계층적)
- 다중 요약 전략 (추출적/추상적/하이브리드)
- 계약서 도메인 특화 카테고리 시스템
- 레벨별 검색 최적화
- 트리 시각화 및 내보내기

Reference: ICLR 2024 - "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"
"""

import os
import json
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np
from datetime import datetime
import hashlib

from app.core.config import settings
from app.core.token_usage_tracker import record_llm_usage


class SummarizationStrategy(Enum):
    """요약 전략"""
    EXTRACTIVE = "extractive"       # 핵심 문장 추출
    ABSTRACTIVE = "abstractive"     # 새로운 문장 생성
    HYBRID = "hybrid"               # 추출 + 생성 혼합
    LEGAL_FOCUSED = "legal_focused" # 법률 조항 중심


class ClusteringMethod(Enum):
    """클러스터링 방법"""
    AGGLOMERATIVE = "agglomerative"   # 계층적 클러스터링
    DBSCAN = "dbscan"                   # 밀도 기반 클러스터링
    KMEANS = "kmeans"                   # K-means
    SEMANTIC = "semantic"               # 의미 기반 (LLM)
    ADAPTIVE = "adaptive"               # 내용에 따라 자동 선택


class SearchLevel(Enum):
    """검색 레벨"""
    LEAF = "leaf"           # 원본 청크만
    SUMMARY = "summary"     # 요약 노드만
    ALL = "all"             # 모든 레벨
    ADAPTIVE = "adaptive"   # 질문에 따라 자동 선택
    TOP_DOWN = "top_down"   # 루트에서 하향 탐색
    BOTTOM_UP = "bottom_up" # 리프에서 상향 탐색


@dataclass
class RAPTORNode:
    """RAPTOR 트리 노드 (고도화)"""
    id: str
    text: str
    level: int
    embedding: Optional[np.ndarray] = None
    children: List[str] = field(default_factory=list)
    parent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 고도화 필드
    category: str = "general"
    summary_type: SummarizationStrategy = SummarizationStrategy.ABSTRACTIVE
    confidence_score: float = 1.0
    token_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    keywords: List[str] = field(default_factory=list)
    legal_references: List[str] = field(default_factory=list)  # 법령 참조

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        return self.parent is None and self.level > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "level": self.level,
            "children": self.children,
            "parent": self.parent,
            "category": self.category,
            "confidence_score": self.confidence_score,
            "keywords": self.keywords,
            "legal_references": self.legal_references,
            "metadata": self.metadata
        }


@dataclass
class RAPTORTree:
    """RAPTOR 트리 구조 (고도화)"""
    nodes: Dict[str, RAPTORNode] = field(default_factory=dict)
    root_ids: List[str] = field(default_factory=list)
    max_level: int = 0

    # 메타데이터
    document_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    total_tokens: int = 0
    category_summary: Dict[str, str] = field(default_factory=dict)

    def add_node(self, node: RAPTORNode):
        self.nodes[node.id] = node
        if node.level > self.max_level:
            self.max_level = node.level
        self.total_tokens += node.token_count

    def get_nodes_at_level(self, level: int) -> List[RAPTORNode]:
        return [n for n in self.nodes.values() if n.level == level]

    def get_nodes_by_category(self, category: str) -> List[RAPTORNode]:
        return [n for n in self.nodes.values() if n.category == category]

    def get_all_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """모든 노드의 임베딩 반환"""
        ids = []
        embeddings = []
        for node_id, node in self.nodes.items():
            if node.embedding is not None:
                ids.append(node_id)
                embeddings.append(node.embedding)
        return ids, np.array(embeddings) if embeddings else np.array([])

    def get_descendants(self, node_id: str) -> List[RAPTORNode]:
        """노드의 모든 자손 반환"""
        descendants = []
        node = self.nodes.get(node_id)
        if not node:
            return descendants

        for child_id in node.children:
            child = self.nodes.get(child_id)
            if child:
                descendants.append(child)
                descendants.extend(self.get_descendants(child_id))
        return descendants

    def get_ancestors(self, node_id: str) -> List[RAPTORNode]:
        """노드의 모든 조상 반환"""
        ancestors = []
        node = self.nodes.get(node_id)
        if not node or not node.parent:
            return ancestors

        parent = self.nodes.get(node.parent)
        if parent:
            ancestors.append(parent)
            ancestors.extend(self.get_ancestors(parent.id))
        return ancestors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "max_level": self.max_level,
            "total_tokens": self.total_tokens,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "root_ids": self.root_ids,
            "category_summary": self.category_summary,
            "created_at": self.created_at.isoformat()
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class RAPTORIndexer:
    """
    RAPTOR 계층적 인덱서 - 상용화 버전

    주요 특징:
    1. 동적 클러스터링: 내용에 따라 최적 클러스터링 방법 선택
    2. 다중 요약 전략: 법률 문서에 최적화된 요약
    3. 레벨별 검색: 질문 유형에 따른 적응형 검색
    4. 법률 참조 추출: 법령/판례 자동 식별
    5. 시각화 지원: Mermaid, JSON 트리 내보내기

    사용법:
        indexer = RAPTORIndexer(strategy=SummarizationStrategy.LEGAL_FOCUSED)
        tree = indexer.build_tree(chunks)
        results = indexer.search(tree, "계약서 전체 요약", level=SearchLevel.ADAPTIVE)
    """

    # 법률 문서 요약 프롬프트
    LEGAL_SUMMARY_PROMPT = """다음은 근로계약서 관련 문서의 일부입니다.
핵심 법적 정보를 보존하면서 통합 요약을 작성하세요.

[필수 포함 항목]
- 관련 법령 조항 (근로기준법 제X조 등)
- 핵심 권리/의무 사항
- 위험 요소 (있는 경우)
- 금액/기간 등 수치 정보

[문서들]
{documents}

[주의사항]
- 법적 용어는 정확히 유지
- 수치 정보 누락 금지
- 위험 조항은 명시적으로 표시

[통합 요약]"""

    # 카테고리별 요약 프롬프트
    CATEGORY_SUMMARY_PROMPTS = {
        "임금": """다음 임금 관련 조항들을 요약하세요.

포함 항목:
- 기본급, 수당 구성
- 지급 시기 및 방법
- 포괄임금제 여부
- 최저임금 준수 여부

조항들:
{documents}

임금 조항 요약:""",

        "근로시간": """다음 근로시간 관련 조항들을 요약하세요.

포함 항목:
- 소정근로시간
- 연장/야간/휴일 근로 규정
- 휴게시간
- 주 52시간 준수 여부

조항들:
{documents}

근로시간 조항 요약:""",

        "휴일휴가": """다음 휴일/휴가 관련 조항들을 요약하세요.

포함 항목:
- 주휴일
- 연차유급휴가
- 특별휴가
- 휴가 사용 조건

조항들:
{documents}

휴일/휴가 조항 요약:""",

        "해고퇴직": """다음 해고/퇴직 관련 조항들을 요약하세요.

포함 항목:
- 해고 사유
- 해고 절차 (예고 등)
- 퇴직 조건
- 퇴직금 규정

조항들:
{documents}

해고/퇴직 조항 요약:"""
    }

    # 계약서 전체 요약 프롬프트
    CONTRACT_OVERVIEW_PROMPT = """다음은 근로계약서의 카테고리별 요약입니다.
전체 계약서의 핵심 내용과 주의사항을 종합 요약하세요.

[카테고리별 요약]
{category_summaries}

[작성 지침]
- 자연스러운 줄글(문단) 형식으로 작성하세요. 번호 매기기나 bullet point를 사용하지 마세요.
- 핵심 키워드나 중요한 정보는 **볼드체**로 강조하세요. (예: **월 200만원**, **주 52시간 초과**)
- 위험한 조항이나 법적 문제가 있는 부분은 반드시 **볼드체**로 표시하세요.
- 3-4개 문단으로 구성하세요:
  1) 계약 개요 (계약 유형, 당사자, 기간)
  2) 핵심 근로조건 (임금, 근로시간, 휴가)
  3) 주의가 필요한 위험 조항들
  4) 종합 평가 및 권고사항

[종합 요약]"""

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        embedding_model: Optional[Any] = None,
        model: str = None,
        cluster_size: int = 5,
        max_levels: int = 3,
        summarization_strategy: SummarizationStrategy = SummarizationStrategy.LEGAL_FOCUSED,
        clustering_method: ClusteringMethod = ClusteringMethod.ADAPTIVE,
        min_cluster_size: int = 2,
        contract_id: Optional[str] = None
    ):
        """
        Args:
            llm_client: OpenAI 클라이언트 (legacy, Gemini 사용 시 무시됨)
            embedding_model: 임베딩 모델
            model: LLM 모델명 (기본값: settings.LLM_RAPTOR_MODEL)
            cluster_size: 한 번에 요약할 청크 수
            max_levels: 최대 트리 레벨
            summarization_strategy: 요약 전략
            clustering_method: 클러스터링 방법
            min_cluster_size: 최소 클러스터 크기
            contract_id: 계약서 ID (토큰 추적용)
        """
        self.model = model if model else settings.LLM_RAPTOR_MODEL
        self.cluster_size = cluster_size
        self.max_levels = max_levels
        self.summarization_strategy = summarization_strategy
        self.clustering_method = clustering_method
        self.min_cluster_size = min_cluster_size
        self.contract_id = contract_id
        self.llm_client = llm_client  # OpenAI fallback

        # Gemini safety settings (완전 완화 - 계약서 분석은 합법적 용도)
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        # Gemini 클라이언트 초기화 (우선 사용)
        self._gemini_model = None
        if "gemini" in self.model.lower():
            try:
                import google.generativeai as genai
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self._gemini_model = genai.GenerativeModel(self.model)
            except ImportError:
                print("google-generativeai package not installed, falling back to OpenAI")
            except Exception as e:
                print(f"Gemini initialization error: {e}")

        # OpenAI fallback
        if self._gemini_model is None and self.llm_client is None:
            try:
                from openai import OpenAI
                self.llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                self.llm_client = None

        # 임베딩 모델 초기화
        if embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer("nlpai-lab/KURE-v1")
            except ImportError:
                self.embedding_model = None
        else:
            self.embedding_model = embedding_model

    def build_tree(
        self,
        chunks: List[str],
        metadata: Optional[List[Dict]] = None,
        document_id: str = ""
    ) -> RAPTORTree:
        """
        청크로부터 RAPTOR 트리 구축

        Args:
            chunks: 원본 텍스트 청크 리스트
            metadata: 각 청크의 메타데이터
            document_id: 문서 ID

        Returns:
            RAPTORTree: 계층적 트리 구조
        """
        tree = RAPTORTree(document_id=document_id or self._generate_doc_id())

        if not chunks:
            return tree

        # Level 0: 원본 청크를 리프 노드로 추가
        current_level_nodes = []
        for i, chunk in enumerate(chunks):
            node_id = f"L0_N{i}"
            embedding = self._generate_embedding(chunk)
            category = self._detect_category(chunk)
            keywords = self._extract_keywords(chunk)
            legal_refs = self._extract_legal_references(chunk)

            node = RAPTORNode(
                id=node_id,
                text=chunk,
                level=0,
                embedding=embedding,
                category=category,
                keywords=keywords,
                legal_references=legal_refs,
                token_count=len(chunk.split()),
                metadata=metadata[i] if metadata and i < len(metadata) else {}
            )
            tree.add_node(node)
            current_level_nodes.append(node)

        # Level 1 이상: 재귀적으로 요약 노드 생성
        level = 1
        while len(current_level_nodes) > 1 and level <= self.max_levels:
            next_level_nodes = self._build_level(tree, current_level_nodes, level)

            if not next_level_nodes:
                break

            current_level_nodes = next_level_nodes
            level += 1

        # 최상위 노드들을 루트로 설정
        tree.root_ids = [n.id for n in current_level_nodes]

        # 카테고리별 요약 생성
        tree.category_summary = self._generate_category_summaries(tree)

        return tree

    def _build_level(
        self,
        tree: RAPTORTree,
        nodes: List[RAPTORNode],
        level: int
    ) -> List[RAPTORNode]:
        """트리의 한 레벨 구축"""
        next_level_nodes = []

        # 클러스터링 방법 선택
        clusters = self._cluster_nodes(nodes)

        for cluster_idx, cluster in enumerate(clusters):
            if len(cluster) < self.min_cluster_size:
                # 너무 작은 클러스터는 상위 레벨에 그대로 전달
                for node in cluster:
                    node.level = level
                    next_level_nodes.append(node)
                continue

            # 클러스터의 텍스트를 요약
            texts = [n.text for n in cluster]
            categories = [n.category for n in cluster]
            dominant_category = max(set(categories), key=categories.count)

            summary = self._summarize(texts, dominant_category)

            if not summary:
                continue

            # 요약 노드 생성
            node_id = f"L{level}_N{cluster_idx}"
            embedding = self._generate_embedding(summary)

            # 자식 노드들의 키워드와 법령 참조 병합
            merged_keywords = list(set(
                kw for node in cluster for kw in node.keywords
            ))[:10]
            merged_legal_refs = list(set(
                ref for node in cluster for ref in node.legal_references
            ))

            summary_node = RAPTORNode(
                id=node_id,
                text=summary,
                level=level,
                embedding=embedding,
                children=[n.id for n in cluster],
                category=dominant_category,
                summary_type=self.summarization_strategy,
                keywords=merged_keywords,
                legal_references=merged_legal_refs,
                token_count=len(summary.split()),
                confidence_score=self._calculate_confidence(cluster),
                metadata={"cluster_size": len(cluster)}
            )

            # 자식 노드의 부모 설정
            for child in cluster:
                child.parent = node_id

            tree.add_node(summary_node)
            next_level_nodes.append(summary_node)

        return next_level_nodes

    def _cluster_nodes(self, nodes: List[RAPTORNode]) -> List[List[RAPTORNode]]:
        """노드를 클러스터로 그룹화"""
        if len(nodes) <= self.cluster_size:
            return [nodes]

        # 클러스터링 방법 결정
        method = self._determine_clustering_method(nodes)

        if method == ClusteringMethod.AGGLOMERATIVE:
            return self._agglomerative_clustering(nodes)
        elif method == ClusteringMethod.DBSCAN:
            return self._dbscan_clustering(nodes)
        elif method == ClusteringMethod.SEMANTIC:
            return self._semantic_clustering(nodes)
        else:  # KMEANS or fallback
            return self._kmeans_clustering(nodes)

    def _determine_clustering_method(self, nodes: List[RAPTORNode]) -> ClusteringMethod:
        """내용에 따라 최적 클러스터링 방법 결정"""
        if self.clustering_method != ClusteringMethod.ADAPTIVE:
            return self.clustering_method

        # 카테고리 다양성 확인
        categories = set(n.category for n in nodes)

        if len(categories) >= 3:
            # 다양한 카테고리 -> 의미 기반
            return ClusteringMethod.SEMANTIC
        elif len(nodes) > 20:
            # 많은 노드 -> DBSCAN
            return ClusteringMethod.DBSCAN
        else:
            # 기본 -> 계층적
            return ClusteringMethod.AGGLOMERATIVE

    def _agglomerative_clustering(self, nodes: List[RAPTORNode]) -> List[List[RAPTORNode]]:
        """계층적 클러스터링"""
        try:
            from sklearn.cluster import AgglomerativeClustering
        except ImportError:
            return self._fallback_clustering(nodes)

        embeddings = [n.embedding for n in nodes if n.embedding is not None]
        if len(embeddings) != len(nodes):
            return self._fallback_clustering(nodes)

        embeddings = np.array(embeddings)
        n_clusters = max(1, len(nodes) // self.cluster_size)

        if n_clusters >= len(nodes):
            return [nodes]

        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average"
        )
        labels = clustering.fit_predict(embeddings)

        clusters_dict = defaultdict(list)
        for node, label in zip(nodes, labels):
            clusters_dict[label].append(node)

        return list(clusters_dict.values())

    def _dbscan_clustering(self, nodes: List[RAPTORNode]) -> List[List[RAPTORNode]]:
        """밀도 기반 클러스터링"""
        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            return self._fallback_clustering(nodes)

        embeddings = [n.embedding for n in nodes if n.embedding is not None]
        if len(embeddings) != len(nodes):
            return self._fallback_clustering(nodes)

        embeddings = np.array(embeddings)

        clustering = DBSCAN(eps=0.3, min_samples=2, metric="cosine")
        labels = clustering.fit_predict(embeddings)

        clusters_dict = defaultdict(list)
        noise_cluster = []

        for node, label in zip(nodes, labels):
            if label == -1:
                noise_cluster.append(node)
            else:
                clusters_dict[label].append(node)

        clusters = list(clusters_dict.values())

        # 노이즈 포인트 처리
        if noise_cluster:
            if len(noise_cluster) >= self.min_cluster_size:
                clusters.append(noise_cluster)
            else:
                # 가장 가까운 클러스터에 추가
                for node in noise_cluster:
                    if clusters:
                        clusters[0].append(node)

        return clusters if clusters else [nodes]

    def _kmeans_clustering(self, nodes: List[RAPTORNode]) -> List[List[RAPTORNode]]:
        """K-means 클러스터링"""
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            return self._fallback_clustering(nodes)

        embeddings = [n.embedding for n in nodes if n.embedding is not None]
        if len(embeddings) != len(nodes):
            return self._fallback_clustering(nodes)

        embeddings = np.array(embeddings)
        n_clusters = max(1, len(nodes) // self.cluster_size)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        clusters_dict = defaultdict(list)
        for node, label in zip(nodes, labels):
            clusters_dict[label].append(node)

        return list(clusters_dict.values())

    def _semantic_clustering(self, nodes: List[RAPTORNode]) -> List[List[RAPTORNode]]:
        """카테고리 기반 의미적 클러스터링"""
        clusters_dict = defaultdict(list)

        for node in nodes:
            clusters_dict[node.category].append(node)

        clusters = []
        for category_nodes in clusters_dict.values():
            if len(category_nodes) <= self.cluster_size:
                clusters.append(category_nodes)
            else:
                # 카테고리 내에서 추가 클러스터링
                sub_clusters = self._agglomerative_clustering(category_nodes)
                clusters.extend(sub_clusters)

        return clusters

    def _fallback_clustering(self, nodes: List[RAPTORNode]) -> List[List[RAPTORNode]]:
        """순차적 그룹화 (폴백)"""
        clusters = []
        for i in range(0, len(nodes), self.cluster_size):
            cluster = nodes[i:i + self.cluster_size]
            clusters.append(cluster)
        return clusters

    def _is_reasoning_model(self) -> bool:
        """reasoning 모델 여부 확인 (temperature 미지원)"""
        reasoning_keywords = ["o1", "o3", "gpt-5"]
        return any(kw in self.model.lower() for kw in reasoning_keywords)

    def _summarize(self, texts: List[str], category: str = "general") -> str:
        """텍스트 리스트를 요약"""
        if not texts:
            return ""

        if len(texts) == 1:
            return texts[0]

        if self._gemini_model is None and self.llm_client is None:
            return "\n\n".join(texts[:3])

        llm_start = time.time()
        try:
            # 카테고리별 프롬프트 선택
            if self.summarization_strategy == SummarizationStrategy.LEGAL_FOCUSED:
                if category in self.CATEGORY_SUMMARY_PROMPTS:
                    prompt_template = self.CATEGORY_SUMMARY_PROMPTS[category]
                else:
                    prompt_template = self.LEGAL_SUMMARY_PROMPT
            else:
                prompt_template = self.LEGAL_SUMMARY_PROMPT

            documents = "\n\n---\n\n".join(texts)
            prompt = prompt_template.format(documents=documents)

            # Gemini 사용 (우선)
            if self._gemini_model is not None:
                full_prompt = "당신은 법률 문서 요약 전문가입니다. 핵심 법적 정보를 정확히 보존하면서 간결하게 요약합니다.\n\n" + prompt
                result = self._gemini_model.generate_content(
                    full_prompt,
                    generation_config={
                        "temperature": 0.3,
                        "max_output_tokens": 800
                    },
                    safety_settings=self.safety_settings
                )
                llm_duration = (time.time() - llm_start) * 1000

                # 토큰 사용량 기록
                if self.contract_id and hasattr(result, 'usage_metadata'):
                    usage = result.usage_metadata
                    record_llm_usage(
                        contract_id=self.contract_id,
                        module="raptor.summarize",
                        model=self.model,
                        input_tokens=getattr(usage, 'prompt_token_count', 0),
                        output_tokens=getattr(usage, 'candidates_token_count', 0),
                        cached_tokens=getattr(usage, 'cached_content_token_count', 0),
                        duration_ms=llm_duration
                    )

                return result.text

            # OpenAI fallback
            else:
                # reasoning 모델은 temperature 미지원
                if self._is_reasoning_model():
                    response = self.llm_client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "user", "content": f"당신은 법률 문서 요약 전문가입니다. 핵심 법적 정보를 정확히 보존하면서 간결하게 요약합니다.\n\n{prompt}"}
                        ],
                        max_completion_tokens=800
                    )
                else:
                    response = self.llm_client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": "당신은 법률 문서 요약 전문가입니다. 핵심 법적 정보를 정확히 보존하면서 간결하게 요약합니다."
                            },
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                        max_completion_tokens=800
                    )

                llm_duration = (time.time() - llm_start) * 1000

                # 토큰 사용량 기록
                if response.usage and self.contract_id:
                    cached = 0
                    if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                        cached = getattr(response.usage.prompt_tokens_details, 'cached_tokens', 0) or 0
                    record_llm_usage(
                        contract_id=self.contract_id,
                        module="raptor.summarize",
                        model=self.model,
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens,
                        cached_tokens=cached,
                        duration_ms=llm_duration
                    )

                return response.choices[0].message.content

        except Exception as e:
            print(f"RAPTOR summarization error: {e}")
            return "\n\n".join(texts[:2])

    def _generate_category_summaries(self, tree: RAPTORTree) -> Dict[str, str]:
        """카테고리별 종합 요약 생성"""
        summaries = {}

        for category in ["임금", "근로시간", "휴일휴가", "해고퇴직", "계약기간", "복리후생"]:
            category_nodes = tree.get_nodes_by_category(category)
            if category_nodes:
                # 레벨이 높은 노드 우선
                sorted_nodes = sorted(category_nodes, key=lambda n: n.level, reverse=True)
                top_node = sorted_nodes[0]
                summaries[category] = top_node.text

        return summaries

    def _generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """텍스트 임베딩 생성"""
        if self.embedding_model is None:
            return None
        try:
            return self.embedding_model.encode(text, normalize_embeddings=True)
        except Exception as e:
            print(f"Embedding error: {e}")
            return None

    def _detect_category(self, text: str) -> str:
        """텍스트에서 카테고리 감지"""
        category_keywords = {
            "임금": ["임금", "급여", "보수", "수당", "상여", "퇴직금", "최저임금", "월급", "연봉", "시급"],
            "근로시간": ["근로시간", "근무시간", "연장근로", "야간근로", "휴일근로", "주52시간", "시간외"],
            "휴일휴가": ["휴일", "휴가", "연차", "월차", "병가", "출산휴가", "육아휴직", "공휴일"],
            "계약기간": ["계약기간", "수습", "정규직", "계약직", "기간", "갱신", "만료"],
            "해고퇴직": ["해고", "퇴직", "해지", "종료", "해촉", "징계", "면직", "사직"],
            "복리후생": ["복리", "후생", "보험", "4대보험", "식대", "교통비", "복지"],
        }

        text_lower = text.lower()
        scores = {}

        for category, keywords in category_keywords.items():
            score = sum(1 for kw in keywords if kw in text)
            scores[category] = score

        if max(scores.values()) == 0:
            return "기타"

        return max(scores, key=scores.get)

    def _extract_keywords(self, text: str) -> List[str]:
        """핵심 키워드 추출"""
        import re

        # 법률 용어 패턴
        legal_patterns = [
            r'근로기준법\s*제?\d+조',
            r'최저임금법\s*제?\d+조',
            r'대법원\s*\d+\.\d+\.\d+',
            r'[가-힣]+수당',
            r'[가-힣]+휴가',
        ]

        keywords = []
        for pattern in legal_patterns:
            matches = re.findall(pattern, text)
            keywords.extend(matches)

        return list(set(keywords))[:10]

    def _extract_legal_references(self, text: str) -> List[str]:
        """법령 참조 추출"""
        import re

        patterns = [
            r'근로기준법\s*제?\s*(\d+)\s*조',
            r'최저임금법\s*제?\s*(\d+)\s*조',
            r'산업안전보건법\s*제?\s*(\d+)\s*조',
            r'남녀고용평등법\s*제?\s*(\d+)\s*조',
        ]

        references = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                law_name = pattern.split(r'\s*제')[0].replace('\\', '')
                references.append(f"{law_name} 제{match}조")

        return list(set(references))

    def _calculate_confidence(self, cluster: List[RAPTORNode]) -> float:
        """클러스터의 신뢰도 점수 계산"""
        if not cluster:
            return 0.0

        # 카테고리 일관성
        categories = [n.category for n in cluster]
        category_consistency = len(set(categories)) == 1

        # 평균 자식 신뢰도
        avg_confidence = sum(n.confidence_score for n in cluster) / len(cluster)

        if category_consistency:
            return min(1.0, avg_confidence * 1.1)
        else:
            return avg_confidence * 0.9

    def _generate_doc_id(self) -> str:
        """문서 ID 생성"""
        return hashlib.md5(datetime.now().isoformat().encode()).hexdigest()[:12]

    def search(
        self,
        tree: RAPTORTree,
        query: str,
        level: SearchLevel = SearchLevel.ADAPTIVE,
        top_k: int = 5,
        include_context: bool = True
    ) -> List[Dict[str, Any]]:
        """
        RAPTOR 트리에서 검색

        Args:
            tree: RAPTOR 트리
            query: 검색 쿼리
            level: 검색 레벨
            top_k: 반환할 결과 수
            include_context: 부모/자식 컨텍스트 포함 여부

        Returns:
            검색 결과 리스트
        """
        query_embedding = self._generate_embedding(query)
        if query_embedding is None:
            return []

        # 검색할 노드 결정
        search_nodes = self._get_search_nodes(tree, query, level)

        if not search_nodes:
            return []

        # 유사도 계산
        results = []
        for node in search_nodes:
            if node.embedding is not None:
                similarity = np.dot(query_embedding, node.embedding)
                result = {
                    "node_id": node.id,
                    "level": node.level,
                    "text": node.text,
                    "score": float(similarity),
                    "is_summary": node.level > 0,
                    "category": node.category,
                    "keywords": node.keywords,
                    "legal_references": node.legal_references,
                    "confidence": node.confidence_score,
                    "metadata": node.metadata
                }

                if include_context:
                    # 부모 컨텍스트
                    ancestors = tree.get_ancestors(node.id)
                    if ancestors:
                        result["parent_context"] = ancestors[0].text[:200]

                    # 자식 컨텍스트 (요약 노드인 경우)
                    if node.level > 0:
                        descendants = tree.get_descendants(node.id)
                        if descendants:
                            result["child_count"] = len(descendants)

                results.append(result)

        # 점수순 정렬
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]

    def _get_search_nodes(
        self,
        tree: RAPTORTree,
        query: str,
        level: SearchLevel
    ) -> List[RAPTORNode]:
        """검색할 노드 결정"""
        if level == SearchLevel.ALL:
            return list(tree.nodes.values())

        elif level == SearchLevel.LEAF:
            return tree.get_nodes_at_level(0)

        elif level == SearchLevel.SUMMARY:
            nodes = []
            for l in range(1, tree.max_level + 1):
                nodes.extend(tree.get_nodes_at_level(l))
            return nodes

        elif level == SearchLevel.TOP_DOWN:
            # 루트에서 시작하여 관련 브랜치 탐색
            return self._top_down_search(tree, query)

        elif level == SearchLevel.BOTTOM_UP:
            # 리프에서 시작하여 상위로 집계
            return tree.get_nodes_at_level(0)

        else:  # ADAPTIVE
            return self._adaptive_level_selection(tree, query)

    def _top_down_search(self, tree: RAPTORTree, query: str) -> List[RAPTORNode]:
        """탑다운 검색: 루트에서 관련 브랜치 탐색"""
        query_embedding = self._generate_embedding(query)
        if query_embedding is None:
            return list(tree.nodes.values())

        selected_nodes = []

        # 루트 노드들에서 시작
        for root_id in tree.root_ids:
            root = tree.nodes.get(root_id)
            if root:
                selected_nodes.append(root)
                # 가장 관련 있는 자식 브랜치 탐색
                self._traverse_relevant_branch(tree, root, query_embedding, selected_nodes)

        return selected_nodes

    def _traverse_relevant_branch(
        self,
        tree: RAPTORTree,
        node: RAPTORNode,
        query_embedding: np.ndarray,
        selected_nodes: List[RAPTORNode],
        threshold: float = 0.5
    ):
        """관련성 높은 브랜치 탐색"""
        if not node.children:
            return

        for child_id in node.children:
            child = tree.nodes.get(child_id)
            if child and child.embedding is not None:
                similarity = np.dot(query_embedding, child.embedding)
                if similarity >= threshold:
                    selected_nodes.append(child)
                    self._traverse_relevant_branch(
                        tree, child, query_embedding, selected_nodes, threshold * 0.9
                    )

    def _adaptive_level_selection(
        self,
        tree: RAPTORTree,
        query: str
    ) -> List[RAPTORNode]:
        """질문 유형에 따른 적응적 레벨 선택"""
        # 거시적 질문 키워드
        macro_keywords = [
            "전체적으로", "전반적으로", "요약", "핵심", "주요", "개요",
            "불리한가", "유리한가", "위험한가", "문제가 있는가", "어떤 계약"
        ]

        # 미시적 질문 키워드
        micro_keywords = [
            "제", "조", "항", "구체적으로", "정확히",
            "몇", "얼마", "언제", "어디", "누가"
        ]

        macro_score = sum(1 for kw in macro_keywords if kw in query)
        micro_score = sum(1 for kw in micro_keywords if kw in query)

        if macro_score > micro_score:
            # 거시적 질문: 상위 레벨 + 일부 하위 레벨
            nodes = []
            for l in range(tree.max_level, -1, -1):
                level_nodes = tree.get_nodes_at_level(l)
                nodes.extend(level_nodes)
                if len(nodes) >= 10:
                    break
            return nodes

        elif micro_score > macro_score:
            # 미시적 질문: 하위 레벨 위주
            return tree.get_nodes_at_level(0)

        else:
            # 균형: 모든 레벨
            return list(tree.nodes.values())

    def to_mermaid(self, tree: RAPTORTree) -> str:
        """트리를 Mermaid 다이어그램으로 변환"""
        lines = ["graph TD"]

        # 스타일 정의
        style_colors = {
            "임금": "#ffcccc",
            "근로시간": "#ccffcc",
            "휴일휴가": "#ccccff",
            "해고퇴직": "#ffcc99",
            "계약기간": "#ffccff",
            "복리후생": "#ccffff",
            "기타": "#eeeeee"
        }

        # 노드 추가
        for node in tree.nodes.values():
            label = node.text[:30].replace('"', "'") + "..."
            lines.append(f'    {node.id}["{label}"]')

        # 엣지 추가
        for node in tree.nodes.values():
            for child_id in node.children:
                lines.append(f'    {node.id} --> {child_id}')

        # 스타일 적용
        for node in tree.nodes.values():
            color = style_colors.get(node.category, "#ffffff")
            lines.append(f'    style {node.id} fill:{color}')

        return "\n".join(lines)


class ContractRAPTOR:
    """
    계약서 전용 RAPTOR 인덱서 - 상용화 버전
    """

    def __init__(self, indexer: RAPTORIndexer):
        self.indexer = indexer

    def build_contract_tree(
        self,
        contract_text: str,
        chunks: List[str]
    ) -> RAPTORTree:
        """계약서 전용 트리 구축"""
        # 기본 트리 구축
        tree = self.indexer.build_tree(chunks, document_id="contract")

        # 계약서 전체 요약 추가 (최상위 레벨)
        if tree.category_summary:
            overview = self._generate_contract_overview(tree.category_summary)

            root_node = RAPTORNode(
                id="L_root_overview",
                text=overview,
                level=tree.max_level + 1,
                embedding=self.indexer._generate_embedding(overview),
                children=tree.root_ids,
                category="계약서요약",
                summary_type=SummarizationStrategy.LEGAL_FOCUSED,
                confidence_score=1.0,
                metadata={"type": "contract_overview"}
            )
            tree.add_node(root_node)

            # 기존 루트 노드들의 부모 설정
            for root_id in tree.root_ids:
                if root_id in tree.nodes:
                    tree.nodes[root_id].parent = "L_root_overview"

            tree.root_ids = ["L_root_overview"]

        return tree

    def _generate_contract_overview(self, category_summaries: Dict[str, str]) -> str:
        """계약서 전체 개요 생성"""
        if self.indexer._gemini_model is None and self.indexer.llm_client is None:
            return "\n\n".join(f"[{k}]\n{v}" for k, v in category_summaries.items())

        llm_start = time.time()
        try:
            summaries_text = "\n\n".join(
                f"[{category}]\n{summary}"
                for category, summary in category_summaries.items()
            )

            prompt = self.indexer.CONTRACT_OVERVIEW_PROMPT.format(
                category_summaries=summaries_text
            )

            # Gemini 사용 (우선)
            if self.indexer._gemini_model is not None:
                full_prompt = "당신은 근로계약서 분석 전문가입니다.\n\n" + prompt
                result = self.indexer._gemini_model.generate_content(
                    full_prompt,
                    generation_config={
                        "temperature": 0.3,
                        "max_output_tokens": 1000
                    },
                    safety_settings=self.indexer.safety_settings
                )
                llm_duration = (time.time() - llm_start) * 1000

                # 토큰 사용량 기록
                if self.indexer.contract_id and hasattr(result, 'usage_metadata'):
                    usage = result.usage_metadata
                    record_llm_usage(
                        contract_id=self.indexer.contract_id,
                        module="raptor.contract_overview",
                        model=self.indexer.model,
                        input_tokens=getattr(usage, 'prompt_token_count', 0),
                        output_tokens=getattr(usage, 'candidates_token_count', 0),
                        cached_tokens=getattr(usage, 'cached_content_token_count', 0),
                        duration_ms=llm_duration
                    )

                return result.text

            # OpenAI fallback
            else:
                # reasoning 모델은 temperature 미지원
                if self.indexer._is_reasoning_model():
                    response = self.indexer.llm_client.chat.completions.create(
                        model=self.indexer.model,
                        messages=[
                            {"role": "user", "content": f"당신은 근로계약서 분석 전문가입니다.\n\n{prompt}"}
                        ],
                        max_completion_tokens=1000
                    )
                else:
                    response = self.indexer.llm_client.chat.completions.create(
                        model=self.indexer.model,
                        messages=[
                            {
                                "role": "system",
                                "content": "당신은 근로계약서 분석 전문가입니다."
                            },
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                        max_completion_tokens=1000
                    )

                llm_duration = (time.time() - llm_start) * 1000

                # 토큰 사용량 기록
                if response.usage and self.indexer.contract_id:
                    cached = 0
                    if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                        cached = getattr(response.usage.prompt_tokens_details, 'cached_tokens', 0) or 0
                    record_llm_usage(
                        contract_id=self.indexer.contract_id,
                        module="raptor.contract_overview",
                        model=self.indexer.model,
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens,
                        cached_tokens=cached,
                        duration_ms=llm_duration
                    )

                return response.choices[0].message.content

        except Exception as e:
            print(f"Contract overview error: {e}")
            return "\n\n".join(f"[{k}]\n{v}" for k, v in category_summaries.items())


# 편의 함수
def create_raptor_indexer(
    strategy: SummarizationStrategy = SummarizationStrategy.LEGAL_FOCUSED,
    clustering: ClusteringMethod = ClusteringMethod.ADAPTIVE
) -> RAPTORIndexer:
    """RAPTOR 인덱서 팩토리 함수"""
    return RAPTORIndexer(
        summarization_strategy=strategy,
        clustering_method=clustering
    )


def build_contract_raptor_tree(
    chunks: List[str],
    document_id: str = ""
) -> RAPTORTree:
    """간편 계약서 RAPTOR 트리 구축"""
    indexer = RAPTORIndexer()
    contract_raptor = ContractRAPTOR(indexer)
    return contract_raptor.build_contract_tree("", chunks)
