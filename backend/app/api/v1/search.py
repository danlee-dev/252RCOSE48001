"""
검색 API - Dify Agent Custom Tool용
우리의 RAG 시스템(CRAG, HyDE, RAPTOR)을 외부에서 호출할 수 있게 하는 엔드포인트
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

router = APIRouter()


class SearchRequest(BaseModel):
    """검색 요청"""
    query: str = Field(..., description="검색 질문", example="포괄임금제가 위법한가요?")
    contract_context: Optional[str] = Field(
        None,
        description="현재 계약서 내용 (선택)",
        example="제5조 급여는 월 250만원으로 하며, 연장근로수당을 포함한다."
    )
    top_k: int = Field(5, description="반환할 문서 수", ge=1, le=20)


class SearchDocument(BaseModel):
    """검색 결과 문서"""
    content: str = Field(..., description="문서 내용")
    source: str = Field(..., description="출처 (판례번호, 법령해석례 등)")
    source_type: str = Field(..., description="문서 타입 (precedent, interpretation, manual)")
    relevance: float = Field(..., description="관련도 점수", ge=0, le=1)


class SearchResponse(BaseModel):
    """검색 응답"""
    documents: List[SearchDocument]
    quality: str = Field(..., description="검색 품질 (correct, ambiguous, incorrect)")
    total_found: int


@router.post(
    "/legal",
    response_model=SearchResponse,
    summary="법률 지식 검색",
    description="""
    법령해석례, 판례, 고용노동부 해설을 검색합니다.

    Dify Agent의 Custom Tool로 사용됩니다.
    우리의 CRAG 파이프라인(벡터 검색 + 그래프 확장)을 활용합니다.
    """
)
async def search_legal_knowledge(request: SearchRequest):
    """
    Dify Agent가 호출할 법률 지식 검색 API
    """
    try:
        from app.ai.crag import CRAGProcessor, RetrievalQuality
        from app.ai.embedding import EmbeddingSearchService

        # 쿼리 구성
        query = request.query
        if request.contract_context:
            query = f"[계약서 컨텍스트]\n{request.contract_context[:1000]}\n\n[질문]\n{query}"

        # CRAG 프로세서로 검색
        crag = CRAGProcessor()
        result = crag.process(query, top_k=request.top_k)

        # 결과 변환
        documents = []
        for doc in result.all_docs:
            # 소스 타입 추출
            source_type = "unknown"
            if "precedent" in doc.source.lower() or "판례" in doc.source:
                source_type = "precedent"
            elif "interpretation" in doc.source.lower() or "해석례" in doc.source:
                source_type = "interpretation"
            elif "manual" in doc.source.lower() or "해설" in doc.source:
                source_type = "manual"

            documents.append(SearchDocument(
                content=doc.text,
                source=doc.source,
                source_type=source_type,
                relevance=min(doc.score, 1.0)  # 0~1 범위로 제한
            ))

        return SearchResponse(
            documents=documents,
            quality=result.quality.value,
            total_found=len(documents)
        )

    except Exception as e:
        # CRAG가 없거나 에러 시 기본 벡터 검색으로 폴백
        try:
            from app.ai.embedding import EmbeddingSearchService

            search_service = EmbeddingSearchService()
            results = search_service.search(request.query, top_k=request.top_k)

            documents = []
            for r in results:
                documents.append(SearchDocument(
                    content=r.get("content", r.get("text", "")),
                    source=r.get("source", "unknown"),
                    source_type=r.get("type", "unknown"),
                    relevance=r.get("score", 0.5)
                ))

            return SearchResponse(
                documents=documents,
                quality="ambiguous",
                total_found=len(documents)
            )

        except Exception as fallback_error:
            raise HTTPException(
                status_code=500,
                detail=f"검색 실패: {str(e)} / 폴백 실패: {str(fallback_error)}"
            )


@router.get(
    "/health",
    summary="검색 서비스 상태 확인"
)
async def search_health():
    """검색 서비스 상태 확인 (Dify Tool 연결 테스트용)"""
    return {
        "status": "healthy",
        "service": "DocScanner Legal Search API",
        "version": "1.0.0"
    }
