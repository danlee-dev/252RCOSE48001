"""
Contract Analysis Tasks (Multi-Format Support)
- PDF, HWP, DOCX, TXT, 이미지 파일 분석 지원
- Celery 비동기 작업 처리
- Advanced AI Pipeline 또는 Dify API 연동
- Cloudflare R2 스토리지 지원 (분산 환경)
"""

import os
import sys
import asyncio
import json
from sqlalchemy.future import select
from app.core.database import AsyncSessionLocal
from app.models.contract import Contract
from app.core.celery_app import celery_app
from app.core.r2_storage import r2_storage
import requests

# Worker 프로세스 내에서 app 모듈 경로를 찾도록 설정
from pathlib import Path
backend_dir = Path(__file__).parent.parent.parent.resolve()
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# 전처리기 및 고급 AI 파이프라인 import
from app.ai.preprocessor import ContractPreprocessor
from app.ai.pipeline import AdvancedAIPipeline, PipelineConfig
from app.ai.vision_parser import VisionParser
from app.ai.document_parser import MultiFormatDocumentParser, DocumentType


def resolve_file_path(file_url: str) -> tuple[str, str]:
    """
    file_url을 실제 파일 경로로 변환

    Args:
        file_url: DB에 저장된 파일 URL
            - R2: "contracts/1/file.pdf"
            - Local: "/storage/contracts/1/file.pdf"

    Returns:
        tuple(file_path, temp_file_path or None)
        - temp_file_path가 있으면 작업 완료 후 삭제 필요
    """
    # R2 object key인 경우 (분산 환경)
    if file_url.startswith("contracts/") and r2_storage.enabled:
        temp_path = r2_storage.download_to_temp_file(file_url)
        if temp_path:
            return temp_path, temp_path  # 두 번째 값이 있으면 삭제 대상
        else:
            raise Exception(f"R2에서 파일 다운로드 실패: {file_url}")

    # 로컬 파일 경로인 경우
    relative_path = file_url.lstrip("/")
    file_path = backend_dir / relative_path
    return str(file_path), None


def cleanup_temp_file(temp_path: str):
    """임시 파일 삭제"""
    if temp_path and os.path.exists(temp_path):
        try:
            os.unlink(temp_path)
            print(f"[Cleanup] Deleted temp file: {temp_path}")
        except Exception as e:
            print(f"[Cleanup] Failed to delete temp file: {e}")


def get_file_extension(file_url: str) -> str:
    """파일 URL에서 확장자 추출"""
    return Path(file_url).suffix.lower()


def is_image_file(extension: str) -> bool:
    """이미지 파일 여부 확인"""
    return extension in {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff', '.tif'}


def is_pdf_file(extension: str) -> bool:
    """PDF 파일 여부 확인"""
    return extension == '.pdf'


def extract_text_from_file(file_path: str, extension: str) -> str:
    """
    파일 형식에 따른 텍스트 추출

    Args:
        file_path: 파일 경로
        extension: 파일 확장자

    Returns:
        추출된 텍스트
    """
    parser = MultiFormatDocumentParser()

    # 이미지 파일은 Vision API 직접 사용
    if is_image_file(extension):
        vision_parser = VisionParser()
        result = vision_parser.parse_image(file_path, extract_tables=True)
        return result.raw_text

    # PDF는 pdfplumber + Vision OCR fallback
    if is_pdf_file(extension):
        processor = ContractPreprocessor()
        text = processor.extract_text(file_path)

        # 텍스트가 부족하면 Vision OCR 사용
        if not text or len(text.strip()) < 100:
            try:
                vision_parser = VisionParser()
                ocr_text = vision_parser.extract_text_from_pdf(file_path)
                if len(ocr_text) > len(text or ""):
                    return ocr_text
            except Exception as e:
                print(f"Vision OCR fallback failed: {e}")

        return text

    # 기타 문서 형식 (HWP, DOCX, TXT 등)
    result = parser.parse(file_path)

    if result.is_empty:
        print(f"Warning: Empty or insufficient text from {extension} file")
        # OCR fallback 시도
        if result.warnings:
            print(f"Parser warnings: {result.warnings}")

    return result.text


@celery_app.task(name="analyze_contract")
def analyze_contract_task(contract_id: int, use_advanced_pipeline: bool = True):
    """
    Celery Task:
    1. 멀티포맷 문서 파싱 (PDF, HWP, DOCX, TXT, 이미지)
    2. Advanced AI Pipeline 또는 Dify API 호출
    3. 결과 파싱 및 DB 저장 (JSONB)

    지원 파일 형식:
    - 문서: PDF, HWP, HWPX, DOCX, DOC, TXT, RTF, MD
    - 이미지: PNG, JPG, JPEG, GIF, WEBP, BMP, TIFF (Vision API OCR)
    """

    async def run_analysis():
        async with AsyncSessionLocal() as db:
            stmt = select(Contract).where(Contract.id == contract_id)
            result = await db.execute(stmt)
            contract = result.scalar_one_or_none()

            if not contract:
                print(f"Error: Contract {contract_id} not found.")
                return

            print(f"[{contract_id}] Processing START for {contract.title}...")

            contract.status = "PROCESSING"
            await db.commit()

            temp_file = None
            try:
                # 파일 경로 해결 (R2 또는 로컬)
                file_path, temp_file = resolve_file_path(contract.file_url)
                extension = get_file_extension(contract.file_url)

                print(f"[{contract_id}] File type: {extension}")
                print(f"[{contract_id}] File path: {file_path}")

                # 텍스트 추출 (형식별 분기 처리)
                full_text = extract_text_from_file(file_path, extension)

                if not full_text or len(full_text.strip()) < 10:
                    raise Exception(f"텍스트 추출 실패: 내용이 너무 적습니다 ({len(full_text) if full_text else 0}자)")

                print(f"[{contract_id}] Extracted text length: {len(full_text)} characters")

                # 청킹 (텍스트 분할)
                processor = ContractPreprocessor()
                chunks = processor.chunk_text(full_text)
                print(f"[{contract_id}] Text chunks: {len(chunks)}")

                # AI 분석 수행
                if use_advanced_pipeline:
                    # Advanced AI Pipeline 사용
                    print(f"[{contract_id}] Using Advanced AI Pipeline...")

                    config = PipelineConfig(
                        enable_pii_masking=True,
                        enable_hyde=True,
                        enable_raptor=True,
                        enable_crag=True,
                        enable_constitutional_ai=True,
                        enable_stress_test=True,
                        enable_redlining=True,
                        enable_judge=True,
                        enable_reasoning_trace=True,
                        enable_dspy=True
                    )

                    pipeline = AdvancedAIPipeline(config=config)
                    pipeline_result = pipeline.analyze(
                        contract_text=full_text,
                        contract_id=str(contract_id),
                        file_path=str(file_path)
                    )

                    # 결과 저장
                    contract.status = "COMPLETED"
                    contract.extracted_text = full_text
                    contract.analysis_result = pipeline_result.to_dict()
                    contract.risk_level = pipeline_result.risk_level

                    print(f"[{contract_id}] Advanced Pipeline completed in {pipeline_result.processing_time:.2f}s")
                    print(f"[{contract_id}] Risk: {pipeline_result.risk_level} ({pipeline_result.risk_score:.2%})")

                else:
                    # Legacy: Dify API 호출
                    DIFY_API_URL = os.getenv("DIFY_API_URL")
                    DIFY_API_KEY = os.getenv("DIFY_API_KEY")

                    payload = {
                        "inputs": {
                            "contract_text": full_text,
                            "file_url": contract.file_url
                        },
                        "query": "이 계약서의 위험 조항을 분석해줘.",
                        "user": str(contract.user_id),
                        "response_mode": "blocking"
                    }

                    headers = {"Authorization": f"Bearer {DIFY_API_KEY}", "Content-Type": "application/json"}

                    print(f"[{contract_id}] Calling Dify API...")
                    response = requests.post(DIFY_API_URL, headers=headers, json=payload, timeout=300)
                    response.raise_for_status()

                    dify_response = response.json()

                    contract.status = "COMPLETED"
                    contract.extracted_text = full_text
                    contract.analysis_result = dify_response
                    contract.risk_level = "Check"

                await db.commit()
                print(f"[{contract_id}] Analysis COMPLETED successfully.")

            except Exception as e:
                contract.status = "FAILED"
                contract.analysis_result = {"error": str(e)}
                print(f"[{contract_id}] Error: {e}")
                await db.commit()

            finally:
                # 임시 파일 정리 (R2에서 다운로드한 경우)
                cleanup_temp_file(temp_file)

    # Windows 환경 호환성 코드
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(run_analysis())


@celery_app.task(name="analyze_contract_quick")
def analyze_contract_quick_task(contract_id: int):
    """
    Celery Task (Quick Mode):
    빠른 분석을 위한 간소화된 파이프라인 (RAPTOR, Reasoning Trace 비활성화)

    지원 파일 형식:
    - 문서: PDF, HWP, HWPX, DOCX, DOC, TXT, RTF, MD
    - 이미지: PNG, JPG, JPEG, GIF, WEBP, BMP, TIFF
    """

    async def run_quick_analysis():
        async with AsyncSessionLocal() as db:
            stmt = select(Contract).where(Contract.id == contract_id)
            result = await db.execute(stmt)
            contract = result.scalar_one_or_none()

            if not contract:
                print(f"Error: Contract {contract_id} not found.")
                return

            print(f"[{contract_id}] Quick Analysis START for {contract.title}...")

            contract.status = "PROCESSING"
            await db.commit()

            temp_file = None
            try:
                # 파일 경로 해결 (R2 또는 로컬)
                file_path, temp_file = resolve_file_path(contract.file_url)
                extension = get_file_extension(contract.file_url)

                print(f"[{contract_id}] File type: {extension}")

                # 텍스트 추출
                full_text = extract_text_from_file(file_path, extension)

                if not full_text or len(full_text.strip()) < 10:
                    raise Exception("텍스트 추출 실패")

                print(f"[{contract_id}] Extracted: {len(full_text)} characters")

                # Quick 모드: 일부 기능 비활성화
                config = PipelineConfig(
                    enable_pii_masking=True,
                    enable_hyde=True,
                    enable_raptor=False,  # 비활성화
                    enable_crag=True,
                    enable_constitutional_ai=True,
                    enable_stress_test=True,
                    enable_redlining=True,
                    enable_judge=True,
                    enable_reasoning_trace=False,  # 비활성화
                    enable_dspy=False  # 비활성화
                )

                pipeline = AdvancedAIPipeline(config=config)
                pipeline_result = pipeline.analyze(
                    contract_text=full_text,
                    contract_id=str(contract_id)
                )

                contract.status = "COMPLETED"
                contract.extracted_text = full_text
                contract.analysis_result = pipeline_result.to_dict()
                contract.risk_level = pipeline_result.risk_level

                await db.commit()
                print(f"[{contract_id}] Quick Analysis COMPLETED in {pipeline_result.processing_time:.2f}s")

            except Exception as e:
                contract.status = "FAILED"
                contract.analysis_result = {"error": str(e)}
                print(f"[{contract_id}] Error: {e}")
                await db.commit()

            finally:
                cleanup_temp_file(temp_file)

    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(run_quick_analysis())


@celery_app.task(name="analyze_image_contract")
def analyze_image_contract_task(contract_id: int):
    """
    Celery Task (Image-Only Mode):
    이미지 파일 전용 분석 (Vision API로 텍스트 및 구조 추출)

    지원 형식: PNG, JPG, JPEG, GIF, WEBP, BMP, TIFF
    """

    async def run_image_analysis():
        async with AsyncSessionLocal() as db:
            stmt = select(Contract).where(Contract.id == contract_id)
            result = await db.execute(stmt)
            contract = result.scalar_one_or_none()

            if not contract:
                print(f"Error: Contract {contract_id} not found.")
                return

            print(f"[{contract_id}] Image Analysis START for {contract.title}...")

            contract.status = "PROCESSING"
            await db.commit()

            temp_file = None
            try:
                # 파일 경로 해결 (R2 또는 로컬)
                file_path, temp_file = resolve_file_path(contract.file_url)
                extension = get_file_extension(contract.file_url)

                if not is_image_file(extension):
                    raise Exception(f"이미지 파일이 아닙니다: {extension}")

                # Vision API로 구조화된 파싱
                vision_parser = VisionParser()
                parse_result = vision_parser.parse_image(file_path, extract_tables=True)

                full_text = parse_result.raw_text
                structured_markdown = parse_result.structured_markdown
                tables = parse_result.tables

                if not full_text or len(full_text.strip()) < 10:
                    raise Exception("이미지에서 텍스트를 추출할 수 없습니다")

                print(f"[{contract_id}] Vision OCR extracted: {len(full_text)} characters")
                print(f"[{contract_id}] Tables found: {len(tables)}")

                # AI 파이프라인 분석
                config = PipelineConfig(
                    enable_pii_masking=True,
                    enable_hyde=True,
                    enable_raptor=False,  # 이미지는 단일 페이지이므로 RAPTOR 불필요
                    enable_crag=True,
                    enable_constitutional_ai=True,
                    enable_stress_test=True,
                    enable_redlining=True,
                    enable_judge=True,
                    enable_reasoning_trace=True,
                    enable_dspy=True
                )

                pipeline = AdvancedAIPipeline(config=config)
                pipeline_result = pipeline.analyze(
                    contract_text=full_text,
                    contract_id=str(contract_id)
                )

                # 이미지 특화 메타데이터 추가
                analysis_result = pipeline_result.to_dict()
                analysis_result["image_analysis"] = {
                    "structured_markdown": structured_markdown,
                    "tables": tables,
                    "checkboxes": parse_result.checkboxes,
                    "signatures": parse_result.signatures
                }

                contract.status = "COMPLETED"
                contract.extracted_text = full_text
                contract.analysis_result = analysis_result
                contract.risk_level = pipeline_result.risk_level

                await db.commit()
                print(f"[{contract_id}] Image Analysis COMPLETED in {pipeline_result.processing_time:.2f}s")

            except Exception as e:
                contract.status = "FAILED"
                contract.analysis_result = {"error": str(e)}
                print(f"[{contract_id}] Error: {e}")
                await db.commit()

            finally:
                cleanup_temp_file(temp_file)

    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(run_image_analysis())
