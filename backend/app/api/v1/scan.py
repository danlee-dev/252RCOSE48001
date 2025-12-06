"""
Quick Scan API for real-time contract risk detection

This API provides fast, lightweight analysis of contract images
captured from the camera. It uses OCR + keyword matching for
rapid risk detection (target: < 3 seconds).

Endpoints:
- POST /api/v1/scan/quick - Quick scan an image for risk clauses
"""

import time
import re
from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.user import User
from app.api.deps import get_current_user


router = APIRouter(prefix="/scan", tags=["Quick Scan"])


# Risk keywords for quick detection (Korean labor law focused)
RISK_KEYWORDS = {
    "HIGH": [
        # Wage-related
        ("위약금", "계약 해지 시 과도한 위약금"),
        ("벌금", "근로기준법 위반 가능성"),
        ("손해배상", "부당한 손해배상 조항"),
        ("최저임금", "최저임금 미달 위험"),
        ("무급", "무급 노동 강요"),
        ("수당 미지급", "법정 수당 미지급"),
        ("초과 근무", "초과 근무 수당 관련"),
        # Contract termination
        ("즉시 해고", "부당 해고 가능성"),
        ("일방적 해지", "일방적 계약 해지 조항"),
        ("무단 결근", "과도한 제재 조항"),
    ],
    "MEDIUM": [
        # Working conditions
        ("연차", "연차 사용 제한"),
        ("휴가", "휴가 사용 제한"),
        ("근무 시간", "근무 시간 조항 확인 필요"),
        ("야간 근무", "야간 근무 수당 확인"),
        ("주휴", "주휴 수당 관련"),
        # Benefits
        ("4대보험", "사회보험 가입 확인"),
        ("퇴직금", "퇴직금 지급 조건"),
        ("수습", "수습 기간 조건"),
        # Contract terms
        ("갱신", "계약 갱신 조건"),
        ("경쟁 금지", "경업 금지 조항"),
        ("비밀 유지", "과도한 비밀 유지 의무"),
    ],
    "LOW": [
        ("업무 범위", "업무 범위 명시 확인"),
        ("근무 장소", "근무 장소 변경 가능성"),
        ("복리후생", "복리후생 조건"),
    ],
}


class DetectedClause(BaseModel):
    text: str
    risk_level: str
    keyword: str
    bbox: Optional[dict] = None


class QuickScanResult(BaseModel):
    risk_level: str  # HIGH, MEDIUM, LOW, SAFE
    detected_clauses: list[DetectedClause]
    summary: str
    scan_time_ms: int


def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Extract text from image using OCR.

    In production, this would use:
    - Google Cloud Vision API
    - Tesseract OCR
    - or other OCR services

    For now, returns empty string (demo mode will use mock data).
    """
    try:
        # TODO: Implement actual OCR
        # Option 1: Google Cloud Vision
        # from google.cloud import vision
        # client = vision.ImageAnnotatorClient()
        # image = vision.Image(content=image_bytes)
        # response = client.text_detection(image=image)
        # return response.text_annotations[0].description if response.text_annotations else ""

        # Option 2: Tesseract (requires pytesseract + Tesseract installed)
        # import pytesseract
        # from PIL import Image
        # import io
        # image = Image.open(io.BytesIO(image_bytes))
        # return pytesseract.image_to_string(image, lang='kor')

        return ""
    except Exception as e:
        print(f">>> OCR error: {e}")
        return ""


def analyze_text_for_risks(text: str) -> tuple[list[DetectedClause], str]:
    """
    Analyze extracted text for risk keywords.
    Returns list of detected clauses and overall risk level.
    """
    detected = []

    # Search for each risk level's keywords
    for level, keywords in RISK_KEYWORDS.items():
        for keyword, description in keywords:
            # Find all occurrences
            pattern = re.compile(f".{{0,30}}{re.escape(keyword)}.{{0,30}}", re.IGNORECASE)
            matches = pattern.findall(text)

            for match in matches:
                detected.append(DetectedClause(
                    text=match.strip(),
                    risk_level=level,
                    keyword=keyword,
                    bbox=None  # Would be populated with actual coordinates from OCR
                ))

    # Determine overall risk level
    if any(c.risk_level == "HIGH" for c in detected):
        overall = "HIGH"
    elif any(c.risk_level == "MEDIUM" for c in detected):
        overall = "MEDIUM"
    elif detected:
        overall = "LOW"
    else:
        overall = "SAFE"

    return detected, overall


@router.post("/quick", response_model=QuickScanResult)
async def quick_scan(
    image: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """
    Perform quick scan on a contract image.

    This endpoint is optimized for speed (< 3 seconds target).
    It uses OCR + keyword matching to detect potential risk clauses.

    For comprehensive analysis, use the full contract upload flow.
    """
    start_time = time.time()

    # Validate file type
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="이미지 파일만 업로드 가능합니다."
        )

    # Read image bytes
    image_bytes = await image.read()

    # Extract text using OCR
    extracted_text = extract_text_from_image(image_bytes)

    # If OCR returned empty (demo mode), use mock analysis
    if not extracted_text:
        # Demo response for testing
        detected_clauses = [
            DetectedClause(
                text="계약 해지 시 위약금 300% 청구",
                risk_level="HIGH",
                keyword="위약금"
            ),
            DetectedClause(
                text="초과 근무 수당은 별도 협의",
                risk_level="HIGH",
                keyword="초과 근무"
            ),
            DetectedClause(
                text="연차 사용은 사전 승인 필수",
                risk_level="MEDIUM",
                keyword="연차"
            ),
        ]
        overall_risk = "HIGH"
        summary = f"{len(detected_clauses)}건의 주의가 필요한 조항이 발견되었습니다."
    else:
        # Analyze extracted text
        detected_clauses, overall_risk = analyze_text_for_risks(extracted_text)

        if detected_clauses:
            high_count = sum(1 for c in detected_clauses if c.risk_level == "HIGH")
            medium_count = sum(1 for c in detected_clauses if c.risk_level == "MEDIUM")

            if high_count > 0:
                summary = f"위험 조항 {high_count}건, 주의 조항 {medium_count}건이 발견되었습니다."
            else:
                summary = f"{len(detected_clauses)}건의 확인이 필요한 조항이 있습니다."
        else:
            summary = "주요 위험 조항이 발견되지 않았습니다."

    # Calculate scan time
    scan_time_ms = int((time.time() - start_time) * 1000)

    return QuickScanResult(
        risk_level=overall_risk,
        detected_clauses=detected_clauses,
        summary=summary,
        scan_time_ms=scan_time_ms
    )


@router.get("/keywords")
async def get_risk_keywords(
    current_user: User = Depends(get_current_user),
):
    """
    Get list of risk keywords used for quick scanning.

    This can be useful for:
    - Displaying to users what the scanner looks for
    - Client-side pre-filtering
    """
    return {
        "high_risk": [kw for kw, _ in RISK_KEYWORDS["HIGH"]],
        "medium_risk": [kw for kw, _ in RISK_KEYWORDS["MEDIUM"]],
        "low_risk": [kw for kw, _ in RISK_KEYWORDS["LOW"]],
    }
