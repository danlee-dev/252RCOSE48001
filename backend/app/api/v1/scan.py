"""
Quick Scan API for real-time contract risk detection

This API provides fast, lightweight analysis of contract images
captured from the camera. It uses Gemini Vision OCR + keyword matching
for rapid risk detection (target: < 3 seconds).

Endpoints:
- POST /api/v1/scan/quick - Quick scan an image for risk clauses
"""

import time
import re
import base64
import asyncio
from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.config import settings
from app.models.user import User
from app.api.deps import get_current_user


router = APIRouter(prefix="/scan", tags=["Quick Scan"])

# Gemini Vision 클라이언트 (lazy loading)
_gemini_model = None

def get_gemini_model():
    """Gemini Flash 모델 가져오기 (속도 최적화)"""
    global _gemini_model
    if _gemini_model is None:
        try:
            import google.generativeai as genai
            genai.configure(api_key=settings.GEMINI_API_KEY)
            # 환경 변수에서 모델명 읽기 (기본값: gemini-2.5-flash)
            model_name = settings.LLM_SCAN_MODEL
            print(f">>> Quick Scan using model: {model_name}")
            _gemini_model = genai.GenerativeModel(model_name)
        except Exception as e:
            print(f">>> Gemini init error: {e}")
            return None
    return _gemini_model


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
    Gemini Vision을 사용한 빠른 OCR

    속도 최적화:
    - Gemini Flash 모델 사용
    - 간단한 프롬프트로 텍스트만 추출
    - 구조화 없이 raw text만 반환
    """
    try:
        import PIL.Image
        import io

        model = get_gemini_model()
        if model is None:
            print(">>> Gemini model not available")
            return ""

        # 이미지 로드
        image = PIL.Image.open(io.BytesIO(image_bytes))

        # 빠른 OCR 프롬프트 (최소한의 지시만)
        prompt = """이 이미지에서 모든 텍스트를 추출하세요.
텍스트만 출력하세요. 설명이나 분석 없이 보이는 텍스트 그대로 추출하세요."""

        # Gemini Vision 호출
        response = model.generate_content(
            [prompt, image],
            generation_config={
                "max_output_tokens": 2000,  # 속도를 위해 제한
                "temperature": 0,  # 일관성을 위해 0
            }
        )

        extracted_text = response.text.strip()
        print(f">>> OCR extracted {len(extracted_text)} chars")
        return extracted_text

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
    Gemini Vision을 사용한 실시간 계약서 스캔

    속도 목표: < 3초
    1. Gemini Flash로 빠른 OCR
    2. 키워드 기반 위험도 판정
    3. 정밀 분석 연결 가능
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

    # 이미지 크기 체크 (너무 크면 리사이즈 필요할 수 있음)
    image_size_mb = len(image_bytes) / (1024 * 1024)
    print(f">>> Quick scan: image size {image_size_mb:.2f}MB")

    # Gemini Vision OCR
    extracted_text = extract_text_from_image(image_bytes)

    # OCR 결과 분석
    if extracted_text:
        # 텍스트가 추출됨 - 키워드 분석 수행
        detected_clauses, overall_risk = analyze_text_for_risks(extracted_text)

        if detected_clauses:
            high_count = sum(1 for c in detected_clauses if c.risk_level == "HIGH")
            medium_count = sum(1 for c in detected_clauses if c.risk_level == "MEDIUM")
            low_count = sum(1 for c in detected_clauses if c.risk_level == "LOW")

            if high_count > 0:
                summary = f"위험 조항 {high_count}건, 주의 조항 {medium_count}건이 발견되었습니다. 정밀 분석을 권장합니다."
            elif medium_count > 0:
                summary = f"주의가 필요한 조항 {medium_count}건이 발견되었습니다."
            else:
                summary = f"{low_count}건의 확인 사항이 있습니다."
        else:
            summary = "주요 위험 키워드가 발견되지 않았습니다. 안전해 보이지만 정밀 분석을 권장합니다."
    else:
        # OCR 실패 또는 텍스트 없음
        # 문서가 아닌 이미지일 수 있음
        detected_clauses = []
        overall_risk = "SAFE"
        summary = "텍스트를 인식하지 못했습니다. 계약서가 선명하게 보이도록 다시 촬영해주세요."

    # Calculate scan time
    scan_time_ms = int((time.time() - start_time) * 1000)
    print(f">>> Quick scan completed in {scan_time_ms}ms, found {len(detected_clauses)} clauses")

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
