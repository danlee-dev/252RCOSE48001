"""
Quick Scan API for real-time contract risk detection

This API provides fast, lightweight analysis of contract images
captured from the camera. Supports two modes:
- LLM mode: Gemini Vision for OCR + contextual risk analysis (SCAN_USE_LLM=true)
- Keyword mode: OCR + keyword matching for faster response (SCAN_USE_LLM=false)

Target: < 5 seconds total response time

Endpoints:
- POST /api/v1/scan/quick - Quick scan an image for risk clauses
"""

import os
import re
import time
import json
from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.models.user import User
from app.api.deps import get_current_user


router = APIRouter(tags=["Quick Scan"])

# Environment variable to toggle LLM mode (default: true)
SCAN_USE_LLM = os.getenv("SCAN_USE_LLM", "true").lower() == "true"

# Gemini Vision 클라이언트 (lazy loading)
_gemini_model = None
_ocr_model = None


def get_gemini_model():
    """Gemini 모델 (LLM 분석용) - 환경변수에서 모델명 로드"""
    global _gemini_model
    if _gemini_model is None:
        try:
            import google.generativeai as genai
            genai.configure(api_key=settings.GEMINI_API_KEY)
            model_name = settings.LLM_SCAN_MODEL
            print(f">>> Quick Scan LLM model: {model_name}")
            _gemini_model = genai.GenerativeModel(model_name)
        except Exception as e:
            print(f">>> Gemini init error: {e}")
            return None
    return _gemini_model


def get_ocr_model():
    """Gemini 모델 (OCR 전용)"""
    global _ocr_model
    if _ocr_model is None:
        try:
            import google.generativeai as genai
            genai.configure(api_key=settings.GEMINI_API_KEY)
            model_name = settings.LLM_SCAN_MODEL
            print(f">>> Quick Scan OCR model: {model_name}")
            _ocr_model = genai.GenerativeModel(model_name)
        except Exception as e:
            print(f">>> Gemini OCR init error: {e}")
            return None
    return _ocr_model


# Risk keywords for keyword-based detection
RISK_KEYWORDS = {
    "HIGH": [
        ("위약금", "계약 해지 시 과도한 위약금"),
        ("벌금", "근로기준법 위반 가능성"),
        ("손해배상", "부당한 손해배상 조항"),
        ("최저임금", "최저임금 미달 위험"),
        ("무급", "무급 노동 강요"),
        ("수당 미지급", "법정 수당 미지급"),
        ("초과 근무", "초과 근무 수당 관련"),
        ("즉시 해고", "부당 해고 가능성"),
        ("일방적 해지", "일방적 계약 해지 조항"),
        ("무단 결근", "과도한 제재 조항"),
    ],
    "MEDIUM": [
        ("연차", "연차 사용 제한"),
        ("휴가", "휴가 사용 제한"),
        ("근무 시간", "근무 시간 조항 확인 필요"),
        ("야간 근무", "야간 근무 수당 확인"),
        ("주휴", "주휴 수당 관련"),
        ("4대보험", "사회보험 가입 확인"),
        ("퇴직금", "퇴직금 지급 조건"),
        ("수습", "수습 기간 조건"),
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


# LLM prompt for contextual analysis
SCAN_PROMPT = """당신은 한국 근로기준법 전문가입니다. 이 계약서 이미지를 분석하세요.

## 작업
1. 이미지에서 텍스트를 추출합니다
2. 근로자에게 불리하거나 확인이 필요한 모든 조항을 식별합니다

## 위험 판단 기준
- HIGH: 근로기준법 위반 가능성 (위약금, 손해배상, 최저임금 미달, 부당해고 등)
- MEDIUM: 확인 필요 (연차/휴가 제한, 포괄임금제, 경쟁금지, 수습기간 등)
- LOW: 참고 사항 (업무범위, 근무장소 변경 등)

## 중요
- 의심되는 조항은 모두 보고하세요 (확실하지 않아도 됨)
- "위약금 없음", "손해배상 청구 안함" 같은 긍정적 문구는 제외
- 각 조항은 개별적으로 분리해서 보고하세요
- clause_number는 계약서에 표시된 조항 번호 (예: "10. 손해배상" -> 10)

## 출력 (JSON)
{"clauses": [{"clause_number": 10, "text": "조항 원문", "risk_level": "HIGH", "reason": "이유"}]}"""


class DetectedClause(BaseModel):
    text: str
    risk_level: str
    reason: str
    clause_number: Optional[int] = None  # 조항 번호 (예: 10조 -> 10)
    keyword: Optional[str] = None
    bbox: Optional[dict] = None


class QuickScanResult(BaseModel):
    risk_level: str  # HIGH, MEDIUM, LOW, SAFE
    detected_clauses: list[DetectedClause]
    summary: str
    scan_time_ms: int
    mode: str  # "llm" or "keyword"


# ============= LLM Mode Functions =============

def analyze_image_with_llm(image_bytes: bytes) -> list[DetectedClause]:
    """
    Single LLM call for OCR + Risk Analysis
    맥락을 이해하여 false positive 방지
    """
    try:
        import PIL.Image
        import io

        model = get_gemini_model()
        if model is None:
            print(">>> Gemini model not available")
            return []

        image = PIL.Image.open(io.BytesIO(image_bytes))

        response = model.generate_content(
            [SCAN_PROMPT, image],
            generation_config={
                "temperature": 0,
                "response_mime_type": "application/json",
            }
        )

        result_text = response.text.strip()
        print(f">>> LLM response: {result_text}")

        # JSON 파싱
        clauses = []
        try:
            result = json.loads(result_text)
            clauses = result.get("clauses", [])
        except json.JSONDecodeError:
            # JSON 블록 추출 시도
            json_match = re.search(r'\{[\s\S]*"clauses"[\s\S]*\}', result_text)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    clauses = result.get("clauses", [])
                except json.JSONDecodeError:
                    print(f">>> Failed to parse extracted JSON")

            # 배열만 있는 경우
            if not clauses:
                array_match = re.search(r'\[[\s\S]*\]', result_text)
                if array_match:
                    try:
                        clauses = json.loads(array_match.group())
                    except json.JSONDecodeError:
                        print(f">>> Failed to parse JSON array")

        if not clauses:
            print(f">>> No clauses parsed from response")
            return []

        detected = []
        for clause in clauses:
            if not isinstance(clause, dict):
                continue
            if not clause.get("text") or not clause.get("risk_level"):
                continue
            detected.append(DetectedClause(
                text=clause["text"],
                risk_level=clause["risk_level"].upper(),
                reason=clause.get("reason", ""),
                clause_number=clause.get("clause_number"),
                keyword=None,
                bbox=None
            ))

        print(f">>> LLM detected {len(detected)} risk clauses")
        return detected

    except Exception as e:
        print(f">>> LLM analysis error: {e}")
        import traceback
        traceback.print_exc()
        return []


# ============= Keyword Mode Functions =============

def extract_text_from_image(image_bytes: bytes) -> str:
    """Gemini Vision OCR (텍스트만 추출)"""
    try:
        import PIL.Image
        import io

        model = get_ocr_model()
        if model is None:
            print(">>> Gemini OCR model not available")
            return ""

        image = PIL.Image.open(io.BytesIO(image_bytes))

        prompt = """이 이미지에서 모든 텍스트를 추출하세요.
텍스트만 출력하세요. 설명이나 분석 없이 보이는 텍스트 그대로 추출하세요."""

        response = model.generate_content(
            [prompt, image],
            generation_config={
                "max_output_tokens": 2000,
                "temperature": 0,
            }
        )

        extracted_text = response.text.strip()
        print(f">>> OCR extracted {len(extracted_text)} chars")
        return extracted_text

    except Exception as e:
        print(f">>> OCR error: {e}")
        return ""


def analyze_text_for_risks(text: str) -> list[DetectedClause]:
    """Keyword-based risk detection"""
    detected = []

    for level, keywords in RISK_KEYWORDS.items():
        for keyword, description in keywords:
            pattern = re.compile(f".{{0,30}}{re.escape(keyword)}.{{0,30}}", re.IGNORECASE)
            matches = pattern.findall(text)

            for match in matches:
                detected.append(DetectedClause(
                    text=match.strip(),
                    risk_level=level,
                    reason=description,
                    keyword=keyword,
                    bbox=None
                ))

    print(f">>> Keyword detected {len(detected)} risk clauses")
    return detected


# ============= API Endpoint =============

@router.post("/quick", response_model=QuickScanResult)
async def quick_scan(
    image: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """
    Gemini Vision을 사용한 실시간 계약서 스캔

    모드 선택 (SCAN_USE_LLM 환경변수):
    - true: LLM 기반 맥락 분석 (정확도 높음)
    - false: 키워드 매칭 (속도 빠름)

    속도 목표: < 5초
    """
    start_time = time.time()

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="이미지 파일만 업로드 가능합니다."
        )

    image_bytes = await image.read()
    image_size_mb = len(image_bytes) / (1024 * 1024)
    print(f">>> Quick scan: image size {image_size_mb:.2f}MB, mode={'LLM' if SCAN_USE_LLM else 'Keyword'}")

    # Choose analysis mode
    if SCAN_USE_LLM:
        detected_clauses = analyze_image_with_llm(image_bytes)
        mode = "llm"
    else:
        extracted_text = extract_text_from_image(image_bytes)
        if extracted_text:
            detected_clauses = analyze_text_for_risks(extracted_text)
        else:
            detected_clauses = []
        mode = "keyword"

    # Determine overall risk level
    if any(c.risk_level == "HIGH" for c in detected_clauses):
        overall_risk = "HIGH"
    elif any(c.risk_level == "MEDIUM" for c in detected_clauses):
        overall_risk = "MEDIUM"
    elif detected_clauses:
        overall_risk = "LOW"
    else:
        overall_risk = "SAFE"

    # Generate summary
    if detected_clauses:
        high_count = sum(1 for c in detected_clauses if c.risk_level == "HIGH")
        medium_count = sum(1 for c in detected_clauses if c.risk_level == "MEDIUM")
        low_count = sum(1 for c in detected_clauses if c.risk_level == "LOW")

        if high_count > 0:
            summary = f"위험 조항 {high_count}건이 발견되었습니다. 정밀 분석을 권장합니다."
        elif medium_count > 0:
            summary = f"확인이 필요한 조항 {medium_count}건이 발견되었습니다."
        else:
            summary = f"{low_count}건의 참고 사항이 있습니다."
    else:
        summary = "주요 위험 조항이 발견되지 않았습니다."

    scan_time_ms = int((time.time() - start_time) * 1000)
    print(f">>> Quick scan completed in {scan_time_ms}ms, found {len(detected_clauses)} clauses")

    return QuickScanResult(
        risk_level=overall_risk,
        detected_clauses=detected_clauses,
        summary=summary,
        scan_time_ms=scan_time_ms,
        mode=mode
    )


@router.get("/keywords")
async def get_risk_keywords(
    current_user: User = Depends(get_current_user),
):
    """
    Get list of risk keywords used for quick scanning.
    """
    return {
        "high_risk": [kw for kw, _ in RISK_KEYWORDS["HIGH"]],
        "medium_risk": [kw for kw, _ in RISK_KEYWORDS["MEDIUM"]],
        "low_risk": [kw for kw, _ in RISK_KEYWORDS["LOW"]],
    }
