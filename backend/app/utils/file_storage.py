"""
Multi-Format File Storage (Production-Grade)
- PDF, HWP, DOCX, TXT, 이미지 파일 저장 지원
- 파일 형식 검증 및 보안 체크
- UUID 기반 파일명 생성
"""

import os
import uuid
import mimetypes
from pathlib import Path
from typing import Set, Tuple, Optional
from fastapi import UploadFile, HTTPException
from app.core.config import settings


# 파일 저장 경로 설정
STORAGE_DIR = Path("storage/contracts")

# 파일 저장을 위한 초기화 (폴더 생성)
if not STORAGE_DIR.exists():
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)


# 지원 파일 형식 정의
SUPPORTED_FORMATS = {
    # 문서 형식
    "document": {
        "extensions": {".pdf", ".hwp", ".hwpx", ".docx", ".doc", ".txt", ".rtf", ".md"},
        "mime_types": {
            "application/pdf",
            "application/x-hwp",
            "application/haansofthwp",
            "application/vnd.hancom.hwp",
            "application/vnd.hancom.hwpx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
            "text/plain",
            "text/markdown",
            "application/rtf",
        }
    },
    # 이미지 형식
    "image": {
        "extensions": {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"},
        "mime_types": {
            "image/png",
            "image/jpeg",
            "image/gif",
            "image/webp",
            "image/bmp",
            "image/tiff",
        }
    }
}

# 전체 지원 확장자 및 MIME 타입
ALL_EXTENSIONS: Set[str] = set()
ALL_MIME_TYPES: Set[str] = set()

for category in SUPPORTED_FORMATS.values():
    ALL_EXTENSIONS.update(category["extensions"])
    ALL_MIME_TYPES.update(category["mime_types"])

# 최대 파일 크기 (50MB)
MAX_FILE_SIZE = 50 * 1024 * 1024

# 파일 시그니처 (매직 바이트)
FILE_SIGNATURES = {
    b'%PDF': '.pdf',
    b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1': '.hwp_or_doc',  # OLE compound
    b'PK\x03\x04': '.docx_or_hwpx',  # ZIP-based
    b'\x89PNG\r\n\x1a\n': '.png',
    b'\xff\xd8\xff': '.jpg',
    b'GIF87a': '.gif',
    b'GIF89a': '.gif',
    b'RIFF': '.webp',
    b'BM': '.bmp',
    b'II*\x00': '.tiff',
    b'MM\x00*': '.tiff',
}


def get_file_category(extension: str, mime_type: str = None) -> Optional[str]:
    """파일 카테고리 반환 (document 또는 image)"""
    ext_lower = extension.lower()

    for category, formats in SUPPORTED_FORMATS.items():
        if ext_lower in formats["extensions"]:
            return category
        if mime_type and mime_type in formats["mime_types"]:
            return category

    return None


def validate_file_signature(content: bytes, expected_extension: str) -> bool:
    """파일 시그니처 검증"""
    for signature, ext in FILE_SIGNATURES.items():
        if content.startswith(signature):
            # OLE compound (HWP 또는 DOC)
            if ext == '.hwp_or_doc':
                return expected_extension.lower() in ['.hwp', '.doc']
            # ZIP-based (DOCX 또는 HWPX)
            if ext == '.docx_or_hwpx':
                return expected_extension.lower() in ['.docx', '.hwpx']
            # 정확한 매칭
            if ext == expected_extension.lower():
                return True

    # 텍스트 파일은 시그니처 없음
    if expected_extension.lower() in ['.txt', '.md', '.rtf']:
        return True

    return False


def sanitize_filename(filename: str) -> str:
    """파일명 안전하게 처리"""
    # 경로 분리자 제거
    filename = filename.replace('/', '_').replace('\\', '_')
    # 위험한 문자 제거
    dangerous_chars = ['..', '~', '$', '`', '|', ';', '&', '>', '<', '*', '?', '"', "'"]
    for char in dangerous_chars:
        filename = filename.replace(char, '_')
    return filename


def get_accept_string() -> str:
    """HTML input accept 속성용 문자열 반환"""
    extensions = sorted(ALL_EXTENSIONS)
    mime_types = sorted(ALL_MIME_TYPES)
    return ",".join(extensions + list(mime_types))


async def validate_upload_file(file: UploadFile) -> Tuple[str, str, bytes]:
    """
    업로드 파일 검증

    Returns:
        Tuple[extension, mime_type, content]

    Raises:
        HTTPException: 검증 실패 시
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일명이 없습니다.")

    # 확장자 추출
    extension = Path(file.filename).suffix.lower()

    if not extension:
        raise HTTPException(status_code=400, detail="파일 확장자가 없습니다.")

    # 확장자 검증
    if extension not in ALL_EXTENSIONS:
        supported = ", ".join(sorted(ALL_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식입니다. 지원 형식: {supported}"
        )

    # MIME 타입 검증 (있는 경우)
    mime_type = file.content_type or mimetypes.guess_type(file.filename)[0] or ""

    # 파일 내용 읽기
    contents = await file.read()

    # 파일 크기 검증
    if len(contents) > MAX_FILE_SIZE:
        max_mb = MAX_FILE_SIZE / (1024 * 1024)
        raise HTTPException(
            status_code=400,
            detail=f"파일 크기가 너무 큽니다. 최대 {max_mb:.0f}MB까지 업로드 가능합니다."
        )

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="빈 파일은 업로드할 수 없습니다.")

    # 파일 시그니처 검증 (보안)
    if not validate_file_signature(contents, extension):
        raise HTTPException(
            status_code=400,
            detail="파일 형식이 올바르지 않습니다. 파일이 손상되었거나 확장자가 변조되었을 수 있습니다."
        )

    # 파일 포인터 리셋
    await file.seek(0)

    return extension, mime_type, contents


async def save_contract_file(user_id: int, file: UploadFile) -> str:
    """
    업로드된 파일을 로컬 저장소에 저장하고 파일 URL을 반환합니다.

    지원 형식:
    - 문서: PDF, HWP, HWPX, DOCX, DOC, TXT, RTF, MD
    - 이미지: PNG, JPG, JPEG, GIF, WEBP, BMP, TIFF

    Returns:
        파일 URL (예: /storage/contracts/1/abc123.pdf)

    Raises:
        HTTPException: 파일 검증 실패 또는 저장 오류 시
    """
    # 파일 검증
    extension, mime_type, contents = await validate_upload_file(file)

    # 사용자별 폴더 생성
    user_storage_path = STORAGE_DIR / str(user_id)
    if not user_storage_path.exists():
        user_storage_path.mkdir(parents=True, exist_ok=True)

    # UUID 기반 안전한 파일명 생성
    original_name = sanitize_filename(Path(file.filename).stem)
    unique_id = uuid.uuid4().hex[:8]
    safe_filename = f"{original_name}_{unique_id}{extension}"

    file_path = user_storage_path / safe_filename

    # 파일 저장
    try:
        with open(file_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 저장 중 오류 발생: {str(e)}")

    # 클라이언트가 접근할 수 있는 URL 반환
    return f"/storage/contracts/{user_id}/{safe_filename}"


def delete_contract_file(file_url: str):
    """
    저장된 파일을 삭제합니다.
    file_url: "/storage/contracts/1/contract.pdf" 형태
    """
    if not file_url:
        return

    # 1. 상대 경로 파싱
    relative_path = file_url.lstrip("/")

    # 2. backend 루트 기준 경로 찾기
    backend_root = Path(__file__).parent.parent.parent
    file_path = backend_root / relative_path

    # 3. 파일 삭제
    if file_path.exists():
        try:
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        except OSError as e:
            print(f"Error deleting file: {e}")
    else:
        print(f"File not found for deletion: {file_path}")


def get_file_info(file_url: str) -> dict:
    """파일 정보 반환"""
    if not file_url:
        return {}

    relative_path = file_url.lstrip("/")
    backend_root = Path(__file__).parent.parent.parent
    file_path = backend_root / relative_path

    if not file_path.exists():
        return {"exists": False}

    extension = file_path.suffix.lower()
    category = get_file_category(extension)

    return {
        "exists": True,
        "filename": file_path.name,
        "extension": extension,
        "category": category,
        "size": file_path.stat().st_size,
        "path": str(file_path)
    }


def get_supported_formats_info() -> dict:
    """지원 형식 정보 반환 (API 응답용)"""
    return {
        "document": {
            "extensions": sorted(SUPPORTED_FORMATS["document"]["extensions"]),
            "description": "PDF, HWP, HWPX, DOCX, DOC, TXT, RTF, MD"
        },
        "image": {
            "extensions": sorted(SUPPORTED_FORMATS["image"]["extensions"]),
            "description": "PNG, JPG, JPEG, GIF, WEBP, BMP, TIFF"
        },
        "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024),
        "accept_string": get_accept_string()
    }
