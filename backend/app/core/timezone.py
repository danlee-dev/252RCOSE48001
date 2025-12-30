"""
Timezone Utility (Extensible)
- 기본: 한국 시간(KST, UTC+9)
- 확장: 사용자별 timezone 설정 지원 가능
"""

from datetime import datetime, timezone, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

# 지원하는 타임존 목록 (확장 가능)
SUPPORTED_TIMEZONES = {
    "Asia/Seoul": "한국 (KST, UTC+9)",
    "Asia/Tokyo": "일본 (JST, UTC+9)",
    "Asia/Shanghai": "중국 (CST, UTC+8)",
    "Asia/Singapore": "싱가포르 (SGT, UTC+8)",
    "America/New_York": "미국 동부 (EST/EDT)",
    "America/Los_Angeles": "미국 서부 (PST/PDT)",
    "Europe/London": "영국 (GMT/BST)",
    "UTC": "UTC",
}

# 기본 타임존 설정
DEFAULT_TIMEZONE = "Asia/Seoul"

# 한국 표준시 (레거시 호환용)
KST = timezone(timedelta(hours=9))
UTC = timezone.utc


def get_timezone(tz_name: Optional[str] = None) -> ZoneInfo:
    """
    타임존 객체 반환

    Args:
        tz_name: 타임존 이름 (예: "Asia/Seoul", "America/New_York")
                 None이면 기본값(Asia/Seoul) 사용

    Returns:
        ZoneInfo 객체
    """
    tz = tz_name or DEFAULT_TIMEZONE
    try:
        return ZoneInfo(tz)
    except Exception:
        return ZoneInfo(DEFAULT_TIMEZONE)


def now(tz_name: Optional[str] = None) -> datetime:
    """
    지정된 타임존의 현재 시간 반환

    Args:
        tz_name: 타임존 이름 (None이면 기본값 사용)

    Returns:
        timezone-aware datetime
    """
    return datetime.now(get_timezone(tz_name))


def now_kst() -> datetime:
    """현재 한국 시간 반환 (timezone-aware)"""
    return datetime.now(get_timezone("Asia/Seoul"))


def now_utc() -> datetime:
    """현재 UTC 시간 반환 (timezone-aware)"""
    return datetime.now(UTC)


def to_timezone(dt: Optional[datetime], tz_name: Optional[str] = None) -> Optional[datetime]:
    """
    datetime을 지정된 타임존으로 변환

    Args:
        dt: 변환할 datetime
        tz_name: 대상 타임존 (None이면 기본값 사용)

    Returns:
        변환된 timezone-aware datetime
    """
    if dt is None:
        return None

    # naive datetime이면 UTC로 가정
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)

    return dt.astimezone(get_timezone(tz_name))


def to_kst(dt: Optional[datetime]) -> Optional[datetime]:
    """datetime을 한국 시간으로 변환"""
    return to_timezone(dt, "Asia/Seoul")


def to_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """datetime을 UTC로 변환"""
    if dt is None:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=get_timezone())

    return dt.astimezone(UTC)


def format_datetime(
    dt: Optional[datetime],
    fmt: str = "%Y-%m-%d %H:%M:%S",
    tz_name: Optional[str] = None
) -> str:
    """
    datetime을 지정된 타임존의 문자열로 포맷팅

    Args:
        dt: datetime 객체
        fmt: 출력 포맷
        tz_name: 타임존 (None이면 기본값 사용)

    Returns:
        포맷된 문자열 또는 빈 문자열
    """
    if dt is None:
        return ""

    converted = to_timezone(dt, tz_name)
    return converted.strftime(fmt)


def format_kst(dt: Optional[datetime], fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """datetime을 한국 시간 문자열로 포맷팅"""
    return format_datetime(dt, fmt, "Asia/Seoul")


def format_relative(dt: Optional[datetime], tz_name: Optional[str] = None) -> str:
    """
    상대적 시간 표시 (예: "3분 전", "2시간 전", "어제")

    Args:
        dt: datetime 객체
        tz_name: 타임존 (None이면 기본값 사용)

    Returns:
        상대 시간 문자열
    """
    if dt is None:
        return ""

    current = now(tz_name)
    target = to_timezone(dt, tz_name)
    diff = current - target

    seconds = diff.total_seconds()

    if seconds < 0:
        return "방금 전"
    elif seconds < 60:
        return "방금 전"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes}분 전"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours}시간 전"
    elif seconds < 172800:
        return "어제"
    elif seconds < 604800:
        days = int(seconds / 86400)
        return f"{days}일 전"
    else:
        return target.strftime("%Y-%m-%d")


def format_kst_relative(dt: Optional[datetime]) -> str:
    """한국 시간 기준 상대 시간 표시"""
    return format_relative(dt, "Asia/Seoul")


def parse_datetime(
    date_str: str,
    fmt: str = "%Y-%m-%d %H:%M:%S",
    tz_name: Optional[str] = None
) -> datetime:
    """
    문자열을 지정된 타임존의 datetime으로 파싱

    Args:
        date_str: 날짜 문자열
        fmt: 입력 포맷
        tz_name: 타임존 (None이면 기본값 사용)

    Returns:
        timezone-aware datetime
    """
    dt = datetime.strptime(date_str, fmt)
    return dt.replace(tzinfo=get_timezone(tz_name))


def parse_kst(date_str: str, fmt: str = "%Y-%m-%d %H:%M:%S") -> datetime:
    """문자열을 KST datetime으로 파싱"""
    return parse_datetime(date_str, fmt, "Asia/Seoul")


def get_supported_timezones() -> dict:
    """지원하는 타임존 목록 반환 (프론트엔드용)"""
    return SUPPORTED_TIMEZONES.copy()


def is_valid_timezone(tz_name: str) -> bool:
    """유효한 타임존인지 확인"""
    try:
        ZoneInfo(tz_name)
        return True
    except Exception:
        return False
