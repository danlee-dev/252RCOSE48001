import json
from pathlib import Path
from typing import List, Dict
import re


def extract_required_fields(chunks_file: Path) -> Dict:
    """standard_contract 청크에서 contract_type별 필수 필드 추출"""

    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    # standard_contract 청크만 필터링
    contract_chunks = [c for c in chunks if c.get('doc_type') == 'standard_contract']

    print(f"총 standard_contract 청크: {len(contract_chunks)}개")

    # contract_type별로 그룹화
    fields_by_type = {}

    for chunk in contract_chunks:
        contract_type = chunk.get('contract_type', 'unknown')
        section = chunk.get('section', '')
        content = chunk.get('content', '')
        is_mandatory = chunk.get('is_mandatory', False)
        clause_number = chunk.get('clause_number')

        # 헤더는 제외
        if section == "헤더":
            continue

        # contract_type별 리스트 초기화
        if contract_type not in fields_by_type:
            fields_by_type[contract_type] = []

        # 필드 정보 구성
        field_info = {
            "field_name": section.strip(),
            "clause_number": clause_number,
            "required": is_mandatory,
            "template": content.strip(),
            "description": ""
        }

        # 필드별 설명 추가
        if "근로개시일" in section or "근로계약기간" in section:
            field_info["description"] = "근로 시작일 또는 계약 기간 명시 필요"
            if "기간제" in contract_type:
                field_info["regulation"] = "계약 시작일과 종료일 명확히 기재"
        elif "근무장소" in section:
            field_info["description"] = "실제 근무할 장소 주소 기재"
        elif "근로시간" in section:
            field_info["description"] = "1일 소정근로시간 명시 (휴게시간 제외)"
            field_info["regulation"] = "1일 8시간, 주 40시간이 법정 기준"
            if "단시간" in contract_type:
                field_info["regulation"] = "주 15시간 이상 시 주휴수당 발생"
        elif "근무일" in section or "휴일" in section or "근로일" in section:
            field_info["description"] = "주 근무일 및 휴일 명시"
            if "단시간" in contract_type:
                field_info["regulation"] = "근로일별 상세 시간표 작성 필요"
        elif "임금" in section or "급여" in section:
            field_info["description"] = "임금 구성항목, 계산방법, 지급방법 명시"
            field_info["regulation"] = "2025년 최저임금: 시급 10,030원"
        elif "연차" in section:
            field_info["description"] = "연차유급휴가 일수 명시"
            field_info["regulation"] = "1년 근무 시 15일 부여"
        elif "사회보험" in section:
            field_info["description"] = "4대보험 적용여부 명시"
            field_info["regulation"] = "고용보험, 산재보험, 국민연금, 건강보험"
        elif "근로계약서" in section and "교부" in content:
            field_info["description"] = "근로계약서 서면 교부 의무"
            field_info["regulation"] = "근로기준법 제17조"
        elif "친권자" in section or "후견인" in section:
            field_info["description"] = "18세 미만 근로자의 친권자 또는 후견인 동의 필요"
            field_info["regulation"] = "근로기준법 제66조"

        fields_by_type[contract_type].append(field_info)

    return fields_by_type


def main():
    project_root = Path(__file__).parent.parent.parent
    chunks_file = project_root / "ai/data/processed/chunks/all_chunks.json"
    output_file = project_root / "ai/data/processed/required_contract_fields.json"

    print(f"청크 파일: {chunks_file}")

    # 필수 필드 추출
    fields_by_type = extract_required_fields(chunks_file)

    # 통계 계산
    total_fields = sum(len(fields) for fields in fields_by_type.values())

    # 최종 구조
    contract_requirements = {
        "document_type": "standard_employment_contract",
        "description": "표준 근로계약서 필수 항목 체크리스트 (계약 유형별)",
        "total_types": len(fields_by_type),
        "total_fields": total_fields,
        "contract_types": fields_by_type
    }

    # 저장
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(contract_requirements, f, ensure_ascii=False, indent=2)

    print(f"\n필수 필드 추출 완료:")
    print(f"  - 계약 유형: {len(fields_by_type)}개")
    print(f"  - 총 필드: {total_fields}개")
    print(f"저장 위치: {output_file}")

    # 미리보기
    print(f"\n=== 계약 유형별 필수 필드 ===")
    for contract_type, fields in fields_by_type.items():
        print(f"\n[{contract_type}] - {len(fields)}개 필드")
        for i, field in enumerate(fields[:5], 1):
            print(f"  {i}. {field['field_name']}")
            if field.get('regulation'):
                print(f"     규정: {field['regulation']}")
        if len(fields) > 5:
            print(f"  ... 외 {len(fields) - 5}개")


if __name__ == "__main__":
    main()
