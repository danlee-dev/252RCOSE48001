"""
국가법령정보 Open API 데이터 수집 스크립트 (본문 포함)

근로계약서 분석을 위한 법률 데이터를 수집합니다:
- 법령해석례 (본문 포함)
- 판례 (본문 포함)
- 고용노동부 법령해설 (본문 포함)
- 노동위원회 판정례 (본문 포함)
- 현행법령 (근로기준법 등)
"""

import requests
import json
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Optional


class LegalDataCollector:
    """법률 데이터 수집기 (본문 포함)"""

    def __init__(self, user_id: str, output_dir: Path):
        self.user_id = user_id
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.search_url = "http://www.law.go.kr/DRF/lawSearch.do"
        self.detail_url = "http://www.law.go.kr/DRF/lawService.do"

        # 근로계약서 관련 검색 키워드 (제목 검색용)
        self.keywords = [
            "근로계약",
            "근로기준법",
            "최저임금",
            "퇴직금",
            "해고",
            "임금",
            "연차",
            "근로시간",
            "휴게시간",
            "휴일",
            "연장근로",
            "통상임금",
            "수습기간",
        ]

    def _request_api(self, url: str, params: Dict) -> Optional[Dict]:
        """API 요청"""
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"  API 요청 실패: {e}")
            return None

    def _extract_data(self, response: Dict) -> Optional[Dict]:
        """응답 JSON에서 실제 데이터 추출"""
        if not response:
            return None
        if len(response) > 0:
            wrapper_key = list(response.keys())[0]
            return response[wrapper_key]
        return None

    def _fetch_detail(self, target: str, item_id: str, additional_params: Dict = None) -> Optional[Dict]:
        """본문 상세 조회"""
        params = {
            "OC": self.user_id,
            "target": target,
            "type": "JSON",
            "ID": item_id,
        }
        if additional_params:
            params.update(additional_params)

        response = self._request_api(self.detail_url, params)
        return self._extract_data(response)

    def collect_interpretations(self, keywords: List[str] = None) -> List[Dict]:
        """법령해석례 수집 (본문 포함)"""
        print("\n=== 법령해석례 수집 시작 ===")

        if keywords is None:
            keywords = self.keywords

        all_data = []

        for keyword in keywords:
            print(f"\n키워드: '{keyword}' 검색 중...")

            params = {
                "OC": self.user_id,
                "target": "expc",
                "type": "JSON",
                "query": keyword,
                "display": 100,
                "search": 1,  # 제목 검색
                "page": 1,
            }

            response = self._request_api(self.search_url, params)
            data = self._extract_data(response)

            if not data or 'totalCnt' not in data:
                print(f"  검색 결과 없음")
                continue

            total_cnt = int(data.get('totalCnt', 0))
            print(f"  총 {total_cnt}건 발견")

            if total_cnt == 0:
                continue

            # 목록 수집
            items = []
            total_pages = min((total_cnt // 100) + 1, 5)  # 최대 500건
            for page in range(1, total_pages + 1):
                params['page'] = page
                page_response = self._request_api(self.search_url, params)
                page_data = self._extract_data(page_response)

                if not page_data or 'expc' not in page_data:
                    break

                page_items = page_data['expc']
                if not isinstance(page_items, list):
                    page_items = [page_items]

                items.extend(page_items)
                time.sleep(0.2)

            # 본문 수집
            print(f"  본문 수집 중... (총 {len(items)}건)")
            for item in tqdm(items, desc=f"  {keyword} 본문"):
                item_id = item.get('법령해석례일련번호')
                if not item_id:
                    continue

                # 본문 조회
                detail = self._fetch_detail("expc", item_id)
                if detail and isinstance(detail, dict):
                    item['질의요지'] = detail.get('질의요지', '')
                    item['회답'] = detail.get('회답', '')
                    item['이유'] = detail.get('이유', '')

                item['검색키워드'] = keyword
                item['수집일시'] = datetime.now().isoformat()
                all_data.append(item)

                time.sleep(0.3)

        print(f"\n총 {len(all_data)}건의 법령해석례 수집 완료")
        return all_data

    def collect_precedents(self, keywords: List[str] = None,
                          start_date: str = "20100101",
                          end_date: str = None) -> List[Dict]:
        """판례 수집 (본문 포함)"""
        print("\n=== 판례 수집 시작 ===")

        if keywords is None:
            keywords = self.keywords

        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        all_data = []

        for keyword in keywords:
            print(f"\n키워드: '{keyword}' 검색 중...")

            params = {
                "OC": self.user_id,
                "target": "prec",
                "type": "JSON",
                "query": keyword,
                "display": 100,
                "search": 1,  # 제목 검색
                "prncYd": f"{start_date}~{end_date}",
                "page": 1,
            }

            response = self._request_api(self.search_url, params)
            data = self._extract_data(response)

            if not data or 'totalCnt' not in data:
                print(f"  검색 결과 없음")
                continue

            total_cnt = int(data.get('totalCnt', 0))
            print(f"  총 {total_cnt}건 발견")

            if total_cnt == 0:
                continue

            # 목록 수집
            items = []
            total_pages = min((total_cnt // 100) + 1, 3)  # 최대 300건
            for page in range(1, total_pages + 1):
                params['page'] = page
                page_response = self._request_api(self.search_url, params)
                page_data = self._extract_data(page_response)

                if not page_data or 'prec' not in page_data:
                    break

                page_items = page_data['prec']
                if not isinstance(page_items, list):
                    page_items = [page_items]

                items.extend(page_items)
                time.sleep(0.2)

            # 본문 수집
            print(f"  본문 수집 중... (총 {len(items)}건)")
            for item in tqdm(items, desc=f"  {keyword} 본문"):
                item_id = item.get('판례일련번호')
                if not item_id:
                    continue

                # 본문 조회
                detail = self._fetch_detail("prec", item_id)
                if detail and isinstance(detail, dict):
                    item['판시사항'] = detail.get('판시사항', '')
                    item['판결요지'] = detail.get('판결요지', '')
                    item['참조조문'] = detail.get('참조조문', '')
                    item['판례내용'] = detail.get('판례내용', '')

                item['검색키워드'] = keyword
                item['수집일시'] = datetime.now().isoformat()
                all_data.append(item)

                time.sleep(0.3)

        print(f"\n총 {len(all_data)}건의 판례 수집 완료")
        return all_data

    def collect_labor_ministry_interpretations(self) -> List[Dict]:
        """고용노동부 법령해설 수집 (본문 포함)"""
        print("\n=== 고용노동부 법령해설 수집 시작 ===")

        all_data = []

        for keyword in self.keywords:
            print(f"\n키워드: '{keyword}' 검색 중...")

            params = {
                "OC": self.user_id,
                "target": "moelCgmExpc",  # 수정: moel -> moelCgmExpc
                "type": "JSON",
                "query": keyword,
                "display": 100,
                "search": 1,  # 제목 검색
                "page": 1,
            }

            response = self._request_api(self.search_url, params)
            data = self._extract_data(response)

            if not data or 'totalCnt' not in data:
                print(f"  검색 결과 없음")
                continue

            total_cnt = int(data.get('totalCnt', 0))
            print(f"  총 {total_cnt}건 발견")

            if total_cnt == 0:
                continue

            # 목록 수집
            items = []
            total_pages = min((total_cnt // 100) + 1, 3)  # 최대 300건
            for page in range(1, total_pages + 1):
                params['page'] = page
                page_response = self._request_api(self.search_url, params)
                page_data = self._extract_data(page_response)

                if not page_data or 'cgmExpc' not in page_data:  # 수정: moel -> cgmExpc
                    break

                page_items = page_data['cgmExpc']  # 수정: moel -> cgmExpc
                if not isinstance(page_items, list):
                    page_items = [page_items]

                items.extend(page_items)
                time.sleep(0.2)

            # 본문 수집
            print(f"  본문 수집 중... (총 {len(items)}건)")
            for item in tqdm(items, desc=f"  {keyword} 본문"):
                item_id = item.get('법령해석일련번호')
                if not item_id:
                    continue

                # 본문 조회
                detail = self._fetch_detail("moelCgmExpc", item_id)
                if detail and isinstance(detail, dict):
                    item['질의요지'] = detail.get('질의요지', '')
                    item['회답'] = detail.get('회답', '')
                    item['이유'] = detail.get('이유', '')
                    item['관련법령'] = detail.get('관련법령', '')

                item['검색키워드'] = keyword
                item['수집일시'] = datetime.now().isoformat()
                all_data.append(item)

                time.sleep(0.3)

        print(f"\n총 {len(all_data)}건의 고용노동부 법령해설 수집 완료")
        return all_data

    def collect_labor_commission(self) -> List[Dict]:
        """노동위원회 판정례 수집 (본문 포함)"""
        print("\n=== 노동위원회 판정례 수집 시작 ===")

        all_data = []

        for keyword in self.keywords:
            print(f"\n키워드: '{keyword}' 검색 중...")

            params = {
                "OC": self.user_id,
                "target": "lwrc",
                "type": "JSON",
                "query": keyword,
                "display": 100,
                "search": 1,  # 제목 검색
                "page": 1,
            }

            response = self._request_api(self.search_url, params)
            data = self._extract_data(response)

            if not data or 'totalCnt' not in data:
                print(f"  검색 결과 없음")
                continue

            total_cnt = int(data.get('totalCnt', 0))
            print(f"  총 {total_cnt}건 발견")

            if total_cnt == 0:
                continue

            # 목록 수집
            items = []
            total_pages = min((total_cnt // 100) + 1, 5)  # 최대 500건
            for page in range(1, total_pages + 1):
                params['page'] = page
                page_response = self._request_api(self.search_url, params)
                page_data = self._extract_data(page_response)

                if not page_data or 'lwrc' not in page_data:
                    break

                page_items = page_data['lwrc']
                if not isinstance(page_items, list):
                    page_items = [page_items]

                items.extend(page_items)
                time.sleep(0.2)

            # 본문 수집
            print(f"  본문 수집 중... (총 {len(items)}건)")
            for item in tqdm(items, desc=f"  {keyword} 본문"):
                item_id = item.get('결정문일련번호')
                if not item_id:
                    continue

                # 본문 조회 (target=nlrc)
                detail = self._fetch_detail("nlrc", item_id)
                if detail and isinstance(detail, dict):
                    item['내용'] = detail.get('내용', '')
                    item['판정사항'] = detail.get('판정사항', '')
                    item['판정요지'] = detail.get('판정요지', '')
                    item['판정결과'] = detail.get('판정결과', '')

                item['검색키워드'] = keyword
                item['수집일시'] = datetime.now().isoformat()
                all_data.append(item)

                time.sleep(0.3)

        print(f"\n총 {len(all_data)}건의 노동위원회 판정례 수집 완료")
        return all_data

    def save_data(self, data: List[Dict], filename: str):
        """데이터 저장"""
        if not data:
            print(f"{filename}: 저장할 데이터 없음")
            return

        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        file_size = filepath.stat().st_size / 1024 / 1024
        print(f"저장 완료: {filepath} ({len(data)}건, {file_size:.2f}MB)")

    def collect_all(self):
        """모든 데이터 수집 및 저장"""
        print("\n" + "="*60)
        print("국가법령정보 Open API 데이터 수집 시작 (본문 포함)")
        print(f"사용자 ID: {self.user_id}")
        print(f"저장 경로: {self.output_dir}")
        print("검색 방식: 제목 검색 (search=1)")
        print("="*60)

        # 1. 법령해석례
        interpretations = self.collect_interpretations()
        self.save_data(
            interpretations,
            f"interpretations_{datetime.now().strftime('%Y%m%d')}.json"
        )

        # 2. 판례
        precedents = self.collect_precedents()
        self.save_data(
            precedents,
            f"precedents_{datetime.now().strftime('%Y%m%d')}.json"
        )

        # 3. 고용노동부 법령해설
        labor_ministry = self.collect_labor_ministry_interpretations()
        self.save_data(
            labor_ministry,
            f"labor_ministry_{datetime.now().strftime('%Y%m%d')}.json"
        )

        # 4. 노동위원회 - API가 JSON 대신 HTML 반환하여 수집 불가
        # labor_commission = self.collect_labor_commission()
        # self.save_data(
        #     labor_commission,
        #     f"labor_commission_{datetime.now().strftime('%Y%m%d')}.json"
        # )

        # 요약 통계
        print("\n" + "="*60)
        print("수집 완료 요약")
        print("="*60)
        print(f"법령해석례:          {len(interpretations):>6}건")
        print(f"판례:                {len(precedents):>6}건")
        print(f"고용노동부 법령해설: {len(labor_ministry):>6}건")
        # print(f"노동위원회 판정례:   {len(labor_commission):>6}건")
        print(f"총합:                {len(interpretations) + len(precedents) + len(labor_ministry):>6}건")
        print("="*60)


def main():
    """메인 실행 함수"""
    import os
    from dotenv import load_dotenv

    load_dotenv()
    user_id = os.getenv('LEGAL_API_USER_ID')

    if not user_id:
        print("오류: LEGAL_API_USER_ID 환경 변수를 설정해주세요.")
        return

    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "data" / "raw" / "api"

    collector = LegalDataCollector(user_id, output_dir)
    collector.collect_all()


if __name__ == "__main__":
    main()
