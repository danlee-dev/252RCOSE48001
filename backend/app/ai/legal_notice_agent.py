import json
from datetime import datetime
from typing import Dict, List, Any
from app.core.llm_client import get_llm_client, LLMTask

class LegalNoticeAgent:
    def __init__(self):
        self.llm = get_llm_client()

    def generate_evidence_guide(self, analysis_result: Dict[str, Any]) -> str:
        """
        계약서 분석 결과(위반 사항)를 바탕으로 맞춤형 증거 수집 가이드 생성함.
        """
        # 분석 결과에서 위반 사항 추출
        # stress_test나 redlining 등 유효한 정보가 있는 곳을 우선 탐색함
        violations = []
        if "stress_test" in analysis_result and "violations" in analysis_result["stress_test"]:
            violations = analysis_result["stress_test"]["violations"]
        elif "violations" in analysis_result:
            violations = analysis_result["violations"]
            
        # 모든 위반 사항을 포함하여 문자열로 변환함 (개수 제한 제거)
        violations_text = json.dumps(violations, ensure_ascii=False)
        
        prompt = f"""
        당신은 노동법 전문 변호사입니다. 
        아래 분석된 근로계약서의 모든 위반 사항들을 입증하기 위해 근로자가 반드시 확보해야 할 '증거 자료'들을 안내해 주세요.

        [위반 사항 목록]
        {violations_text}

        [작성 지침]
        1. 목록에 있는 **모든 위반 사항**에 대해 빠짐없이 다루세요.
        2. 각 위반 항목별로 '필수 증거'와 '보조 증거'를 구분하여 목록화하세요.
        3. 증거 수집 방법(예: 녹음 시 주의사항, 카톡 캡처 방법, 교통카드 내역 조회법, 근로자 노트 작성법 등)을 아주 구체적이고 실천적인 꿀팁으로 제공하세요.
        4. 마크다운(Markdown) 형식으로 가독성 있게 작성하세요.
        """
        
        # Reasoning 모델(GPT) 사용하여 정확하고 상세한 가이드 생성
        response = self.llm.generate(
            prompt=prompt,
            task=LLMTask.REASONING,
            temperature=0.5,
            max_tokens=4000  # 토큰 제한을 4000으로 늘려 답변이 잘리지 않게 함
        )
        return response.content

    def chat_for_collection(self, current_info: Dict, user_message: str, history: List[Dict]) -> Dict[str, Any]:
        """
        사용자와 대화하며 내용증명 작성에 필요한 필수 정보를 수집함.
        """
        # 내용증명 작성에 필요한 필수 정보 목록 정의
        required_fields = [
            "sender_name (발신인 이름)", 
            "sender_address (발신인 주소)", 
            "recipient_company (수신 회사명)", 
            "recipient_representative (대표자 이름)",
            "recipient_address (수신인 주소)",
            "employment_start_date (입사일)", 
            "employment_end_date (퇴사일 - 해당시)",
            "main_damage_summary (주요 피해 내용 및 요구사항)"
        ]
        
        # 시스템 프롬프트: 정보 수집가 역할 부여
        system_prompt = f"""
        당신은 내용증명 작성을 돕는 AI 법률 비서입니다.
        현재까지 수집된 정보: {json.dumps(current_info, ensure_ascii=False)}
        
        수집해야 할 필수 정보: {', '.join(required_fields)}

        [당신의 목표]
        1. 사용자와의 대화에서 위 필수 정보를 추출하세요.
        2. 아직 수집되지 않은 정보가 있다면, 한 번에 1~2개씩 자연스럽게 질문하세요.
        3. 모든 정보가 수집되었다면 "모든 정보가 수집되었습니다."라고 말하고 is_complete를 true로 설정하세요.
        
        [답변 형식 - JSON Only]
        반드시 아래 JSON 형식으로만 응답하세요. 마크다운 태그 없이 순수 JSON만 반환하세요.
        {{
            "ai_message": "사용자에게 건넬 말 (친절하고 전문적인 어조)",
            "extracted_info": {{ "필드명": "추출된 값" }},
            "is_complete": true 또는 false
        }}
        """

        # 최근 대화 5개만 컨텍스트로 사용하여 토큰 효율성 증대
        chat_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-5:]])
        
        prompt = f"""
        [이전 대화]
        {chat_context}
        
        [사용자 입력]
        {user_message}
        
        위 입력에서 정보를 추출하고 JSON으로 응답하세요.
        """

        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                task=LLMTask.REASONING,
                temperature=0.1, # 정보 추출은 정확해야 하므로 낮은 온도 설정
                max_tokens=1000
            )
            # JSON 파싱 전처리 (마크다운 코드블록 제거)
            content = response.content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception as e:
            print(f"Extraction Error: {e}")
            return {
                "ai_message": "죄송합니다. 정보를 처리하는 중 오류가 발생했습니다. 다시 한 번 말씀해 주시겠어요?",
                "extracted_info": {},
                "is_complete": False
            }

    def write_legal_notice(self, collected_info: Dict, analysis_summary: str) -> str:
        """
        수집된 정보와 분석 결과를 종합하여 최종 내용증명 본문을 작성함.
        """
        today_date = datetime.now().strftime("%Y년 %m월 %d일")

        prompt = f"""
        아래 정보를 바탕으로 법적 효력이 있는 '내용증명서(통지서)'를 작성해 주세요.

        [작성 기준일(오늘)]
        {today_date}
        
        [수신인/발신인 및 사실관계]
        {json.dumps(collected_info, ensure_ascii=False, indent=2)}
        
        [계약서 분석 결과 (참고용 위반 사실)]
        {analysis_summary}
        
        [작성 지침]
        1. 제목, 수신인, 발신인, 본문, 작성일({today_date}), 발신인 서명란 형식을 갖추세요.
        2. 어조는 엄중하고 단호하며, 법률적 용어를 사용하세요.
        3. '육하원칙'에 따라 사실관계를 명확히 기술하세요.
        4. 미지급 임금 지급 등 구체적인 요구사항과 이행 기한(예: 수령 후 7일 이내)을 명시하세요.
        5. 불이행 시 고용노동부 진정 및 민/형사상 법적 조치를 취하겠다는 강력한 경고를 포함하세요.
        6. 전체 내용은 마크다운(Markdown) 형식으로 작성하세요.
        """
        
        response = self.llm.generate(
            prompt=prompt,
            task=LLMTask.REASONING, 
            temperature=0.3,
            max_tokens=4000  # 내용증명서도 길어질 수 있으므로 토큰 제한 늘림
        )
        return response.content