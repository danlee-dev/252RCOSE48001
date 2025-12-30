import os
from dotenv import load_dotenv
from neo4j import GraphDatabase


# 1. 환경 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
load_dotenv(os.path.join(root_dir, ".env"))

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
AUTH = (USER, PASSWORD)

class OntologyBuilder:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)

    def close(self):
        self.driver.close()

    def create_indexes(self):
        print("⚙️ 인덱스 생성 중...")
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:ClauseType) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (r:RiskPattern) REQUIRE r.name IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (r:RiskPattern) ON (r.riskLevel)"
        ]
        with self.driver.session() as session:
            for q in queries:
                session.run(q)
        print("✅ 인덱스 설정 완료")

    def create_schema(self):
        print("🧠 온톨로지(지식 체계) 구축 시작...")
        
        # 1. 조항 유형 (ClauseType) 데이터 정의
        clause_types = [
            {
                "name": "임금", 
                "isRequired": True, 
                "desc": "근로기준법 제17조에 따라 임금의 구성항목(기본급, 제수당), 계산방법, 지급방법이 구체적으로 명시되어야 합니다."
            },
            {
                "name": "근로시간", 
                "isRequired": True, 
                "desc": "소정근로시간, 업무의 시작과 종료 시각, 그리고 4시간 근무 시 30분 이상의 휴게시간이 명시되어야 합니다."
            },
            {
                "name": "휴일_휴가", 
                "isRequired": True, 
                "desc": "주휴일(제55조) 및 연차유급휴가(제60조)의 발생 조건과 부여 일수가 명확히 기재되어야 합니다."
            },
            {
                "name": "계약기간", 
                "isRequired": True, 
                "desc": "근로계약의 시작일과 종료일(기간제 근로자의 경우)이 명시되어야 하며, 수습기간이 있다면 그 기간도 포함해야 합니다."
            },
            {
                "name": "해고_퇴직", 
                "isRequired": False, 
                "desc": "해고의 사유와 절차는 근로기준법 제23조(해고 등의 제한)에 부합해야 하며, 퇴직금 지급 규정이 포함되어야 합니다."
            },
            {
                "name": "손해배상", 
                "isRequired": False, 
                "desc": "근로자의 실수로 인한 손해배상 책임을 미리 약정하는 것은 금지됩니다(위약금 예정 금지)."
            }
        ]

        # 2. 위험 패턴 (RiskPattern) 데이터 정의
        risk_patterns = [
            {
                "name": "포괄임금제",
                "riskLevel": "High",
                "explanation": "연장·야간·휴일근로수당을 실제 근로시간과 관계없이 일정액으로 고정하여 지급하는 방식입니다. 이는 근로자의 실제 일한 만큼의 수당 청구권을 제한하고, 장시간 '공짜 야근'을 유발할 수 있는 매우 불리한 조항입니다.",
                "triggers": ["포괄하여", "포함하여 지급", "모든 수당", "제수당 포함"], 
                "law_keywords": ["제56조", "연장근로", "통상임금", "시간외근로"],
                "type": "임금"
            },
            {
                "name": "과도한_위약금",
                "riskLevel": "High",
                "explanation": "근로계약 불이행 시 위약금이나 손해배상액을 미리 정해놓는 것은 근로기준법 제20조(위약금 예정 금지) 위반입니다. 이는 근로자의 자유로운 퇴직을 가로막고 강제 근로를 유발할 수 있어 법적으로 무효입니다.",
                "triggers": ["배상하여야", "위약금", "반환", "손해를 배상", "월급을 공제"],
                "law_keywords": ["제20조", "위약금", "손해배상액", "강제근로"],
                "type": "손해배상"
            },
            {
                "name": "최저임금_미달",
                "riskLevel": "High",
                "explanation": "수습기간이라 하더라도 최저임금의 90% 미만으로 지급하거나, 단순노무직종에게 감액 적용하는 것은 최저임금법 위반입니다. 약정된 임금이 법정 최저임금보다 낮을 경우 그 부분은 무효가 됩니다.",
                "triggers": ["최저임금", "수습기간", "90%", "감액"],
                "law_keywords": ["최저임금법", "제6조", "수습근로자"], 
                "type": "임금"
            },
            {
                "name": "부당_해고_조항",
                "riskLevel": "Medium",
                "explanation": "'갑의 판단에 따라', '즉시 해고' 등 사용자가 임의로 해고할 수 있다고 명시한 조항은 근로기준법 제23조(정당한 이유 없는 해고 금지) 위반 소지가 큽니다. 해고는 반드시 정당한 사유와 절차(서면 통지 등)를 거쳐야 합니다.",
                "triggers": ["즉시 해고", "임의로 해지", "일방적으로", "갑의 판단"],
                "law_keywords": ["제23조", "해고의 제한", "정당한 이유", "서면통지"], 
                "type": "해고_퇴직"
            }
        ]

        with self.driver.session() as session:
            # Step 1: ClauseType 생성
            print("   Step 1: 조항 유형 생성 중...")
            for ct in clause_types:
                session.run("""
                MERGE (c:ClauseType {name: $name})
                SET c.isRequired = $required, c.explanation = $desc
                """, name=ct["name"], required=ct["isRequired"], desc=ct["desc"])

            # Step 2: RiskPattern 생성 및 IS_A_TYPE_OF 연결
            print("   Step 2: 위험 패턴 생성 및 유형 연결 중...")
            for rp in risk_patterns:
                session.run("""
                MERGE (r:RiskPattern {name: $name})
                SET r.riskLevel = $level,
                    r.explanation = $exp,
                    r.triggers = $triggers
                
                WITH r
                MATCH (c:ClauseType {name: $typeName})
                MERGE (r)-[:IS_A_TYPE_OF]->(c)
                """, 
                name=rp["name"], level=rp["riskLevel"], 
                exp=rp["explanation"], triggers=rp["triggers"], typeName=rp["type"])

            # Step 3: 위험 -> 근거 자료(판례/해석) 연결 (멀티홉 1단계)
            print("   Step 3: 위험 패턴과 근거 자료(판례/해석) 연결 중...")
            for rp in risk_patterns:
                # 1. 위험 -> 판례 (HAS_CASE) 연결
                query_case = """
                MATCH (r:RiskPattern {name: $riskName})
                MATCH (p:Precedent) // 🔴 라벨링 된 Precedent 노드를 사용
                WHERE ANY(t IN $triggers WHERE p.content CONTAINS t)
                MERGE (r)-[:HAS_CASE]->(p)
                """
                session.run(query_case, riskName=rp["name"], triggers=rp["triggers"])
                
                # 2. 위험 -> 행정해석 (HAS_INTERPRETATION) 연결
                query_interp = """
                MATCH (r:RiskPattern {name: $riskName})
                MATCH (i:Interpretation) // 🔴 라벨링 된 Interpretation 노드를 사용
                WHERE ANY(t IN $triggers WHERE i.content CONTAINS t)
                MERGE (r)-[:HAS_INTERPRETATION]->(i)
                """
                session.run(query_interp, riskName=rp["name"], triggers=rp["triggers"])

        print("✅ 온톨로지 구축 완료!")

def main():
    builder = OntologyBuilder(URI, AUTH)
    try:
        builder.create_indexes()
        builder.create_schema()
    finally:
        builder.close()

if __name__ == "__main__":
    main()