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

class RelationshipBuilder:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.verify_connection()

    def verify_connection(self):
        try:
            self.driver.verify_connectivity()
            print("✅ Neo4j 접속 성공!")
        except Exception as e:
            print(f"❌ Neo4j 접속 실패: {e}")
            raise e

    def close(self):
        self.driver.close()

    def classify_nodes(self):
        """
        Document 노드에 type 속성을 기반으로 추가 라벨을 부여합니다.
        (멀티홉 검색의 시작점 역할을 명확히 하기 위함)
        """
        print("🏷️ 노드 라벨 세분화 중...")
        queries = [
            # 1. 판례 (Precedent)
            "MATCH (d:Document) WHERE d.type = 'precedent' SET d:Precedent",
            # 2. 행정해석 (Interpretation) - 'interpretation' 또는 'labor_ministry'
            "MATCH (d:Document) WHERE d.type IN ['interpretation', 'labor_ministry'] SET d:Interpretation",
            # 3. 실무 매뉴얼 (Manual) - 'manual', 'leaflet', 'guide'
            "MATCH (d:Document) WHERE d.type IN ['manual', 'leaflet', 'guide'] SET d:Manual",
            # 4. 법령 (Law) - '근로기준법', '최저임금법' 등을 포함한 문서에 Law 라벨 부여
            "MATCH (d:Document) WHERE d.category IN ['근로기준법', '최저임금법'] SET d:Law",
        ]
        
        with self.driver.session() as session:
            for q in queries:
                session.run(q)
        print("✅ 노드 라벨링 완료!")


    def create_category_relationships(self):
        """
        Document와 Category 노드 간의 관계를 생성합니다.
        """
        print("🔗 카테고리 관계 생성 중... (Document)-[:CATEGORIZED_AS]->(Category)")
        
        query_create_categories = """
        MATCH (d:Document)
        WHERE d.category IS NOT NULL AND d.category <> 'General'
        WITH DISTINCT d.category AS catName
        MERGE (c:Category {name: catName})
        """
        
        query_link_documents = """
        MATCH (d:Document)
        WHERE d.category IS NOT NULL AND d.category <> 'General'
        WITH d
        MATCH (c:Category {name: d.category})
        MERGE (d)-[:CATEGORIZED_AS]->(c)
        """

        with self.driver.session() as session:
            print("   Step 1: 카테고리 중심점(Hub) 만드는 중...")
            session.run(query_create_categories)
            
            print("   Step 2: 문서들과 카테고리 연결하는 중...")
            session.run(query_link_documents)
            
        print("✅ 카테고리 연결 완료!")

    def create_source_relationships(self):
        """
        Source 노드를 만들고 Document와 연결합니다.
        """
        print("🔗 출처 관계 생성 중...")
        
        query = """
        MATCH (d:Document)
        WHERE d.source IS NOT NULL AND d.source <> 'Unknown'
        WITH d
        MERGE (s:Source {name: d.source})
        MERGE (d)-[:SOURCE_IS]->(s)
        """
        with self.driver.session() as session:
            session.run(query)
        print("✅ 출처 연결 완료!")

def main():
    builder = RelationshipBuilder(URI, AUTH)
    try:
        builder.classify_nodes() 
        builder.create_category_relationships()
        builder.create_source_relationships()
        print("\n🎉 그래프 관계 구축 및 분류 완료!")
    finally:
        builder.close()

if __name__ == "__main__":
    main()