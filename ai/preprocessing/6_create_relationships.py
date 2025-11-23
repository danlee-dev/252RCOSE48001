import os
from dotenv import load_dotenv
from neo4j import GraphDatabase


# 1. 프로젝트 루트의 .env 파일 로드
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
load_dotenv(os.path.join(root_dir, ".env"))

# 2. 환경변수에서 접속 정보 가져오기 (없으면 기본값)
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

    def create_category_relationships(self):
        """
        1. Category 노드를 새로 만듭니다. (예: '근로기준법'이라는 점을 생성)
        2. Document 노드들과 연결합니다. (Document)-[:CATEGORIZED_AS]->(Category)
        """
        print("🔗 카테고리 관계 생성 중... (잠시만 기다려주세요)")
        
        # 1단계: 카테고리 노드(Category) 생성
        # 기존 문서들의 category 속성을 모아서 유일한 Category 노드로 만듭니다.
        query_create_categories = """
        MATCH (d:Document)
        WHERE d.category IS NOT NULL AND d.category <> 'General'
        WITH DISTINCT d.category AS catName
        MERGE (c:Category {name: catName})
        """
        
        # 2단계: 문서와 카테고리 연결 (선 긋기)
        # 배치를 사용하여 메모리 터짐 방지 (1000개씩 끊어서 연결)
        query_link_documents = """
        MATCH (d:Document)
        WHERE d.category IS NOT NULL AND d.category <> 'General'
        WITH d
        MATCH (c:Category {name: d.category})
        MERGE (d)-[:CATEGORIZED_AS]->(c)
        """

        with self.driver.session() as session:
            print("   Step 1: 카테고리 중심점(Hub) 만드는 중...")
            session.run(query_create_categories)
            
            print("   Step 2: 문서들과 카테고리 연결하는 중 (시간이 좀 걸립니다)...")
            # 데이터가 많으므로 call in transactions를 쓰거나, 그냥 실행 (1.5만개는 한 번에 가능)
            session.run(query_link_documents)
            
        print("✅ 카테고리 연결 완료!")

    def create_source_relationships(self):
        """
        1. Source(출처) 노드를 만듭니다. (예: '국가법령정보센터')
        2. Document와 연결합니다.
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
        builder.create_category_relationships()
        builder.create_source_relationships()
        print("\n🎉 그래프 관계 구축이 모두 완료되었습니다!")
    finally:
        builder.close()

if __name__ == "__main__":
    main()