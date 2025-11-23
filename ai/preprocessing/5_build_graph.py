import os
import json
import glob
from dotenv import load_dotenv
from neo4j import GraphDatabase
from tqdm import tqdm


# 1. 프로젝트 루트의 .env 파일 로드
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
load_dotenv(os.path.join(root_dir, ".env"))

# 2. 환경변수에서 접속 정보 가져오기 (없으면 기본값)
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
AUTH = (USER, PASSWORD)

class GraphBuilder:
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

    def create_indexes(self):
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.category)",
            "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.type)"
        ]
        with self.driver.session() as session:
            for q in queries:
                session.run(q)
        print("✅ 인덱스 설정 완료")

    def load_processed_data(self):
        # 경로: ai/data/processed
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, "..", "data", "processed")
        
        print(f"🔍 데이터 경로: {os.path.abspath(data_path)}")
        files = glob.glob(os.path.join(data_path, "**", "*.json"), recursive=True)
        all_chunks = []
        
        print(f"📂 파일 스캔 중... ({len(files)}개 발견)")
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_chunks.extend(data)
                    else:
                        all_chunks.append(data)
            except Exception as e:
                print(f"⚠️ 읽기 실패: {file_path}")
        
        # [디버깅용] 첫 번째 데이터 구조 확인
        if len(all_chunks) > 0:
            print(f"👀 첫 번째 데이터 샘플 (키 확인): {list(all_chunks[0].keys())}")
            
        print(f"📊 총 {len(all_chunks)}개의 데이터 준비 완료")
        return all_chunks

    def create_nodes(self, chunks):
        # [수정됨] coalesce 함수를 사용하여 metadata가 있든 없든 데이터를 찾아내도록 변경
        query = """
        UNWIND $batch AS row
        MERGE (d:Document {id: row.chunk_id})
        SET d.content = row.content,
            d.source = coalesce(row.metadata.source, row.source, 'Unknown'),
            d.category = coalesce(row.metadata.category, row.category, 'General'),
            d.type = coalesce(row.metadata.type, row.type, 'document'),
            d.page = coalesce(row.metadata.page, row.page, 1)
        """
        batch_size = 500
        
        cleaned = []
        for i, c in enumerate(chunks):
            if 'chunk_id' not in c:
                c['chunk_id'] = f"unknown_{i}"
            cleaned.append(c)

        print("🚀 Neo4j에 데이터 저장 시작...")
        with self.driver.session() as session:
            for i in tqdm(range(0, len(cleaned), batch_size), desc="Graph Node 생성"):
                batch = cleaned[i:i+batch_size]
                session.run(query, batch=batch)
        print("🎉 저장 완료!")

def main():
    builder = GraphBuilder(URI, AUTH)
    try:
        builder.create_indexes()
        chunks = builder.load_processed_data()
        if chunks:
            builder.create_nodes(chunks)
        else:
            print("❌ 데이터가 없습니다.")
    finally:
        builder.close()

if __name__ == "__main__":
    main()