"""
Application Initializers

This module handles initialization of external services on application startup:
- Elasticsearch index creation and data loading
- Neo4j schema, ontology, and data loading

These run once when the application starts, ensuring all required
infrastructure is in place before handling requests.

Environment Variables:
- INIT_LOAD_DATA: Set to "true" to load pre-processed data on startup
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any


def get_data_path() -> Path:
    """Get path to pre-processed data directory"""
    possible_paths = [
        Path(__file__).parent.parent.parent.parent / "ai" / "data" / "processed",
        Path("/app/ai/data/processed"),
    ]
    for path in possible_paths:
        if path.exists():
            return path
    return possible_paths[0]


# Elasticsearch initialization
def get_es_client():
    """Create Elasticsearch client with proper authentication."""
    from elasticsearch import Elasticsearch

    es_url = os.getenv("ES_URL", "http://localhost:9200")
    es_api_key = os.getenv("ES_API_KEY")

    if es_api_key:
        # Cloud authentication with API key
        return Elasticsearch(es_url, api_key=es_api_key)
    else:
        # Local without auth
        return Elasticsearch(es_url)


async def init_elasticsearch() -> bool:
    """
    Initialize Elasticsearch index with proper mappings.
    Creates the index if it doesn't exist.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        es = get_es_client()

        if not es.ping():
            print(">>> [Init] Elasticsearch not available, skipping...")
            return False

        INDEX_NAME = "docscanner_chunks"

        # Check if index exists
        if es.indices.exists(index=INDEX_NAME):
            print(f">>> [Init] Elasticsearch index '{INDEX_NAME}' already exists")
            return True

        # Create index with mappings
        index_settings = {
            "mappings": {
                "properties": {
                    "text": {
                        "type": "text",
                        "analyzer": "nori"  # Korean morphological analyzer
                    },
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 1024  # MUVERA FDE vector dimension
                    },
                    "source": {
                        "type": "keyword"  # For filtering
                    }
                }
            }
        }

        es.indices.create(index=INDEX_NAME, body=index_settings)
        print(f">>> [Init] Elasticsearch index '{INDEX_NAME}' created successfully")
        return True

    except Exception as e:
        print(f">>> [Init] Elasticsearch initialization failed: {e}")
        return False


async def load_elasticsearch_data() -> bool:
    """Load pre-processed embedding data into Elasticsearch."""
    if os.getenv("INIT_LOAD_DATA", "false").lower() != "true":
        return True

    try:
        from elasticsearch import helpers

        es = get_es_client()

        if not es.ping():
            return False

        INDEX_NAME = "docscanner_chunks"

        # Check if data already exists
        try:
            count = es.count(index=INDEX_NAME)["count"]
            if count > 0:
                print(f">>> [Init] ES already has {count} documents, skipping load")
                return True
        except:
            pass

        # Find embedding file
        data_path = get_data_path() / "embeddings"
        if not data_path.exists():
            print(f">>> [Init] Embeddings directory not found: {data_path}")
            return False

        embedding_files = sorted(
            data_path.glob("all_chunks_with_muvera_embeddings_*.json"),
            reverse=True
        )

        if not embedding_files:
            print(">>> [Init] No embedding files found")
            return False

        file_path = embedding_files[0]
        print(f">>> [Init] Loading embeddings from: {file_path.name}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f">>> [Init] Loaded {len(data)} chunks for ES")

        def generate_actions():
            for item in data:
                if 'content' not in item or 'embedding' not in item:
                    continue
                source_value = item.get('source', item.get('doc_type', 'unknown'))
                yield {
                    "_index": INDEX_NAME,
                    "_source": {
                        "text": item["content"],
                        "embedding": item["embedding"],
                        "source": source_value
                    }
                }

        success, failed = helpers.bulk(
            es.options(request_timeout=120),
            generate_actions(),
            chunk_size=500
        )

        print(f">>> [Init] ES indexing complete: {success} success, {failed} failed")
        return True

    except Exception as e:
        print(f">>> [Init] ES data load failed: {e}")
        return False


async def init_neo4j() -> bool:
    """
    Initialize Neo4j database with indexes and constraints.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        from neo4j import GraphDatabase

        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")

        driver = GraphDatabase.driver(uri, auth=(user, password))

        # Verify connection
        driver.verify_connectivity()
        print(">>> [Init] Neo4j connection verified")

        with driver.session() as session:
            # Create indexes and constraints for Document nodes
            index_queries = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.category)",
                "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.type)",
                # Ontology indexes
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:ClauseType) REQUIRE c.name IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (r:RiskPattern) REQUIRE r.name IS UNIQUE",
                "CREATE INDEX IF NOT EXISTS FOR (r:RiskPattern) ON (r.riskLevel)"
            ]

            for query in index_queries:
                session.run(query)

            print(">>> [Init] Neo4j indexes and constraints created")

        driver.close()
        return True

    except Exception as e:
        print(f">>> [Init] Neo4j initialization failed: {e}")
        return False


async def seed_ontology() -> bool:
    """
    Seed the Neo4j database with ontology data (ClauseTypes and RiskPatterns).
    This provides the knowledge base for contract analysis.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        from neo4j import GraphDatabase

        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")

        driver = GraphDatabase.driver(uri, auth=(user, password))

        # Clause types (contract clause categories)
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

        # Risk patterns (dangerous contract clauses)
        risk_patterns = [
            {
                "name": "포괄임금제",
                "riskLevel": "High",
                "explanation": "연장/야간/휴일근로수당을 실제 근로시간과 관계없이 일정액으로 고정하여 지급하는 방식입니다. 이는 근로자의 실제 일한 만큼의 수당 청구권을 제한하고, 장시간 '공짜 야근'을 유발할 수 있는 매우 불리한 조항입니다.",
                "triggers": ["포괄하여", "포함하여 지급", "모든 수당", "제수당 포함"],
                "type": "임금"
            },
            {
                "name": "과도한_위약금",
                "riskLevel": "High",
                "explanation": "근로계약 불이행 시 위약금이나 손해배상액을 미리 정해놓는 것은 근로기준법 제20조(위약금 예정 금지) 위반입니다. 이는 근로자의 자유로운 퇴직을 가로막고 강제 근로를 유발할 수 있어 법적으로 무효입니다.",
                "triggers": ["배상하여야", "위약금", "반환", "손해를 배상", "월급을 공제"],
                "type": "손해배상"
            },
            {
                "name": "최저임금_미달",
                "riskLevel": "High",
                "explanation": "수습기간이라 하더라도 최저임금의 90% 미만으로 지급하거나, 단순노무직종에게 감액 적용하는 것은 최저임금법 위반입니다. 약정된 임금이 법정 최저임금보다 낮을 경우 그 부분은 무효가 됩니다.",
                "triggers": ["최저임금", "수습기간", "90%", "감액"],
                "type": "임금"
            },
            {
                "name": "부당_해고_조항",
                "riskLevel": "Medium",
                "explanation": "'갑의 판단에 따라', '즉시 해고' 등 사용자가 임의로 해고할 수 있다고 명시한 조항은 근로기준법 제23조(정당한 이유 없는 해고 금지) 위반 소지가 큽니다. 해고는 반드시 정당한 사유와 절차(서면 통지 등)를 거쳐야 합니다.",
                "triggers": ["즉시 해고", "임의로 해지", "일방적으로", "갑의 판단"],
                "type": "해고_퇴직"
            }
        ]

        with driver.session() as session:
            # Check if ontology already exists
            result = session.run("MATCH (c:ClauseType) RETURN count(c) as count")
            count = result.single()["count"]

            if count > 0:
                print(f">>> [Init] Ontology already seeded ({count} ClauseTypes found)")
                driver.close()
                return True

            # Create ClauseTypes
            for ct in clause_types:
                session.run("""
                MERGE (c:ClauseType {name: $name})
                SET c.isRequired = $required, c.explanation = $desc
                """, name=ct["name"], required=ct["isRequired"], desc=ct["desc"])

            # Create RiskPatterns and link to ClauseTypes
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

            print(">>> [Init] Ontology seeded successfully")

        driver.close()
        return True

    except Exception as e:
        print(f">>> [Init] Ontology seeding failed: {e}")
        return False


async def run_all_initializers():
    """
    Run all initialization tasks.
    Called on application startup.
    """
    print("=" * 60)
    print(">>> [Init] Starting application initialization...")
    print("=" * 60)

    # Initialize Elasticsearch
    es_ok = await init_elasticsearch()

    # Initialize Neo4j
    neo4j_ok = await init_neo4j()

    # Seed ontology (only if Neo4j is available)
    if neo4j_ok:
        await seed_ontology()

    print("=" * 60)
    print(f">>> [Init] Initialization complete (ES: {es_ok}, Neo4j: {neo4j_ok})")
    print("=" * 60)
