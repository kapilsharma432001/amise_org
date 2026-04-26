import os
from typing import List, Dict, Any
from .database import get_db_connection
from .ingestion import generate_embeddings
import psycopg

async def hybrid_search(query: str, limit: int = 5, rrf_k: int = 60) -> List[Dict[str, Any]]:
    """
    Executes a Hybrid Search using Reciprocal Rank Fusion (RRF) directly in PostgreSQL.
    This combines pgvector Cosine Distance with PostgreSQL Full Text Search (BM25 proxy).
    """
    query_embedding = await generate_embeddings(query)
    conn = await get_db_connection()

    # Production Pattern: Executing RRF inside a single SQL query using CTEs (Common Table Expressions)
    # This avoids pulling thousands of rows into Python memory before ranking.

    sql_query = """
      WITH semantic_search AS (
        SELECT id, content, embedding <=> %s::vector AS vector_distance, 
             RANK() OVER (ORDER BY embedding <=> %s::vector) AS semantic_rank
        FROM documents
        ORDER BY vector_distance LIMIT %s
        ),
    keyword_search AS (
    SELECT id, content, ts_rank_cd(fts_tokens, websearch_to_tsquery('english', %s)) AS keyword_score,
               RANK() OVER (ORDER BY ts_rank_cd(fts_tokens, websearch_to_tsquery('english', %s)) DESC) AS keyword_rank
        FROM documents
        WHERE fts_tokens @@ websearch_to_tsquery('english', %s)
        ORDER BY keyword_score DESC LIMIT %s
    )
    SELECT 
        COALESCE(s.id, k.id) AS doc_id,
        COALESCE(s.content, k.content) AS content,
        -- RRF Formula: 1 / (k + rank)
        COALESCE(1.0 / (%s + s.semantic_rank), 0.0) + 
        COALESCE(1.0 / (%s + k.keyword_rank), 0.0) AS rrf_score
    FROM semantic_search s
    FULL OUTER JOIN keyword_search k ON s.id = k.id
    ORDER BY rrf_score DESC
    LIMIT %s;
    """

    try:
        async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            await cur.execute(sql_query, (
                query_embedding, query_embedding, limit * 2, # Semantic params
                query, query, query, limit * 2,              # Keyword params
                rrf_k, rrf_k,                                # RRF constant params
                limit                                        # Final limit
            ))
            results = await cur.fetchall()
            return results
    finally:
        await conn.close()