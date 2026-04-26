import os
from typing import List
from litellm import aembedding
from .database import get_db_connection


async def generate_embeddings(text: str) -> List[float]:
    """Generates a dense vector via LiteLLM standard API"""
    
    response = await aembedding(
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        input = text
    )
    return response.data[0]["embedding"]

async def ingest_document(content: str):
    """Embed and saves a document to PostgreSQL"""
    embedding = await generate_embeddings(content)
    conn = await get_db_connection()
    try:
        await conn.execute(
            "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
            (content, embedding)
        )
        await conn.commit()
    finally:
        await conn.close()


