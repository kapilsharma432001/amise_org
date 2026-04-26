import os
import psycopg
from pgvector.psycopg import register_vector_async

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/amise_db")

async def get_db_connection():
    """Provides an async connection to the PostgreSQL and registers pgvector."""
    conn = await psycopg.AsyncConnection.connect(DATABASE_URL)
    await register_vector_async(conn)
    return conn

async def setup_database():
    """Initializes the AMISE document table with vector and FTS indexes."""
    conn = await psycopg.AsyncConnection.connect(DATABASE_URL, autocommit=True)
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        await register_vector_async(conn)

        # We store the raw text, the embedding, and a tsvector for FTS (Sparse Search)
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding vector({os.getenv("EMBEDDING_DIMENSIONS", 1536)}),
                fts_tokens tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
            );
        """)

        # Production index for exact keyword search
        await conn.execute("CREATE INDEX IF NOT EXISTS docs_fts_idx ON documents USING GIN (fts_tokens);")

        # Production index for dense vector search (HNSW is faster/more scalable than IVFFlat)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS docs_vector_idx ON documents 
            USING hnsw (embedding vector_cosine_ops);
        """)
    finally:
        await conn.close()


