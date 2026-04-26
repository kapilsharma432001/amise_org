import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/amise_db")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "dummy-openai-key")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
