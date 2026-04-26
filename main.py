import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
from litellm import acompletion, exceptions
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio
from amise_org.data.database import setup_database, DATABASE_URL
from amise_org.data.ingestion import ingest_document
from amise_org.data.retriever import hybrid_search



# Environment variables for API keys
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "dummy-anthropic-key")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "dummy-openai-key")

class LLMGateway:
    """Resilient interface for executing standard and fallback LLM queries."""

    def __init__(self, primary: str = "gpt-4o", fallbacks: list = ["claude-3-haiku-20240307"]):
        self.primary = primary
        self.fallbacks = fallbacks
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def generate(self, prompt: str, system_prompt: str = None, temperature: float = 0.0) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await acompletion(
                model = self.primary,
                messages = messages,
                temperature = temperature,
                max_tokens = 1024,
                fallbacks = self.fallbacks
            )
            return response.choices[0].message.content
        except exceptions.RateLimitError as e:
            print(f"[Gateway] Rate Limited: {e}")
            raise
        except exceptions.AuthenticationError as e:
            print(f"[Gateway] Authentication Failed, check API Keys: {e}")
            raise
        except Exception as e:
            print(f"[Gateway] Unhandled exception: {e}")
            raise

# FastAPI app
app = FastAPI(title = "AMISE API", version = "0.1.0")
gateway = LLMGateway()

class QueryRequest(BaseModel):
    prompt: str = Field(..., description = "The user's query for AMISE")
    system: str = Field(default="You are AMISE, an AI market intelligence assistant.", description = "System behavior")

@app.post("/api/v1/query")
async def process_query(req: QueryRequest):
    """Entry point for testing the LLM Gateway"""
    try:
        result = await gateway.generate(prompt=req.prompt, system_prompt=req.system)
        return {"status": "success", "data": result}
    except exceptions.AuthenticationError:
        raise HTTPException(status_code=401, detail="Authentication failed. Check API keys.")
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))

async def run_smoke_test():
    print("--- AMISE Day 2: Production Hybrid RAG DB Test ---")
    print("Setting up pgvector tables and HNSW indexes...")
    try:
        await setup_database()
    except Exception as e:
        print(f"DB Connection failed. Ensure PostgreSQL is running at {DATABASE_URL}")
        print(f"Error: {e}")
        return

    # Dummy Data Ingestion
    docs = [
        "Tiger Analytics released a new enterprise AI product in 2025.",
        "The Q3 revenue for AAPL dropped by 2% due to supply chain issues.",
        "PostgreSQL 16 introduces better query optimization for JSONB.",
        "Large Language Models struggle with exact math and require external tools.",
        "FastAPI integrates perfectly with Starlette for asynchronous execution."
    ]
    print("Ingesting and embedding documents...")
    try:
        for doc in docs:
            await ingest_document(doc)
    except Exception as e:
        print(f"Embedding failed. Ensure valid OPENAI_API_KEY. Error: {e}")
        return
    test_query = "enterprise AI products 2025"
    print(f"\nExecuting Hybrid Search for: '{test_query}'")
    results = await hybrid_search(test_query, limit=2)

    for i, res in enumerate(results, 1):
        print(f"\nRank {i} (RRF Score: {res['rrf_score']:.4f})")
        print(f"Content: {res['content']}")

if __name__ == "__main__":
    asyncio.run(run_smoke_test())
