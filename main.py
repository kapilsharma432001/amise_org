import os
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
from litellm import acompletion, exceptions
from tenacity import retry, stop_after_attempt, wait_exponential

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
    

if __name__ == "__main__":
    print("AMISE Day 1")

    client = TestClient(app)

    payload = {
        "prompt": "In one sentence, what is the core advantage of a multi-agent AI system?",
        "system": "You are a senior AI architect."
    }
    print(f"Sending POST /api/v1/query with payload:\n{payload}\n")
    response = client.post("/api/v1/query", json=payload)


    print(f"Status Code: {response.status_code}")
    print(f"Response JSON: {response.json()}")
    
    if response.status_code == 401:
        print("\nNote: Smoke test caught the 401 Auth Error gracefully. Add real API keys to see LLM output.")
