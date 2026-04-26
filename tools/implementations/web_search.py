import httpx
from pydantic import BaseModel, Field
from tools.base import BaseTool


class WebSearchSchema(BaseModel):
    query: str = Field(..., description="The search query string.")
    max_results: int = Field(default = 3, description = "Number of results to return.")

class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Searches the web for current events, news, and general knowledge."
    args_schema = WebSearchSchema

    async def execute(self, query: str, max_results: int) -> str:
        print(f"[Network] Executing web search for: '{query}'")

        results = []

        # DuckDuckGo search
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(f"{r['title']} - {r['href']}")

        if not results:
            return "No results found."

        return "\n".join(results)


