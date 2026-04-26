import asyncio
import json
from tools.registry import ToolRegistry
from tools.implementations.web_search import WebSearchTool
from tools.implementations.stock_search import StockQuoteTool


async def run_smoke_test():
    print("--- AMISE Day 3: Tool Registry System ---")

    # 1. Initialize and Register
    registry = ToolRegistry()
    registry.register(WebSearchTool())
    registry.register(StockQuoteTool())

    # 2. View the Schema (This is what we will send to the LLM tomorrow)
    print("\n--- Generated Tool Schemas for LLM ---")
    schemas = registry.get_all_schemas()
    print(json.dumps(schemas, indent=2))

    # 3. Simulate an LLM Function Call Response
    print("\n--- Simulating LLM Function Call Execution ---")
    mock_llm_call = {
        "tool_name": "stock_quote",
        "arguments": {"ticker": "AAPL"}
    }
    
    print(f"LLM requested to run '{mock_llm_call['tool_name']}' with args {mock_llm_call['arguments']}")
    result = await registry.invoke(mock_llm_call["tool_name"], mock_llm_call["arguments"])
    print(f"Tool Result: {result}")
    
    # 4. Test Pydantic Validation Failure
    print("\n--- Testing LLM Hallucination Prevention ---")
    try:
        # Missing the required 'query' argument
        bad_args = {"wrong_param": "test"}
        await registry.invoke("web_search", bad_args)
    except Exception as e:
        print(f"Caught LLM hallucination: {e}")

if __name__ == "__main__":
    asyncio.run(run_smoke_test())
    

