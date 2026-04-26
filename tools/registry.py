from typing import Dict, List, Any
from .base import BaseTool
class ToolRegistry:
    """
    Central hub for registering, discovering and invoking tools.
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool
    
    def get_all_schemas(self) -> List[dict]:
        return [tool.get_openai_schema() for tool in self._tools.values()]
    
    async def invoke(self, tool_name: str, arguments: dict) -> Any:
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' not found in registry.")
        tool = self._tools[tool_name]
        
        # Pydantic validation: Ensure LLM args match our schema
        validated_args = tool.arg_schema(**arguments)

        try:
            # Execute the tool with validated arguments
            return await tool.execute(**validated_args.model_dump())
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"
        
        
        
    
