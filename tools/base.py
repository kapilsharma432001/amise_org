from abc import ABC, abstractmethod
from typing import Any, Type
from pydantic import BaseModel

class BaseTool(ABC):
    """
    Abstract base class for all AMISE tools.
    Enforces a srict contract for execution and schema generation.
    """

    name: str
    description: str
    args_schema: Type[BaseModel]

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """The actual tool logic. Must be implemented by subclasses."""
        pass

    def get_openai_schema(self) -> dict:
        """Generates the exact schema required by LiteLLM / OpenAI"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_schema.model_json_schema(),
            },
        }
    