import os
from typing import Any

from litellm import completion
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from utils.logging_config import setup_logging
from dotenv import load_dotenv

load_dotenv()

logger = setup_logging(__name__)


class LLMGateway:
    """
    Abstractions for LLM calls using LiteLLM.
    Allows switching between Groq, OpenAI, and others via .env.
    """

    def __init__(self):
        self.model = os.getenv("LLM_MODEL", "z-ai/glm4.7")
        self.api_key = os.getenv("NVIDIA_API_KEY")
        
        self.client = ChatNVIDIA(
            model=self.model,
            api_key=self.api_key,
            temperature=0, # Keep this 0 for agentic logic
        )

    def chat(self, messages: list, temperature: float = 0) -> Any:
        """Standard chat completion call."""
        # return completion(model=self.model, messages=messages, temperature=temperature)
        return self.client.invoke(messages)

    def get_message_text(self, response: Any) -> str:
        """Extracts the first assistant message text from a completion response."""
        if hasattr(response, "content"):
            return str(response.content).strip()
        return ""
        # litellm responses can vary in structure, so we add some robustness here
        # response_dict: dict[str, Any] = {}
        # if isinstance(response, dict):
        #     response_dict = response
        # elif hasattr(response, "model_dump"):
        #     response_dict = response.model_dump()
        # elif hasattr(response, "dict"):
        #     response_dict = response.dict()
        
        # choices = response_dict.get("choices") or []
        # if not choices:
        #     return ""
        # message = choices[0].get("message") or {}
        # return (message.get("content") or "").strip()

    def extract_json(self, prompt: str):
        """Helper to ensure we get cleaner JSON responses."""
        # messages = [
        #     {
        #         "role": "system",
        #         "content": "You are a precise data extractor. Return only valid JSON.",
        #     },
        #     {"role": "user", "content": prompt},
        # ]
        # response = self.chat(messages)
        # content = self.get_message_text(response)
        # if content.startswith("```json"):
        #     content = content.replace("```json", "").replace("```", "").strip()
        # return content
        
        # messages = [{"role": "user", "content": prompt}]
        messages = [
            {"role": "system", "content": "You are a precise data extractor. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ]
        response = self.chat(messages)
        content = self.get_message_text(response)
        
        # Cleanup Markdown formatting
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        return content
