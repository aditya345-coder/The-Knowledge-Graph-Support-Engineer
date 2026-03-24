import os
from litellm import completion
from dotenv import load_dotenv

load_dotenv()


class LLMGateway:
    """
    Abstractions for LLM calls using LiteLLM.
    Allows switching between Groq, OpenAI, and others via .env.
    """

    def __init__(self):
        self.model = os.getenv("LLM_MODEL", "groq/llama-3.1-8b-instant")

    def chat(self, messages: list, temperature: float = 0):
        """Standard chat completion call."""
        return completion(model=self.model, messages=messages, temperature=temperature)

    def extract_json(self, prompt: str):
        """Helper to ensure we get cleaner JSON responses."""
        messages = [
            {
                "role": "system",
                "content": "You are a precise data extractor. Return only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ]
        response = self.chat(messages)
        # Basic cleanup in case of markdown blocks
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        return content
