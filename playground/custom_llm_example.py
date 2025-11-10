import requests
import json
import asyncio
from deepeval.models import DeepEvalBaseLLM

class MyCustomLLM(DeepEvalBaseLLM):
    def __init__(self):
        self.api_url = "https://your-api.com/chat"
        self.api_key = "your-api-key"

    def get_model_name(self):
        return "My Custom LLM"

    def load_model(self):
        return self

    def generate(self, prompt: str) -> str:
        response = requests.post(
            self.api_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"message": prompt}
        )
        return response.json()["response"]

    async def a_generate(self, prompt: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt)