import asyncio
from deepeval.models import DeepEvalBaseLLM
from client import VtcChatClient, ZephyrChatClient
from langchain_ollama import ChatOllama
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
import os


class BastetOllama(DeepEvalBaseLLM):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = ChatOllama(
            model=model_name,
            temperature=0.1,
            base_url="http://bastet:11434",
        )
        
    def load_model(self):   # type: ignore
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()
        chain = model | StrOutputParser()
        response = chain.invoke(
            [
                HumanMessage(content=prompt),
            ]
        )
        return response

    async def a_generate(self, prompt: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt)

    def get_model_name(self):
        return self.model_name


class VTCSingleChatLLM(DeepEvalBaseLLM):
    def __init__(self):
        self.vtc_client = VtcChatClient()

    def get_model_name(self):
        return "vtc-chatbot"

    def load_model(self):
        return self

    def generate(self, prompt: str) -> str:
        response = self.vtc_client.chat_single(prompt)
        return response

    async def a_generate(self, prompt: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt)

class VTCMultiChatLLM(DeepEvalBaseLLM):
    def __init__(self):
        self.vtc_client = VtcChatClient()
        self.session_id = None

    def get_model_name(self):
        return "vtc-chatbot"

    def load_model(self):
        return self

    def generate(self, prompt: str) -> str:
        response, _ = self.vtc_client.chat_multi(prompt)
        return response, self.session_id

    async def a_generate(self, prompt: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt)

class ZephyrMultiChatLLM(DeepEvalBaseLLM):
    def __init__(self):
        self.zephyr_client = ZephyrChatClient()
        self.session_id = None

    def get_model_name(self):
        return "zephyr-chatbot"

    def load_model(self):
        return self

    def generate(self, prompt: str) -> str:
        response, self.session_id = self.zephyr_client.chat(prompt)
        return response

    async def a_generate(self, prompt: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt)