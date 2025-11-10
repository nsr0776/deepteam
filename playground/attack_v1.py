import asyncio
from datetime import datetime
from pathlib import Path
from pprint import pprint
from re import I
from socket import timeout
import timeit
from typing import cast

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from deepeval.models import DeepEvalBaseLLM
from custom_llm import BastetOllama


# llm = get_bastet_llm()

# async def model_callback(input: str) -> str:
#     response = await llm.ainvoke([
#         HumanMessage(content=input)
#     ])
#     return cast(str, response.content)


from deepteam import red_team
from deepteam.red_teamer import RedTeamer
import deepteam.attacks.single_turn as SingleTurn
import deepteam.attacks.multi_turn as MultiTurn
import deepteam.vulnerabilities as Vuln
from client import ZephyrChatClient, VtcChatClient


async def zephyr_callback(input: str) -> str:
    zephyr_client = ZephyrChatClient()
    response, _ = zephyr_client.chat(input)
    return response

async def vtcchatbot_callback(input: str) -> str:
    vtcchatbot_client = VtcChatClient()
    response = vtcchatbot_client.chat_single(input, timeout=86400.0)
    return response

def get_minimal(evaluation_model: DeepEvalBaseLLM, simulator_model: DeepEvalBaseLLM):
    vulnerabilities = [
        Vuln.Bias(
            purpose="test", 
            types=["race"],
            evaluation_model=evaluation_model,
            simulator_model=simulator_model,
        )
    ]
    attacks = [
        SingleTurn.PromptInjection(),
    ]    
    return vulnerabilities, attacks

def get_owasp_llm_01():
    """
    LLM01: Prompt Injection
    """
    vulnerabilities = [
        Vuln.Bias(types=["race"])
    ]
    attacks = [
        SingleTurn.PromptInjection(),
        SingleTurn.Base64(),
        SingleTurn.ROT13(),
        SingleTurn.Leetspeak()
    ]    
    return vulnerabilities, attacks

# def get_owasp_llm_02():
#     """
#     LLM02: Sensitive Data Exposure
#     """
#     vulnerabilities = [
#         Vuln.PIILeakage(types=["direct disclosure", "session leak"]),
#         Vuln.PromptLeakage(types=["secrets and credentials", "instructions"]),
#         Vuln.IntellectualProperty(types=["patent disclosure", "copyright violations"])
#     ]
#     attacks = [
#         SingleTurn.PromptInjection(),
#         SingleTurn.Roleplay(),
#     ]    
#     return vulnerabilities, attacks

# def get_owasp_llm_03():
#     vulnerabilities = [
#         Vuln.Bias(types=["race", "gender", "politics"]),
#         Vuln.Toxicity(types=["profanity", "insults"]),
#         Vuln.Misinformation(types=["factual errors"]),
#         Vuln.Robustness(types=["hijacking"])
#     ]
#     attacks=[
#         SingleTurn.PromptInjection(),
#         SingleTurn.Roleplay(),
#         MultiTurn.LinearJailbreaking(),
#     ]
#     return vulnerabilities, attacks

# def get_owasp_llm_04():
#     vulnerabilities = [
#         Vuln.Bias(types=["race", "gender", "religion"]),
#         Vuln.Toxicity(types=["profanity", "threats"]),
#         Vuln.Misinformation(types=["factual errors", "unsupported claims"]),
#         Vuln.IllegalActivity(types=["cybercrime", "violent crimes"])
#     ]
#     attacks = [
#         SingleTurn.PromptInjection(),
#         SingleTurn.Roleplay(),
#         MultiTurn.CrescendoJailbreaking(),
#     ]
#     return vulnerabilities, attacks

# def get_owasp_llm_05():
#     vulnerabilities = [
#         Vuln.ShellInjection(types=["command_injection", "system_command_execution"]),
#         Vuln.SQLInjection(types=["blind_sql_injection", "union_based_injection"])
#     ]
#     attacks = [
#         SingleTurn.PromptInjection(), 
#         SingleTurn.GrayBox(),
#     ]
#     return vulnerabilities, attacks

# def get_owasp_llm_06():
#     vulnerabilities = [
#         Vuln.ExcessiveAgency(types=["functionality", "permissions", "autonomy"]),
#         Vuln.RBAC(types=["role_bypass", "privilege_escalation"]),
#         Vuln.BFLA(types=["function_bypass", "authorization_bypass"]),
#         Vuln.BOLA(types=["object_access_bypass", "cross_customer_access"])
#     ]
#     attacks = [
#         SingleTurn.PromptInjection(),
#         SingleTurn.Roleplay(),
#     ]
#     return vulnerabilities, attacks

# def get_owasp_llm_07():
#     vulnerabilities = [
#         Vuln.PromptLeakage(types=[
#             "secrets and credentials",
#             "instructions",
#             "guard exposure",
#             "permissions and roles"
#         ])
#     ]
#     attacks = [
#         SingleTurn.PromptInjection(),
#         # SingleTurn.PromptProbing(),
#         # SingleTurn.Base64(),
#     ]
#     return vulnerabilities, attacks

# def get_owasp_llm_08():
#     vulnerabilities = [
#         Vuln.Misinformation(types=["factual errors"]),  # May indicate poisoned knowledge base
#         Vuln.PIILeakage(types=["direct disclosure"])    # May indicate vector database leakage
#     ]
#     attacks = [
#         SingleTurn.PromptInjection(),
#         SingleTurn.GrayBox(),
#     ]
#     return vulnerabilities, attacks

# def get_owasp_llm_09():
#     vulnerabilities = [
#     Vuln.Misinformation(types=[
#         "factual errors",
#         "unsupported claims",
#         "expertize misrepresentation"
#     ]),
#     Vuln.Competition(types=["discreditation"])
#     ]
#     attacks = [
#         SingleTurn.PromptInjection(),
#         SingleTurn.Roleplay(),
#         SingleTurn.PromptProbing(),
#     ]
#     return vulnerabilities, attacks

# def get_owasp_llm_10():
#     vulnerabilities = []
#     attacks = [
#         SingleTurn.MathProblem(),
#         MultiTurn.LinearJailbreaking(),
#         MultiTurn.CrescendoJailbreaking(),
#     ]
#     return vulnerabilities, attacks

async def main():
    # simulator_model = CustomDeepSeek("deepseek-chat")
    # evaluation_model = CustomDeepSeek("deepseek-reasoner")
    simulator_model = BastetOllama(
        model_name="gpt-oss:20b",
    )
    evaluation_model = BastetOllama(
        model_name="gpt-oss:20b",
    )

    vulnerabilities, attacks = get_minimal(evaluation_model=evaluation_model, simulator_model=simulator_model)

    print(f"Vulnerabilities: {vulnerabilities}")
    print(f"Attacks: {attacks}")

    red_teamer = RedTeamer(
        async_mode=True,
        max_concurrent=1,
        target_purpose=None,
        simulator_model=simulator_model,
        evaluation_model=evaluation_model,
    )
    risk_assessment = red_teamer.red_team(
        model_callback=vtcchatbot_callback,   # type: ignore
        vulnerabilities=vulnerabilities,      # type: ignore
        attacks=attacks,                      # type: ignore
        attacks_per_vulnerability_type=len(attacks),
        ignore_errors=False,
    )
    
    RESULT_DIR = Path(__file__).parent / "deepteam-results"
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    result_path = RESULT_DIR

    risk_assessment.save(to=str(result_path))
    print(risk_assessment.overview)
    


if __name__ == "__main__":
    start_time = timeit.default_timer()
    asyncio.run(main())
    end_time = timeit.default_timer()
    print(f"Time taken: {end_time - start_time} seconds")