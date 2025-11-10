
# ----------- Part 1 - Define the Target LLM Wrapper ------------
import httpx

async def model_callback_single_turn(input: str) -> str:
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            f"https://<target POST endpoint>t", 
            json={"prompt": input}
        )
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")


_session_id: str | None = None

async def model_callback_multi_turn(input: str) -> str:
    global _session_id
    headers = {"X-Session-Id": _session_id} if _session_id else {}
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            f"https://<target POST endpoint>t", 
            json={"prompt": input}, 
            headers=headers
        )
        r.raise_for_status()
        _session_id = r.headers.get("X-Session-Id", _session_id)
        return r.json().get("response", "")
# ------------ End of Part 1 ------------

# ----------- Part 2 - Define Custom Ollama Interfaces ------------
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_ollama import ChatOllama


# Inherit from the DeepEvalBaseLLM class and implement the methods
# `load_model`, `generate`, `a_generate`, and `get_model_name`
class CustomOllama(DeepEvalBaseLLM):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = ChatOllama(
            model=model_name,    # Pass the Ollama model name, e.g., "deepseek-r1:8b"
            temperature=0.1,
            base_url="http://<ollama_server>:11434",  # Url of your Ollama server
        )
        
    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()
        response = model.invoke(prompt)
        return response     # type: ignore

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name
# ------------ End of Part 2 ------------


# ------ Part 3 - Customize (Vuln + Attack) Combinations to Use in Scanning --------
import deepteam.attacks.single_turn as SingleTurn
import deepteam.attacks.multi_turn as MultiTurn
import deepteam.vulnerabilities as Vuln

def get_minimal():
    vulnerabilities = [
        Vuln.Bias(types=["race"])
    ]
    attacks = [
        # weight is an int that determines the weighted probability that a particular attack method will be randomly selected for simulation. Defaulted to 1.
        SingleTurn.PromptInjection(weight=1),
    ]    
    return vulnerabilities, attacks

def custom_owasp_llm_01():
    """
    LLM01: Prompt Injection
    """
    vulnerabilities = [
        Vuln.Bias(types=["race"])
    ]
    attacks = [
        SingleTurn.PromptInjection(weight=1),
        SingleTurn.Base64(weight=1),
        SingleTurn.ROT13(weight=1),
        SingleTurn.Leetspeak(weight=1)
    ]    
    return vulnerabilities, attacks

def custom_owasp_llm_02():
    """
    LLM02: Sensitive Data Exposure
    """
    vulnerabilities = [
        Vuln.PIILeakage(types=["direct disclosure", "session leak"]),
        Vuln.PromptLeakage(types=["secrets and credentials", "instructions"]),
        Vuln.IntellectualProperty(types=["patent disclosure", "copyright violations"])
    ]
    attacks = [
        SingleTurn.PromptInjection(weight=1),
        SingleTurn.Roleplay(weight=1),
    ]    
    return vulnerabilities, attacks

def custom_owasp_llm_03():
    vulnerabilities = [
        Vuln.Bias(types=["race", "gender", "politics"]),
        Vuln.Toxicity(types=["profanity", "insults"]),
        Vuln.Misinformation(types=["factual errors"]),
        Vuln.Robustness(types=["hijacking"])
    ]
    attacks=[
        SingleTurn.PromptInjection(weight=1),
        SingleTurn.Roleplay(weight=1),
        MultiTurn.LinearJailbreaking(weight=1, turns=5),
    ]
    return vulnerabilities, attacks

def custom_owasp_llm_04():
    vulnerabilities = [
        Vuln.Bias(types=["race", "gender", "religion"]),
        Vuln.Toxicity(types=["profanity", "threats"]),
        Vuln.Misinformation(types=["factual errors", "unsupported claims"]),
        Vuln.IllegalActivity(types=["cybercrime", "violent crimes"])
    ]
    attacks = [
        SingleTurn.PromptInjection(weight=1),
        SingleTurn.Roleplay(weight=1),
        MultiTurn.CrescendoJailbreaking(weight=1, max_rounds=5),
    ]
    return vulnerabilities, attacks

def custom_owasp_llm_05():
    vulnerabilities = [
        Vuln.ShellInjection(types=["command_injection", "system_command_execution"]),
        Vuln.SQLInjection(types=["blind_sql_injection", "union_based_injection"])
    ]
    attacks = [
        SingleTurn.PromptInjection(weight=1), 
        SingleTurn.GrayBox(weight=1),
    ]
    return vulnerabilities, attacks

def custom_owasp_llm_06():
    vulnerabilities = [
        Vuln.ExcessiveAgency(types=["functionality", "permissions", "autonomy"]),
        Vuln.RBAC(types=["role_bypass", "privilege_escalation"]),
        Vuln.BFLA(types=["function_bypass", "authorization_bypass"]),
        Vuln.BOLA(types=["object_access_bypass", "cross_customer_access"])
    ]
    attacks = [
        SingleTurn.PromptInjection(weight=1),
        SingleTurn.Roleplay(weight=1),
    ]
    return vulnerabilities, attacks

def custom_owasp_llm_07():
    vulnerabilities = [
        Vuln.PromptLeakage(types=[
            "secrets and credentials",
            "instructions",
            "guard exposure",
            "permissions and roles"
        ])
    ]
    attacks = [
        SingleTurn.PromptInjection(weight=1),
        SingleTurn.PromptProbing(weight=1),
        SingleTurn.Base64(weight=1),
    ]
    return vulnerabilities, attacks

def custom_owasp_llm_08():
    vulnerabilities = [
        Vuln.Misinformation(types=["factual errors"]),  # May indicate poisoned knowledge base
        Vuln.PIILeakage(types=["direct disclosure"])    # May indicate vector database leakage
    ]
    attacks = [
        SingleTurn.PromptInjection(weight=1),
        SingleTurn.GrayBox(weight=1),
    ]
    return vulnerabilities, attacks

def custom_owasp_llm_09():
    vulnerabilities = [
    Vuln.Misinformation(types=[
        "factual errors",
        "unsupported claims",
        "expertize misrepresentation"
    ]),
    Vuln.Competition(types=["discreditation"])
    ]
    attacks = [
        SingleTurn.PromptInjection(weight=1),
        SingleTurn.Roleplay(weight=1),
        SingleTurn.PromptProbing(weight=1),
    ]
    return vulnerabilities, attacks

def custom_owasp_llm_10():
    vulnerabilities = []
    attacks = [
        SingleTurn.MathProblem(weight=1),
        MultiTurn.LinearJailbreaking(weight=1, turns=5),
        MultiTurn.CrescendoJailbreaking(weight=1, max_rounds=5),
    ]
    return vulnerabilities, attacks
# ------------ End of Part 3 ------------


# ----------- Part 4 - Define the Main Function ------------
import asyncio
import timeit

from deepteam.red_teamer import RedTeamer
from deepteam.frameworks import OWASPTop10


async def main(use_default_owasp: bool):
    # The model that generates the payload
    simulator_model = CustomOllama("<Ollama model name> (e.g., gemma3:4b)")

    # The model that evaluates the payload
    evaluation_model = CustomOllama("<Ollama model name> (e.g., deepseek-r1:8b)")

    # Option 1: Use the default DeepTeam OWASP Top 10 vulnerabilities and attacks
    if use_default_owasp:
        framework_config = OWASPTop10()
        vulnerabilities = framework_config.vulnerabilities
        attacks = framework_config.attacks

    # Option 2: Use the custom vulnerabilities and attacks
    else:
        vulnerabilities, attacks = get_minimal()

    print(f"Vulnerabilities: {vulnerabilities}")
    print(f"Attacks: {attacks}")


    """
    [Optional] target_purpose: a string specifying your target LLM application's intended purpose. This affects the passing and failing of simulated attacks that are evaluated. Defaulted to None.

    [Optional] simulator_model: a string specifying which of OpenAI's GPT models to use, OR any custom LLM model of type DeepEvalBaseLLM for simulating attacks. Defaulted to "gpt-3.5-turbo-0125".

    [Optional] evaluation_model: a string specifying which of OpenAI's GPT models to use, OR any custom LLM model of type DeepEvalBaseLLM for evaluation. Defaulted to "gpt-4o".

    [Optional] async_mode: a boolean specifying whether to enable async mode. Defaulted to True.

    [Optional] max_concurrent: an integer that determines the maximum number of coroutines that can be ran in parallel. You can decrease this value if your models are running into rate limit errors. Defaulted to 10.
    """
    red_teamer = RedTeamer(
        simulator_model=simulator_model,
        evaluation_model=evaluation_model,
        async_mode=True,
        max_concurrent=10,
        target_purpose=None,
    )

    """
    model_callback: a callback of type Callable[[str], str] that wraps around the target LLM system you wish to red team.

    vulnerabilities: a list of type BaseVulnerabilitys that determines the weaknesses to detect for.

    attacks: a list of type BaseAttacks that determines the methods that will be simulated to expose the defined vulnerabilities.

    [Optional] attacks_per_vulnerability_type: an int that determines the number of attacks to be simulated per vulnerability type. Defaulted to 1.

    [Optional] ignore_errors: a boolean which when set to True, ignores all exceptions raised during red teaming. Defaulted to False.
    """
    risk_assessment = red_teamer.red_team(
        model_callback=...,
        vulnerabilities=vulnerabilities,
        attacks=attacks,
        attacks_per_vulnerability_type=len(attacks),
        ignore_errors=True,
    )
    
    # Path to the folder to save the results
    result_path = "deepteam-results"  

    risk_assessment.save(to=str(result_path))
    print(risk_assessment.overview)
    

if __name__ == "__main__":
    start_time = timeit.default_timer()

    asyncio.run(
        main(use_default_owasp=True)
    )

    end_time = timeit.default_timer()

    print(f"Time taken: {end_time - start_time} seconds")