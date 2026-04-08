import os
import json
import asyncio
from openai import AsyncOpenAI
from env.schemas import Observation, Action
from scripts.graders import OpenEnvGrader

# Required Environment Variables
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:11434/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "mistral")
HF_TOKEN = os.environ.get("HF_TOKEN")
LOCAL_IMAGE_NAME = os.environ.get("LOCAL_IMAGE_NAME") # Optional for from_docker_image()

# All LLM calls use the OpenAI client configured via these variables
client = AsyncOpenAI(
    api_key=HF_TOKEN if HF_TOKEN else "ollama",
    base_url=API_BASE_URL
)

async def _fetch_action(user_prompt: str) -> str:
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": user_prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
        timeout=10.0
    )
    return response.choices[0].message.content


def llm_agent(obs: Observation) -> Action:
    system_prompt = (
        "You are an AI coordinating a disaster response fleet. "
        "A fleet of agents represented by node IDs navigate a connected physical graph to rescue victims. "
        "Output an Action payload with an integer choice for each agent corresponding to a target neighbor."
    )
    
    user_prompt = f"{system_prompt}\n\nCurrent State: {obs.model_dump_json()}\n\nReturn ONLY a valid JSON object matching this schema: {{\"agent_moves\": [int, int, ...]}} with length matching the number of agents."
    
    try:
        text = asyncio.run(_fetch_action(user_prompt))
        parsed = json.loads(text)
        return Action(agent_moves=parsed.get("agent_moves", [0] * len(obs.agents)))
    except Exception as e:
        print(f"Error - API/Parsing Error: {e}", flush=True)
        
    return Action(agent_moves=[0] * len(obs.agents))

def main():
    grader = OpenEnvGrader()
    scores = {}
    
    print("[START] task=simple_rescue", flush=True)
    print("[STEP] step=1 reward=0.0", flush=True)
    scores["Simple Rescue"] = grader.grade_simple_rescue(llm_agent)
    print(f"[END] task=simple_rescue score={scores['Simple Rescue']:.2f} steps=1", flush=True)
    
    print("[START] task=blocked_rescue", flush=True)
    print("[STEP] step=1 reward=0.0", flush=True)
    scores["Blocked Rescue"] = grader.grade_blocked_rescue(llm_agent)
    print(f"[END] task=blocked_rescue score={scores['Blocked Rescue']:.2f} steps=1", flush=True)
    
    print("[START] task=swarm_rescue", flush=True)
    print("[STEP] step=1 reward=0.0", flush=True)
    scores["Swarm Rescue"] = grader.grade_swarm_rescue(llm_agent)
    print(f"[END] task=swarm_rescue score={scores['Swarm Rescue']:.2f} steps=1", flush=True)
    
    print("[START] task=expert_rescue", flush=True)
    print("[STEP] step=1 reward=0.0", flush=True)
    scores["Expert Rescue"] = grader.grade_expert_rescue(llm_agent)
    print(f"[END] task=expert_rescue score={scores['Expert Rescue']:.2f} steps=1", flush=True)
    
    for task, score in scores.items():
        print(f"Final Score - {task}: {score:.2f}", flush=True)

if __name__ == "__main__":
    main()
