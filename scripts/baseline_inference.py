import os
import json
from openai import OpenAI
from env.schemas import Observation, Action
from scripts.graders import OpenEnvGrader

# Required Environment Variables
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:11434/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "mistral")
HF_TOKEN = os.environ.get("HF_TOKEN")
LOCAL_IMAGE_NAME = os.environ.get("LOCAL_IMAGE_NAME") # Optional for from_docker_image()

# All LLM calls use the OpenAI client configured via these variables
client = OpenAI(
    api_key=HF_TOKEN if HF_TOKEN else "ollama",
    base_url=API_BASE_URL
)

def llm_agent(obs: Observation) -> Action:
    print("STEP - Planning action for current observation")
    system_prompt = (
        "You are an AI coordinating a disaster response fleet. "
        "A fleet of agents represented by node IDs navigate a connected physical graph to rescue victims. "
        "Output an Action payload with an integer choice for each agent corresponding to a target neighbor."
    )
    
    user_prompt = f"{system_prompt}\n\nCurrent State: {obs.model_dump_json()}\n\nReturn ONLY a valid JSON object matching this schema: {{\"agent_moves\": [int, int, ...]}} with length matching the number of agents."
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
            timeout=120.0
        )
        
        text = response.choices[0].message.content
        parsed = json.loads(text)
        return Action(agent_moves=parsed.get("agent_moves", [0] * len(obs.agents)))
    except Exception as e:
        print(f"STEP - API/Parsing Error: {e}")
        
    return Action(agent_moves=[0] * len(obs.agents))

def main():
    print("START - OpenEnv Baseline Inference")
    
    grader = OpenEnvGrader()
    scores = {}
    
    print("STEP - Grading simple_rescue")
    scores["Simple Rescue"] = grader.grade_simple_rescue(llm_agent)
    print(f"STEP - Result: {scores['Simple Rescue']:.2f}/1.00")
    
    print("STEP - Grading blocked_rescue")
    scores["Blocked Rescue"] = grader.grade_blocked_rescue(llm_agent)
    print(f"STEP - Result: {scores['Blocked Rescue']:.2f}/1.00")
    
    print("STEP - Grading swarm_rescue")
    scores["Swarm Rescue"] = grader.grade_swarm_rescue(llm_agent)
    print(f"STEP - Result: {scores['Swarm Rescue']:.2f}/1.00")
    
    print("STEP - Grading expert_rescue")
    scores["Expert Rescue"] = grader.grade_expert_rescue(llm_agent)
    print(f"STEP - Result: {scores['Expert Rescue']:.2f}/1.00")
    
    print("STEP - Final Baseline Scores")
    for task, score in scores.items():
        print(f"STEP - {task}: {score:.2f}")
        
    print("END - OpenEnv Baseline Inference")

if __name__ == "__main__":
    main()
