import os
import json
from openai import OpenAI
from env.schemas import Observation, Action
from scripts.graders import OpenEnvGrader

# Detect if user provided an actual OpenAI API Key, otherwise fallback to local Ollama
api_key = os.environ.get("OPENAI_API_KEY")

if not api_key or api_key.lower() == "ollama":
    print("Using Local Ollama endpoint. Provide OPENAI_API_KEY to benchmark on OpenAI models.")
    client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
    model_name = "mistral"
else:
    print("Detected OPENAI_API_KEY. Using OpenAI gpt-4o-mini.")
    client = OpenAI(api_key=api_key)
    model_name = "gpt-4o-mini"

def llm_agent(obs: Observation) -> Action:
    system_prompt = (
        "You are an AI coordinating a disaster response fleet. "
        "A fleet of agents represented by node IDs navigate a connected physical graph to rescue victims. "
        "Output an Action payload with an integer choice for each agent corresponding to a target neighbor."
    )
    
    # Construct context dynamically based on environment observation
    user_prompt = f"{system_prompt}\n\nCurrent State: {obs.model_dump_json()}\n\nReturn ONLY a valid JSON object matching this schema: {{\"agent_moves\": [int, int, ...]}} with length matching the number of agents."
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
            timeout=120.0
        )
        
        text = response.choices[0].message.content
        # Parse the JSON response
        parsed = json.loads(text)
        return Action(agent_moves=parsed.get("agent_moves", [0] * len(obs.agents)))
    except Exception as e:
        print(f"API/Parsing Error: {e}")
        
    # Return fallback heuristic: stand still
    return Action(agent_moves=[0] * len(obs.agents))

def main():
    print("Disclaimer: Using OpenAI python client pointing to local Ollama Mistral model on http://localhost:11434")
    
    grader = OpenEnvGrader()
    
    scores = {}
    print("--- Starting OpenEnv Baseline Inference (Ollama) ---")
    
    scores["Simple Rescue"] = grader.grade_simple_rescue(llm_agent)
    print(f"Result -> {scores['Simple Rescue']:.2f}/1.00\n")
    
    scores["Blocked Rescue"] = grader.grade_blocked_rescue(llm_agent)
    print(f"Result -> {scores['Blocked Rescue']:.2f}/1.00\n")
    
    scores["Swarm Rescue"] = grader.grade_swarm_rescue(llm_agent)
    print(f"Result -> {scores['Swarm Rescue']:.2f}/1.00\n")
    
    scores["Expert Rescue"] = grader.grade_expert_rescue(llm_agent)
    print(f"Result -> {scores['Expert Rescue']:.2f}/1.00\n")
    
    print("--- Final Baseline Scores ---")
    for task, score in scores.items():
        print(f"{task}: {score:.2f}")

if __name__ == "__main__":
    main()
