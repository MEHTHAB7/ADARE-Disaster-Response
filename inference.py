import os
import json
from openai import OpenAI
from env.schemas import Observation, Action
from scripts.graders import OpenEnvGrader

# Required Environment Variables
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# Initialize strictly using the exact expected class and variables.
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

def _fetch_action(user_prompt: str) -> str:
    response = client.chat.completions.create(
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
    
    text = _fetch_action(user_prompt)
    parsed = json.loads(text)
    return Action(agent_moves=parsed.get("agent_moves", [0] * len(obs.agents)))

def run_task(grader, task_name, task_args, max_steps):
    print(f"[START] task={task_name} env=disaster model={MODEL_NAME}", flush=True)
    env = grader._create_env(**task_args)
    obs = env.reset()
    
    rewards_list = []
    success = False
    
    for step in range(1, max_steps + 1):
        action_str = ""
        error_msg = "null"
        done = False
        reward_val = 0.0
        
        try:
            action = llm_agent(obs)
            action_dict = action.model_dump()
            action_str = json.dumps(action_dict).replace(' ', '')
            
            obs, reward_obj, done, info = env.step(action)
            # Handle reward both from pydantic Model or raw float
            reward_val = reward_obj.value if hasattr(reward_obj, 'value') else float(reward_obj)
            
        except Exception as e:
            error_msg = str(e).replace(' ', '_').replace('\n', '_')
            if not action_str:
                action_str = "error"
            done = True
            
        rewards_list.append(reward_val)
        
        print(f"[STEP] step={step} action={action_str} reward={reward_val:.2f} done={str(done).lower()} error={error_msg}", flush=True)
        
        if done:
            break
            
    # Calculate success and score
    raw_env = env._core_env if hasattr(env, '_core_env') else env
    success = (raw_env.rescued_count == raw_env.total_victims) and (raw_env.total_victims > 0)
    
    survival_rate = raw_env.rescued_count / max(1, raw_env.total_victims)
    time_penalty = (raw_env.current_time / max_steps) * 0.15
    final_score = survival_rate - (time_penalty if survival_rate > 0.0 else 0)
    final_score = max(0.01, min(0.99, final_score))
    
    rewards_str = ",".join([f"{r:.2f}" for r in rewards_list])
    print(f"[END] task={task_name} score={final_score:.2f} success={str(success).lower()} steps={len(rewards_list)} rewards={rewards_str}", flush=True)

def main():
    grader = OpenEnvGrader()
    
    tasks = [
        ("simple_rescue", {"location": "Grid-Simple", "num_agents": 1, "num_victims": 1, "use_obstacles": False, "seed": 42}, 15),
        ("blocked_rescue", {"location": "Grid-Blocked", "num_agents": 2, "num_victims": 5, "use_obstacles": True, "seed": 42}, 30),
        ("swarm_rescue", {"location": "Grid-Swarm", "num_agents": 4, "num_victims": 40, "use_obstacles": True, "seed": 42}, 50),
        ("expert_rescue", {"location": "Grid-Expert", "num_agents": 8, "num_victims": 100, "use_obstacles": True, "seed": 99}, 100)
    ]
    
    for task_name, task_args, max_steps in tasks:
        run_task(grader, task_name, task_args, max_steps)

if __name__ == "__main__":
    main()
