import os
import gymnasium as gym
from stable_baselines3 import PPO
from data.pipeline import OSMLoader, ScenarioGenerator
from env.core import OpenEnv

def train():
    os.makedirs("rl/models", exist_ok=True)
    
    # Initialize basic data sources
    loader = OSMLoader()
    gen = ScenarioGenerator(loader)
    
    # Stage 1: Basic Learning (Difficulty 1)
    scenario_easy = gen.generate(difficulty=1)
    env = OpenEnv(scenario_easy)
    
    print(f"--- Phase 1: Basic Training (Easy) ---")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./rl/logs/")
    model.learn(total_timesteps=10000)
    
    # Stage 2: Medium Difficulty
    print(f"--- Phase 2: Adaptation (Medium) ---")
    scenario_med = gen.generate(difficulty=3)
    env = OpenEnv(scenario_med) # New env instance with changed graph
    model.set_env(env)
    model.learn(total_timesteps=20000, reset_num_timesteps=False)
    
    # Final Stage: Expert
    print(f"--- Phase 3: Expert Challenge ---")
    scenario_expert = gen.generate(difficulty=5)
    env = OpenEnv(scenario_expert)
    model.set_env(env)
    model.learn(total_timesteps=20000, reset_num_timesteps=False)
    
    model.save("rl/models/ppo_disaster_v1")
    print("ADRAE++ RL Brain Saved.")

if __name__ == "__main__":
    train()
