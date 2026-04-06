import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

from data.pipeline import OSMLoader, ScenarioGenerator
from env.core import OpenEnv
from agents.heuristic_agent import HeuristicAgent
from agents.rl_agent import RLAgent
from agents.hybrid_agent import HybridAgent

def run_benchmark(n_episodes=5, difficulty=3):
    loader = OSMLoader()
    gen = ScenarioGenerator(loader)
    
    # Define Agents
    agents = {
        "Heuristic++": lambda g: HeuristicAgent("Heuristic", g),
        "Hybrid (LLM+HL)": lambda g: HybridAgent("Hybrid", g),
        "Random": lambda g: None # We'll handle random in the loop
    }
    
    results = []
    
    for agent_name, agent_factory in agents.items():
        print(f"Benchmarking {agent_name}...")
        
        for ep in range(n_episodes):
            scenario = gen.generate(difficulty=difficulty)
            env = OpenEnv(scenario)
            agent = agent_factory(env.graph) if agent_name != "Random" else None
            
            obs, info = env.reset()
            total_reward = 0
            start_time = time.time()
            
            while True:
                if agent_name == "Random":
                    action = env.action_space.sample()
                else:
                    action = agent.act(env)
                    
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                if terminated or truncated:
                    break
            
            end_time = time.time()
            
            results.append({
                "Agent": agent_name,
                "Episode": ep,
                "Survival Rate": (env.rescued_count / env.total_victims) * 100 if env.total_victims > 0 else 0,
                "Time Taken": env.current_time,
                "Reward": total_reward,
                "Wall Clock (s)": end_time - start_time
            })
            
    df = pd.DataFrame(results)
    
    # Save results
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/benchmark_metrics.csv", index=False)
    
    # Generate Summary Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Agent", y="Survival Rate", palette="viridis")
    plt.title(f"Survival Rate Comparison (Difficulty {difficulty})")
    plt.savefig("results/survival_comparison.png")
    
    print("\nBenchmark Summary:")
    print(df.groupby("Agent")[["Survival Rate", "Time Taken", "Reward"]].mean())
    return df

if __name__ == "__main__":
    run_benchmark(n_episodes=3, difficulty=2)
