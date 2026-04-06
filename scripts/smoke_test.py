from data.pipeline import OSMLoader, ScenarioGenerator
from env.core import OpenEnv
from agents.heuristic_agent import HeuristicAgent
import networkx as nx

def main():
    print("Initializing ADRAE++ Smoke Test...")
    loader = OSMLoader()
    gen = ScenarioGenerator(loader)
    
    # Generate scenario
    scenario = gen.generate(difficulty=1)
    print(f"Generated scenario for: {scenario['location']}")
    
    # Create environment
    env = OpenEnv(scenario)
    obs, info = env.reset()
    
    # Run a few steps with Heuristic Agent
    agent = HeuristicAgent("Heuristic-Alpha", scenario['graph'])
    
    total_reward = 0
    for i in range(20):
        action = agent.act(env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {i}: Action={action}, Reward={reward:.2f}, Rescued={info['rescued']}")
        if terminated:
            print("Scenario Complete!")
            break
            
    print(f"Smoke Test Finished. Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    main()
