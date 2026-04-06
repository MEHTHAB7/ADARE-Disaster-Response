from data.pipeline import OSMLoader, ScenarioGenerator
from env.core import OpenEnv
from agents.heuristic_agent import HeuristicAgent
from agents.rl_agent import RLAgent
from agents.hybrid_agent import HybridAgent
import time

def test_models():
    print("Testing all agent models...")
    loader = OSMLoader()
    gen = ScenarioGenerator(loader)
    
    # Generate a small scenario
    scenario = gen.generate(difficulty=1)
    env = OpenEnv(scenario)
    
    models_to_test = {
        "Heuristic": HeuristicAgent("Heuristic", env.graph),
        "RL (Untrained Fallback)": RLAgent("RL"),
        "Hybrid (LLM Fallback)": HybridAgent("Hybrid", env.graph)
    }
    
    results = {}
    
    for name, agent in models_to_test.items():
        print(f"\n--- Testing {name} ---")
        try:
            env.reset()
            # Just test first 3 steps to ensure no crashes
            for step in range(3):
                action = agent.act(env)
                _, reward, done, _, info = env.step(action)
                print(f"Step {step+1}: Action taken = {action}, Reward = {reward:.2f}")
                if done: break
            results[name] = "Working"
        except Exception as e:
            print(f"Error testing {name}: {e}")
            results[name] = f"Error: {e}"
            
    print("\n--- Summary ---")
    for name, status in results.items():
        print(f"{name}: {status}")

if __name__ == "__main__":
    test_models()
