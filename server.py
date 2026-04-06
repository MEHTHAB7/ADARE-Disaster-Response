import threading
import time
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from typing import List, Dict, Any

from data.pipeline import OSMLoader, ScenarioGenerator
from env.core import OpenEnv
from agents.heuristic_agent import HeuristicAgent
from agents.rl_agent import RLAgent
from agents.hybrid_agent import HybridAgent
from stable_baselines3 import PPO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared State
class SimState:
    def __init__(self):
        self.env = None
        self.agent = None
        self.running = False
        self.thread = None
        self.last_state = {}
        self.lock = threading.Lock()

    def loop(self):
        while True:
            with self.lock:
                if not self.running:
                    break
                    
                if self.env and self.agent:
                    try:
                        action = self.agent.act(self.env)
                        _, reward, terminated, _, info = self.env.step(action)
                    except Exception as e:
                        print(f"Agent/Env execution error: {e}")
                        terminated = True
                        reward = 0
                    
                    # Process state for UI
                    agents_data = []
                    for ap in self.env.agents_pos:
                        node_data = self.env.graph.nodes[ap]
                        lat = node_data.get('y') or node_data.get('lat') or 0
                        lng = node_data.get('x') or node_data.get('lon') or 0
                        agents_data.append({"lat": lat, "lng": lng})
                    
                    victims_data = []
                    for v in self.env.victims_status:
                        node_data = self.env.graph.nodes[v['node']]
                        lat = node_data.get('y') or node_data.get('lat') or 0
                        lng = node_data.get('x') or node_data.get('lon') or 0
                        victims_data.append({
                            "lat": lat, "lng": lng,
                            "status": v['status'], "severity": v['severity']
                        })
                    
                    self.last_state = {
                        "agents": agents_data,
                        "victims": victims_data,
                        "time": self.env.current_time,
                        "rescued": self.env.rescued_count,
                        "total": self.env.total_victims,
                        "reward": reward,
                        "terminated": terminated,
                        "location": self.env.location_name
                    }
                    
                    if terminated:
                        self.running = False
                        
            time.sleep(0.5) # Sim speed

state_manager = SimState()

class StartConfig(BaseModel):
    location: str = None
    difficulty: int = 1
    agent_type: str = "heuristic"

@app.post("/start")
async def start_sim(config: StartConfig):
    with state_manager.lock:
        state_manager.running = False
    
    # Wait for old thread
    if state_manager.thread:
        state_manager.thread.join(timeout=2)
    
    loader = OSMLoader()
    gen = ScenarioGenerator(loader)
    scenario = gen.generate(difficulty=config.difficulty, location=config.location)
    
    state_manager.env = OpenEnv(scenario)
    
    if config.agent_type == "heuristic":
        state_manager.agent = HeuristicAgent("Heuristic", state_manager.env.graph)
    elif config.agent_type == "hybrid":
        state_manager.agent = HybridAgent("Hybrid", state_manager.env.graph)
    elif config.agent_type == "rl":
        try:
            model = PPO.load("rl/models/ppo_disaster_v1")
            state_manager.agent = RLAgent("RL-PPO", model=model)
        except Exception:
            print("PPO Model not found, falling back to untrained RL Agent.")
            state_manager.agent = RLAgent("RL-PPO")
    else:
        state_manager.agent = HeuristicAgent("Default", state_manager.env.graph)
        
    state_manager.running = True
    state_manager.last_state = {"status": "starting", "location": state_manager.env.location_name}
    state_manager.thread = threading.Thread(target=state_manager.loop, daemon=True)
    state_manager.thread.start()
    
    return {"status": "started", "location": state_manager.env.location_name}

@app.get("/state")
async def get_state():
    return state_manager.last_state

@app.post("/stop")
async def stop_sim():
    with state_manager.lock:
        state_manager.running = False
    return {"status": "stopped"}

import os
if os.path.exists("ui/dist"):
    app.mount("/", StaticFiles(directory="ui/dist", html=True), name="static")
else:
    print("Warning: ui/dist not found. React App will not be served.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
