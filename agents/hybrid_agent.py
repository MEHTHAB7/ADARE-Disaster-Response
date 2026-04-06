import requests
import json
import random
from .base import BaseAgent
from .heuristic_agent import HeuristicAgent

class HybridAgent(BaseAgent):
    """
    Hybrid RL + LLM Agent.
    Planner: LLM (Strategic high-level planning)
    Executor: Heuristic/RL (Tactical local execution)
    """
    def __init__(self, name: str, graph, ollama_url: str = "http://localhost:11434/api/generate"):
        super().__init__(name)
        self.ollama_url = ollama_url
        self.executor = HeuristicAgent(f"{self.name}-executor", graph)
        self.current_goal_node = None
        self.plan_life = 0
        
    def _fetch_llm_strategy(self, env):
        waiting = [v for v in env.victims_status if v['status'] == 'waiting']
        if not waiting: return None
        
        # Scenario context for LLM
        context = f"Disaster Response Digital Twin. Location: {env.location_name}. Survivors: {len(waiting)}. "
        context += f"Heuristic executor active. Objective: Optimal resource allocation."
        
        payload = {
            "model": "mistral",
            "prompt": f"Strategy Request: {context} Current Top Priority Node Suggestions: {[v['node'] for v in waiting[:3]]}. Output the best objective node ID.",
            "stream": False
        }
        
        try:
            resp = requests.post(self.ollama_url, json=payload, timeout=4.0)
            if resp.status_code == 200:
                data = resp.json()
                text = data.get("response", "")
                # Attempt to extract a node ID from the text
                for v in waiting:
                    if str(v['node']) in text:
                        return v['node']
            return waiting[0]['node']
        except Exception as e:
            print(f"LLM Strategy fallback (Timeout/Error: {e})")
            # Fallback: Highest severity survivor
            waiting.sort(key=lambda x: x['severity'], reverse=True)
            return waiting[0]['node']

    def act(self, env):
        # Check if current goal is still actively waiting
        if self.current_goal_node is not None:
            still_waiting = any(v['node'] == self.current_goal_node and v['status'] == 'waiting' for v in env.victims_status)
            if not still_waiting:
                self.plan_life = 0  # Force immediate re-plan

        # Update high-level plan every 20 steps
        if self.current_goal_node is None or self.plan_life <= 0:
            self.current_goal_node = self._fetch_llm_strategy(env)
            self.plan_life = 20
        
        self.plan_life -= 1
        
        # Guide the heuristic executor to the LLM-selected target
        self.executor.target_victim = {'node': self.current_goal_node, 'status': 'waiting'}
        return self.executor.act(env)
