import numpy as np
import random
from .base import BaseAgent

class RLAgent(BaseAgent):
    """
    Reinforcement Learning Agent using Stable-Baselines3 interface.
    """
    def __init__(self, name: str, model=None):
        super().__init__(name)
        self.model = model
            
    def act(self, env):
        obs = env._get_obs()
        actions = []
        
        # Primary Agent Neural Inference
        if self.model:
            action, _states = self.model.predict(obs, deterministic=True)
            actions.append(int(action))
        else:
            actions.append(random.randint(0, env.action_space.n - 1))
            
        # Secondary Agents (Baseline Swarm Automation)
        for i in range(1, env.num_agents):
            current_node = env.agents_pos[i]
            neighbors = list(env.graph.neighbors(current_node))
            actions.append(random.randint(0, len(neighbors) - 1) if neighbors else 0)
            
        return actions
