import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import random
from typing import Dict, List, Tuple, Any

class OpenEnv(gym.Env):
    """
    Digital Twin Environment for Disaster Response.
    Supports Graph-based navigation, multiple victims, and dynamic obstacles.
    """
    def __init__(self, scenario: Dict):
        super(OpenEnv, self).__init__()
        self.scenario = scenario
        # Convert copy into an undirected MultiGraph to allow emergency reverse traversals out of dead ends
        self.graph = nx.MultiGraph(scenario["graph"].copy())
        
        self.victims = scenario["victims"]
        self.shelters = scenario["shelters"]
        self.obstacles = set(scenario["obstacles"])
        self.location_name = scenario.get("location", "Unknown City")
        
        # Physically block obstacle paths in the structural graph to force A* detours
        for edge in self.obstacles:
            u, v = edge[0], edge[1]
            if self.graph.has_edge(u, v):
                keys = list(self.graph[u][v].keys())
                for k in keys:
                    self.graph.remove_edge(u, v, k)
            if self.graph.has_edge(v, u):
                keys = list(self.graph[v][u].keys())
                for k in keys:
                    self.graph.remove_edge(v, u, k)
        
        # Attributes for mapping
        # Ensure graph has coordinates
        self.node_ids = list(self.graph.nodes())
        self.node_to_idx = {nid: i for i, nid in enumerate(self.node_ids)}
        self.idx_to_node = {i: nid for nid, i in self.node_to_idx.items()}
        
        # Calculate bounding box for normalization
        # Calculate bounding box for normalization with robust key detection
        lats = [self.graph.nodes[nid].get('y') or self.graph.nodes[nid].get('lat') or 0 for nid in self.node_ids]
        lons = [self.graph.nodes[nid].get('x') or self.graph.nodes[nid].get('lon') or 0 for nid in self.node_ids]
        
        if not lats or not lons:
            # Fallback for grid graphs without coords
            self.min_lat, self.max_lat = 0, 10
            self.min_lon, self.max_lon = 0, 10
        else:
            self.min_lat, self.max_lat = min(lats), max(lats)
            self.min_lon, self.max_lon = min(lons), max(lons)
        
        self.num_nodes = len(self.node_ids)
        self.num_agents = len(self.shelters)
        
        # Action space: Mapping to neighbor index
        self.max_degree = max([self.graph.degree(n) for n in self.node_ids]) if self.node_ids else 1
        self.action_space = spaces.Discrete(self.max_degree)
        
        # Observation space: [agent_pos, victim_pos x 5, time]
        self.obs_size = (self.num_agents * 2) + (5 * 3) + 1 
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self.obs_size,), dtype=np.float32
        )
        
        self.reset()

    def _get_normalized_pos(self, node_id: Any) -> Tuple[float, float]:
        data = self.graph.nodes[node_id]
        lat = data.get('y') or data.get('lat') or 0
        lon = data.get('x') or data.get('lon') or 0
        lat_norm = (lat - self.min_lat) / (self.max_lat - self.min_lat + 1e-9)
        lon_norm = (lon - self.min_lon) / (self.max_lon - self.min_lon + 1e-9)
        return (lat_norm * 2 - 1, lon_norm * 2 - 1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_time = 0
        self.agents_pos = [s for s in self.shelters]
        self.victims_status = [v.copy() for v in self.victims]
        self.rescued_count = 0
        self.total_victims = len(self.victims)
        
        return self._get_obs(), {}

    def _get_obs(self):
        obs = []
        # Agent positions
        for ap in self.agents_pos:
            obs.extend(self._get_normalized_pos(ap))
            
        # Top 5 waiting victims
        waiting_victims = [v for v in self.victims_status if v['status'] == 'waiting']
        
        # Rank primarily by severity (descending), then by euclidean distance estimate
        def dist_est(v):
            p1 = self.graph.nodes[self.agents_pos[0]]
            p2 = self.graph.nodes[v['node']]
            dist = (p1.get('x',0)-p2.get('x',0))**2 + (p1.get('y',0)-p2.get('y',0))**2
            return (-v['severity'], dist)
        
        waiting_victims.sort(key=dist_est)
        
        for v in waiting_victims[:5]:
            obs.extend(self._get_normalized_pos(v['node']))
            obs.append(v['severity'] / 10.0)
            
        # Pad if fewer than 5 victims
        while len(obs) < self.obs_size - 1:
            obs.append(0.0)
            
        # Time
        obs.append(self.current_time / 1000.0)
        
        return np.array(obs, dtype=np.float32)

    def step(self, actions):
        if isinstance(actions, (int, np.integer)):
            # Backward compatibility / single agent RL fallback
            acts = [actions]
            for _ in range(self.num_agents - 1):
                # Idle others
                acts.append(0)
        else:
            acts = actions

        reward = 0.0
        
        for agent_idx in range(self.num_agents):
            current_node = self.agents_pos[agent_idx]
            neighbors = list(self.graph.neighbors(current_node))
            action = acts[agent_idx] if agent_idx < len(acts) else 0
            
            if action < len(neighbors):
                next_node = neighbors[action]
                # Check for obstacles
                if (current_node, next_node) in self.obstacles or (next_node, current_node) in self.obstacles:
                    reward -= 2.0 # Higher penalty for obstacle collision
                    next_node = current_node
                else:
                    reward -= 0.05 # Step penalty
            else:
                next_node = current_node
                reward -= 0.5 # Invalid action penalty
                
            self.agents_pos[agent_idx] = next_node
            
            # Rescue logic (done per agent immediately to avoid concurrent counting bounds)
            for v in self.victims_status:
                if v['status'] == 'waiting' and v['node'] == next_node:
                    v['status'] = 'rescued'
                    # Reward scales significantly with severity so RL agents learn severity priority
                    reward += 10.0 + (v['severity'] * 5.0) 
                    self.rescued_count += 1
                    
        self.current_time += 1
                
        # Health tracking (Infinite time for rescue to ensure simulation completes visually)
        any_survivors = False
        for v in self.victims_status:
            if v['status'] == 'waiting':
                any_survivors = True
                    
        # Check termination (prevent 1000 step timeout since 40 targets can take 2000+ steps on big city maps)
        terminated = self.rescued_count == self.total_victims or not any_survivors or self.current_time > 5000
        truncated = False
        
        if self.rescued_count == self.total_victims and self.total_victims > 0:
            reward += 100.0
            
        return self._get_obs(), float(reward), terminated, truncated, {"rescued": self.rescued_count}

    def render(self):
        pass
