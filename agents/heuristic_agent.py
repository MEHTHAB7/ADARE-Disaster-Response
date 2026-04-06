import networkx as nx
import random
from .base import BaseAgent

class HeuristicAgent(BaseAgent):
    """
    Heuristic Agent using A* Shortest Path to reach nearest rescue targets.
    """
    def __init__(self, name: str, graph: nx.MultiDiGraph):
        super().__init__(name)
        self.graph = graph
        self.target_victim = None
        
    def act(self, env):
        waiting = [v for v in env.victims_status if v['status'] == 'waiting']
        actions = []
        claimed_nodes = set()
        
        # If there's an LLM override, assign it to the structurally closest agent
        override_node = None
        if hasattr(self, 'override_target') and getattr(self, 'override_target') is not None:
            override_node = self.override_target['node']
            
        override_assigned_to = -1
        if override_node is not None:
            best_d = float('inf')
            for i in range(env.num_agents):
                try:
                    d = nx.shortest_path_length(env.graph, env.agents_pos[i], override_node, weight='length')
                    if d < best_d:
                        best_d = d
                        override_assigned_to = i
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass
            
        # Process each agent
        for agent_idx in range(env.num_agents):
            current_node = env.agents_pos[agent_idx]
            
            if not waiting:
                 neighbors = list(env.graph.neighbors(current_node))
                 actions.append(random.randint(0, len(neighbors) - 1) if neighbors else 0)
                 continue
                 
            target_victim = None
            
            if agent_idx == override_assigned_to:
                target_victim = self.override_target
                claimed_nodes.add(target_victim['node'])
            else:
                best_score = None
                for v in waiting:
                    if v['node'] in claimed_nodes:
                        continue
                    try:
                        d = nx.shortest_path_length(env.graph, current_node, v['node'], weight='length')
                        # Prioritize higher severity first by scoring with a tuple: (-severity, distance) 
                        score = (-v['severity'], d)
                        if best_score is None or score < best_score:
                            best_score = score
                            target_victim = v
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue
                
                if target_victim:
                    claimed_nodes.add(target_victim['node'])
                    
            if target_victim:
                try:
                    path = nx.shortest_path(env.graph, current_node, target_victim['node'], weight='length')
                    if len(path) > 1:
                        next_node = path[1]
                        neighbors = list(env.graph.neighbors(current_node))
                        if next_node in neighbors:
                            actions.append(neighbors.index(next_node))
                            continue
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass
                    
            # Fallback to random if no path or no target
            neighbors = list(env.graph.neighbors(current_node))
            actions.append(random.randint(0, len(neighbors) - 1) if neighbors else 0)
            
        return actions
