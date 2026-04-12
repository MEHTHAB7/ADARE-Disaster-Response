from .core import OpenEnv
from .schemas import *
from typing import Dict, Any, Tuple
import copy

class DisasterOpenEnv:
    """
    OpenEnv Compliant Adapter.
    Translates raw physics bounds into strict Pydantic Data schemas for benchmark grading.
    """
    def __init__(self, scenario: Dict[str, Any]):
        self._core_env = OpenEnv(scenario)
        self._current_obs: Observation = None
        self._reward_tracker = 0.0
        self._terminated = False
        self._info = {}

    def _convert_obs(self) -> Observation:
        graph = self._core_env.graph
        
        agents_pos = []
        for i, node in enumerate(self._core_env.agents_pos):
            data = graph.nodes[node]
            lat = data.get('y') or data.get('lat') or 0.0
            lng = data.get('x') or data.get('lon') or 0.0
            agents_pos.append(AgentState(id=i, position=Coordinates(lat=lat, lng=lng, node_id=node)))
            
        victims_pos = []
        for i, v in enumerate(self._core_env.victims_status):
            data = graph.nodes[v['node']]
            lat = data.get('y') or data.get('lat') or 0.0
            lng = data.get('x') or data.get('lon') or 0.0
            victims_pos.append(VictimState(
                id=i, 
                position=Coordinates(lat=lat, lng=lng, node_id=v['node']),
                severity=v['severity'],
                status=v['status']
            ))
            
        return Observation(
            agents=agents_pos,
            victims=victims_pos,
            time=self._core_env.current_time,
            rescued_count=self._core_env.rescued_count
        )

    def reset(self) -> Observation:
        self._core_env.reset()
        self._reward_tracker = 0.0
        self._terminated = False
        self._current_obs = self._convert_obs()
        return self._current_obs

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        # Unwrap Pydantic Action into raw list for the core physics bounds
        raw_actions = action.agent_moves
        _, reward_val, self._terminated, _, self._info = self._core_env.step(raw_actions)
        
        self._reward_tracker += reward_val
        self._current_obs = self._convert_obs()
        
        return self._current_obs, reward_val, self._terminated, self._info

    def state(self) -> OpenEnvState:
        return OpenEnvState(
            obs=self._current_obs if self._current_obs else self.reset(),
            reward=self._reward_tracker,
            done=self._terminated,
            info=self._info
        )
