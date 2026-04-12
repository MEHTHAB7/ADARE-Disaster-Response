from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class Coordinates(BaseModel):
    lat: float = Field(..., description="Latitude coordinate")
    lng: float = Field(..., description="Longitude coordinate")
    node_id: Optional[int] = Field(None, description="Underlying structural graph node ID")

class AgentState(BaseModel):
    id: int
    position: Coordinates
    status: str = "active"

class VictimState(BaseModel):
    id: int
    position: Coordinates
    severity: int = Field(..., description="Severity level from 1 to 10")
    status: str = Field(..., description="waiting, rescued, or deceased")

class Observation(BaseModel):
    agents: List[AgentState]
    victims: List[VictimState]
    time: int = Field(0, description="Elapsed simulation steps")
    rescued_count: int = Field(0, description="Total number of successfully evacuated victims")

class Action(BaseModel):
    agent_moves: List[int] = Field(..., description="An array of target topology neighbors mapped identically to the active agents array. e.g., [0, 2, 1, 0]")

class Reward(BaseModel):
    value: float = Field(0.0, description="The float representation of the step's success logic.")

class OpenEnvState(BaseModel):
    obs: Observation
    reward: float = Field(0.0, description="The float representation of the step's success logic.")
    done: bool
    info: Dict = Field(default_factory=dict)
