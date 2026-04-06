from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name
        
    @abstractmethod
    def act(self, env):
        """
        Takes the environment instance and returns an action.
        This allowing agents to access graph and victim data.
        """
        pass
