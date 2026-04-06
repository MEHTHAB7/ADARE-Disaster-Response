import osmnx as ox
import networkx as nx
import pickle
import os
import random
import requests
from typing import Dict, List, Tuple

class OSMLoader:
    def __init__(self, cache_dir: str = "data/raw"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.default_locations = [
            "Manhattan, New York, USA",
            "London, UK",
            "Tokyo, Japan",
            "Paris, France",
            "Mumbai, India",
            "Berlin, Germany"
        ]

    def get_graph(self, location: str = None, dist: int = 1000) -> nx.MultiDiGraph:
        if not location:
            location = random.choice(self.default_locations)
        
        file_path = os.path.join(self.cache_dir, f"{location.replace(' ', '_').replace(',', '')}_{dist}.pkl")
        
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        
        try:
            # Download graph
            graph = ox.graph_from_address(location, dist=dist, network_type='drive')
            with open(file_path, 'wb') as f:
                pickle.dump(graph, f)
            return graph
        except Exception as e:
            print(f"Error loading graph for {location}: {e}")
            # Fallback to a simple grid graph for testing
            return nx.grid_2d_graph(20, 20)

class WeatherAPI:
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"

    def get_weather(self, location: str) -> Dict:
        if not self.api_key:
            # Simulation mode
            return {
                "status": random.choice(["clear", "rain", "storm", "flood"]),
                "intensity": random.uniform(0, 1)
            }
        
        try:
            params = {"q": location, "appid": self.api_key}
            response = requests.get(self.base_url, params=params)
            data = response.json()
            return {
                "status": data.get("weather", [{}])[0].get("main", "unknown").lower(),
                "intensity": data.get("rain", {}).get("1h", 0) / 10 # Sample intensity
            }
        except:
            return {"status": "clear", "intensity": 0}

class ScenarioGenerator:
    def __init__(self, loader: OSMLoader):
        self.loader = loader

    def generate(self, difficulty: int = 1, location: str = None) -> Dict:
        """
        Difficulty levels (1 to 7)
        1: Easy - Small area, few victims, no obstacles
        7: Expert - Large area, many high-severity victims, multiple flood/roadblocks
        """
        dist = 500 + (difficulty * 200)
        num_victims = 5 + (difficulty * 5)
        num_teams = 2 if difficulty < 4 else 4
        
        graph = self.loader.get_graph(location, dist=dist)
        nodes = list(graph.nodes())
        
        # Identify hospitals/shelters or just pick random nodes for now
        shelters = random.sample(nodes, num_teams)
        
        # Spawn victims
        victims = []
        for _ in range(num_victims):
            node = random.choice(nodes)
            severity = random.randint(1, 10)
            time_left = 100 - (severity * 10) + random.randint(0, 50)
            victims.append({
                "node": node,
                "severity": severity,
                "time_left": time_left,
                "status": "waiting"
            })
            
        # Spawn obstacles (flood zones/blocked roads)
        obstacles = []
        if difficulty > 2:
            num_obstacles = difficulty - 2
            blocked_nodes = random.sample(nodes, num_obstacles * 5)
            # Find edges connected to these nodes
            for edge in graph.edges(blocked_nodes):
                obstacles.append(edge)
                
        return {
            "graph": graph,
            "victims": victims,
            "shelters": shelters,
            "obstacles": obstacles,
            "difficulty": difficulty,
            "location": location or "Randomized"
        }
