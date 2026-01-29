# utils.py
import numpy as np
import json
import os

def load_scenario(scenario_path):
    """Load and validate scenario file"""
    if not os.path.isabs(scenario_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        scenario_path = os.path.join(script_dir, scenario_path)
    
    with open(scenario_path, 'r') as f:
        config = json.load(f)
    
    # Validate required fields
    required_fields = ['nodes', 'edges', 'modules', 'starting_node']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")
    
    return config

def normalize_value(value, min_val, max_val):
    """Normalize value to [0, 1] range"""
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val) if max_val > min_val else 0.0))

def calculate_energy_to_bss(graph, current_node, vehicle_params, current_load):
    """Calculate energy needed to reach nearest BSS"""
    bss_nodes = [node['id'] for node in graph.nodes.values() if node['type'] == 'bss']
    
    if not bss_nodes:
        return float('inf')
    
    min_energy = float('inf')
    for bss in bss_nodes:
        # Simplified - in practice, you'd use the full energy calculation
        distance = graph.get_distance_km(current_node, bss)
        if distance < float('inf'):
            # Approximate energy (0.2 kWh per km is a rough estimate)
            energy = distance * 0.2
            if energy < min_energy:
                min_energy = energy
    
    return min_energy