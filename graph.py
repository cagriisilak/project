import math

class Graph:
    def __init__(self, nodes, edges):
        """
        Initialize graph from nodes and edges
        
        Args:
            nodes: list of dicts [{'id': 'D', 'type': 'depot', 'x': 0, 'y': 0}, ...]
            edges: list of [node1, node2, distance_km, traffic_factor]
        """
        self.nodes = {node['id']: node for node in nodes}
        self.edges = {}
        
        # Build adjacency with distances and traffic factors
        for edge in edges:
            if len(edge) == 4:  # Format: [from, to, distance, traffic]
                node1, node2, distance, traffic = edge
                self.edges[(node1, node2)] = {
                    'distance_km': distance,
                    'traffic_factor': traffic
                }
                self.edges[(node2, node1)] = {
                    'distance_km': distance,
                    'traffic_factor': traffic  # Undirected edges
                }
            elif isinstance(edge, dict):  # Format: {"from": "D", "to": "1", "distance": 8.0, "traffic_factor": 1.0}
                node1, node2 = edge['from'], edge['to']
                self.edges[(node1, node2)] = {
                    'distance_km': edge['distance'],
                    'traffic_factor': edge['traffic_factor']
                }
                self.edges[(node2, node1)] = {
                    'distance_km': edge['distance'],
                    'traffic_factor': edge['traffic_factor']
                }
    
    def get_edge_info(self, from_node, to_node):
        """Get edge information if exists"""
        return self.edges.get((from_node, to_node))
    
    def get_distance_km(self, from_node, to_node):
        """Get distance between nodes in km"""
        edge = self.get_edge_info(from_node, to_node)
        return edge['distance_km'] if edge else float('inf')
    
    def get_traffic_factor(self, from_node, to_node):
        """Get traffic factor for edge"""
        edge = self.get_edge_info(from_node, to_node)
        return edge['traffic_factor'] if edge else 1.0
    
    def calculate_actual_speed_kmh(self, from_node, to_node, base_speed):
        """Calculate actual speed on edge (km/h)"""
        traffic = self.get_traffic_factor(from_node, to_node)
        return base_speed * traffic
    
    def calculate_travel_time_minutes(self, from_node, to_node, base_speed):
        """Calculate travel time in minutes"""
        distance_km = self.get_distance_km(from_node, to_node)
        speed_kmh = self.calculate_actual_speed_kmh(from_node, to_node, base_speed)
        
        if speed_kmh == 0:
            return float('inf')
        
        time_hours = distance_km / speed_kmh
        return time_hours * 60  # Convert to minutes
    
    def get_neighbors(self, node):
        """Get all reachable neighbors from node"""
        neighbors = []
        for (from_node, to_node) in self.edges:
            if from_node == node:
                neighbors.append(to_node)
        return list(set(neighbors))  # Remove duplicates
    
    def calculate_euclidean_distance(self, node1_id, node2_id):
        """Calculate Euclidean distance between nodes"""
        node1 = self.nodes[node1_id]
        node2 = self.nodes[node2_id]
        
        dx = node1['x'] - node2['x']
        dy = node1['y'] - node2['y']
        
        return math.sqrt(dx**2 + dy**2)
    
    def get_nearest_bss_distance(self, from_node):
        """Get distance to nearest BSS station"""
        bss_nodes = [node_id for node_id, node_data in self.nodes.items() 
                    if node_data['type'] == 'bss']
        
        if not bss_nodes:
            return float('inf')
        
        min_distance = float('inf')
        for bss_node in bss_nodes:
            dist = self.get_distance_km(from_node, bss_node)
            if dist < min_distance:
                min_distance = dist
        
        return min_distance