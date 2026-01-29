import gymnasium as gym
import numpy as np
import json
from gymnasium import spaces
from energy import calculate_energy_consumption
from battery import BatterySystem
from graph import Graph

class EVRPEnvironment(gym.Env):
    """Electric Vehicle Routing Problem with Battery Swapping Environment"""
    
    metadata = {'render_modes': ['human'], 'render_fps': 4}
    
    def __init__(self, config_path='scenario_small.json'):
        super().__init__()

        # fix for no such file or directory error (works i think)
        import os

        if not os.path.isabs(config_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, config_path)

        print(f"Loading environment from: {config_path}")  # Debug line

        # Load scenario
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize components
        self.graph = Graph(self.config['nodes'], self.config['edges'])
        self.battery = BatterySystem(self.config['modules'])
        
        # Environment parameters
        self.base_speed = self.config['base_speed']  # km/h
        self.starting_node = self.config['starting_node']
        self.vehicle_params = self.config.get('vehicle_params', self.config.get('vehicle', {}))
        
        # Get all nodes and customers
        self.all_nodes = [node['id'] for node in self.config['nodes']]
        self.customer_nodes = [node['id'] for node in self.config['nodes'] 
                              if node['type'] == 'customer']
        
        # observation space bug fix
        self.num_customers = len(self.customer_nodes)
        self.num_modules = len(self.config['modules'])
        #self.state_dim = 4 + 1 + self.num_customers + 6 
        self.state_dim = 4 + 1 + (self.num_modules * 2) + self.num_customers + 7

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.state_dim,), dtype=np.float32
        )
        print(f"Observation space shape set to: {self.observation_space.shape}")


        # State variables
        self.current_node = None
        self.remaining_customers = None
        self.time_elapsed = None  # minutes
        self.total_swap_cost = None
        self.current_load_kg = None
        self.route_history = None
        self.arrival_times = None
        self.just_swapped = None
        self.previous_total_cost = None
        self.total_cost = None

        #added, fix 2
        self.max_steps = 150
        self.step_count = 0
        
        # Action space: choose next node (all nodes)
        self.action_space = spaces.Discrete(len(self.all_nodes))
        
        
        
        # Initialize
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        self.step_count = 0 # new fix 2

        # Reset state variables
        self.current_node = self.starting_node
        self.remaining_customers = set(self.customer_nodes)
        self.time_elapsed = 0.0  # minutes
        self.total_swap_cost = 0.0
        self.packages_delivered = 0
        self.current_load_kg = len(self.customer_nodes) * 5  # 5kg per package
        self.route_history = [self.current_node]
        self.arrival_times = {self.current_node: 0.0}
        self.just_swapped = False
        self.step_count = 0
        self.previous_total_cost = 0.0
        self.total_cost = 0.0
        
        # reset batteries bugfix
        import copy
        modules_config = copy.deepcopy(self.config['modules'])
        
        self.battery = BatterySystem(modules_config)
        
        # Get initial state
        state = self._get_state()

        info = self._get_info()
        
        return state, info
    
    def step(self, action):
        """Take action (move to target node)"""
        # 1. Initial setup and truncation check
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        
        target_node = self.all_nodes[action]

        # test tracking customers before action
        self.remaining_customers_before_action = self.remaining_customers.copy()

        
        # 2. Invalid action handling
        # If somehow an invalid action is selected (shouldn't happen with masking),
        # we'll treat it as a terminal state with heavy penalty
        if not self._is_action_valid(target_node):
            print(f"Warning: Invalid action {target_node} selected despite masking!")
            state = self._get_state()
            info = self._get_info()
            return state, -100.0, True, True, info
        
        # 3. Track costs and execute action
        cost_before = self._calculate_current_cost()
        self._execute_action(target_node)
        cost_after = self._calculate_current_cost()
        
        # 4. Calculate base reward from cost difference
        reward = -(cost_after - cost_before)
        
        # 5. Apply reward shaping
        reward = self._apply_reward_shaping(reward, target_node)
        
        # 6. Check termination conditions
        terminated = self._check_termination()
        
        # 7. Scale reward for RL stability
        reward = reward / 100.0
        
        # 8. Prepare return values
        state = self._get_state()
        info = self._get_info()
        
        return state, reward, terminated, truncated, info
    #following are helper methods for step (next 4)

    def _handle_invalid_action(self):
        """Handle invalid action with penalty"""
        reward = -50.0 /100
        state = self._get_state()
        info = self._get_info()
        truncated = self.step_count >= self.max_steps
        return state, reward, False, truncated, info

    def _execute_action(self, target_node):
        """Execute all effects of moving to target node"""
        # Move to node (updates position, time, battery)
        travel_time, energy_used = self._move_to_node(target_node)
        
        # Handle customer delivery
        if target_node in self.remaining_customers:
            self.remaining_customers.remove(target_node)
            self.current_load_kg -= 5
        
        # Handle battery swap if at BSS
        if self._get_node_type(self.current_node) == 'bss':
            self._handle_battery_swap()

    def _handle_battery_swap(self):
        """Handle battery swapping logic"""
        total_soc = self.battery.get_total_soc()
        should_swap = total_soc < 40.0 or self.battery.needs_swap(threshold_percent=20.0)
        
        if should_swap:
            swap_time, swap_cost = self.battery.swap_modules()
            self.time_elapsed += swap_time
            self.total_swap_cost += swap_cost
            self.just_swapped = True

    def _apply_reward_shaping(self, base_reward, target_node):
        """Apply additional reward shaping on top of base cost-based reward"""
        reward = base_reward

        if target_node in self.remaining_customers_before_action:  # Was delivered in _execute_action
            reward += 200.0  # Delivery bonus
            
            if len(self.remaining_customers) == 0:  # All delivered
                reward += 500.0  # Completion bonus
        
        # Battery management incentives
        soc = self.battery.get_total_soc()
        current_type = self._get_node_type(self.current_node)
        
        if soc < 30.0 and current_type == 'bss':
            reward += 5.0  # at BSS when battery low
        
        if soc < 10.0 and current_type != 'bss':
            reward -= 2.0  # Battery critically low penalty
        
        # Exploration incentive
        if target_node not in self.route_history:
            reward += 1.0
        
        # Return to depot after all deliveries
        if len(self.remaining_customers) == 0 and self.current_node == 'D':
            reward += 1000.0  # Final completion bonus
            print(f"  Return to depot bonus: +1000.0")
        
        return reward
    

    def _check_termination(self):
        """Check if episode should end"""
        # Success condition: all delivered and returned to depot
        if len(self.remaining_customers) == 0 and self.current_node == 'D':
            return True
        
        # Failure condition: battery depleted with no reachable BSS
        if self.battery.get_total_soc() <= 0:
            # Check if any BSS is reachable
            for node in self.all_nodes:
                if self._get_node_type(node) == 'bss' and self._is_action_valid(node):
                    return False  # Still have reachable BSS, don't terminate
            return True  # No reachable BSS, terminate
        
        return False

    def _calculate_current_cost(self):
        """Calculate current total cost (for incremental reward calculation)
        Returns: current sum of arrival times + swap costs"""
        # Sum of arrival times for all visited nodes
        arrival_time_sum = sum(self.arrival_times.values())
    
        # Current swap cost
        swap_cost = self.battery.swapped_modules_count * 50

        # Debug
        if hasattr(self, 'debug_print') and self.debug_print:
            print(f"  Arrival times sum: {arrival_time_sum:.1f}")
            print(f"  Swap cost: {swap_cost:.1f}")
    
        return arrival_time_sum + swap_cost
    
    def _is_action_valid(self, target_node):
        """Check if moving to target node is valid"""
        # Can't stay at same node
        if target_node == self.current_node:
            return False
        
        # Edge must exist
        if not self.graph.get_edge_info(self.current_node, target_node):
            return False
        
        # Can't revisit delivered customers
        if target_node in self.customer_nodes and target_node not in self.remaining_customers:
            return False
        
        # Can only return to depot after all deliveries
        if target_node == 'D' and len(self.remaining_customers) > 0:
            return False
        
        # Must have enough battery to reach target
        energy_needed = self._calculate_edge_energy(self.current_node, target_node)
        available_energy = self.battery.get_available_energy_kwh()


        # Debug
        #print(f"  {self.current_node} -> {target_node}: need {energy_needed:.2f}kWh, have {available_energy:.2f}kWh")

        if energy_needed > available_energy:
            return False
        
        return True
    
    def get_priority_actions(self):
        """Get action indices sorted by priority (for guided exploration)"""
        action_priorities = []
        
        for i, node in enumerate(self.all_nodes):
            if self._is_action_valid(node):
                priority = 0  # Base priority
                
                # Higher priority for customers that need delivery
                if node in self.remaining_customers:
                    priority += 2
                
                # Higher priority for BSS when battery is low
                node_type = self._get_node_type(node)
                if node_type == 'bss' and self.battery.get_total_soc() < 30.0:
                    priority += 1
                
                # Lower priority for depot unless all deliveries done
                if node == 'D' and len(self.remaining_customers) > 0:
                    priority -= 1
                
                action_priorities.append((i, priority))
        
        # Sort by priority (descending)
        action_priorities.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in action_priorities]

    def get_valid_actions(self):
        """Get list of valid action indices"""
        valid_actions = []
        
        for i, node in enumerate(self.all_nodes):
            if self._is_action_valid(node):
                valid_actions.append(i)
    
        return valid_actions
    
    def get_action_mask(self):
        """Get binary mask for valid actions (1=valid, 0=invalid)"""
        mask = np.zeros(len(self.all_nodes), dtype=np.int8)
        for i in self.get_valid_actions():
            mask[i] = 1
        return mask
    
    def _move_to_node(self, target_node):
        """Execute movement to target node"""
        # Calculate travel time
        travel_time = self.graph.calculate_travel_time_minutes(
            self.current_node, target_node, self.base_speed
        )
    
        # Calculate energy consumption
        energy_used = self._calculate_edge_energy(self.current_node, target_node)
    
        # Consume energy from total pool (simplified!)
        success = self.battery.consume_energy(energy_used)
    
        if not success:
            # Battery depleted
            print(f"Battery depleted! Needed {energy_used:.2f}kWh, "
              f"had {self.battery.get_available_energy_kwh():.2f}kWh")
            # Could end episode here or continue with penalty
    
        # Update time
        self.time_elapsed += travel_time
    
        # Update current node and arrival time
        self.current_node = target_node
        self.route_history.append(target_node)
        self.arrival_times[target_node] = self.time_elapsed
    
        return travel_time, energy_used
    
    def _calculate_edge_energy(self, from_node, to_node):
        """Calculate energy needed for edge (in kWh)"""
        # Get edge information
        edge = self.graph.get_edge_info(from_node, to_node)
        if not edge:
            return float('inf')
        
        # Get parameters
        distance_km = edge['distance_km']
        traffic_factor = edge['traffic_factor']
        
        # Calculate actual speed in m/s
        actual_speed_kmh = self.base_speed * traffic_factor
        actual_speed_ms = actual_speed_kmh * (1000 / 3600)  # km/h → m/s
        
        # Convert distance to meters
        distance_m = distance_km * 1000
        
        # Get vehicle parameters
        M = self.vehicle_params.get('M', self.vehicle_params.get('base_mass', 1500))
        f = self.vehicle_params.get('f', 0.01)
        rho = 1.205  # air density
        Cx = self.vehicle_params.get('Cx', 0.3)
        A = self.vehicle_params.get('A', 2.5)
        m = self.vehicle_params.get('m', 100)  # mass_factor
        alpha = 0.86  # angle
        
        # Add current load to vehicle mass
        M_total = M + self.current_load_kg
        
        # Calculate energy using your equation
        energy_kwh = calculate_energy_consumption(
            M=M_total, f=f, rho=rho, Cx=Cx, A=A,
            v=actual_speed_ms, m=m, alpha=alpha, d=distance_m
        )
        
        # debug line:
        #print(f"  Energy needed from {from_node}→{to_node}: {energy_kwh:.4f} kWh")

        return energy_kwh
    
    def _calculate_final_cost(self):
        """
        Calculate final cost as per your specification:
        cost = sum(arrival_times_at_all_nodes) + (50 x modules_swapped)
        """
        total_arrival_time = sum(self.arrival_times.values())  # minutes
        swap_cost = self.battery.swapped_modules_count * 50  # dollars
    
        total_cost = total_arrival_time + swap_cost

        print(f"  Modules swapped: {self.battery.swapped_modules_count}")
        print(f"  Swap cost: {swap_cost:.2f}")
        print(f"  TOTAL COST: {total_cost:.1f}")
    
        return total_cost
    
    def _get_node_type(self, node_id):
        """Get type of node"""
        for node in self.config['nodes']:
            if node['id'] == node_id:
                return node['type']
        return 'intersection'
    
    def _get_state(self):
        """Convert internal state to observation vector"""
        state = []
    
        # 1. Current node type (4 values)
        node_types = ['depot', 'customer', 'bss', 'intersection']
        current_type = self._get_node_type(self.current_node)
        node_onehot = [0.0] * 4
        node_onehot[node_types.index(current_type)] = 1.0
        state.extend(node_onehot)
    
        # 2. TOTAL battery SOC (1 value)
        state.append(self.battery.get_total_soc() / 100.0)
    
        # 3. Individual module states (critical for sequential discharge)
        module_charges = []
        for module in self.battery.modules:
            module_charges.append(module['charge'] / 100.0)
        state.extend(module_charges)

        # 4. Active module index (which module is currently being discharged)
        # Find first module with charge > 0 (sequential discharge)
        active_module_idx = 0
        for i, module in enumerate(self.battery.modules):
            if module['charge'] > 0:
                active_module_idx = i
                break
        active_module_onehot = [0.0] * len(self.battery.modules)
        active_module_onehot[active_module_idx] = 1.0
        state.extend(active_module_onehot)

        # 5. Customers remaining (binary)
        for customer in self.customer_nodes:
            state.append(1.0 if customer in self.remaining_customers else 0.0)
    
        # 6. Current payload ratio (1 value)
        max_load = len(self.customer_nodes) * 5
        state.append(self.current_load_kg / max_load if max_load > 0 else 0.0)
    
        # 7. Time elapsed ratio (1 value)
        state.append(min(self.time_elapsed / 480.0, 1.0))
    
        # 8. Distance to nearest BSS ratio (1 value)
        min_bss_dist = self.graph.get_nearest_bss_distance(self.current_node)
        state.append(min(min_bss_dist / 50.0, 1.0))
    
        # 9. Return requirement flag (1 value)
        state.append(1.0 if len(self.remaining_customers) == 0 else 0.0)
        
        # 10. Distance to nearest remaining customer (normalized)
        if self.remaining_customers:
            min_cust_dist = min(self.graph.get_distance_km(self.current_node, cust) 
            for cust in self.remaining_customers)
            state.append(min(min_cust_dist / 50.0, 1.0))
        else:
            state.append(0.0)

        # 11. Distance to depot (normalized)
        depot_dist = self.graph.get_distance_km(self.current_node, 'D')
        state.append(min(depot_dist / 50.0, 1.0))

        # 12. Energy needed to reach nearest BSS (normalized)
        nearest_bss = self._find_nearest_bss()
        if nearest_bss:
            bss_energy = self._calculate_edge_energy(self.current_node, nearest_bss)
            available_energy = self.battery.get_available_energy_kwh()
            if available_energy > 0:
                state.append(min(bss_energy / available_energy, 1.0))
            else:
                state.append(1.0)
        else:
            state.append(1.0)

        

        return np.array(state, dtype=np.float32)
    
    def _find_nearest_bss(self):
        """Find nearest BSS station"""
        bss_nodes = [node['id'] for node in self.config['nodes'] 
                    if node['type'] == 'bss']
        
        if not bss_nodes:
            return None
        
        nearest = None
        min_dist = float('inf')
        
        for bss in bss_nodes:
            dist = self.graph.get_distance_km(self.current_node, bss)
            if dist < min_dist:
                min_dist = dist
                nearest = bss
        
        return nearest
    
    def _get_info(self):
        """Get additional info"""
        return {
            'current_node': self.current_node,
            'remaining_customers': list(self.remaining_customers),
            'time_elapsed': self.time_elapsed,
            'total_swap_cost': self.total_swap_cost,
            'battery_soc': self.battery.get_total_soc(),
            'route': self.route_history.copy(),
            'action_mask': self.get_action_mask() 
        }
    
    def render(self):
        """Render environment state"""
        print(f"\n=== EVRP Environment ===")
        print(f"Current node: {self.current_node}")
        print(f"Remaining customers: {list(self.remaining_customers)}")
        print(f"Time elapsed: {self.time_elapsed:.1f} min")
        print(f"Battery SOC: {self.battery.get_total_soc():.1f}%")
        print(f"Swap cost: {self.total_swap_cost:.2f}")
        print(f"Current load: {self.current_load_kg} kg")
        print(f"Route: {' → '.join(self.route_history[-5:])}")  # Last 5 nodes
        print("=" * 30)
    
    def close(self):
        """Clean up resources"""
        pass