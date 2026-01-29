from evrp_env import EVRPEnvironment

def debug_environment():
    """Debug the environment to see what's wrong"""
    env = EVRPEnvironment("scenario_small.json")
    
    print("\n" + "="*60)
    print("DEBUGGING ENVIRONMENT")
    print("="*60)
    
    # Reset and show initial state
    obs, info = env.reset()
    
    print(f"\nInitial state:")
    print(f"  Current node: {env.current_node}")
    print(f"  Battery SOC: {env.battery.get_total_soc():.1f}%")
    print(f"  Total charge: {env.battery.total_charge_kwh:.2f}kWh / {env.battery.total_capacity_kwh:.2f}kWh")
    print(f"  Remaining customers: {env.remaining_customers}")
    
    # Test all possible actions
    print(f"\nTesting all actions from {env.current_node}:")
    for i, node in enumerate(env.all_nodes):
        print(f"\nAction {i} -> {node}:")
        
        if node == env.current_node:
            print("  ❌ INVALID: Can't stay at same node")
            continue
        
        # Check edge
        edge = env.graph.get_edge_info(env.current_node, node)
        if not edge:
            print("  ❌ INVALID: No edge exists")
            continue
        
        # Check if customer already delivered
        if node in env.customer_nodes and node not in env.remaining_customers:
            print("  ❌ INVALID: Customer already delivered")
            continue
        
        # Check depot return
        if node == 'D' and len(env.remaining_customers) > 0:
            print("  ❌ INVALID: Can't return to depot before all deliveries")
            continue
        
        # Check energy
        energy_needed = env._calculate_edge_energy(env.current_node, node)
        available_energy = env.battery.get_available_energy_kwh()
        
        print(f"  Edge distance: {edge['distance_km']}km")
        print(f"  Energy needed: {energy_needed:.4f}kWh")
        print(f"  Available energy: {available_energy:.4f}kWh")
        
        if energy_needed > available_energy:
            print(f"  ❌ INVALID: Not enough energy (need {energy_needed:.4f}, have {available_energy:.4f})")
        else:
            print(f"  ✅ VALID")
            
            # Try the action
            print(f"  Trying action...")
            old_node = env.current_node
            old_soc = env.battery.get_total_soc()
            obs, reward, terminated, truncated, info = env.step(i)
            
            print(f"  Moved from {old_node} to {env.current_node}")
            print(f"  Battery SOC: {old_soc:.1f}% -> {env.battery.get_total_soc():.1f}%")
            print(f"  Reward: {reward:.3f}")
            
            # Reset for next test
            env.reset()
    
    print("\n" + "="*60)
    print("DEBUG COMPLETE")
    print("="*60)

if __name__ == "__main__":
    debug_environment()