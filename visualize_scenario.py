import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def visualize_scenario(scenario_file, route=None, save_fig=False):
    """Visualize scenario nodes, edges, and optional route"""
    
    with open(scenario_file, 'r') as f:
        config = json.load(f)
    
    nodes = config['nodes']
    edges = config['edges']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot nodes with different colors for types
    node_colors = {
        'depot': 'red',
        'customer': 'blue',
        'bss': 'green',
        'intersection': 'gray'
    }
    
    node_markers = {
        'depot': 's',  # square
        'customer': 'o',  # circle
        'bss': '^',  # triangle
        'intersection': 'x'  # x
    }
    
    node_sizes = {
        'depot': 200,
        'customer': 150,
        'bss': 150,
        'intersection': 100
    }
    
    # Plot nodes
    for node in nodes:
        x, y = node['x'], node['y']
        node_type = node['type']
        color = node_colors.get(node_type, 'black')
        marker = node_markers.get(node_type, 'o')
        size = node_sizes.get(node_type, 100)
        
        ax.scatter(x, y, c=color, marker=marker, s=size, zorder=5, edgecolors='black')
        ax.annotate(node['id'], (x, y), xytext=(5, 5), 
                   textcoords='offset points', fontsize=9, fontweight='bold')
    
    # Plot edges
    for edge in edges:
        if isinstance(edge, list) and len(edge) >= 4:
            from_node, to_node, distance, traffic = edge[:4]
        elif isinstance(edge, dict):
            from_node, to_node = edge['from'], edge['to']
            distance = edge['distance']
            traffic = edge.get('traffic_factor', 1.0)
        else:
            continue
        
        # Find node coordinates
        from_coords = next((n for n in nodes if n['id'] == from_node), None)
        to_coords = next((n for n in nodes if n['id'] == to_node), None)
        
        if from_coords and to_coords:
            x1, y1 = from_coords['x'], from_coords['y']
            x2, y2 = to_coords['x'], to_coords['y']
            
            # Color by traffic factor
            if traffic < 0.9:
                color = 'green'  # Good traffic
            elif traffic < 1.1:
                color = 'orange'  # Normal traffic
            else:
                color = 'red'  # Heavy traffic
            
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.7, zorder=1)
            
            # Add distance label
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.annotate(f"{distance}km", (mid_x, mid_y), 
                       xytext=(0, 5), textcoords='offset points',
                       fontsize=8, ha='center', backgroundcolor='white')
    
    # Plot route if provided
    if route:
        route_coords = []
        for node_id in route:
            node = next((n for n in nodes if n['id'] == node_id), None)
            if node:
                route_coords.append((node['x'], node['y']))
        
        if route_coords:
            route_x, route_y = zip(*route_coords)
            ax.plot(route_x, route_y, 'r-', linewidth=3, zorder=2, marker='o', 
                   markersize=8, markerfacecolor='yellow', markeredgecolor='red')
            
            # Add route numbers
            for i, (x, y) in enumerate(route_coords):
                ax.annotate(str(i), (x, y), xytext=(0, -15),
                           textcoords='offset points', fontsize=10, 
                           fontweight='bold', ha='center',
                           backgroundcolor='yellow')
    
    # Create legend
    legend_patches = []
    for node_type, color in node_colors.items():
        legend_patches.append(mpatches.Patch(color=color, label=node_type))
    
    # Traffic legend
    traffic_patches = [
        mpatches.Patch(color='green', label='Good traffic (<0.9)'),
        mpatches.Patch(color='orange', label='Normal traffic (0.9-1.1)'),
        mpatches.Patch(color='red', label='Heavy traffic (>1.1)')
    ]
    
    ax.legend(handles=legend_patches + traffic_patches, loc='upper right')
    
    # Set labels and title
    ax.set_xlabel('X Coordinate (km)')
    ax.set_ylabel('Y Coordinate (km)')
    ax.set_title(f'Scenario: {scenario_file}\n' + 
                (f'Route: {" → ".join(route)}' if route else ''))
    
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_fig:
        filename = scenario_file.replace('.json', '')
        if route:
            filename += '_route'
        plt.savefig(f'{filename}.png', dpi=150)
        print(f"Saved visualization as {filename}.png")
    
    plt.show()

def visualize_model_route(model_path, scenario_file, num_episodes=1):
    """Run model and visualize the route it takes"""
    
    from sb3_contrib import MaskablePPO
    from evrp_env import EVRPEnvironment
    from sb3_contrib.common.wrappers import ActionMasker
    
    def mask_fn(env):
        return env.get_action_mask()
    
    # Load environment
    env = EVRPEnvironment(scenario_file)
    env = ActionMasker(env, mask_fn)
    
    # Load model
    model = MaskablePPO.load(model_path)
    
    # Run model
    obs, info = env.reset()
    done = False
    route = [env.env.current_node]
    
    print(f"Running model on {scenario_file}")
    print(f"Starting route: {route[-1]}")
    
    while not done:
        action_mask = env.action_masks()
        action, _ = model.predict(obs, action_masks=action_mask, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        route.append(env.env.current_node)
        
        print(f"Step {len(route)-1}: {route[-2]} → {route[-1]}")
    
    print(f"\nFinal route: {' → '.join(route)}")
    print(f"Total steps: {len(route)-1}")
    print(f"Remaining customers: {env.env.remaining_customers}")
    print(f"Final battery SOC: {env.env.battery.get_total_soc():.1f}%")
    
    # Visualize
    visualize_scenario(scenario_file, route, save_fig=True)
    
    env.close()
    return route

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize scenarios and routes")
    parser.add_argument("--scenario", default="scenario_small.json", 
                       help="Scenario file to visualize")
    parser.add_argument("--model", help="Model to run and visualize route")
    parser.add_argument("--route", nargs='+', help="Route to visualize (list of nodes)")
    parser.add_argument("--save", action="store_true", help="Save visualization as PNG")
    
    args = parser.parse_args()
    
    if args.model:
        # Run model and visualize its route
        visualize_model_route(args.model, args.scenario)
    elif args.route:
        # Visualize provided route
        visualize_scenario(args.scenario, args.route, args.save)
    else:
        # Just visualize scenario
        visualize_scenario(args.scenario, save_fig=args.save)