import argparse
import numpy as np
from sb3_contrib import MaskablePPO
from evrp_env import EVRPEnvironment
from sb3_contrib.common.wrappers import ActionMasker

def mask_fn(env):
    return env.get_action_mask()

def test_model_interactive(scenario_file, model_path):
    """Test model interactively with step-by-step output"""
    
    # Load environment
    env = EVRPEnvironment(scenario_file)
    env = ActionMasker(env, mask_fn)
    
    # Load model
    model = MaskablePPO.load(model_path)
    
    obs, info = env.reset()
    done = False
    total_reward = 0
    step = 0
    
    print("\n" + "="*60)
    print("TESTING MODEL - INTERACTIVE MODE")
    print("="*60)
    
    env.render()
    
    while not done and step < 150:
        # Get action mask
        action_mask = env.action_masks()
        
        # Show valid actions
        valid_actions = [i for i, valid in enumerate(action_mask) if valid]
        print(f"\nValid actions (node indices): {valid_actions}")
        print(f"Valid nodes: {[env.all_nodes[i] for i in valid_actions]}")
        
        # Predict with action mask
        action, _ = model.predict(obs, action_masks=action_mask, deterministic=True)
        
        print(f"\nStep {step + 1}: Choosing action {action} -> {env.all_nodes[action]}")
        
        # Take action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1
        
        # Render state
        env.render()
        
        if done:
            print("\n" + "="*40)
            print("EPISODE COMPLETED!")
            print(f"Total steps: {step}")
            print(f"Total reward: {total_reward:.3f}")
            print(f"Final cost: {env._calculate_current_cost():.1f}")
            print(f"Final route: {' -> '.join(env.route_history)}")
            print("="*40)
            break
    
    if not done:
        print(f"\nMax steps reached ({step})")
        print(f"Total reward: {total_reward:.3f}")
        print(f"Final cost: {env._calculate_current_cost():.1f}")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="scenario_small.json", help="Scenario file")
    parser.add_argument("--model", default="evrp_trained_model", help="Model file")
    
    args = parser.parse_args()
    test_model_interactive(args.scenario, args.model)