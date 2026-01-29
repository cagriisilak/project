import argparse
import numpy as np
from sb3_contrib import MaskablePPO
from evrp_env import EVRPEnvironment
from sb3_contrib.common.wrappers import ActionMasker

# Sample how to run: python inference.py --scenario scenario_small.json --model evrp_trained_model --episodes 3

def mask_fn(env):
    return env.get_action_mask()

def run_inference(scenario_file, model_path, num_episodes=3, render=True):
    """Run trained model on scenario with action masking"""
    
    # Load environment
    env = EVRPEnvironment(scenario_file)
    
    # IMPORTANT: Get the action mask BEFORE wrapping with ActionMasker
    # because we need to call get_action_mask() on the original env
    env = ActionMasker(env, mask_fn)  # Wrap with action masking
    
    try:
        model = MaskablePPO.load(model_path)
    except ValueError as e:
        print(f"Error loading model: {e}")
        print("Make sure you're loading a MaskablePPO model, not a regular PPO model.")
        env.close()
        return
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        print(f"\n{'='*50}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*50}")
        
        # Get the original environment from the wrapper
        original_env = env.env
        
        if render:
            original_env.render()
        
        while not done and step < 150:
            # Get action mask - use action_masks() on the wrapped env
            action_mask = env.action_masks()
            
            # Predict with action mask
            action, _ = model.predict(obs, action_masks=action_mask, deterministic=True)
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1
            
            if render and step % 5 == 0:  # Render every 5 steps
                print(f"\nStep {step}:")
                print(f"  Action: {action} -> {original_env.current_node}")
                print(f"  Reward: {reward:.3f}")
                print(f"  Battery SOC: {original_env.battery.get_total_soc():.1f}%")
                print(f"  Remaining customers: {original_env.remaining_customers}")
            
            if done:
                print(f"\n{'='*50}")
                print(f"Episode completed in {step} steps!")
                print(f"Total reward: {total_reward:.3f}")
                print(f"Final node: {original_env.current_node}")
                print(f"Remaining customers: {len(original_env.remaining_customers)}")
                print(f"Battery SOC: {original_env.battery.get_total_soc():.1f}%")
                
                # Calculate final cost
                final_cost = original_env._calculate_current_cost()
                print(f"Total cost: {final_cost:.1f}")
                break
        
        if not done:
            print(f"\nEpisode truncated after {step} steps (max steps reached)")
            print(f"Total reward: {total_reward:.3f}")
    
    print(f"\n{'='*50}")
    print(f"Inference completed for {num_episodes} episodes")
    print(f"{'='*50}")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EVRP-BSS Inference with Action Masking")
    parser.add_argument("--scenario", default="scenario_small.json",
                       help="Scenario file to test")
    parser.add_argument("--model", default="evrp_trained_model",
                       help="Trained MaskablePPO model to use")
    parser.add_argument("--episodes", type=int, default=3,
                       help="Number of episodes to run")
    parser.add_argument("--no-render", action="store_true",
                       help="Disable rendering during inference")
    
    args = parser.parse_args()
    run_inference(args.scenario, args.model, args.episodes, not args.no_render)