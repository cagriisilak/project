import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from evrp_env import EVRPEnvironment
from config import Config
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
#from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import torch
import os
#import torch.nn as nn
#import logging

torch.set_num_threads(4)

def mask_fn(env):
    return env.get_action_mask()



def train_model(scenario_file, model_save_path="evrp_ppo_model"):
    """Train PPO model on EVRP environment"""
    
    # Create environment
    env = EVRPEnvironment(scenario_file)
    env = ActionMasker(env, mask_fn)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    
    # Get dimensions from environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"Training Configuration:")
    print(f"  State dimension: {state_dim}")
    print(f"  Action dimension: {action_dim}")
    print(f"  Network architecture: {Config.NET_ARCH}")
    print(f"  Learning rate: {Config.LEARNING_RATE}")
    print(f"  Total timesteps: {Config.TOTAL_TIMESTEPS}")
    print(f"  Entropy coefficient: {Config.ENT_COEF}")
    
    # Create model with PROPER initialization
    model = MaskablePPO(
        Config.POLICY,
        env,
        learning_rate=Config.LEARNING_RATE,
        n_steps=Config.N_STEPS,
        batch_size=Config.BATCH_SIZE,
        n_epochs=Config.N_EPOCHS,
        gamma=Config.GAMMA,
        gae_lambda=Config.GAE_LAMBDA,
        clip_range=Config.CLIP_RANGE,
        ent_coef=Config.ENT_COEF,  # Exploration
        vf_coef=Config.VF_COEF,  # Value function weight
        max_grad_norm=Config.MAX_GRAD_NORM,  # Gradient clipping
        verbose=Config.VERBOSE,
        tensorboard_log="./tensorboard/",
        policy_kwargs={
            "net_arch": Config.NET_ARCH,
            "activation_fn": torch.nn.ReLU,
            "ortho_init": True,
        }
    )
    
    # Debug print model info
    print(f"\nModel created:")
    print(f"  Policy network: {model.policy}")
    print(f"  Value network: {model.policy.value_net}")
    print(f"  Learning rate: {model.learning_rate}")
    print(f"  Entropy coefficient: {model.ent_coef}")
    
    # Callbacks
    callbacks = []
    
    # Debug callback (was causing error)
    #debug_callback = DebugCallback()
    #callbacks.append(debug_callback)

    callbacks = []

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=min(5000, Config.TOTAL_TIMESTEPS // 10),
        save_path="./logs/checkpoints/",
        name_prefix="evrp_model"
    )
    callbacks.append(checkpoint_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=min(5000, Config.TOTAL_TIMESTEPS // 10),
        save_path="./logs/checkpoints/",
        name_prefix="evrp_model"
    )
    callbacks.append(checkpoint_callback)
    
    # Train
    print(f"\nStarting training for {Config.TOTAL_TIMESTEPS} steps...")
    print("=" * 50)
    
    model.learn(
        total_timesteps=Config.TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=True,
        tb_log_name=f"evrp_{scenario_file.replace('.json', '')}"
    )
    
    # Save model
    #model.save("evrp_trained_model")
    #print(f"\nModel saved as evrp_trained_model.zip") <- old, worked but not for multiple scenarios
    if model_save_path is None:
        # Create model name based on scenario
        scenario_name = scenario_file.replace('.json', '')
        model_save_path = f"evrp_model_{scenario_name}"

    model.save(model_save_path)
    print(f"\nModel saved as {model_save_path}.zip")
    
    # Test the trained model
    print("\nTesting trained model...")
    #test_trained_model(env, model) <- old, testing new vers
    test_trained_model(model_save_path, scenario_file)
    
    env.close()
    return model

def test_trained_model(model, scenario_file, num_episodes=5):
    """Test the trained model - ENSURE same scenario"""
    
    print(f"\n{'='*60}")
    print(f"TESTING MODEL on {scenario_file}")
    print(f"{'='*60}")
    
    # Check if model exists
    if not os.path.exists(f"{model}.zip"):
        print(f"Error: Model {model}.zip not found!")
        return
    
    # Create environment for testing
    test_env = EVRPEnvironment(scenario_file)
    test_env = ActionMasker(test_env, mask_fn)
    
    # Get the original environment
    original_env = test_env.env
    
    # Load model
    try:
        model = MaskablePPO.load(model)
    except Exception as e:
        print(f"Error loading model: {e}")
        test_env.close()
        return
    
    # Track performance
    results = {
        'episodes': [],
        'average_reward': 0,
        'success_rate': 0
    }
    
    # Run episodes
    for episode in range(num_episodes):
        obs, info = test_env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print(f"Starting at: {original_env.current_node}")
        print(f"Remaining customers: {sorted(original_env.remaining_customers)}")
        
        route = [original_env.current_node]
        
        while not done and steps < 150:
            # Get action mask
            action_mask = test_env.action_masks()
            
            # Ensure obs is 1D
            if obs.ndim > 1:
                obs = obs.flatten()
            
            # Predict
            action, _ = model.predict(obs, action_masks=action_mask, deterministic=True)
            
            # Take step
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            episode_reward += float(reward)
            steps += 1
            
            route.append(original_env.current_node)
            
            if done:
                success = (len(original_env.remaining_customers) == 0 and 
                          original_env.current_node == 'D')
                
                episode_result = {
                    'episode': episode + 1,
                    'success': success,
                    'steps': steps,
                    'reward': episode_reward,
                    'cost': original_env._calculate_current_cost(),
                    'route': ' -> '.join(route),
                    'swaps': original_env.battery.swapped_modules_count,
                    'final_soc': original_env.battery.get_total_soc()
                }
                results['episodes'].append(episode_result)
                
                print(f"\n{'='*30}")
                print(f"Episode completed in {steps} steps")
                print(f"Success: {'Yes' if success else 'No'}")
                print(f"Reward: {episode_reward:.3f}")
                print(f"Cost: {episode_result['cost']:.1f}")
                print(f"Route: {episode_result['route']}")
                print(f"Battery swaps: {episode_result['swaps']}")
                print(f"Final SOC: {episode_result['final_soc']:.1f}%")
                break
        
        if steps >= 150:
            print(f"\nEpisode truncated after {steps} steps")
    
    # Calculate statistics
    if results['episodes']:
        results['average_reward'] = sum(ep['reward'] for ep in results['episodes']) / len(results['episodes'])
        results['success_rate'] = sum(1 for ep in results['episodes'] if ep['success']) / len(results['episodes'])
        
        print(f"\n{'='*60}")
        print(f"SUMMARY:")
        print(f"Average reward: {results['average_reward']:.3f}")
        print(f"Success rate: {results['success_rate']:.1%}")
        print(f"{'='*60}")
    
    test_env.close()
    return results

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="scenario_small.json", 
                       help="Scenario file to train on")
    parser.add_argument("--model", default="evrp_ppo_model",
                       help="Output model name")
    
    args = parser.parse_args()
    train_model(args.scenario, args.model)
