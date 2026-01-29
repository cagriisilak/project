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
    model.save("evrp_trained_model")
    print(f"\nModel saved as evrp_trained_model.zip")
    
    # Test the trained model
    print("\nTesting trained model...")
    test_trained_model(env, model)
    
    env.close()
    return model

def test_trained_model(env, model, num_episodes=5):
    """Test the trained model on a few episodes"""
    total_reward_sum = 0
    
    for episode in range(num_episodes):
        # We need to create a fresh environment for testing (not the vectorized one)
        from evrp_env import EVRPEnvironment
        from sb3_contrib.common.wrappers import ActionMasker
        
        # Create a new environment for testing
        test_env = EVRPEnvironment()
        test_env = ActionMasker(test_env, mask_fn)
        
        obs, info = test_env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        print(f"\n{'='*50}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*50}")
        
        # Render initial state
        test_env.render()
        
        while not done and steps < 150:
            # Get action mask from the environment
            action_mask = test_env.action_masks()
            
            # Predict with action mask
            # Note: need to reshape obs if needed
            if len(obs.shape) == 1:
                obs = obs.reshape(1, -1)
            action, _ = model.predict(obs, action_masks=action_mask, deterministic=False)
            
            # Take action
            obs, reward, terminated, truncated, info = test_env.step(action[0])
            episode_reward += float(reward)
            steps += 1
            
            # Show progress every 5 steps
            if steps % 5 == 0:
                print(f"\nStep {steps}:")
                print(f"  Action: {action[0]} -> {test_env.current_node}")
                print(f"  Step reward: {reward:.3f}")
                print(f"  Battery SOC: {test_env.battery.get_total_soc():.1f}%")
                print(f"  Remaining customers: {test_env.remaining_customers}")
            
            if terminated or truncated:
                print(f"\n{'='*30}")
                print(f"Episode completed in {steps} steps!")
                print(f"Total episode reward: {episode_reward:.3f}")
                print(f"Final cost: {test_env._calculate_current_cost():.1f}")
                print(f"Route: {' -> '.join(test_env.route_history)}")
                break
        
        total_reward_sum += episode_reward
        
        if steps >= 150:
            print(f"\nEpisode truncated after {steps} steps (max reached)")
            print(f"Total episode reward: {episode_reward:.3f}")
            print(f"Final cost: {test_env._calculate_current_cost():.1f}")
            print(f"Route: {' -> '.join(test_env.route_history)}")
        
        test_env.close()
    
    print(f"\n{'='*50}")
    print(f"Average reward over {num_episodes} episodes: {total_reward_sum/num_episodes:.3f}")
    print(f"{'='*50}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="scenario_small.json", 
                       help="Scenario file to train on")
    parser.add_argument("--model", default="evrp_ppo_model",
                       help="Output model name")
    
    args = parser.parse_args()
    train_model(args.scenario, args.model)
