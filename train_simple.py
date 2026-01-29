import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from evrp_env import EVRPEnvironment
from config import Config
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import numpy as np
import torch

torch.set_num_threads(4)

class ValidActionEnv(EVRPEnvironment):
    """Custom environment that only returns valid actions"""
    
    def __init__(self, config_path='scenario_small.json'):
        super().__init__(config_path)
        self.invalid_action_count = 0
        
    def step(self, action):
        """Override step to handle invalid actions better"""
        target_node = self.all_nodes[action]
        
        # Check if action is valid
        if not self._is_action_valid(target_node):
            self.invalid_action_count += 1
            
            # For training: smaller penalty that doesn't end episode immediately
            # This allows the model to learn from mistakes
            state = self._get_state()
            info = self._get_info()
            
            # Penalty based on how "bad" the action is
            if target_node == self.current_node:
                penalty = -5.0  # Staying in place
            elif not self.graph.get_edge_info(self.current_node, target_node):
                penalty = -10.0  # No edge exists
            elif target_node == 'D' and len(self.remaining_customers) > 0:
                penalty = -15.0  # Returning to depot early
            else:
                # Not enough energy or other reasons
                penalty = -20.0
            
            # Scale penalty
            penalty = penalty / 100.0
            
            # Don't terminate immediately during training
            # Let the model try other actions
            return state, penalty, False, False, info
        
        # If valid, proceed with normal step
        return super().step(action)

def train_model_simple(scenario_file, model_save_path="evrp_ppo_model"):
    """Train PPO model on EVRP environment - Simplified version"""
    
    # Create environment
    env = ValidActionEnv(scenario_file)
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
    
    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=Config.LEARNING_RATE,
        n_steps=Config.N_STEPS,
        batch_size=Config.BATCH_SIZE,
        n_epochs=Config.N_EPOCHS,
        gamma=Config.GAMMA,
        gae_lambda=Config.GAE_LAMBDA,
        clip_range=Config.CLIP_RANGE,
        ent_coef=Config.ENT_COEF,
        verbose=Config.VERBOSE,
        tensorboard_log="./tensorboard/",
        policy_kwargs={
            "net_arch": Config.NET_ARCH,
            "activation_fn": torch.nn.ReLU,
        }
    )
    
    # Train
    print(f"\nStarting training for {Config.TOTAL_TIMESTEPS} steps...")
    print("=" * 50)
    
    model.learn(
        total_timesteps=Config.TOTAL_TIMESTEPS,
        progress_bar=True,
        tb_log_name=f"evrp_simple_{scenario_file.replace('.json', '')}"
    )
    
    # Save model
    model.save("evrp_trained_model_simple")
    print(f"\nModel saved as evrp_trained_model_simple.zip")
    
    # Test the trained model
    print("\nTesting trained model...")
    test_model_simple(env, model)
    
    env.close()
    return model

def test_model_simple(env, model, num_episodes=3):
    """Test the trained model"""
    
    # Create a fresh environment for testing (not the wrapped one)
    test_env = EVRPEnvironment()
    
    for episode in range(num_episodes):
        obs, info = test_env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"\nEpisode {episode + 1}:")
        print(f"Starting at {test_env.current_node}")
        print(f"Battery SOC: {test_env.battery.get_total_soc():.1f}%")
        print(f"Remaining customers: {test_env.remaining_customers}")
        
        # Show valid actions
        valid_actions = test_env.get_valid_actions()
        print(f"Valid actions from {test_env.current_node}: {[test_env.all_nodes[i] for i in valid_actions]}")
        
        while not done and steps < 50:
            # Reshape obs for model prediction
            obs_reshaped = obs.reshape(1, -1)
            action, _ = model.predict(obs_reshaped, deterministic=True)
            action = action[0]  # Get scalar action
            
            print(f"\nStep {steps + 1}:")
            print(f"  Choosing action {action} -> {test_env.all_nodes[action]}")
            
            # Check if action is valid
            target_node = test_env.all_nodes[action]
            if not test_env._is_action_valid(target_node):
                print(f"  WARNING: Action {action} ({target_node}) is INVALID!")
                energy_needed = test_env._calculate_edge_energy(test_env.current_node, target_node)
                available_energy = test_env.battery.get_available_energy_kwh()
                print(f"  Energy needed: {energy_needed:.2f}kWh, Available: {available_energy:.2f}kWh")
            
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            steps += 1
            
            print(f"  New node: {test_env.current_node}")
            print(f"  Step reward: {reward:.3f}")
            print(f"  Battery SOC: {test_env.battery.get_total_soc():.1f}%")
            print(f"  Remaining customers: {test_env.remaining_customers}")
            
            if done:
                print(f"\n  Episode completed!")
                print(f"  Total steps: {steps}")
                print(f"  Total reward: {total_reward:.3f}")
                print(f"  Route: {' -> '.join(test_env.route_history)}")
                print(f"  Final cost: {test_env._calculate_current_cost():.1f}")
                break
        
        if steps >= 50:
            print(f"\nEpisode truncated after {steps} steps (max reached)")
            print(f"Total reward: {total_reward:.3f}")
            print(f"Route: {' -> '.join(test_env.route_history)}")
    
    print(f"\nTesting completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="scenario_small.json", 
                       help="Scenario file to train on")
    parser.add_argument("--model", default="evrp_ppo_model_simple",
                       help="Output model name")
    
    args = parser.parse_args()
    train_model_simple(args.scenario, args.model)