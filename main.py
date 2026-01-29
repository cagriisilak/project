import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="EVRP-BSS RL System")
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument("--inference", action="store_true", help="Run inference with trained model")
    parser.add_argument("--scenario", default="scenario_small.json", help="Scenario file")
    parser.add_argument("--model", default="evrp_ppo_model", help="Model file")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes for inference")

    
    args = parser.parse_args()
    
    if args.train:
        from train import train_model
        train_model(args.scenario, args.model)
    
    
    elif args.inference:
        from inference import run_inference
        run_inference(args.scenario, args.model, args.episodes)
    
    else:
        print("Please specify --train or --inference")
        sys.exit(1)

if __name__ == "__main__":
    main()