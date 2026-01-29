EVRP BSS (Electric Vehicle Routing Problem with Battery
Swapping Station)

downloads:
- download python on Visual Studio Code\
run: 
- python -m pip install --upgrade pip
- pip install gymnasium stable-baselines3[extra] sb3-contrib torch numpy tensorboard


config.py             # Training parameters (timestep length etc)\
evrp_env.py           # Main Gymnasium environment\
battery.py            # Battery system with sequential discharge\
energy.py             # Energy consumption calculations\
graph.py              # Graph representation of nodes/edges\
train.py              # PPO training with action masking\
inference.py          # Model testing and inference\
visualize_scenario.py # Scenario and route visualization\
main.py               # Command-line interface\
utils.py              # Utility functions\
sample_scenario.json  # Larger sample scenario (7 nodes)\
scenario_small.json   # Small test scenario (5 nodes)\
requirements.txt      # Python dependencies\
test_model.py         # For debugging\


**How to run:**
train.py:\
- run directly to use scenario_small.json, for other scenarios use format: python train.py --scenario sample_scenario.json

inference.py:\
- python main.py --inference --scenario scenario_small.json --model evrp_model --episodes 3

visualize_scenario.py:\
- only visualize and save: python visualize_scenario.py --scenario sample_scenario.json --save
- visualize and run model: python visualize_scenario.py --scenario sample_scenario.json --model evrp_sample_model
