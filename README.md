EVRP BSS (Electric Vehicle Routing Problem with Battery
Swapping Station)

to run, first:
download python on Visual Studio Code
run: 
- python -m pip install --upgrade pip
- pip install gymnasium stable-baselines3[extra] sb3-contrib torch numpy tensorboard


battery.py          # Battery system model with modular energy management
  methods:
  - consume_energy(), swap_modules(), needs_swap(), get_total_soc()
config.py           # Parameters for training (timestep length etc) timesteps should be 100000 minimum but it runs crazy slow on the laurier virtual desktop, takes about 1 minute on my pc
energy.py           # Energy calculations
evrp_env.py         # Environment with a million functions
  methods:
  - a LOT
graph.py            # Graph representation of nodes and edges
  methods:
  - calculate_travel_time_minutes(), get_distance_km()
inference.py        # Run trained models with action masking (or well it should but I don't think it works), to run: python main.py --inference --scenario scenario_small.json --model evrp_model --episodes 3
main.py             # Main
train.py            # Main training script, run directly to use scenario_small.json, for other scenarios use format: python train.py --scenario sample_scenario.json
train_simple.py     # Simplified training without action masking (using this to debug issues with main training script)
utils.py            # Utility functions
debug_env.py        # Environment debugging
test_model.py       # Supposed to test the model but don't think this works properly
scenario_small.json # Small test scenario (5 nodes)
sample_scenario.json # Larger sample scenario (7 nodes)
