class Config:
    """Configuration for training"""
    
    # Training parameters
    TOTAL_TIMESTEPS = 1000  # Increased from 100000, debugging currently from 500000
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 64
    GAMMA = 0.99
    N_STEPS = 2048
    N_EPOCHS = 10
    GAE_LAMBDA = 0.95
    ENT_COEF = 0.2 # from 0.01, for more exploration
    CLIP_RANGE = 0.2
    
    # Model architecture
    POLICY = "MlpPolicy"
    #NET_ARCH = dict(pi=[64, 64], vf=[64, 64])
    NET_ARCH = dict(pi=[256, 256], vf=[256, 256])  # Larger network
    
    # Environment
    MAX_EPISODE_STEPS = 150
    VERBOSE = 1

    # New: Gradient clipping
    MAX_GRAD_NORM = 0.5
    
    # New: Value function coefficient
    VF_COEF = 0.5

    # testing
    LEARNING_RATE_SCHEDULE = "constant"  # or "linear"

    DEBUG = False