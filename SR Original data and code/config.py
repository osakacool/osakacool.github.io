"""
config.py
Global hyperparameter configuration for ET-NODE-LNN manuscript reproduction.
All values are strictly extracted from Sections 4, 5.1, and 5.2 of the manuscript.
This file serves as the Single Source of Truth for all scripts.
"""

class Config:
    # ==========================================================================
    # 1. Synthetic Data Generation (Section 5.1)
    # ==========================================================================
    SAMPLING_RATE_HZ = 100          # Ambient & downsampled seismic sampling rate
    FE_TIME_STEP_S = 0.005          # Original FE implicit integration time step
    PGA_MIN_G = 0.05                # Minimum Peak Ground Acceleration
    PGA_MAX_G = 0.8                 # Maximum Peak Ground Acceleration
    NUM_GROUND_MOTIONS = 120        # Total records from PEER NGA-West2
    AMBIENT_DURATION_RANGE_H = (24, 168)  # Continuous series length: 24h to 7 days
    
    # Environmental drift model parameters
    TEMP_DAILY_CYCLE_AMP = 5.0      # Sinusoidal daily temperature amplitude
    HUMIDITY_BASELINE = 60.0        # Mean relative humidity (%)
    
    # Ground truth vulnerability index
    VUL_INDEX_NORM_RANGE = (0.0, 1.0)  # Normalization range for training
    
    # Dataset split ratios
    TRAIN_SPLIT = 0.70
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15

    # ==========================================================================
    # 2. Model Architecture (Sections 4.1 - 4.4 & 5.2)
    # ==========================================================================
    # Encoder
    ENCODER_HIDDEN_DIM = 128
    LATENT_DIM = 64                 # Shared latent space dimension (d)
    
    # Neural ODE
    ODE_HIDDEN_UNITS = 128          # Per layer in 3-layer FC network
    ODE_STATE_DIM = 64              # Latent state dimension for ODE
    ODE_SOLVER_METHOD = "dopri5"    # Dormand-Prince adaptive step size
    ODE_REL_TOL = 1e-3
    ODE_ABS_TOL = 1e-5
    
    # Liquid Neural Network
    LNN_STATE_DIM = 64
    LNN_LOW_RANK = 16               # Rank r for low-rank factorization
    LNN_TIME_CONSTANT_INIT = 0.1    # Initial τ (seconds), learned during training
    
    # Optimized LSTM
    LSTM_HIDDEN_DIM = 128
    ATTENTION_DIM = 64              # Attention mechanism dimension
    
    # Gated Fusion
    FUSION_GATE_ACTIVATION = "sigmoid