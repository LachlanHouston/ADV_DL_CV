# src/config.py

"""
Copyright (C) ADLCV 2025

This file defines general config loader
"""


import yaml
import os
import torch

# Default configuration dictionary
DEFAULT_CONFIG = {
    # Basic project settings
    "experiment_name": "layer_gen_basic",
    "experiment_dir": "experiments/",
    
    # Data settings
    "data_path": "data/processed/kenney_dataset_1000.h5",
    
    # Model settings
    "image_size": 224,
    "patch_size": 16,
    "channels": 4,  # RGBA
    "embed_dim": 768,
    "num_heads": 8,
    "num_layers": 6,
    
    # Training settings
    "batch_size": 32,
    "learning_rate": 1e-4,
    "num_epochs": 100,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def load_config(config_path=None):
    """
    Load configuration from a YAML file or use defaults.
    
    Args:
        config_path: Path to YAML config file (optional)
        
    Returns:
        Dictionary containing configuration
    """
    config = DEFAULT_CONFIG.copy()
    
    # Load from file if provided
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            file_config = yaml.safe_load(f)
            # Update default config with file values
            config.update(file_config)
    
    return config