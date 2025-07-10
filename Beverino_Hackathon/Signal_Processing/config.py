"""
Configuration file for SNN Speech Recognition System

This file contains all the configurable parameters for the system.
"""

import torch

# Data Configuration
DATA_CONFIG = {
    "root_dir": "/path/to/speech_commands_dataset",  # Update this path <------------------------------------------------------
    "commands": ["yes", "no", "up", "down"],  # Speech commands to classify
    "sample_rate": 16000,
    "max_length": 16000,  # Maximum audio length in samples
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "num_workers": 4,
}

# Preprocessing Configuration
PREPROCESSING_CONFIG = {
    "n_mels": 16,
    "n_mfcc": 16,
    "n_fft": 1024, # this is the window size, 1024 is the default and represents 25ms of audio
    "hop_length": 512, # this is the hop size, 512 is the default and represents 12.5ms of audio, this is the step size of the window and is used to overlap the windows
    "win_length": 1024, # this is the window size, 1024 is the default and represents 25ms of audio
    "f_min": 0,
    "f_max": 8000,
    "spike_encoding": True,
}

# Model Configuration
MODEL_CONFIG = {
    "input_size": 16,
    "hidden_size": 32,
    "output_size": 4,
    "beta_input_min": 0.5,
    "beta_input_max": 0.95,
    "beta_hidden": 0.85,
    "beta_output": 0.9,
    "threshold": 1.0,
    "spike_grad_slope": 25,
    "output_type": "mem_potential",  # "mem_potential" or "spk_count"
}

# Training Configuration
TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 100,
    "loss_type": "MSE",  # "MSE" or "CE" where CE is cross entropy loss
    "scheduler_step_size": 10,
    "scheduler_gamma": 0.7,
    "early_stopping_patience": 15,
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
}

# Experiment Configuration
EXPERIMENT_CONFIG = {
    "seed": 42,
    "save_model": True,
    "save_history": True,
    "log_interval": 5,  # Print progress every N epochs
    "save_interval": 10,  # Save model every N epochs
}

# Preprocessing Methods Configuration
PREPROCESSING_METHODS = {
    "mel": {
        "description": "MEL Spectrogram - Standard audio feature extraction",
        "recommended_for": "Quick experiments, baseline performance",
        "computation_cost": "Low",
    },
    "mfcc": {
        "description": "MFCC - Mel-frequency cepstral coefficients",
        "recommended_for": "Compact features, standard speech recognition",
        "computation_cost": "Low",
    },
    "gammatone": {
        "description": "Gammatone Filterbank - Auditory-inspired frequency analysis",
        "recommended_for": "Biologically realistic performance",
        "computation_cost": "Medium",
    },
    "cochleagram": {
        "description": "Cochleagram - Cochlea-inspired representation",
        "recommended_for": "Best biological realism, research",
        "computation_cost": "High",
    }
}

# Beta Value Presets (for different frequency sensitivities)
BETA_PRESETS = {
    "human_ear": {
        "description": "Mimics human ear frequency sensitivity",
        "beta_input_min": 0.5,
        "beta_input_max": 0.95,
    },
    "uniform": {
        "description": "Uniform frequency sensitivity",
        "beta_input_min": 0.8,
        "beta_input_max": 0.8,
    },
    "low_freq_sensitive": {
        "description": "Very sensitive to low frequencies",
        "beta_input_min": 0.3,
        "beta_input_max": 0.9,
    },
    "high_freq_sensitive": {
        "description": "Very sensitive to high frequencies",
        "beta_input_min": 0.7,
        "beta_input_max": 0.99,
    }
}

def get_config(preprocessing="mel", output_type="mem_potential", loss_type="MSE", 
               beta_preset="human_ear", **kwargs):
    """
    Get configuration with custom parameters
    
    Args:
        preprocessing: Preprocessing method
        output_type: Output type ("mem_potential" or "spk_count")
        loss_type: Loss type ("MSE" or "CE")
        beta_preset: Beta value preset
        **kwargs: Additional parameters to override
        
    Returns:
        config: Complete configuration dictionary
    """
    config = {
        "data": DATA_CONFIG.copy(),
        "preprocessing": PREPROCESSING_CONFIG.copy(),
        "model": MODEL_CONFIG.copy(),
        "training": TRAINING_CONFIG.copy(),
        "experiment": EXPERIMENT_CONFIG.copy(),
    }
    
    # Update with custom parameters
    config["preprocessing"]["method"] = preprocessing
    config["model"]["output_type"] = output_type
    config["training"]["loss_type"] = loss_type
    
    # Apply beta preset
    if beta_preset in BETA_PRESETS:
        preset = BETA_PRESETS[beta_preset]
        config["model"]["beta_input_min"] = preset["beta_input_min"]
        config["model"]["beta_input_max"] = preset["beta_input_max"]
    
    # Apply additional kwargs
    for key, value in kwargs.items():
        if "." in key:
            # Nested key (e.g., "model.input_size")
            section, param = key.split(".", 1)
            if section in config and param in config[section]:
                config[section][param] = value
        else:
            # Top-level key
            if key in config:
                config[key].update(value)
    
    return config

def print_config(config):
    """Print configuration in a readable format"""
    print("="*60)
    print("SNN Speech Recognition Configuration")
    print("="*60)
    
    for section, params in config.items():
        print(f"\n{section.upper()}:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    
    print("\n" + "="*60)

def validate_config(config):
    """Validate configuration parameters"""
    errors = []
    
    # Check data configuration
    if not config["data"]["root_dir"] or config["data"]["root_dir"] == "/path/to/speech_commands_dataset":
        errors.append("DATA_CONFIG['root_dir'] must be set to a valid path")
    
    if len(config["data"]["commands"]) != config["model"]["output_size"]:
        errors.append("Number of commands must match output_size")
    
    # Check model configuration
    if config["model"]["output_type"] not in ["mem_potential", "spk_count"]:
        errors.append("output_type must be 'mem_potential' or 'spk_count'")
    
    if config["training"]["loss_type"] not in ["MSE", "CE"]:
        errors.append("loss_type must be 'MSE' or 'CE'")
    
    # Check beta values
    if config["model"]["beta_input_min"] >= config["model"]["beta_input_max"]:
        errors.append("beta_input_min must be less than beta_input_max")
    
    if any(beta < 0 or beta > 1 for beta in [
        config["model"]["beta_input_min"],
        config["model"]["beta_input_max"],
        config["model"]["beta_hidden"],
        config["model"]["beta_output"]
    ]):
        errors.append("All beta values must be between 0 and 1")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
    
    return True 