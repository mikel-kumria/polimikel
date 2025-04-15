import os
import itertools
import ray
from ray import tune
import json
from datetime import datetime
import LSM_main
import numpy as np
from pathlib import Path
import re
from typing import Optional, Dict, Any, List
import torch
import shutil

class MatrixConfig:
    """Configuration class containing all valid parameter values"""
    
    # Valid topology types and their corresponding ratios
    VALID_TOPOLOGIES = {
        "Random": {"ratios": ["80_20", "90_10"]},
        "Dale": {"ratios": ["80_20"]},
        "Small_world": {"ratios": ["80_20"]}
    }
    
    # Valid spectral radius values (as they appear in paths)
    VALID_SPECTRAL_RADII = [
        "rho0x1", "rho0x5", "rho1x0", "rho1x5", "rho2x0"
    ]
    
    # Valid sparsity values
    VALID_SPARSITIES = [str(round(x, 1)) for x in np.arange(0.1, 1.1, 0.1)] # 0.1 to 1.0 in steps of 0.1
    VALID_SPARSITIES_ALTERNATIVE = ["1"]  # Alternative format for sparsity 1.0
    
    # Valid seeds
    VALID_SEEDS = list(range(1, 11))  # 1 to 10
    
    # Valid input weight seeds
    VALID_INPUT_SEEDS = list(range(1, 11))  # 1 to 10
    
    @classmethod
    def validate_topology(cls, topology: str, ratio: str) -> bool:
        """Validate topology and its ratio"""
        if topology not in cls.VALID_TOPOLOGIES:
            raise ValueError(f"Invalid topology: {topology}. Must be one of: {list(cls.VALID_TOPOLOGIES.keys())}")
        if ratio not in cls.VALID_TOPOLOGIES[topology]["ratios"]:
            raise ValueError(f"Invalid ratio {ratio} for topology {topology}. Must be one of: {cls.VALID_TOPOLOGIES[topology]['ratios']}")
        return True
    
    @classmethod
    def validate_spectral_radius(cls, rho: str) -> bool:
        """Validate spectral radius value"""
        if rho not in cls.VALID_SPECTRAL_RADII:
            raise ValueError(f"Invalid spectral radius: {rho}. Must be one of: {cls.VALID_SPECTRAL_RADII}")
        return True
    
    @classmethod
    def validate_sparsity(cls, sparsity: str) -> bool:
        """Validate sparsity value"""
        if sparsity not in cls.VALID_SPARSITIES and sparsity not in cls.VALID_SPARSITIES_ALTERNATIVE:
            raise ValueError(f"Invalid sparsity: {sparsity}. Must be one of: {cls.VALID_SPARSITIES} or {cls.VALID_SPARSITIES_ALTERNATIVE}")
        return True
    
    @classmethod
    def validate_seed(cls, seed: int) -> bool:
        """Validate matrix seed value"""
        if seed not in cls.VALID_SEEDS:
            raise ValueError(f"Invalid seed: {seed}. Must be one of: {cls.VALID_SEEDS}")
        return True
    
    @classmethod
    def validate_input_seed(cls, seed: int) -> bool:
        """Validate input weight seed value"""
        if seed not in cls.VALID_INPUT_SEEDS:
            raise ValueError(f"Invalid input seed: {seed}. Must be one of: {cls.VALID_INPUT_SEEDS}")
        return True

class PathParser:
    @staticmethod
    def parse_input_weights_path(path: str) -> Optional[Dict[str, Any]]:
        """Parse input weights path to extract seed information.
        Example: ".../nnLinear_weights_10seeds/nnLinear_weights_seed1.npy" -> {"seed": 1}
        """
        filename = os.path.basename(path)
        seed_match = re.search(r'seed(\d+)\.npy$', filename)
        if not seed_match:
            return None
            
        seed = int(seed_match.group(1))
        # Validate seed
        MatrixConfig.validate_input_seed(seed)
        
        return {
            "path": path,
            "seed": seed,
            "type": "input_weights"
        }

    @staticmethod
    def parse_connectivity_matrix_path(path: str) -> Optional[Dict[str, Any]]:
        """Parse connectivity matrix path to extract topology, ratio, spectral radius, sparsity, and seed.
        Example: ".../Random_80_20/rho1x0/80_20_weights_sparsity_0.1_rho1/weight_matrix_seed_1.npy"
        """
        parts = Path(path).parts
        
        # Extract topology and ratio
        topology_part = None
        for part in parts:
            for valid_topology in MatrixConfig.VALID_TOPOLOGIES.keys():
                if part.startswith(valid_topology):
                    topology_part = part
                    topology = valid_topology
                    break
            if topology_part:
                break
        
        if not topology_part:
            raise ValueError(f"No valid topology found in path: {path}")
            
        # Extract ratios from topology
        ratio_match = re.search(r'_(\d+_\d+)$', topology_part)
        if not ratio_match:
            raise ValueError(f"No valid ratio found in topology part: {topology_part}")
        
        ratio = ratio_match.group(1)
        # Validate topology and ratio combination
        MatrixConfig.validate_topology(topology, ratio)
        
        # Extract spectral radius from directory name (e.g., "rho0x5")
        rho_dir = None
        for part in parts:
            if part.startswith('rho'):
                rho_dir = part
                break
        
        if not rho_dir:
            raise ValueError(f"No spectral radius directory found in path: {path}")
        
        # Validate spectral radius directory name
        MatrixConfig.validate_spectral_radius(rho_dir)
        
        # Extract sparsity and actual rho value from the weights directory
        weights_dir = None
        for part in parts:
            if "weights_sparsity" in part:
                weights_dir = part
                break
                
        if not weights_dir:
            raise ValueError(f"No weights directory found in path: {path}")
            
        # Extract sparsity value
        sparsity_match = re.search(r'sparsity_(\d+(?:\.\d+)?)', weights_dir)
        if not sparsity_match:
            raise ValueError(f"No valid sparsity found in weights directory: {weights_dir}")
        
        sparsity = sparsity_match.group(1)
        # Normalize sparsity format (convert "1" to "1.0" for consistency)
        if sparsity == "1":
            sparsity = "1.0"
            
        # Validate sparsity
        MatrixConfig.validate_sparsity(sparsity)
        
        # Extract seed
        seed_match = re.search(r'seed_(\d+)\.npy$', os.path.basename(path))
        if not seed_match:
            raise ValueError(f"No valid seed found in filename: {path}")
        
        seed = int(seed_match.group(1))
        # Validate seed
        MatrixConfig.validate_seed(seed)
        
        # Extract excitatory/inhibitory ratio from the ratio string
        exc_inh_ratio = ratio.split('_')
        
        return {
            "path": path,
            "topology": topology.lower(),
            "ratio": ratio,
            "excitatory_ratio": int(exc_inh_ratio[0]),
            "inhibitory_ratio": int(exc_inh_ratio[1]),
            "spectral_radius": float(rho_dir[3].replace('x', '.') + rho_dir[4]),
            "sparsity": float(sparsity),
            "seed": seed,
            "type": "connectivity_matrix"
        }

def discover_weight_matrices(base_path):
    """Automatically discover and parse all weight matrices in the given base path."""
    matrices = {
        "input_weights": [],
        "connectivity_matrices": []
    }
    
    # Walk through all directories
    for root, _, files in os.walk(base_path):
        for file in files:
            if not file.endswith('.npy'):
                continue
                
            full_path = os.path.join(root, file)
            
            # Try parsing as input weights
            if 'nnLinear_weights' in file:
                parsed = PathParser.parse_input_weights_path(full_path)
                if parsed:
                    matrices["input_weights"].append(parsed)
            
            # Try parsing as connectivity matrix
            elif 'weight_matrix' in file:
                parsed = PathParser.parse_connectivity_matrix_path(full_path)
                if parsed:
                    matrices["connectivity_matrices"].append(parsed)
    
    return matrices

# Configuration for different experiment parameters
EXPERIMENT_CONFIG = {
    # Simulation parameters
    "simulation": {
        "reservoir_size": 100,
        "reset_delay": 0,
        "input_lif_beta": 0.0,
        "threshold_range": [0.0, 2.0],
        "beta_reservoir_range": [0.0, 1.0],
        "grid_points": 5  # Number of points in the grid search
    },
    
    # Base path for weight matrices
    "base_path": "/Users/mikel/Documents/GitHub/polimikel/UCR/Weight_matrices",
    
    # Experiment parameter subsets
    "parameters": {
        # Specify which reset mechanisms to use
        "reset_mechanisms": ["zero"],
        
        # Specify which topologies and their ratios to use
        "topologies": {
            "Random": ["80_20"],
            #"Dale": ["80_20"], ["90_10"]         # Example: Dale topology with 80/20 and 90/10ratio
        },
        
        # Specify which spectral radii to use
        "spectral_radii": ["rho0x5"],  # Example: only these three
        
        # Specify which sparsity values to use
        "sparsities": ["0.1", "0.9"],
        
        # Specify which matrix seeds to use
        "matrix_seeds": [1, 2],
        
        # Specify which input weight seeds to use
        "input_seeds": [1, 2]
    }
}

def validate_parameters(params: Dict) -> bool:
    """Validate that all parameters in a path exist in our experiment configuration."""
    
    def in_topologies(topology: str, ratio: str) -> bool:
        return (topology in EXPERIMENT_CONFIG["parameters"]["topologies"] and
                ratio in EXPERIMENT_CONFIG["parameters"]["topologies"][topology])
    
    def in_spectral_radii(rho: str) -> bool:
        return rho in EXPERIMENT_CONFIG["parameters"]["spectral_radii"]
    
    def in_sparsities(sparsity: str) -> bool:
        return sparsity in EXPERIMENT_CONFIG["parameters"]["sparsities"]
    
    def in_matrix_seeds(seed: int) -> bool:
        return seed in EXPERIMENT_CONFIG["parameters"]["matrix_seeds"]
    
    def in_input_seeds(seed: int) -> bool:
        return seed in EXPERIMENT_CONFIG["parameters"]["input_seeds"]
    
    return all([
        in_topologies(params.get("topology", ""), params.get("ratio", "")),
        in_spectral_radii(params.get("spectral_radius_str", "")),
        in_sparsities(params.get("sparsity", "")),
        in_matrix_seeds(params.get("seed", 0)),
        params.get("reset_mechanism", "") in EXPERIMENT_CONFIG["parameters"]["reset_mechanisms"]
    ])

def parse_path(path: str) -> Optional[Dict[str, Any]]:
    """Parse a file path to extract parameters."""
    try:
        parts = Path(path).parts
        
        # Extract topology and ratio
        topology_part = next((part for part in parts 
                            if any(t in part for t in EXPERIMENT_CONFIG["parameters"]["topologies"].keys())), None)
        if not topology_part:
            return None
            
        topology = next(t for t in EXPERIMENT_CONFIG["parameters"]["topologies"].keys() if t in topology_part)
        ratio = re.search(r'_(\d+_\d+)$', topology_part).group(1)
        
        # Extract spectral radius from directory name
        rho_dir = next((part for part in parts if part.startswith('rho')), None)
        if not rho_dir:
            return None
            
        # Convert directory rho format (e.g., "rho0x5") to match config format
        rho_dir_normalized = rho_dir
        if rho_dir not in EXPERIMENT_CONFIG["parameters"]["spectral_radii"]:
            # Try to normalize the format (e.g., convert "rho0.5" to "rho0x5")
            if '.' in rho_dir:
                whole, decimal = rho_dir[3:].split('.')
                rho_dir_normalized = f"rho{whole}x{decimal}"
        
        if rho_dir_normalized not in EXPERIMENT_CONFIG["parameters"]["spectral_radii"]:
            return None
        
        # Extract sparsity
        sparsity_match = re.search(r'sparsity_(\d+(?:\.\d+)?)', str(parts))
        if not sparsity_match:
            return None
        sparsity = sparsity_match.group(1)
        
        # Normalize sparsity format
        if sparsity == "1":
            sparsity = "1.0"
            
        if sparsity not in EXPERIMENT_CONFIG["parameters"]["sparsities"]:
            return None
        
        # Extract seed
        seed_match = re.search(r'seed_(\d+)\.npy$', os.path.basename(path))
        if not seed_match:
            return None
        seed = int(seed_match.group(1))
        
        if seed not in EXPERIMENT_CONFIG["parameters"]["matrix_seeds"]:
            return None
        
        params = {
            "path": path,
            "topology": topology,
            "ratio": ratio,
            "spectral_radius_str": rho_dir_normalized,
            "spectral_radius": float(rho_dir[3:].replace('x', '.')),
            "sparsity": sparsity,
            "seed": seed
        }
        
        return params
        
    except Exception as e:
        print(f"Error parsing path {path}: {str(e)}")
        return None

def discover_matrices(base_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Discover and parse weight matrices that match our experiment parameters."""
    matrices = {
        "input_weights": [],
        "connectivity_matrices": []
    }
    
    for root, _, files in os.walk(base_path):
        for file in files:
            if not file.endswith('.npy'):
                continue
                
            full_path = os.path.join(root, file)
            
            # Handle input weights
            if 'nnLinear_weights' in file:
                seed_match = re.search(r'seed(\d+)\.npy$', file)
                if seed_match:
                    seed = int(seed_match.group(1))
                    if seed in EXPERIMENT_CONFIG["parameters"]["input_seeds"]:
                        matrices["input_weights"].append({
                            "path": full_path,
                            "seed": seed
                        })
            
            # Handle connectivity matrices
            elif 'weight_matrix' in file:
                parsed = parse_path(full_path)
                if parsed:
                    matrices["connectivity_matrices"].append(parsed)
    
    return matrices

def create_experiment_folder(base_dir: str, params: Dict) -> str:
    """Create a hierarchical folder structure for the experiment."""
    folder_path = os.path.join(
        base_dir,
        f"topology_{params['topology'].lower()}_{params['ratio']}",
        f"rho_{params['spectral_radius']:.1f}".replace('.', 'x'),
        f"sparsity_{params['sparsity']}",
        f"seed_{params['connectivity_seed']}",
        f"input_seed_{params['input_seed']}",
        f"reset_{params['reset_mechanism']}"
    )
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

@ray.remote  # Remove GPU requirement from the default function
def run_single_experiment(params):
    """Run a single experiment with the given parameters."""
    # Create experiment folder
    output_folder = create_experiment_folder(
        params["base_output_dir"],
        params
    )
    
    # Create storage directory for Ray
    ray_storage = os.path.join(output_folder, "ray_storage")
    os.makedirs(ray_storage, exist_ok=True)
    
    # Prepare hyperparameters for this experiment
    hyperparams = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",  # Use CUDA if available
        "output_base_dir": output_folder,
        "storage_path": ray_storage,  # Use the correct parameter name
        "connectivity_matrix_path": params["connectivity_matrix_path"],
        "input_weights_path": params["input_weights_path"],
        "reservoir_size": EXPERIMENT_CONFIG["simulation"]["reservoir_size"],
        "reset_delay": EXPERIMENT_CONFIG["simulation"]["reset_delay"],
        "input_lif_beta": EXPERIMENT_CONFIG["simulation"]["input_lif_beta"],
        "reset_mechanism": params["reset_mechanism"],
        "threshold_range": EXPERIMENT_CONFIG["simulation"]["threshold_range"],
        "beta_reservoir_range": EXPERIMENT_CONFIG["simulation"]["beta_reservoir_range"],
        "grid_points": EXPERIMENT_CONFIG["simulation"]["grid_points"]
    }
    
    # Run the experiment
    LSM_main.run_experiment(hyperparams)
    
    return {
        "status": "completed",
        "output_folder": output_folder,
        "params": params
    }

def print_experiment_summary():
    """Print a summary of the current experiment configuration."""
    print("\nExperiment Configuration Summary:")
    print("=" * 50)
    
    params = EXPERIMENT_CONFIG["parameters"]
    
    print("\nTopologies and Ratios:")
    for topology, ratios in params["topologies"].items():
        print(f"  {topology}: {ratios}")
    
    print("\nSpectral Radii:")
    print(f"  {params['spectral_radii']}")
    
    print("\nSparsities:")
    print(f"  {params['sparsities']}")
    
    print("\nMatrix Seeds:")
    print(f"  {params['matrix_seeds']}")
    
    print("\nInput Seeds:")
    print(f"  {params['input_seeds']}")
    
    print("\nReset Mechanisms:")
    print(f"  {params['reset_mechanisms']}")
    
    # Calculate total number of experiments
    n_topologies = sum(len(ratios) for ratios in params["topologies"].values())
    total_experiments = (n_topologies * 
                        len(params["spectral_radii"]) * 
                        len(params["sparsities"]) * 
                        len(params["matrix_seeds"]) * 
                        len(params["input_seeds"]) * 
                        len(params["reset_mechanisms"]))
    
    print("\nTotal Experiments:")
    print(f"  {total_experiments} experiments will be run")
    print("=" * 50)

def main():
    # Print experiment configuration
    print_experiment_summary()
    
    # Ask for confirmation
    response = input("\nDo you want to proceed with these experiments? (y/n): ")
    if response.lower() != 'y':
        print("Experiment cancelled.")
        return
    
    # Create base output directory first
    base_output_dir = os.path.join(
        os.path.dirname(EXPERIMENT_CONFIG["base_path"]),
        "Simulations",
        f"parallel_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Create a shorter path for Ray's temporary files
    ray_temp_dir = os.path.expanduser("~/ray_temp")
    os.makedirs(ray_temp_dir, exist_ok=True)
    
    # Initialize Ray with proper configuration
    try:
        # Try to initialize with all available resources and local temp dir
        ray.init(
            _temp_dir=ray_temp_dir,
            ignore_reinit_error=True,
            include_dashboard=False,  # Disable dashboard to avoid port conflicts
            log_to_driver=False  # Disable logging to driver to avoid file access issues
        )
        
        # Check available resources
        resources = ray.available_resources()
        has_gpu = 'GPU' in resources and resources['GPU'] > 0
        
        if not has_gpu:
            print("\nNo GPU detected. Running experiments on CPU only.")
            # Redefine the run_single_experiment without GPU requirement
            global run_single_experiment
            @ray.remote
            def run_single_experiment(params):
                try:
                    # Create experiment folder
                    output_folder = create_experiment_folder(
                        params["base_output_dir"],
                        params
                    )
                    
                    # Create storage directory for Ray
                    ray_storage = os.path.join(output_folder, "ray_storage")
                    os.makedirs(ray_storage, exist_ok=True)
                    
                    # Prepare hyperparameters for this experiment
                    hyperparams = {
                        "device": "cpu",  # Force CPU usage
                        "output_base_dir": output_folder,
                        "storage_path": ray_storage,  # Use the correct parameter name
                        "connectivity_matrix_path": params["connectivity_matrix_path"],
                        "input_weights_path": params["input_weights_path"],
                        "reservoir_size": EXPERIMENT_CONFIG["simulation"]["reservoir_size"],
                        "reset_delay": EXPERIMENT_CONFIG["simulation"]["reset_delay"],
                        "input_lif_beta": EXPERIMENT_CONFIG["simulation"]["input_lif_beta"],
                        "reset_mechanism": params["reset_mechanism"],
                        "threshold_range": EXPERIMENT_CONFIG["simulation"]["threshold_range"],
                        "beta_reservoir_range": EXPERIMENT_CONFIG["simulation"]["beta_reservoir_range"],
                        "grid_points": EXPERIMENT_CONFIG["simulation"]["grid_points"]
                    }
                    
                    # Run the experiment
                    LSM_main.run_experiment(hyperparams)
                    
                    return {
                        "status": "completed",
                        "output_folder": output_folder,
                        "params": params
                    }
                except Exception as e:
                    return {
                        "status": "failed",
                        "error": str(e),
                        "params": params
                    }
    except Exception as e:
        print(f"\nError initializing Ray: {str(e)}")
        print("Trying to initialize Ray without any specific resource requirements...")
        # Try to initialize Ray without any specific resource requirements
        ray.shutdown()
        ray.init(
            _temp_dir=ray_temp_dir,
            ignore_reinit_error=True,
            include_dashboard=False,
            log_to_driver=False
        )
    
    # Discover matrices matching our parameters
    print("\nDiscovering weight matrices...")
    matrices = discover_matrices(EXPERIMENT_CONFIG["base_path"])
    
    print(f"Found {len(matrices['input_weights'])} matching input weight matrices")
    print(f"Found {len(matrices['connectivity_matrices'])} matching connectivity matrices")
    
    if len(matrices['connectivity_matrices']) == 0:
        print("\nError: No matching connectivity matrices found. Please check your configuration.")
        print("Writing configuration summary anyway...")
        # Save experiment summary even if no matrices were found
        summary_path = os.path.join(base_output_dir, "experiment_summary.json")
        with open(summary_path, "w") as f:
            json.dump({
                "config": EXPERIMENT_CONFIG,
                "matrices_found": matrices,
                "error": "No matching connectivity matrices found",
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        print(f"\nConfiguration saved in: {base_output_dir}")
        ray.shutdown()
        return
    
    # Launch experiments in parallel
    experiment_futures = []
    
    # Create all combinations of parameters
    for conn_matrix in matrices["connectivity_matrices"]:
        for input_weights in matrices["input_weights"]:
            for reset_mechanism in EXPERIMENT_CONFIG["parameters"]["reset_mechanisms"]:
                params = {
                    "base_output_dir": base_output_dir,
                    "connectivity_matrix_path": conn_matrix["path"],
                    "input_weights_path": input_weights["path"],
                    "reset_mechanism": reset_mechanism,
                    "topology": conn_matrix["topology"],
                    "ratio": conn_matrix["ratio"],
                    "spectral_radius": conn_matrix["spectral_radius"],
                    "sparsity": conn_matrix["sparsity"],
                    "connectivity_seed": conn_matrix["seed"],
                    "input_seed": input_weights["seed"]
                }
                
                # Launch experiment
                future = run_single_experiment.remote(params)
                experiment_futures.append(future)
    
    # Wait for all experiments to complete
    print(f"\nLaunched {len(experiment_futures)} experiments. Waiting for completion...")
    try:
        results = ray.get(experiment_futures)
        
        # Check for failed experiments
        failed_experiments = [r for r in results if r["status"] == "failed"]
        if failed_experiments:
            print("\nWarning: Some experiments failed:")
            for failed in failed_experiments:
                print(f"  Error in experiment with params: {failed['params']}")
                print(f"  Error message: {failed['error']}")
    except Exception as e:
        print(f"\nError while running experiments: {str(e)}")
        results = []
    
    # Save experiment summary
    summary_path = os.path.join(base_output_dir, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "config": EXPERIMENT_CONFIG,
            "matrices_found": matrices,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    # Clean up Ray
    ray.shutdown()
    
    print(f"\nAll experiments completed. Results saved in: {base_output_dir}")

if __name__ == "__main__":
    main() 