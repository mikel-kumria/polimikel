import os
import numpy as np
import torch
import json
from torch.utils.tensorboard import SummaryWriter
import LSM_imports as defs
from LSM_imports import SpikingReservoirLoaded, generate_input_signal, create_unique_folder, set_seed
import ray
from ray import tune
import uuid
from ray.air import session
from datetime import datetime

def save_hyperparameters_log(hyperparams, output_folder):
    """
    Saves hyperparameters in both JSON and human-readable text formats.
    """
    # Save as JSON (preserve existing functionality)
    json_path = os.path.join(output_folder, "hyperparameters.json")
    with open(json_path, "w") as f:
        json.dump(hyperparams, f, indent=2)
    
    # Create a human-readable text file
    txt_path = os.path.join(output_folder, "hyperparameters_summary.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("EXPERIMENT HYPERPARAMETERS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("HARDWARE AND PATHS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Device: {hyperparams['device']}\n")
        f.write(f"Output Directory: {hyperparams['output_base_dir']}\n")
        f.write(f"Weight Matrix Path: {hyperparams['connectivity_matrix_path']}\n")
        if hyperparams.get('input_weights_path'):
            f.write(f"Input Weights Path: {hyperparams['input_weights_path']}\n")
        f.write("\n")
        
        f.write("NETWORK ARCHITECTURE\n")
        f.write("-" * 80 + "\n")
        f.write(f"Reservoir Size: {hyperparams['reservoir_size']} neurons\n")
        f.write(f"Reset Mechanism: {hyperparams['reset_mechanism']}\n")
        f.write(f"Reset Delay: {hyperparams['reset_delay']}\n")
        f.write(f"Input LIF Beta: {hyperparams['input_lif_beta']}\n")
        f.write("\n")
        
        f.write("PARAMETER RANGES\n")
        f.write("-" * 80 + "\n")
        f.write(f"Spectral Radius Range: [{hyperparams['spectral_radius_range'][0]}, {hyperparams['spectral_radius_range'][1]}]\n")
        f.write(f"Beta Reservoir Range: [{hyperparams['beta_reservoir_range'][0]}, {hyperparams['beta_reservoir_range'][1]}]\n")
        f.write("\n")
        
        # Load and add spectral radius information
        try:
            # Try to load the first seed's matrix to get properties
            first_seed = hyperparams.get('seeds', [1])[0]
            matrix_path = hyperparams["connectivity_matrix_path"].format(seed_num=first_seed)
            W = np.load(matrix_path)
            
            f.write("CONNECTIVITY MATRIX PROPERTIES\n")
            f.write("-" * 80 + "\n")
            f.write(f"Spectral Radius: {np.max(np.abs(np.linalg.eigvals(W))):.4f}\n")
            f.write(f"Matrix Shape: {W.shape}\n")
            f.write(f"Matrix Sparsity: {1 - np.count_nonzero(W) / W.size:.4f}\n")
            f.write("\n")
        except Exception as e:
            f.write("CONNECTIVITY MATRIX PROPERTIES\n")
            f.write("-" * 80 + "\n")
            f.write(f"Could not load matrix: {str(e)}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("End of Hyperparameters Summary\n")
        f.write("=" * 80 + "\n")

def scale_matrix_to_radius(W, desired_radius):
    """
    Rescale the matrix W to have the desired spectral radius.
    
    Parameters:
        W (np.ndarray): The weight matrix.
        desired_radius (float): Target spectral radius.
        
    Returns:
        W_rescaled (np.ndarray): The rescaled matrix.
    """
    eigenvalues = np.linalg.eigvals(W)
    current_radius = np.max(np.abs(eigenvalues))
    scaling_factor = desired_radius / current_radius if current_radius != 0 else 1.0
    return W * scaling_factor

def run_trial(config):
    defs.set_seed(42)
    trial_id = uuid.uuid4().hex[:6] # Unique trial ID for logging.

    # Create a folder for this trial inside the output folder.
    trial_folder = os.path.join(config["output_folder"], f"trial_{trial_id}")
    os.makedirs(trial_folder, exist_ok=True)

    # Create a TensorBoard writer for logging trial-specific data.
    writer = SummaryWriter(log_dir=os.path.join(trial_folder, "tensorboard"))
    writer.add_text("Hyperparameters", json.dumps(config, indent=2))

    # Load and rescale the weight matrix to desired spectral radius
    W_initial = np.load(config["connectivity_matrix_path"])
    W_rescaled = scale_matrix_to_radius(W_initial, config["spectral_radius"])
    
    # Save the rescaled matrix temporarily
    temp_matrix_path = os.path.join(trial_folder, "temp_rescaled_matrix.npy")
    np.save(temp_matrix_path, W_rescaled)

    model = SpikingReservoirLoaded(
        threshold=1.0,  # Fixed threshold
        beta_reservoir=config["beta_reservoir"],
        reservoir_size=config["reservoir_size"],
        device=config["device"],
        reset_delay=config["reset_delay"],
        input_lif_beta=config["input_lif_beta"],
        reset_mechanism=config["reset_mechanism"],
        connectivity_matrix_path=temp_matrix_path,
        input_weights_path=config.get("input_weights_path", None)
    )
    model.eval()

    # Generate 50-time-step input signal.
    x = generate_input_signal(V_threshold=1.0, time_steps=50)  # Fixed threshold

    # Run the simulation.
    avg_firing_rate, spike_record, mem_record = model(x)

    # Log per-time-step average firing rate and histograms to TensorBoard.
    T = spike_record.shape[0]
    for t in range(T):
        avg_rate_t = np.mean(spike_record[t])
        writer.add_scalar("FiringRate/TimeStep", avg_rate_t, global_step=t)
    writer.add_histogram("MembraneVoltage/Time0", mem_record[0].flatten(), global_step=0)
    writer.add_histogram("MembraneVoltage/Final", mem_record[-1].flatten(), global_step=T-1)
    writer.flush()
    writer.close()

    # Save the raw spike and membrane potential data as .npy files.
    firing_rates_file = os.path.join(trial_folder, f"firing_rates_trial_{trial_id}.npy")
    mem_potentials_file = os.path.join(trial_folder, f"membrane_potentials_trial_{trial_id}.npy")
    np.save(firing_rates_file, spike_record)
    np.save(mem_potentials_file, mem_record)

    # Aggregate a per-time-step average firing rate (averaged across neurons).
    avg_rate_time = np.mean(spike_record, axis=2).squeeze(1)  # shape: (time_steps,)

    # Clean up temporary matrix file
    os.remove(temp_matrix_path)

    # Report the trial result (and extra fields) via session.report.
    session.report({
        "avg_firing_rate": avg_firing_rate,
        "firing_rate_time": avg_rate_time,
        "firing_rates_file": firing_rates_file,
        "mem_potentials_file": mem_potentials_file
    })

def run_experiment(hyperparams):
    """
    Run a single LSM experiment with the given hyperparameters.
    This function is called by both the main script and parallel experiments.
    """
    # Create output folder if it doesn't exist
    output_folder = hyperparams.get("output_folder", 
                                  create_unique_folder(hyperparams["output_base_dir"]))
    hyperparams["output_folder"] = output_folder
    print(f"Running experiment in output folder: {output_folder}")

    # Save detailed hyperparameters log
    save_hyperparameters_log(hyperparams, output_folder)

    # Define grid search arrays for spectral_radius and beta_reservoir
    n_points = hyperparams.get("grid_points", 50)
    spectral_radius_vals = np.linspace(hyperparams["spectral_radius_range"][0], 
                                     hyperparams["spectral_radius_range"][1], 
                                     n_points).tolist()
    beta_vals = np.linspace(hyperparams["beta_reservoir_range"][0], 
                           hyperparams["beta_reservoir_range"][1], 
                           n_points).tolist()

    # Get the list of seeds to run
    seeds = hyperparams.get("seeds", [1, 2, 3, 4, 5])
    
    # Initialize arrays to store results from all seeds
    all_FR_time = []
    all_convergence_times = []
    
    # Run experiments for each seed
    for seed in seeds:
        print(f"\nRunning experiment for seed {seed}")
        
        # Create seed-specific output folder
        seed_folder = os.path.join(output_folder, f"seed_{seed}")
        os.makedirs(seed_folder, exist_ok=True)
        
        # Update connectivity matrix path for this seed
        seed_hyperparams = hyperparams.copy()
        seed_hyperparams["output_folder"] = seed_folder
        seed_hyperparams["connectivity_matrix_path"] = hyperparams["connectivity_matrix_path"].format(seed_num=seed)
        
        # Create configuration for ray tune
        config = {
            "spectral_radius": tune.grid_search(spectral_radius_vals),
            "beta_reservoir": tune.grid_search(beta_vals),
            "reservoir_size": seed_hyperparams["reservoir_size"],
            "device": seed_hyperparams["device"],
            "reset_delay": seed_hyperparams["reset_delay"],
            "input_lif_beta": seed_hyperparams["input_lif_beta"],
            "reset_mechanism": seed_hyperparams["reset_mechanism"],
            "connectivity_matrix_path": seed_hyperparams["connectivity_matrix_path"],
            "input_weights_path": seed_hyperparams.get("input_weights_path", None),
            "output_folder": seed_hyperparams["output_folder"],
        }

        # Initialize ray and run the grid search
        ray.init(ignore_reinit_error=True)
        analysis = tune.run(
            run_trial,
            config=config,
            resources_per_trial={"gpu": 1} if seed_hyperparams["device"]=="cuda" else {"cpu": 1},
            metric="avg_firing_rate",
            mode="max",
            storage_path=hyperparams.get("storage_path", os.path.join(seed_folder, "ray_storage"))
        )
        ray.shutdown()

        # Process results for this seed
        trials = analysis.trials
        time_steps = 50  # This should match the number of time steps you used
        FR_time = np.zeros((time_steps, len(spectral_radius_vals), len(beta_vals)))
        convergence_times = np.zeros((len(spectral_radius_vals), len(beta_vals)))
        
        for trial in trials:
            rho_val = trial.config["spectral_radius"]
            b_val = trial.config["beta_reservoir"]
            i = spectral_radius_vals.index(rho_val)
            j = beta_vals.index(b_val)
            FR_time[:, i, j] = trial.last_result["firing_rate_time"]
            
            # Calculate convergence time (you'll need to implement this)
            convergence_times[i, j] = calculate_convergence_time(trial.last_result["firing_rate_time"])

        # Save the seed-specific results
        fr_time_file = os.path.join(seed_folder, "FR_time.npy")
        np.save(fr_time_file, FR_time)
        
        conv_time_file = os.path.join(seed_folder, "convergence_times.npy")
        np.save(conv_time_file, convergence_times)
        
        # Store results for averaging
        all_FR_time.append(FR_time)
        all_convergence_times.append(convergence_times)
        
        # Generate plots for this seed
        import LSM_plots as plots
        plots.plot_all_static_3d_surfaces(fr_time_file, beta_vals, spectral_radius_vals, seed_folder)
        plots.plot_all_static_2d_heatmaps(fr_time_file, beta_vals, spectral_radius_vals, seed_folder)
        plots.plot_interactive_animated_3d_from_file(fr_time_file, beta_vals, spectral_radius_vals, seed_folder)
        plots.plot_interactive_animated_2d_from_file(fr_time_file, beta_vals, spectral_radius_vals, seed_folder)
        plots.plot_convergence_time_heatmap(conv_time_file, beta_vals, spectral_radius_vals, seed_folder)

    # Calculate and save averaged results
    avg_FR_time = np.mean(all_FR_time, axis=0)
    avg_convergence_times = np.mean(all_convergence_times, axis=0)
    
    # Save averaged results
    avg_fr_time_file = os.path.join(output_folder, "avg_FR_time.npy")
    np.save(avg_fr_time_file, avg_FR_time)
    
    avg_conv_time_file = os.path.join(output_folder, "avg_convergence_times.npy")
    np.save(avg_conv_time_file, avg_convergence_times)
    
    # Generate averaged plots
    import LSM_plots as plots
    plots.plot_all_static_3d_surfaces(avg_fr_time_file, beta_vals, spectral_radius_vals, output_folder, prefix="avg_")
    plots.plot_all_static_2d_heatmaps(avg_fr_time_file, beta_vals, spectral_radius_vals, output_folder, prefix="avg_")
    plots.plot_interactive_animated_3d_from_file(avg_fr_time_file, beta_vals, spectral_radius_vals, output_folder, prefix="avg_")
    plots.plot_interactive_animated_2d_from_file(avg_fr_time_file, beta_vals, spectral_radius_vals, output_folder, prefix="avg_")
    plots.plot_convergence_time_heatmap(avg_conv_time_file, beta_vals, spectral_radius_vals, output_folder, prefix="avg_")

    print(f"Experiment completed. All results saved in: {output_folder}")
    return output_folder

def calculate_convergence_time(firing_rates, threshold=0.01):
    """
    Calculate the convergence time for a given firing rate time series.
    
    Parameters:
        firing_rates (np.ndarray): Array of firing rates over time
        threshold (float): Threshold for considering convergence
        
    Returns:
        int: Time step at which convergence is reached
    """
    # Calculate the difference between consecutive time steps
    diffs = np.abs(np.diff(firing_rates))
    
    # Find the first time step where the difference is below threshold
    for t in range(len(diffs)):
        if np.all(diffs[t:] < threshold):
            return t + 1
    
    return len(firing_rates)  # Return last time step if no convergence

if __name__ == '__main__':
    # Default hyperparameters when running as main script
    hyperparams = {
        "device": "cpu",
        "output_base_dir": "/Users/mikel/Documents/GitHub/polimikel/UCR/Simulations/LSM_impulse/results",
        "connectivity_matrix_path": "/Users/mikel/Documents/GitHub/polimikel/UCR/generated_matrices/Random_Gaussian/E50I50/W_res_seed{seed_num}.npy",
        "input_weights_path": "/Users/mikel/Documents/GitHub/polimikel/UCR/Weight_matrices/nnLinear_Vectors/nnLinear_weights_seed1.npy",
        "reservoir_size": 100,
        "reset_delay": 0,
        "input_lif_beta": 0.01,
        "reset_mechanism": "zero",
        "spectral_radius_range": [0.9, 1.1],
        "beta_reservoir_range": [0.8, 1.0],
        "grid_points": 2,
        "seeds": [1, 2, 3]  # List of seeds to run
    }

    run_experiment(hyperparams)
