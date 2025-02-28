import os
import numpy as np
import optuna
import torch
import json
from torch.utils.tensorboard import SummaryWriter
import LSM_imports as defs
import LSM_plots
import matplotlib
matplotlib.use('Agg')

# Define the objective function for Optuna.
def objective(trial):
    # Grid search parameters from hyperparams = {...}
    threshold = trial.suggest_float("threshold", hyperparams["threshold_range"][0], hyperparams["threshold_range"][1])
    beta_reservoir = trial.suggest_float("beta_reservoir", hyperparams["beta_reservoir_range"][0], hyperparams["beta_reservoir_range"][1])
    
    # Instantiate the model with hyperparameters.
    model = defs.SpikingReservoir(
        threshold=threshold,
        beta_reservoir=beta_reservoir,
        reservoir_size=hyperparams["reservoir_size"],
        device=hyperparams["device"],
        reset_delay=hyperparams["reset_delay"],
        input_lif_beta=hyperparams["input_lif_beta"],
        reset_mechanism=hyperparams["reset_mechanism"],
        init_weight_a=hyperparams["init_weight_a"],
        init_weight_b=hyperparams["init_weight_b"],
        spectral_radius=hyperparams["spectral_radius"]
    )
    model.eval()
    
    # Select dataset based on hyperparams["dataset_type"].
    if hyperparams["dataset_type"] == "TSV":
        x = defs.load_tsv_input(tsv_file, sample_index=hyperparams["tsv_sample_index"], output_folder=output_folder)
    elif hyperparams["dataset_type"] == "synthetic":
        x = defs.generate_synthetic_input(num_steps=hyperparams["synthetic_num_steps"], threshold=max(hyperparams["threshold_range"]), pattern=hyperparams["synthetic_pattern"], output_folder=output_folder, noise_mean=0.0, noise_std=1.0)
    else:
        raise ValueError("Unknown dataset type")
    
    avg_firing_rate, spike_record, mem_record = model(x)
    
    # Compute the firing rate at each time step.
    # spike_record shape: (T, batch_size, reservoir_size); assume batch_size==1.
    firing_rate_time = np.mean(spike_record.squeeze(1), axis=1)  # shape (T,)
    trial.set_user_attr("firing_rate_time", firing_rate_time)
    trial.set_user_attr("spike_record", spike_record)
    trial.set_user_attr("mem_record", mem_record)
    trial.set_user_attr("weights", model.get_recurrent_weights())
    
    return avg_firing_rate

if __name__ == '__main__':
    # Define all hyperparameters in one dictionary.
    hyperparams = {
        # Device and dataset selection.
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dataset_type": "synthetic",  # "TSV" or "synthetic"
        "tsv_file": "/home/workspaces/polimikel/data/UCR_dataset/Wafer/Wafer_TRAIN.tsv",
        "tsv_sample_index": 0,
        "synthetic_num_steps": 250,
        "synthetic_pattern": "dirac",
        "output_base_dir": "/home/workspaces/polimikel/UCR/Simulation_Results",
        
        # Model architecture.
        "reservoir_size": 100,
        "reset_delay": 0,
        "input_lif_beta": 0.01,
        "reset_mechanism": "zero",
        
        # Spectral initialization.
        "init_weight_a": -1.0,
        "init_weight_b": 1.0,
        "spectral_radius": 1.0,
        
        # Grid search ranges for parameters to be tuned.
        "threshold_range": [0.1, 2.0],
        "beta_reservoir_range": [0.01, 0.99],
        
        # Grid search resolution.
        "n_grid_points": 50
    }
    
    # Create the output folder only once.
    base_output_dir = hyperparams["output_base_dir"]
    output_folder = defs.create_unique_folder(base_output_dir)
    print("Output folder:", output_folder)
    
    # Save hyperparameters to a JSON file.
    hyperparams_file = os.path.join(output_folder, "hyperparameters.json")
    with open(hyperparams_file, "w") as f:
        json.dump(hyperparams, f, indent=4)
    
    # Initialize TensorBoard writer.
    writer = SummaryWriter(log_dir=output_folder)
    
    # Define the TSV file path.
    tsv_file = hyperparams["tsv_file"]
    
    # Create grid search arrays.
    n_points = hyperparams["n_grid_points"]
    threshold_values = np.linspace(hyperparams["threshold_range"][0], hyperparams["threshold_range"][1], n_points).tolist()
    beta_values = np.linspace(hyperparams["beta_reservoir_range"][0], hyperparams["beta_reservoir_range"][1], n_points).tolist()
    search_space = {"threshold": threshold_values, "beta_reservoir": beta_values}
    
    # Set up the grid sampler.
    sampler = optuna.samplers.GridSampler(search_space)
    study = optuna.create_study(sampler=sampler, direction="maximize")
    
    # Pass additional attributes to each trial.
    def objective_with_attrs(trial):
        trial.set_user_attr("output_folder", output_folder)
        trial.set_user_attr("tsv_file", tsv_file)
        return objective(trial)
    
    study.optimize(objective_with_attrs, n_trials=len(threshold_values) * len(beta_values), n_jobs=10) # NOTE: n_jobs=3 (for example) for parallel execution, reduce if it is too slow, or set to 1 for serial execution.
    
    # Extract grid results (static average firing rate).
    trials = study.trials
    firing_rate_grid = np.zeros((len(threshold_values), len(beta_values)))
    for trial in trials:
        t_val = trial.params["threshold"]
        b_val = trial.params["beta_reservoir"]
        i = threshold_values.index(t_val)
        j = beta_values.index(b_val)
        firing_rate_grid[i, j] = trial.value
    
    # Build a 3D array for time evolution.
    rep_trial = trials[0]
    T = len(rep_trial.user_attrs["firing_rate_time"])
    FR_time = np.zeros((T, len(threshold_values), len(beta_values)))
    for trial in trials:
        t_val = trial.params["threshold"]
        b_val = trial.params["beta_reservoir"]
        i = threshold_values.index(t_val)
        j = beta_values.index(b_val)
        FR_time[:, i, j] = trial.user_attrs["firing_rate_time"]
    
    # Save trial hyperparameters (grid ranges and representative values).
    rep_hyperparams_file = os.path.join(output_folder, "hyperparameters.txt")
    with open(rep_hyperparams_file, "w") as f:
        f.write("Grid Search Ranges:\n")
        f.write(f"Threshold Range: {hyperparams['threshold_range']}\n")
        f.write(f"Beta Reservoir Range: {hyperparams['beta_reservoir_range']}\n\n")
    
    # Retrieve detailed data from the representative trial.
    spike_record = rep_trial.user_attrs["spike_record"]   # shape: (T, 1, reservoir_size)
    mem_record = rep_trial.user_attrs["mem_record"]
    weights = rep_trial.user_attrs["weights"]
    spike_record = spike_record[:, 0, :]  # remove batch dimension
    mem_record = mem_record[:, 0, :]
    
    # Compute inter-spike intervals.
    all_intervals = []
    for neuron in range(spike_record.shape[1]):
        spike_times = np.where(spike_record[:, neuron] > 0)[0]
        if len(spike_times) > 1:
            intervals = np.diff(spike_times)
            all_intervals.extend(intervals)
    
    # Generate static plots.
    LSM_plots.plot_static_3d_surface(firing_rate_grid, beta_values, threshold_values, hyperparams["spectral_radius"], output_folder, writer)
    LSM_plots.plot_static_heatmap(firing_rate_grid, beta_values, threshold_values, hyperparams["spectral_radius"], output_folder, writer)
    LSM_plots.plot_static_spike_raster(spike_record, output_folder, writer)
    LSM_plots.plot_static_membrane_traces(mem_record, output_folder, writer)
    LSM_plots.plot_static_isi_histogram(all_intervals, output_folder, writer)
    LSM_plots.plot_static_weight_matrix(weights, output_folder, writer)
    LSM_plots.plot_static_eigenvalues(np.linalg.eigvals(weights), output_folder, writer)
    
    # Generate interactive plots.
    LSM_plots.plot_interactive_ts_input(tsv_file, output_folder)
    LSM_plots.plot_interactive_3d_surface(firing_rate_grid, beta_values, threshold_values, hyperparams["spectral_radius"], output_folder)
    LSM_plots.plot_interactive_heatmap(firing_rate_grid, beta_values, threshold_values, hyperparams["spectral_radius"], output_folder)
    LSM_plots.plot_interactive_spike_raster(spike_record, output_folder)
    LSM_plots.plot_interactive_membrane_traces(mem_record, output_folder)
    LSM_plots.plot_interactive_isi_histogram(all_intervals, output_folder)
    LSM_plots.plot_interactive_weight_matrix(weights, output_folder)
    LSM_plots.plot_interactive_eigenvalues(np.linalg.eigvals(weights), output_folder)
    
    # Generate animated plots (HTML).
    LSM_plots.plot_interactive_animated_3d_surface(FR_time, beta_values, threshold_values, hyperparams["spectral_radius"], output_folder)
    LSM_plots.plot_interactive_animated_heatmap(FR_time, beta_values, threshold_values, hyperparams["spectral_radius"], output_folder)
    
    # Also export animated plots as videos (MP4) with adjustable frame rate.
    LSM_plots.animate_3d_video(FR_time, beta_values, threshold_values, hyperparams["spectral_radius"], output_folder, fps=10)
    LSM_plots.animate_heatmap_video(FR_time, beta_values, threshold_values, hyperparams["spectral_radius"], output_folder, fps=10)
    
    writer.close()
    print("All static and interactive plots, videos, logs, and hyperparameters have been saved.")
