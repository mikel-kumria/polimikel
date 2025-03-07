import os
import json
import numpy as np
import torch
import optuna
import datetime
import matplotlib
matplotlib.use('Agg')

# Import model and plotting functions from your codebase.
from LSM_imports import SpikingReservoirLoaded, generate_input_signal
from LSM_plots import (
    animate_reservoir_activity,
    animate_HPO_heatmap,
    animate_HPO_3D_surface,
    plot_static_heatmap_reach_time,
)

# =====================================================
# 1. Define Hyperparameters and Output Folder
# =====================================================
hyperparams = {
    # Device and connectivity.
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "connectivity_matrix_path": "/home/workspaces/polimikel/UCR/Weight_matrices/matrix_seed_42_uniform_20250306_145913/original/W_original.npy",
    
    # Model parameters.
    "reservoir_size": 100,
    "reset_delay": 0,
    "input_lif_beta": 0.01,
    "reset_mechanism": "zero",
    
    # Tuning ranges.
    "threshold_range": [0, 2],          # Range for V_threshold.
    "beta_reservoir_range": [0.01, 0.99], # Range for beta_reservoir.
    "n_grid_points": 3,                 # Number of grid points per parameter.
    
    # Simulation settings.
    "time_steps": 200
}

# Create an output folder.
base_output_dir = "/home/workspaces/polimikel/UCR/Simulations/pos/reservoir_delta/Results"
unique_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_folder_name = f"Results_{unique_datetime}"
output_folder = os.path.join(base_output_dir, results_folder_name)
os.makedirs(output_folder, exist_ok=True)

# Save hyperparameters to a JSON file.
with open(os.path.join(output_folder, "hyperparameters.json"), "w") as f:
    json.dump(hyperparams, f, indent=4)

# Create grid search arrays.
n_points = hyperparams["n_grid_points"]
threshold_values = np.linspace(hyperparams["threshold_range"][0], hyperparams["threshold_range"][1], n_points).tolist()
beta_values = np.linspace(hyperparams["beta_reservoir_range"][0], hyperparams["beta_reservoir_range"][1], n_points).tolist()
# The search space keys must match the parameter names used in the objective.
search_space = {"V_threshold": threshold_values, "beta_reservoir": beta_values}

# =====================================================
# 2. Define the Optuna Objective Function
# =====================================================
def objective(trial):
    # Attach the output folder to the trial (if needed later).
    trial.set_user_attr("output_folder", output_folder)
    
    # Suggest hyperparameters. (GridSampler will force these values.)
    V_threshold = trial.suggest_float("V_threshold", hyperparams["threshold_range"][0], hyperparams["threshold_range"][1])
    beta_reservoir = trial.suggest_float("beta_reservoir", hyperparams["beta_reservoir_range"][0], hyperparams["beta_reservoir_range"][1])
    
    # Instantiate the spiking reservoir model.
    model = SpikingReservoirLoaded(
        threshold=V_threshold,
        beta_reservoir=beta_reservoir,
        reservoir_size=hyperparams["reservoir_size"],
        device=hyperparams["device"],
        reset_delay=hyperparams["reset_delay"],
        input_lif_beta=hyperparams["input_lif_beta"],
        reset_mechanism=hyperparams["reset_mechanism"],
        connectivity_matrix_path=hyperparams["connectivity_matrix_path"]
    )
    model.eval()
    
    # Generate the input signal. Note: your generate_input_signal function now multiplies V_threshold by 2 for the first time step.
    x_input = generate_input_signal(V_threshold, time_steps=hyperparams["time_steps"])
    
    # Run the reservoir simulation.
    avg_fr, spike_record, mem_record = model(x_input)
    
    # Save an animation of the reservoir's spiking activity.
    activity_video_path = os.path.join(output_folder, f"activity_{V_threshold}_{beta_reservoir}.mp4")
    animate_reservoir_activity(spike_record, activity_video_path, fps=10)
    print(f"Saved reservoir activity video to {activity_video_path}")
    
    # Record additional data in trial user attributes.
    firing_rate_time = np.mean(spike_record.squeeze(1), axis=1)  # Firing rate at each time step.
    trial.set_user_attr("firing_rate_time", firing_rate_time)
    trial.set_user_attr("spike_record", spike_record)
    trial.set_user_attr("mem_record", mem_record)
    # Save recurrent weights for reference.
    weights = model.reservoir_lif.recurrent.weight.cpu().detach().numpy()
    trial.set_user_attr("weights", weights)
    
    # Return the average firing rate as the objective.
    return avg_fr

# =====================================================
# 3. Run the Optuna Study with Grid Sampling
# =====================================================
sampler = optuna.samplers.GridSampler(search_space)
study = optuna.create_study(sampler=sampler, direction="maximize")
n_trials = len(threshold_values) * len(beta_values)
study.optimize(objective, n_trials=n_trials, n_jobs=1)

# =====================================================
# 4. Process Results and Generate Plots
# =====================================================
trials = study.trials

# Build a 2D grid for the average firing rate.
firing_rate_grid = np.zeros((len(threshold_values), len(beta_values)))
for trial in trials:
    v_th = trial.params["V_threshold"]
    beta_val = trial.params["beta_reservoir"]
    i = threshold_values.index(v_th)
    j = beta_values.index(beta_val)
    firing_rate_grid[i, j] = trial.value

# Build a 3D array for time evolution of firing rate.
# We assume each trial stores a "firing_rate_time" array of length T.
rep_trial = trials[0]
T = len(rep_trial.user_attrs["firing_rate_time"])
FR_time = np.zeros((T, len(threshold_values), len(beta_values)))
for trial in trials:
    v_th = trial.params["V_threshold"]
    beta_val = trial.params["beta_reservoir"]
    i = threshold_values.index(v_th)
    j = beta_values.index(beta_val)
    FR_time[:, i, j] = trial.user_attrs["firing_rate_time"]

# Define a target firing rate for the static heatmap (using the mean of the grid).
target_rate = np.mean(firing_rate_grid)

# Generate the animated heatmap for hyperparameter evolution.
hpo_heatmap_video = os.path.join(output_folder, "HPO_heatmap.mp4")
animate_HPO_heatmap(FR_time, beta_values, threshold_values, hpo_heatmap_video, fps=2)
print(f"Saved animated HPO heatmap to {hpo_heatmap_video}")

# Generate the animated 3D surface plot.
hpo_3d_video = os.path.join(output_folder, "HPO_3D_surface.mp4")
animate_HPO_3D_surface(FR_time, beta_values, threshold_values, hpo_3d_video, fps=2)
print(f"Saved animated HPO 3D surface to {hpo_3d_video}")

# Generate the static heatmap showing time-to-reach target firing rate.
static_heatmap_path = os.path.join(output_folder, "static_heatmap.png")
plot_static_heatmap_reach_time(FR_time, beta_values, threshold_values, target_rate, static_heatmap_path)
print(f"Saved static heatmap to {static_heatmap_path}")

print("Optuna HPO run completed; all results and plots have been saved.")
