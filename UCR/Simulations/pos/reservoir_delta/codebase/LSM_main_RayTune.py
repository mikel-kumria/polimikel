import os
import numpy as np
import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import matplotlib
import datetime

# Ensure matplotlib uses a non-interactive backend.
os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from ray.air import session

# Import your model and input signal generator.
from LSM_imports import SpikingReservoirLoaded, generate_input_signal, set_seed

# --- Import plotting functions from LSM_plots.py ---
from LSM_plots import (
    animate_reservoir_activity,
    animate_HPO_heatmap,
    animate_HPO_3D_surface,
    plot_static_heatmap_reach_time,
)

# ---------------------------------------------------------------
# Set up the results directory structure.
# ---------------------------------------------------------------
# Manually set the connectivity matrix file path (only one file now).
connectivity_matrix_path = "/home/workspaces/polimikel/UCR/Weight_matrices/matrix_seed_42_uniform_20250306_145913/original/W_original.npy"

# Extract a unique datetime string.
unique_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# setting all seeds
set_seed(42)

# Extract base name and derive a folder name.
base_name = os.path.basename(connectivity_matrix_path)
if base_name.startswith("W_rescaled_"):
    param_str = base_name[len("W_rescaled_"):]
else:
    param_str = base_name
param_str = os.path.splitext(param_str)[0].replace(".", "x")
results_folder_name = f"Results_{param_str}_{unique_datetime}"

# Define the base results directory (adjust as needed).
base_results_dir = "/home/workspaces/polimikel/UCR/Simulations/pos/reservoir_delta/Results"
base_folder = os.path.join(base_results_dir, results_folder_name)

# Create subfolders: RayTune, TensorBoard, and Plots.
raytune_folder = os.path.join(base_folder, "RayTune")
tensorboard_folder = os.path.join(base_folder, "TensorBoard")
plots_folder = os.path.join(base_folder, "Plots")

os.makedirs(raytune_folder, exist_ok=True)
os.makedirs(tensorboard_folder, exist_ok=True)
os.makedirs(plots_folder, exist_ok=True)

# ---------------------------------------------------------------
# Define the objective function for Ray Tune.
# ---------------------------------------------------------------
def objective_reservoir(config):
    """
    Ray Tune objective function.
    Loads a connectivity matrix based on config, creates an input signal,
    runs the reservoir simulation, and reports the average firing rate.
    """
    V_threshold = config["V_threshold"]
    beta_reservoir = config["beta_reservoir"]
    connectivity_matrix_path = config["connectivity_matrix_path"]
    reservoir_size = config["reservoir_size"]
    device = config["device"]
    reset_delay = config["reset_delay"]
    input_lif_beta = config["input_lif_beta"]
    reset_mechanism = config["reset_mechanism"]
    results_folder = config["results_folder"]

    model = SpikingReservoirLoaded(
        threshold=V_threshold,
        beta_reservoir=beta_reservoir,
        reservoir_size=reservoir_size,
        device=device,
        reset_delay=reset_delay,
        input_lif_beta=input_lif_beta,
        reset_mechanism=reset_mechanism,
        connectivity_matrix_path=connectivity_matrix_path,
    )
    model.eval()
    # Generate input signal: first time step equals V_threshold, rest are zeros.
    x_input = generate_input_signal(V_threshold, time_steps=200)
    avg_fr, spike_record, _ = model(x_input)
    
    # Save an animation of the reservoir activity in the "Plots" folder.
    output_video = os.path.join(results_folder, "Plots", f"activity_{V_threshold}_{beta_reservoir}.mp4")
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    animate_reservoir_activity(spike_record, output_video, fps=10)
    print(f"Animation saved to {output_video}")
    
    # Report the average firing rate for hyperparameter optimization.
    tune.report({"avg_firing_rate": avg_fr})

# ---------------------------------------------------------------
# Define the configuration for Ray Tune.
# ---------------------------------------------------------------
config = {
    # Use GPU if available.
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "connectivity_matrix_path": connectivity_matrix_path,
    "reservoir_size": 100,
    "reset_delay": 0,
    "input_lif_beta": 0.01,
    "reset_mechanism": "zero",
    "V_threshold": tune.grid_search(np.linspace(0, 2, 5).tolist()),
    "beta_reservoir": tune.grid_search(np.linspace(0.01, 0.99, 5).tolist()),
    # Pass the base folder so the objective function can store plots in "Plots".
    "results_folder": base_folder,
}

# Set the SummaryWriter to log to the "TensorBoard" subfolder.
writer = SummaryWriter(log_dir=tensorboard_folder)

scheduler = ASHAScheduler(metric="avg_firing_rate", mode="max", max_t=1, grace_period=1)

analysis = tune.run(
    objective_reservoir,
    config=config,
    num_samples=1,
    scheduler=scheduler,
    storage_path=raytune_folder,  # Store Ray Tune trial outputs in the "RayTune" folder.
    resources_per_trial={"cpu": 2, "gpu": 1} if torch.cuda.is_available() else {"cpu": 2},
)

df = analysis.results_df
print("Best trial metrics:")
print(df.sort_values("avg_firing_rate", ascending=False).head())

writer.close()

# ---------------------------------------------------------------
# After the HPO run, generate additional plots using the results.
# ---------------------------------------------------------------
# Note: Ray Tune stores config keys with a "config/" prefix.
df_pivot = df.pivot_table(
    values="avg_firing_rate", 
    index="config/V_threshold", 
    columns="config/beta_reservoir", 
    aggfunc="mean"
)
df_pivot = df_pivot.sort_index().sort_index(axis=1)

# Extract sorted hyperparameter values.
threshold_values = np.sort(df_pivot.index.to_numpy())
beta_values = np.sort(df_pivot.columns.to_numpy())

# Get the final metric grid (2D array).
final_metric = df_pivot.values

# For animation, create a time series (T frames) that linearly interpolates from 0 to final_metric.
T = 200  # number of time frames for the animation
metric_grid_time = np.array([final_metric * ((t + 1) / T) for t in range(T)])

# Define a target firing rate for the static heatmap (using the mean value).
target_rate = np.mean(final_metric)

# Generate animated heatmap for hyperparameter optimization.
hpo_heatmap_video = os.path.join(plots_folder, "HPO_heatmap.mp4")
animate_HPO_heatmap(metric_grid_time, beta_values, threshold_values, hpo_heatmap_video, fps=2)
print(f"HPO heatmap animation saved to {hpo_heatmap_video}")

# Generate animated 3D surface for hyperparameter optimization.
hpo_3d_video = os.path.join(plots_folder, "HPO_3D_surface.mp4")
animate_HPO_3D_surface(metric_grid_time, beta_values, threshold_values, hpo_3d_video, fps=2)
print(f"HPO 3D surface animation saved to {hpo_3d_video}")

# Generate static heatmap showing time to reach target firing rate.
static_heatmap_path = os.path.join(plots_folder, "static_heatmap.png")
plot_static_heatmap_reach_time(metric_grid_time, beta_values, threshold_values, target_rate, static_heatmap_path)
print(f"Static heatmap saved to {static_heatmap_path}")

print("HPO run completed; all results, plots, and TensorBoard logs have been saved.")
