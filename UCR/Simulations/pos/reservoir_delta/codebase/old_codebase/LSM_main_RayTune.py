import os
import numpy as np
import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import matplotlib
import datetime
import glob

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
    plot_spike_record_static,
    plot_spike_record_interactive,
    plot_mem_records_all_runs,
)

# ---------------------------------------------------------------
# Set up the results directory structure.
# ---------------------------------------------------------------
connectivity_matrix_path = "/home/workspaces/polimikel/UCR/Weight_matrices/matrix_seed_42_uniform_20250306_145913/original/W_original.npy"

unique_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

set_seed(42)

base_name = os.path.basename(connectivity_matrix_path)
if base_name.startswith("W_rescaled_"):
    param_str = base_name[len("W_rescaled_"):]
else:
    param_str = base_name
param_str = os.path.splitext(param_str)[0].replace(".", "x")
results_folder_name = f"Results_{param_str}_{unique_datetime}"

base_results_dir = "/home/workspaces/polimikel/UCR/Simulations/pos/reservoir_delta/Results"
base_folder = os.path.join(base_results_dir, results_folder_name)

raytune_folder = os.path.join(base_folder, "RayTune")
tensorboard_folder = os.path.join(base_folder, "TensorBoard")
plots_folder = os.path.join(base_folder, "Plots")

os.makedirs(raytune_folder, exist_ok=True)
os.makedirs(tensorboard_folder, exist_ok=True)
os.makedirs(plots_folder, exist_ok=True)

# Create folder to store mem_record files for each run.
mem_records_folder = os.path.join(base_folder, "mem_records")
os.makedirs(mem_records_folder, exist_ok=True)

# ---------------------------------------------------------------
# Define the objective function for Ray Tune.
# ---------------------------------------------------------------
def objective_reservoir(config):
    set_seed(42)
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
    x_input = generate_input_signal(V_threshold, time_steps=20)
    # Get mem_record as well now.
    avg_fr, spike_record, mem_record = model(x_input)
    
    # Save an animation of the reservoir activity.
    output_video = os.path.join(results_folder, "Plots", f"activity_{V_threshold}_{beta_reservoir}.mp4")
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    animate_reservoir_activity(spike_record, output_video, fps=10)
    print(f"Animation saved to {output_video}")
    
    # Generate spike record plots (static and interactive).
    static_spike_path = os.path.join(results_folder, "Plots", f"spike_record_{V_threshold}_{beta_reservoir}.png")
    plot_spike_record_static(spike_record, static_spike_path)
    print(f"Spike record static plot saved to {static_spike_path}")
    
    interactive_spike_path = os.path.join(results_folder, "Plots", f"spike_record_{V_threshold}_{beta_reservoir}.html")
    plot_spike_record_interactive(spike_record, interactive_spike_path)
    print(f"Spike record interactive plot saved to {interactive_spike_path}")
    
    # Save mem_record to file for later aggregation.
    mem_record_file = os.path.join(mem_records_folder, f"mem_record_{V_threshold}_{beta_reservoir}.npy")
    np.save(mem_record_file, mem_record)
    print(f"Mem record saved to {mem_record_file}")
    
    # Report the average firing rate for hyperparameter optimization.
    tune.report({"avg_firing_rate": avg_fr})

# ---------------------------------------------------------------
# Define the configuration for Ray Tune.
# ---------------------------------------------------------------
config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "connectivity_matrix_path": connectivity_matrix_path,
    "reservoir_size": 100,
    "reset_delay": 0,
    "input_lif_beta": 0.01,
    "reset_mechanism": "zero",
    "V_threshold": tune.grid_search(np.linspace(0, 2, 5).tolist()),
    "beta_reservoir": tune.grid_search(np.linspace(0.01, 0.99, 5).tolist()),
    "results_folder": base_folder,
}

writer = SummaryWriter(log_dir=tensorboard_folder)

scheduler = ASHAScheduler(metric="avg_firing_rate", mode="max", max_t=1, grace_period=1)

analysis = tune.run(
    objective_reservoir,
    config=config,
    num_samples=1,
    scheduler=scheduler,
    storage_path=raytune_folder,
    resources_per_trial={"cpu": 2, "gpu": 1} if torch.cuda.is_available() else {"cpu": 2},
)

df = analysis.results_df
print("Best trial metrics:")
print(df.sort_values("avg_firing_rate", ascending=False).head())

writer.close()

# ---------------------------------------------------------------
# After the HPO run, generate additional plots using the results.
# ---------------------------------------------------------------
# Create aggregated plots for HPO hyperparameters from mem_record files.
mem_files = glob.glob(os.path.join(mem_records_folder, "mem_record_*.npy"))
mem_records_list = []
for file in mem_files:
    # Expected filename format: "mem_record_{V_threshold}_{beta_reservoir}.npy"
    basename = os.path.basename(file)
    parts = basename.replace("mem_record_", "").replace(".npy", "").split("_")
    if len(parts) >= 2:
        try:
            V_threshold_val = float(parts[0])
            beta_reservoir_val = float(parts[1])
            mem_record_data = np.load(file)
            mem_records_list.append((V_threshold_val, beta_reservoir_val, mem_record_data))
        except ValueError:
            continue

if mem_records_list:
    mem_records_plot_path = os.path.join(plots_folder, "mem_records_all_runs.png")
    plot_mem_records_all_runs(mem_records_list, mem_records_plot_path)
    print(f"Mem records aggregated plot saved to {mem_records_plot_path}")
else:
    print("No mem_record files found for aggregation.")

# Generate animated heatmap for hyperparameter optimization.
df_pivot = df.pivot_table(
    values="avg_firing_rate", 
    index="config/V_threshold", 
    columns="config/beta_reservoir", 
    aggfunc="mean"
)
df_pivot = df_pivot.sort_index().sort_index(axis=1)

threshold_values = np.sort(df_pivot.index.to_numpy())
beta_values = np.sort(df_pivot.columns.to_numpy())
final_metric = df_pivot.values

T = 200  # number of time frames for the animation
metric_grid_time = np.array([final_metric * ((t + 1) / T) for t in range(T)])
target_rate = np.mean(final_metric)

hpo_heatmap_video = os.path.join(plots_folder, "HPO_heatmap.mp4")
animate_HPO_heatmap(metric_grid_time, beta_values, threshold_values, hpo_heatmap_video, fps=2)
print(f"HPO heatmap animation saved to {hpo_heatmap_video}")

hpo_3d_video = os.path.join(plots_folder, "HPO_3D_surface.mp4")
animate_HPO_3D_surface(metric_grid_time, beta_values, threshold_values, hpo_3d_video, fps=2)
print(f"HPO 3D surface animation saved to {hpo_3d_video}")

static_heatmap_path = os.path.join(plots_folder, "static_heatmap.png")
plot_static_heatmap_reach_time(metric_grid_time, beta_values, threshold_values, target_rate, static_heatmap_path)
print(f"Static heatmap saved to {static_heatmap_path}")

print("HPO run completed; all results, plots, and TensorBoard logs have been saved.")
