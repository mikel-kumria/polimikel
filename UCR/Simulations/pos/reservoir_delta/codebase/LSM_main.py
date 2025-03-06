# ray_tune_hpo.py
import os
import numpy as np
import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

# Import the modified reservoir simulation model and input generator.
from LSM_imports import SpikingReservoirLoaded, generate_input_signal

def objective_reservoir(config):
    """
    Ray Tune objective function.
    Loads a connectivity matrix based on config, creates an input signal (first time step = V_threshold, rest=0),
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
    
    model = SpikingReservoirLoaded(threshold=V_threshold,
                                   beta_reservoir=beta_reservoir,
                                   reservoir_size=reservoir_size,
                                   device=device,
                                   reset_delay=reset_delay,
                                   input_lif_beta=input_lif_beta,
                                   reset_mechanism=reset_mechanism,
                                   connectivity_matrix_path=connectivity_matrix_path)
    model.eval()
    # Generate input signal of 200 time steps (first = V_threshold, rest = 0)
    x_input = generate_input_signal(V_threshold, time_steps=200)
    avg_fr, spike_record, _ = model(x_input)
    # Report average firing rate as metric.
    tune.report(avg_firing_rate=avg_fr)

# Grid search configuration.
config = {
    "device": "cpu",  # Change to "cuda" to use GPU.
    "connectivity_matrix_path": tune.grid_search([
        "path_to_W_rho_0x1.npy",
        "path_to_W_rho_0x5.npy",
        "path_to_W_rho_1x0.npy",
        "path_to_W_rho_2x0.npy",
        "path_to_W_rho_10x0.npy"
    ]),
    "reservoir_size": 100,
    "reset_delay": 0,
    "input_lif_beta": 0.01,  # Not used in direct injection but retained for compatibility.
    "reset_mechanism": "zero",
    "V_threshold": tune.grid_search(np.linspace(0, 2, 50).tolist()),
    "beta_reservoir": tune.grid_search(np.linspace(0.01, 0.99, 50).tolist())
}

log_dir = "/path/to/your/log_dir"  # Update this path.
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

scheduler = ASHAScheduler(metric="avg_firing_rate", mode="max", max_t=1, grace_period=1)

analysis = tune.run(
    objective_reservoir,
    config=config,
    num_samples=1,
    scheduler=scheduler,
    local_dir=log_dir,
    resources_per_trial={"cpu": 1, "gpu": 1},  # Adjust if GPU is available.
    verbose=1
)

df = analysis.results_df
print("Best trial metrics:")
print(df.sort_values("avg_firing_rate", ascending=False).head())

writer.close()
print("HPO run completed; all results and TensorBoard logs have been saved.")
