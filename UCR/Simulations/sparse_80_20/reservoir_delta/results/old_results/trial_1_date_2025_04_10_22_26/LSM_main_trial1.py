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

# ------------------------
# Define the trial function that ray.tune will run
def run_trial(config):
    # Set a fixed random seed for reproducibility.
    defs.set_seed(42)
    trial_id = uuid.uuid4().hex[:6]

    # Create a folder for this trial inside the output folder.
    trial_folder = os.path.join(config["output_folder"], f"trial_{trial_id}")
    os.makedirs(trial_folder, exist_ok=True)

    # Create a TensorBoard writer for logging trial-specific data.
    writer = SummaryWriter(log_dir=os.path.join(trial_folder, "tensorboard"))
    writer.add_text("Hyperparameters", json.dumps(config, indent=2))

    # Instantiate your spiking reservoir model.
    model = SpikingReservoirLoaded(
        threshold=config["threshold"],
        beta_reservoir=config["beta_reservoir"],
        reservoir_size=config["reservoir_size"],
        device=config["device"],
        reset_delay=config["reset_delay"],
        input_lif_beta=config["input_lif_beta"],
        reset_mechanism=config["reset_mechanism"],
        connectivity_matrix_path=config["connectivity_matrix_path"],
        input_weights_path=config.get("input_weights_path", None)
    )
    model.eval()

    # Generate your 50-time-step input signal.
    x = generate_input_signal(V_threshold=config["threshold"], time_steps=50)

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

    # Report the trial result (and extra fields) via session.report.
    session.report({
        "avg_firing_rate": avg_firing_rate,
        "firing_rate_time": avg_rate_time,
        "firing_rates_file": firing_rates_file,
        "mem_potentials_file": mem_potentials_file
    })

# ------------------------
# Main block: set up hyperparameters, run trials, aggregate FR_time, then plot.
if __name__ == '__main__':
    # Define experiment hyperparameters.
    hyperparams = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "output_base_dir": "/Users/mikel/Documents/GitHub/polimikel/UCR/Simulations/sparse_80_20/reservoir_delta/results",
        "connectivity_matrix_path": "/Users/mikel/Documents/GitHub/polimikel/UCR/Weight_matrices/Random_80_20/rho1x0/80_20_weights_sparsity_1_rho1/weight_matrix_seed_1.npy",
        "input_weights_path": "/Users/mikel/Documents/GitHub/polimikel/UCR/Weight_matrices/nnLinear_weights.npy",
        "reservoir_size": 100,
        "reset_delay": 0,
        "input_lif_beta": 0.01,
        "reset_mechanism": "zero",
        "threshold_range": [0.0, 2.0],
        "beta_reservoir_range": [0.0, 1.0],
    }

    # Create a unique folder for this experiment.
    output_folder = defs.create_unique_folder(hyperparams["output_base_dir"])
    hyperparams["output_folder"] = output_folder
    print("Output folder:", output_folder)

    # Save hyperparameters for future reference.
    with open(os.path.join(output_folder, "hyperparameters.json"), "w") as f:
        json.dump(hyperparams, f, indent=2)

    # Define grid search arrays for threshold and beta_reservoir.
    threshold_vals = np.linspace(hyperparams["threshold_range"][0], hyperparams["threshold_range"][1], 50).tolist()
    beta_vals = np.linspace(hyperparams["beta_reservoir_range"][0], hyperparams["beta_reservoir_range"][1], 50).tolist()

    # Create a configuration for ray tune that uses grid search.
    config = {
        "threshold": tune.grid_search(threshold_vals),
        "beta_reservoir": tune.grid_search(beta_vals),
        "reservoir_size": hyperparams["reservoir_size"],
        "device": hyperparams["device"],
        "reset_delay": hyperparams["reset_delay"],
        "input_lif_beta": hyperparams["input_lif_beta"],
        "reset_mechanism": hyperparams["reset_mechanism"],
        "connectivity_matrix_path": hyperparams["connectivity_matrix_path"],
        "input_weights_path": hyperparams.get("input_weights_path", None),
        "output_folder": hyperparams["output_folder"],
    }

    # Initialize ray and run the grid search.
    ray.init(ignore_reinit_error=True)
    analysis = tune.run(
        run_trial,
        config=config,
        resources_per_trial={"gpu": 1} if hyperparams["device"]=="cuda" else {"cpu": 1},
        metric="avg_firing_rate",
        mode="max",
        storage_path=output_folder
    )
    ray.shutdown()

    # ------------------------
    # Aggregate the trial results into a single FR_time array.
    trials = analysis.trials
    time_steps = 50  # This should match the number of time steps you used.
    FR_time = np.zeros((time_steps, len(threshold_vals), len(beta_vals)))
    for trial in trials:
        t_val = trial.config["threshold"]
        b_val = trial.config["beta_reservoir"]
        i = threshold_vals.index(t_val)
        j = beta_vals.index(b_val)
        # We assume that the trial reports "firing_rate_time" in its last result.
        FR_time[:, i, j] = trial.last_result["firing_rate_time"]

    # Save the aggregated FR_time for later plotting.
    fr_time_file = os.path.join(output_folder, "FR_time.npy")
    np.save(fr_time_file, FR_time)
    print("Grid search completed and aggregated FR_time saved as", fr_time_file)

    # ------------------------
    # Compute the spectral radius from the connectivity matrix.
    W = np.load(hyperparams["connectivity_matrix_path"])
    spectral_radius = np.max(np.abs(np.linalg.eigvals(W)))
    print("Computed spectral radius:", spectral_radius)

    # ------------------------
    # Now call the plotting functions (from LSM_plots.py) to generate the plots.
    import LSM_plots as plots

    # Generate static plots (PNG) at time step 0.
    plots.plot_static_3d_surface_from_file(fr_time_file, beta_vals, threshold_vals, spectral_radius, output_folder, time_step=0)
    plots.plot_static_2d_heatmap_from_file(fr_time_file, beta_vals, threshold_vals, spectral_radius, output_folder, time_step=0)

    # Generate interactive animated plots (HTML files).
    plots.plot_interactive_animated_3d_from_file(fr_time_file, beta_vals, threshold_vals, spectral_radius, output_folder)
    plots.plot_interactive_animated_2d_from_file(fr_time_file, beta_vals, threshold_vals, spectral_radius, output_folder)

    print("All plots have been saved in the output folder:", output_folder)
