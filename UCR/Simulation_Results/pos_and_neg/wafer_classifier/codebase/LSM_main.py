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
from sklearn.metrics import f1_score
import pandas as pd
import matplotlib.pyplot as plt


def load_tsv_dataset(tsv_path):
    data = pd.read_csv(tsv_path, sep='\t', header=0)
    labels = data.iloc[:, 0].values.astype(int)
    features = data.iloc[:, 1:].values.astype(np.float32)
    X, y = [], []
    for i in range(features.shape[0]):
        x_sample = features[i]
        x_tensor = torch.tensor(x_sample, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        X.append(x_tensor)
        y.append(labels[i])
    return X, np.array(y)

def objective_classification_batch(trial):
    threshold = trial.suggest_float("threshold", hyperparams["threshold_range"][0], hyperparams["threshold_range"][1])
    beta_reservoir = trial.suggest_float("beta_reservoir", hyperparams["beta_reservoir_range"][0], hyperparams["beta_reservoir_range"][1])
    
    # Instantiate the reservoir.
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
    

    # TRAIN

    X_train, y_train = load_tsv_dataset(hyperparams["tsv_file_train"])
    reservoir_features_train = []
    for x in X_train:
        _, spike_record, _ = model(x)
        features = np.mean(spike_record.squeeze(1), axis=0)
        reservoir_features_train.append(features)
    X_res_train = np.vstack(reservoir_features_train)
    
    N_train = len(y_train)
    Y_targets_train = np.zeros((N_train, 2))
    for i, label in enumerate(y_train):
        if label == 1:
            Y_targets_train[i, 0] = 1
        elif label == -1:
            Y_targets_train[i, 1] = 1
        else:
            raise ValueError("Unexpected training label.")
    
    W_readout = np.linalg.pinv(X_res_train) @ Y_targets_train
    trial.set_user_attr("W_readout", W_readout)
    

    # TEST 

    X_test, y_test = load_tsv_dataset(hyperparams["tsv_file_test"])
    reservoir_features_test = []
    all_fr_times = []
    rep_test_fr = None
    for idx, x in enumerate(X_test):
        _, spike_record, _ = model(x)
        features = np.mean(spike_record.squeeze(1), axis=0)
        reservoir_features_test.append(features)
        fr_time = np.mean(spike_record.squeeze(1), axis=1)
        all_fr_times.append(fr_time)
        if idx == 0:
            rep_test_fr = fr_time
    X_res_test = np.vstack(reservoir_features_test)
    avg_fr_time = np.mean(np.vstack(all_fr_times), axis=0)
    trial.set_user_attr("rep_test_firing_rate", rep_test_fr)
    trial.set_user_attr("avg_test_firing_rate", avg_fr_time)
    
    Y_pred = X_res_test @ W_readout
    pred_classes = np.argmax(Y_pred, axis=1)
    true_classes = np.array([0 if lab == 1 else 1 for lab in y_test])
    
    f1_abnormal = f1_score(true_classes, pred_classes, pos_label=1)
    trial.set_user_attr("f1_abnormal", f1_abnormal)
    
    return f1_abnormal



# MAIN

if __name__ == '__main__':
    hyperparams = {
        #"device": "cuda" if torch.cuda.is_available() else "cpu",
        "device": "cpu",
        "dataset_type": "TSV",
        "tsv_file_train": "/home/workspaces/polimikel/data/UCR_dataset/Wafer/Wafer_TRAIN_mini.tsv",
        "tsv_file_test": "/home/workspaces/polimikel/data/UCR_dataset/Wafer/Wafer_TEST_mini.tsv",

        "output_base_dir": "/home/workspaces/polimikel/UCR/Simulation_Results/pos_and_neg/wafer_classifier/Vmem_input",
        
        "reservoir_size": 100,
        "reset_delay": 0,
        "input_lif_beta": 0.01,
        "reset_mechanism": "zero",
        
        "init_weight_a": -1.0,
        "init_weight_b": 1.0,

        "spectral_radius": 2.0,
        
        "threshold_range": [0.1, 2.0],
        "beta_reservoir_range": [0.01, 0.99],

        "n_grid_points": 50
    }
    
    base_output_dir = hyperparams.get("output_base_dir", "./classification_output")
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    output_folder = defs.create_unique_folder(base_output_dir)
    print("Output folder:", output_folder)
    
    hyperparams_file = os.path.join(output_folder, "hyperparameters.json")
    with open(hyperparams_file, "w") as f:
        json.dump(hyperparams, f, indent=4)
    
    writer = SummaryWriter(log_dir=output_folder)
    
    n_points = hyperparams["n_grid_points"]
    threshold_values = np.linspace(hyperparams["threshold_range"][0], hyperparams["threshold_range"][1], n_points).tolist()
    beta_values = np.linspace(hyperparams["beta_reservoir_range"][0], hyperparams["beta_reservoir_range"][1], n_points).tolist()
    search_space = {"threshold": threshold_values, "beta_reservoir": beta_values}
    
    sampler = optuna.samplers.GridSampler(search_space)
    study = optuna.create_study(sampler=sampler, direction="maximize")
    
    def objective_with_attrs(trial):
        trial.set_user_attr("output_folder", output_folder)
        return objective_classification_batch(trial)
    
    total_trials = len(threshold_values) * len(beta_values)
    study.optimize(objective_with_attrs, n_trials=total_trials, n_jobs=5)

trials = study.trials

# Build a 2D grid of F1 scores over hyperparameters.
f1_grid = np.zeros((len(threshold_values), len(beta_values)))
for trial in trials:
    t_val = trial.params["threshold"]
    b_val = trial.params["beta_reservoir"]
    i = threshold_values.index(t_val)
    j = beta_values.index(b_val)
    f1_grid[i, j] = trial.value

# Build a 3D array for the test reservoir firing rate evolution.
# FR_time has shape (T, num_threshold, num_beta), where T is the number of time steps.
rep_trial = trials[0]
T = len(rep_trial.user_attrs["avg_test_firing_rate"])
FR_time = np.zeros((T, len(threshold_values), len(beta_values)))
for trial in trials:
    t_val = trial.params["threshold"]
    b_val = trial.params["beta_reservoir"]
    i = threshold_values.index(t_val)
    j = beta_values.index(b_val)
    FR_time[:, i, j] = trial.user_attrs["avg_test_firing_rate"]

# Convert the 3D FR_time array into a 2D static grid by averaging over time.
static_FR_grid = np.mean(FR_time, axis=0)

# Save grid ranges for record.
rep_hyperparams_file = os.path.join(output_folder, "hyperparameters.txt")
with open(rep_hyperparams_file, "w") as f:
    f.write("Grid Search Ranges:\n")
    f.write(f"Threshold Range: {hyperparams['threshold_range']}\n")
    f.write(f"Beta Reservoir Range: {hyperparams['beta_reservoir_range']}\n\n")

# Plot static classification performance (F1 score) over hyperparameters.
LSM_plots.plot_static_3d_surface_classification(f1_grid, beta_values, threshold_values, output_folder, writer)
LSM_plots.plot_static_heatmap_classification(f1_grid, beta_values, threshold_values, output_folder, writer)

# Plot reservoir spiking activity using the static_FR_grid (averaged over time).
LSM_plots.plot_static_3d_surface(static_FR_grid, beta_values, threshold_values, hyperparams["spectral_radius"], output_folder, writer)
LSM_plots.plot_static_heatmap(static_FR_grid, beta_values, threshold_values, hyperparams["spectral_radius"], output_folder, writer)

# Plot the readout weight matrix from a representative trial.
rep_W = trials[0].user_attrs["W_readout"]
plt.figure(figsize=(6, 5))
plt.imshow(rep_W, cmap='inferno', aspect='auto')
plt.colorbar(label='Weight Value')
plt.xlabel('Output Neuron (Normal vs Abnormal)')
plt.ylabel('Reservoir Neuron')
plt.title('Readout Weight Matrix (After Regression)')
weight_path = os.path.join(output_folder, 'readout_weight_matrix.png')
plt.savefig(weight_path, dpi=600)
writer.add_figure("Readout_Weights", plt.gcf())
plt.close()

# Optional: Generate animated plots (using the original FR_time, if desired).
LSM_plots.plot_interactive_animated_3d_surface(FR_time, beta_values, threshold_values, hyperparams["spectral_radius"], output_folder)
LSM_plots.plot_interactive_animated_heatmap(FR_time, beta_values, threshold_values, hyperparams["spectral_radius"], output_folder)
LSM_plots.animate_3d_video(FR_time, beta_values, threshold_values, hyperparams["spectral_radius"], output_folder, fps=10)
LSM_plots.animate_heatmap_video(FR_time, beta_values, threshold_values, hyperparams["spectral_radius"], output_folder, fps=10)

writer.close()
print("All static and interactive plots, videos, logs, and hyperparameters have been saved.")
