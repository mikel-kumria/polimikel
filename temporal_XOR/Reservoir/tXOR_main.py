import os
import sys
import json
import csv
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# Ensure parent directory is in sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tXOR_dataset import TemporalXORDataset
from tXOR_plots import plot_sample_traces
from tXOR_reservoir import TemporalXORReservoirNetwork


def extract_states_and_targets(model, loader, device):
    """
    Forward the reservoir on each batch to collect states and binary targets.
    Returns X (num_samples x res_size) and y (num_samples x output_size).
    """
    X_list, y_list = [], []
    model.eval()
    with torch.no_grad():
        for seq, targets in loader:
            # seq: [batch, seq_len, input_size], targets: [batch, output_size]
            seq = seq.permute(1,0,2).to(device)  # to [seq_len, batch, input]
            batch_size = seq.shape[1]
            # Forward through reservoir only
            V = torch.zeros(batch_size, model.res_size, device=device)
            S = torch.zeros_like(V)
            spk_rec = []
            for t in range(seq.shape[0]):
                inp = seq[t]
                I_in = model.input_layer(inp)
                I_rec = S @ model.W
                V = model.beta * V + I_in + I_rec
                S = (V >= model.threshold).float()
                if model.reset_mechanism=='zero': V = V * (1-S)
                elif model.reset_mechanism=='subtract': V = V - S*model.threshold
                spk_rec.append(S.cpu().numpy())
            spk_rec = np.stack(spk_rec, axis=0)  # [seq_len, batch, res]
            # summarize reservoir state: e.g. average spike rate over all timesteps
            R_avg = spk_rec.mean(axis=0)       # [batch, res]
            X_list.append(R_avg)
            y_list.append(targets.numpy())
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y


def main():
    # Reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Paths
    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, 'Results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Add min_gap and max_gap to folder name
    ds_params = dict(
        num_samples=1000,
        seq_len=50,
        v_th=1.0,
        min_gap=0,
        max_gap=30,
    )
    exp_dir = os.path.join(
        results_dir,
        f'exp_{timestamp}_min{ds_params["min_gap"]}_max{ds_params["max_gap"]}'
    )
    os.makedirs(exp_dir, exist_ok=True)

    # Dataset
    train_ds = TemporalXORDataset(**ds_params)
    val_ds   = TemporalXORDataset(**ds_params)
    batch_size = 64
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    plot_sample_traces(train_ds, exp_dir, timestamp, prefix='Train')

    # Reservoir model (frozen)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net_params = dict(
        input_size=3,
        reservoir_size=100,
        output_size=2,
        threshold=1.0,
        beta_reservoir=0.924,
        spectral_radius=1.273, #1.273 has 145 steps
        connectivity_matrix_path='/Users/mikel/Documents/GitHub/polimikel/UCR/generated_matrices/Symmetric_Uniform/E80I20/W_res_seed1.npy',
        reset_mechanism='zero',
        device=device
    )
    model = TemporalXORReservoirNetwork(**net_params).to(device)
    # Freeze reservoir and input weights
    for p in model.parameters(): p.requires_grad = False

    # Extract reservoir states
    X_train, y_train = extract_states_and_targets(model, train_loader, device)
    X_val,   y_val   = extract_states_and_targets(model, val_loader,   device)

    # Train readout via Ridge regression
    alpha = 1.0  # ridge regularization
    clf = Ridge(alpha=alpha, fit_intercept=False)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred_train = clf.predict(X_train)
    acc_train = ((y_pred_train.argmax(axis=1)==y_train.argmax(axis=1)).mean()*100)
    y_pred_val = clf.predict(X_val)
    acc_val   = ((y_pred_val.argmax(axis=1)==y_val.argmax(axis=1)).mean()*100)

    # Save results in unique experiment folder
    with open(os.path.join(exp_dir, 'results.txt'), 'w') as f:
        f.write(f"Train accuracy: {acc_train:.2f}%\n")
        f.write(f"Val   accuracy: {acc_val:.2f}%\n")

    # Plot and save train/val accuracy as large text
    plt.figure(figsize=(6, 4))
    plt.text(0.5, 0.7, f"Train Accuracy: {acc_train:.2f}%", fontsize=18, ha='center')
    plt.text(0.5, 0.3, f"Val Accuracy: {acc_val:.2f}%", fontsize=18, ha='center')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'accuracy_text.png'))
    plt.close()

    # Save experiment parameters
    net_params_serializable = net_params.copy()
    if isinstance(net_params_serializable.get('device', None), torch.device):
        net_params_serializable['device'] = str(net_params_serializable['device'])
    experiment_info = {
        'timestamp': timestamp,
        'dataset_params': ds_params,
        'net_params': net_params_serializable,
        'seed': seed,
        'alpha': alpha,
    }
    with open(os.path.join(exp_dir, 'experiment_info.json'), 'w') as f:
        json.dump(experiment_info, f, indent=4)
    with open(os.path.join(exp_dir, 'experiment_info.txt'), 'w') as f:
        for k, v in experiment_info.items():
            f.write(f'{k}: {v}\n')

    # Save states and readout
    np.savez(os.path.join(exp_dir, 'states.npz'),
             X_train=X_train, y_train=y_train,
             X_val=X_val,     y_val=y_val)
    np.savez(os.path.join(exp_dir, 'readout_coef.npz'), coef=clf.coef_)

    print(f"Done. Train acc: {acc_train:.2f}%, Val acc: {acc_val:.2f}%")

if __name__=='__main__':
    main()