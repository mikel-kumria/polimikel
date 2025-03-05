import torch
from torch import nn
import os
import json
# import nbimporter  # use this to import from Jupyter notebooks
from datetime import datetime
import random
import numpy as np
import tqdm.notebook as tqdm
import snntorch as snn
import torch.nn.functional as F
from plot_functions import *
from dataset_generation import *
from training_validation_testing import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Create "simulations" folder two levels up from the current directory, if it doesn't exist already
parent_folder = os.path.dirname(os.getcwd())  # Get the parent directory
grandparent_folder = os.path.dirname(parent_folder) # Go up another level
simulations_folder = os.path.join(grandparent_folder, 'thesis_simulations')
os.makedirs(simulations_folder, exist_ok=True)

# Create unique folder name inside the "simulations" folder
now = datetime.now()
unique_folder_name = os.path.join(simulations_folder, now.strftime("PY_wave_order_recognition_%Yy%mm%dd_%Hh%Mm"))
os.makedirs(unique_folder_name, exist_ok=True)

# Seeds for reproducibility
seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if device == 'cuda':
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)



"""

                    HYPERPARAMETERS

"""

std_dev = 0.0
num_samples = 1000
train_ratio = 0.7
test_ratio = 0.1
validation_ratio = 0.2

freq_min = 2
freq_max = 2
amp_min = 1.1
amp_max = 1.1
offset = 1.1

sample_rate = 20 * freq_max
duration = 3
deltaT = 1 / sample_rate
num_steps = int(duration / deltaT)

input_size = 1
hidden_size = 70
output_size = 2

threshold_hidden_min = 1.0
threshold_hidden_max = 1.1
learn_threshold_hidden = True

threshold_output_min = 1.0
threshold_output_max = 1.1
learn_threshold_output = True

N_hidden_weights_gaussian, N_hidden_weights_std = 10, 1
N_output_weights_gaussian, N_output_weights_std = 0.3, 0.1

gaussian_mean_hidden_weights = N_hidden_weights_gaussian / sample_rate
gaussian_std_hidden_weights = N_hidden_weights_std / sample_rate

gaussian_mean_output_weights = N_output_weights_gaussian / (sample_rate * hidden_size)
gaussian_std_output_weights = N_output_weights_std / (sample_rate * hidden_size)

hidden_reset_mechanism = 'zero'
output_reset_mechanism = 'zero'

weights_hidden_min_clamped = 0.0
weights_hidden_max_clamped = 2.0
weights_output_min_clamped = 0.0
weights_output_max_clamped = 2.0

N_hidden_tau, N_output_tau = 0.1, 0.1
tau_hidden = torch.Tensor(hidden_size).uniform_(N_hidden_tau * freq_min, N_hidden_tau * freq_max)
tau_output = torch.Tensor(output_size).uniform_(N_output_tau * freq_min, N_output_tau * freq_max)

beta_hidden = torch.exp(-deltaT / tau_hidden)
beta_output = torch.exp(-deltaT / tau_output)

learn_beta_hidden = True
learn_beta_output = True

threshold_hidden = np.random.uniform(threshold_hidden_min, threshold_hidden_max, hidden_size)
threshold_output = np.random.uniform(threshold_output_min, threshold_output_max, output_size)

# Training parameters
num_epochs = 100
learning_rate = 0.02
batch_size = 32                     # 32 is Yann LeCun's magic number
optimizer_betas = (0.99, 0.999)     # Adam optimizer's betas, first value is for the gradient and the second for the gradient squared
scheduler_step_size = 30            # Decrease the learning rate by scheduler_gamma every scheduler_step_size epochs
scheduler_gamma = 0.5
penalty_weight = 1                  # Weight of the spike_count regularization term
L1_lambda = 0.001                   # L1 regularization strength

# Save hyperparameters
hyperparams = {
    "seed": seed,
    "std_dev": std_dev,
    "offset": offset,
    "num_samples": num_samples,
    "train_ratio": train_ratio,
    "test_ratio": test_ratio,
    "validation_ratio": validation_ratio,
    "freq_min": freq_min,
    "freq_max": freq_max,
    "amp_min": amp_min,
    "amp_max": amp_max,
    "sample_rate": sample_rate,
    "duration": duration,
    "deltaT": deltaT,
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "num_steps": num_steps,
    "threshold_hidden_min": threshold_hidden_min,
    "threshold_hidden_max": threshold_hidden_max,
    "threshold_output_min": threshold_output_min,
    "threshold_output_max": threshold_output_max,
    "gaussian_mean_hidden_weights": gaussian_mean_hidden_weights,
    "gaussian_std_hidden_weights": gaussian_std_hidden_weights,
    "gaussian_mean_output_weights": gaussian_mean_output_weights,
    "gaussian_std_output_weights": gaussian_std_output_weights,
    "hidden_reset_mechanism": hidden_reset_mechanism,
    "output_reset_mechanism": output_reset_mechanism,
    "weights_hidden_min_clamped": weights_hidden_min_clamped,
    "weights_hidden_max_clamped": weights_hidden_max_clamped,
    "weights_output_min_clamped": weights_output_min_clamped,
    "weights_output_max_clamped": weights_output_max_clamped,
    "num_epochs": num_epochs,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "optimizer_betas": optimizer_betas,
    "scheduler_step_size": scheduler_step_size,
    "scheduler_gamma": scheduler_gamma,
    "beta_hidden": beta_hidden.tolist(),
    "beta_output": beta_output.tolist(),
    "threshold_hidden": threshold_hidden.tolist(),
    "threshold_output": threshold_output.tolist(),
    "N_hidden_weights_gaussian": N_hidden_weights_gaussian,
    "N_hidden_weights_std": N_hidden_weights_std,
    "N_output_weights_gaussian": N_output_weights_gaussian,
    "N_output_weights_std": N_output_weights_std,
    "learn_beta_hidden": learn_beta_hidden,
    "learn_beta_output": learn_beta_output,
    "learn_threshold_hidden": learn_threshold_hidden,
    "learn_threshold_output": learn_threshold_output,
    "hidden_tau": N_hidden_tau,
    "output_tau": N_output_tau,
    "phase1": "random_uniform_0_to_2pi",
    "phase2": "random_uniform_0_to_2pi"
}

with open(os.path.join(unique_folder_name, 'hyperparameters.json'), 'w') as f:
    json.dump(hyperparams, f)

# Define the Spiking Neural Network
class Wave_Order_Recognition_SNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.lif1 = snn.Leaky(beta=beta_hidden, threshold=threshold_hidden, learn_beta=learn_beta_hidden, learn_threshold=learn_threshold_hidden, reset_mechanism=hidden_reset_mechanism, reset_delay=False)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)
        self.lif2 = snn.Leaky(beta=beta_output, threshold=threshold_output, learn_beta=learn_beta_output, learn_threshold=learn_threshold_output, reset_mechanism=output_reset_mechanism, reset_delay=False)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.fc1.weight, mean=gaussian_mean_hidden_weights, std=gaussian_std_hidden_weights)
        nn.init.normal_(self.fc2.weight, mean=gaussian_mean_output_weights, std=gaussian_std_output_weights)

    def forward(self, x, mem1=None, mem2=None):
        batch_size = x.size(0)
        if mem1 is None:
            mem1 = torch.zeros(batch_size, self.fc1.out_features, device=x.device)
        if mem2 is None:
            mem2 = torch.zeros(batch_size, self.fc2.out_features, device=x.device)

        spk1_rec = []
        mem1_rec = []
        spk2_rec = []
        mem2_rec = []
        hidden_spike_count = 0
        output_spike_count = 0

        for step in range(num_steps):
            cur1 = self.fc1(x[:, step].unsqueeze(1))  # Process one time step at a time
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk1_rec.append(spk1)
            mem1_rec.append(mem1)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

            hidden_spike_count += spk1.sum().item()
            output_spike_count += spk2.sum().item()

        return torch.stack(spk1_rec, dim=0), torch.stack(mem1_rec, dim=0), torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0), hidden_spike_count, output_spike_count

def main():
    dataset = Wave_Order_Dataset(num_samples, sample_rate, duration, freq_min, freq_max, amp_min, amp_max, std_dev, offset)  # Initialize the dataset
    train_dataset, validation_dataset, test_dataset = split_dataset(dataset, train_ratio, validation_ratio, test_ratio)  # Split the dataset
    train_loader, validation_loader, test_loader = get_dataloaders(train_dataset, validation_dataset, test_dataset, batch_size)  # Get dataloaders

    model = Wave_Order_Recognition_SNN(input_size, hidden_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate, betas=optimizer_betas)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    validation_accuracies = []
    test_accuracies = []
    loss_hist = []
    test_loss_hist = []
    beta1_values = []
    beta2_values = []
    weights_hidden_layer = []
    weights_output_layer = []
    threshold_hidden_layer = []
    threshold_output_layer = []
    hidden_spike_count = []
    output_spike_count = []
    output_spike_counts_neuron0 = []
    output_spike_counts_neuron1 = []
    equal_prediction_values = []
    pbar = tqdm.trange(num_epochs, desc='Epochs')

    for epoch in pbar:
        loss, avg_hidden_spike_count, avg_output_spike_count, avg_output_spike_counts_neuron0, avg_output_spike_counts_neuron1 = train_model(model, train_loader, criterion, optimizer, epoch, num_epochs, batch_size, hidden_size, output_size, weights_hidden_min_clamped, weights_hidden_max_clamped, weights_output_min_clamped, weights_output_max_clamped, penalty_weight, L1_lambda, device)

        validation_accuracy, validation_loss = validate_model(model, validation_loader, criterion, device)
        test_metrics = test_model(model, test_loader, criterion, device)
        test_accuracy = test_metrics['Accuracy']
        equal_prediction = test_metrics['Equal Prediction']
        total_predictions = test_metrics['Total Predictions']
        validation_accuracies.append(validation_accuracy)
        test_accuracies.append(test_accuracy)
        equal_prediction_values.append(equal_prediction)
        beta1_values.append(model.lif1.beta.detach().cpu().numpy().flatten())
        beta2_values.append(model.lif2.beta.detach().cpu().numpy().flatten())
        threshold_hidden_layer.append(model.lif1.threshold.detach().cpu().numpy().flatten())
        threshold_output_layer.append(model.lif2.threshold.detach().cpu().numpy().flatten())
        weights_hidden_layer.append(model.fc1.weight.detach().cpu().numpy().copy())
        weights_output_layer.append(model.fc2.weight.detach().cpu().numpy().copy())
        hidden_spike_count.append(avg_hidden_spike_count)
        output_spike_count.append(avg_output_spike_count)
        output_spike_counts_neuron0.append(avg_output_spike_counts_neuron0)
        output_spike_counts_neuron1.append(avg_output_spike_counts_neuron1)
        loss_hist.append(loss)
        test_loss_hist.append(validation_loss)
        validate_model(model, validation_loader, criterion, device)
        scheduler.step()

        equal_pred_ratio = equal_prediction / total_predictions * 100
        equal_pred_str = f'{equal_prediction}/{total_predictions} ({equal_pred_ratio:.2f}%)'

        if epoch == 0 or epoch == num_epochs - 1:
            print(f'Epoch [{epoch + 1}/{num_epochs}]  ||  Test Accuracy: {test_accuracy:.2f}%  ||  Validation Accuracy: {validation_accuracy:.2f}%  ||  Loss: {loss:.4f}  ||  Validation Loss: {validation_loss:.4f}  ||  Equal Prediction: {equal_pred_str}')
        elif epoch % 5 == 0:
            print(f'Epoch [{epoch}/{num_epochs}]  ||  Test Accuracy: {test_accuracy:.2f}%  ||  Validation Accuracy: {validation_accuracy:.2f}%  ||  Loss: {loss:.4f}  ||  Validation Loss: {validation_loss:.4f}  ||  Equal Prediction: {equal_pred_str}')
        pbar.set_postfix({'Test Accuracy': test_accuracy, 'Validation Accuracy': validation_accuracy, 'Loss': loss})


    torch.save(model.state_dict(), os.path.join(unique_folder_name, 'model.pth'))

    # Plot results
    #plot_wave(train_loader, save_path=os.path.join(unique_folder_name, 'wave_samples.png'))
    plot_accuracies(num_epochs, test_accuracies, validation_accuracies, os.path.join(unique_folder_name, 'accuracy_plot'))
    plot_loss_curve(loss_hist, test_loss_hist, num_epochs, os.path.join(unique_folder_name, 'loss_curve'))
    plot_equal_prediction_values(equal_prediction_values, num_epochs, os.path.join(unique_folder_name, 'equal_prediction_values'))
    #plot_beta_values(beta1_values, num_epochs, os.path.join(unique_folder_name, 'beta_hidden_layer'), layer_name='Hidden')
    #plot_beta_values(beta2_values, num_epochs, os.path.join(unique_folder_name, 'beta_output_layer'), layer_name='Output')
    plot_tau_values(beta1_values, num_epochs, deltaT, os.path.join(unique_folder_name, 'tau_hidden_layer'), layer_name='Hidden')
    plot_tau_values(beta2_values, num_epochs, deltaT, os.path.join(unique_folder_name, 'tau_output_layer'), layer_name='Output')
    plot_layer_weights(weights_hidden_layer, num_epochs, os.path.join(unique_folder_name, 'weights_hidden_layer'), layer_name='Hidden')
    plot_layer_weights(weights_output_layer, num_epochs, os.path.join(unique_folder_name, 'weights_output_layer'), layer_name='Output')
    plot_spike_counts(hidden_spike_count, output_spike_count, output_spike_counts_neuron0, output_spike_counts_neuron1, num_epochs, os.path.join(unique_folder_name, 'spike_counts'))
    plot_snn_spikes(model, test_loader, device, os.path.join(unique_folder_name, 'hidden_layer_spikes'), layer_name='Hidden', layer_size=hidden_size, num_steps=num_steps)
    plot_snn_spikes(model, test_loader, device, os.path.join(unique_folder_name, 'output_layer_spikes'), layer_name='Output', layer_size=output_size, num_steps=num_steps)
    plot_membrane_potentials(model, test_loader, device, 'Hidden', hidden_size, num_steps, os.path.join(unique_folder_name, 'hidden_membrane_potentials'))
    plot_membrane_potentials(model, test_loader, device, 'Output', output_size, num_steps, os.path.join(unique_folder_name, 'output_membrane_potentials'))
    plot_threshold_potentials(threshold_hidden_layer, num_epochs, os.path.join(unique_folder_name, 'threshold_hidden_layer'), 'Hidden')
    plot_threshold_potentials(threshold_output_layer, num_epochs, os.path.join(unique_folder_name, 'threshold_output_layer'), 'Output')
    plot_evaluation(test_metrics,os.path.join(unique_folder_name, 'evaluation_plots'))

if __name__ == '__main__':
    main()