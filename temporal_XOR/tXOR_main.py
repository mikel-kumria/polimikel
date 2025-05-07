import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import snntorch as snn
from snntorch import surrogate
from tXOR_imports import validate, monitor_spike_activity
from tXOR_plots import plot_spike_pattern, plot_training_metrics, plot_input_pattern

# Dataset: temporal XOR encoding
class TemporalXORDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.seq_len = 30
        self.data = []
        self.labels = []
        v_th = 3.0
        for _ in range(num_samples):
            a = torch.randint(0, 2, (1,)).item()
            b = torch.randint(0, 2, (1,)).item()
            xor = a ^ b
            # Initialize sequence: [30 timesteps x 3 input channels]
            seq = torch.zeros((self.seq_len, 3), dtype=torch.float)
            
            # Input neuron 1 (A): first 10 timesteps active, rest 0
            seq[0:10, 0] = v_th if a == 1 else v_th / 2
            
            # Input neuron 2 (B): timesteps 10-20 active, rest 0
            seq[10:20, 1] = v_th if b == 1 else v_th / 2
            
            # Input neuron 3 (R): last 10 timesteps active with v_th
            seq[20:30, 2] = v_th
            
            self.data.append(seq)
            # One-hot label for XOR result
            label = torch.zeros(2, dtype=torch.float)
            label[xor] = 1.0
            self.labels.append(label)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Spiking neural network model
class TemporalXORNetwork(nn.Module):
    def __init__(self, input_size=3, hidden_size=100, output_size=2, beta=0.8):
        super().__init__()
        # Input to hidden layer (no bias)
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        # Leaky (LIF) neuron with beta=0.8, threshold=1.0
        spike_grad = surrogate.fast_sigmoid(slope=25)  # Steeper slope for better gradients
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=1.0, reset_mechanism="zero")
        # Hidden to output (analog)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)
        self.sig = nn.Sigmoid()
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x_seq):
        # Initialize membrane potential and outputs
        batch_size = x_seq.size(1)
        mem1 = torch.zeros(batch_size, self.fc1.out_features, device=x_seq.device)
        spk_rec = []
        outs = []
        
        # Temporal processing
        for t in range(x_seq.size(0)):
            cur = self.fc1(x_seq[t])
            spk1, mem1 = self.lif1(cur, mem1)
            out = self.sig(self.fc2(spk1))
            spk_rec.append(spk1)
            outs.append(out)
        
        # Stack time dimension first
        spk_rec = torch.stack(spk_rec)
        outs = torch.stack(outs)
        
        # Average over last 10 timesteps
        avg_out = outs[-10:].mean(dim=0)
        
        return spk_rec, avg_out  # Return spike recording instead of full output sequence

# Training loop helper
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for data, labels in loader:
        data = data.to(device)
        labels = labels.to(device)
        data = data.permute(1, 0, 2)  # [seq_len, batch_size, input_size]
        
        optimizer.zero_grad()
        _, avg_out = model(data)
        loss = criterion(avg_out, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(loader)

def main():
    # Hyperparameters
    num_samples = 1000
    batch_size = 16
    learning_rate = 3e-4  # Reduced learning rate
    num_epochs = 200  # Increased epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset & Loader
    dataset = TemporalXORDataset(num_samples)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model, optimizer, and loss
    model = TemporalXORNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    # Training loop with validation
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_spikes = []

    for epoch in range(1, num_epochs + 1):
        # Training
        train_loss = train(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Validation
        val_loss, val_acc, val_spike = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_spikes.append(val_spike)
        
        print(f'Epoch {epoch}/{num_epochs} - '
              f'Train Loss: {train_loss:.4f} - '
              f'Val Loss: {val_loss:.4f} - '
              f'Val Acc: {val_acc:.2f}% - '
              f'Avg Spikes: {val_spike:.2f}')
        
        # Visualize spike patterns for a sample every X epochs
        if epoch % num_epochs == 0:
            sample_data, _ = next(iter(val_loader))
            spike_data = monitor_spike_activity(model, sample_data, device)
            plot_spike_pattern(spike_data, title=f'Spike Pattern - Epoch {epoch}')
            plot_input_pattern(sample_data[0], title=f'Input Pattern - Epoch {epoch}')
    
    # Plot final training metrics
    plot_training_metrics(train_losses, val_losses, val_accuracies, val_spikes)

if __name__ == '__main__':
    main()