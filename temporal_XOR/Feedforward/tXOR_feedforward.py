import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class TemporalXORNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, beta, threshold, spike_grad_slope, weight_gain):
        super().__init__()
        
        # Initialize layers
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)
        
        # Initialize LIF neurons
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad_slope)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight, gain=weight_gain)
        nn.init.xavier_uniform_(self.fc2.weight, gain=weight_gain)
        
        # Record spikes for visualization
        self.spike_rec = []
        
    def forward(self, x):
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        
        # Record spikes
        spk1_rec = []
        mem1_rec = []
        
        # Loop through time
        for step in range(x.size(0)):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            
            # Record spikes
            spk1_rec.append(spk1)
            mem1_rec.append(mem1)
            
            # Detach membrane potential
            mem1 = mem1.detach()
        
        # Stack spikes
        spk1_rec = torch.stack(spk1_rec, dim=0)
        mem1_rec = torch.stack(mem1_rec, dim=0)
        
        # Store spikes for visualization
        self.spike_rec = spk1_rec
        
        # Average spikes over last 10 timesteps
        spk1_avg = spk1_rec[-10:].mean(0)
        
        # Final classification
        out = self.fc2(spk1_avg)
        
        return out 