import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class TemporalXORNetwork(nn.Module):
    def __init__(self, input_size=3, hidden_size=100, output_size=2, beta=0.9):
        super().__init__()
        # Input to hidden layer (no bias)
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        # Leaky (LIF) neuron with lower threshold
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=0.5, reset_mechanism="zero")
        # Hidden to output (analog)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)
        self.sig = nn.Sigmoid()
        
        # Initialize weights with larger values
        nn.init.xavier_uniform_(self.fc1.weight, gain=2.0)
        nn.init.xavier_uniform_(self.fc2.weight, gain=2.0)

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
        
        return spk_rec, avg_out 