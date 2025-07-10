"""
Spiking Neural Network for Speech Recognition using snnTorch
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import numpy as np


class SNNSpeechClassifier(nn.Module):
    """
    Spiking Neural Network for Speech Recognition
    
    Architecture:
    - Input layer: 16 LIF neurons with logarithmically spaced beta values
    - Hidden layer: 32 LIF neurons
    - Output layer: 4 LIF neurons (for 4 speech commands)
    
    Output types:
    - "spk_count": Use spike counts for classification
    - "mem_potential": Use final membrane potential for classification
    """
    
    def __init__(self, input_size=16, hidden_size=32, output_size=4, 
                 beta_input_min=0.5, beta_input_max=0.95, beta_hidden=0.85, beta_output=0.9,
                 threshold=1.0, spike_grad_slope=25, output_type="mem_potential", device='cpu'):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.output_type = output_type
        self.device = device
        
        # Create logarithmically spaced beta values for input layer (like human ear)
        # Lower beta = more sensitive to low frequencies (longer time constant)
        self.beta_input = torch.logspace(
            torch.log10(torch.tensor(beta_input_min)), 
            torch.log10(torch.tensor(beta_input_max)), 
            input_size
        ).to(device)
        
        # Linear layers
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)
        
        # Initialize weights with Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)
        
        # Create LIF neurons for each layer
        # Input layer: each neuron has its own beta value
        self.lif_input = snn.Leaky(
            beta=self.beta_input,
            threshold=threshold,
            spike_grad=surrogate.fast_sigmoid(slope=spike_grad_slope),
            reset_mechanism="zero"
        )
        
        # Hidden layer: single beta value for all neurons
        self.lif_hidden = snn.Leaky(
            beta=beta_hidden,
            threshold=threshold,
            spike_grad=surrogate.fast_sigmoid(slope=spike_grad_slope),
            reset_mechanism="zero"
        )
        
        # Output layer: single beta value for all neurons
        self.lif_output = snn.Leaky(
            beta=beta_output,
            threshold=threshold,
            spike_grad=surrogate.fast_sigmoid(slope=spike_grad_slope),
            reset_mechanism="zero"
        )
        
        # Store spike recordings for analysis
        self.spike_recordings = []
        
    def forward(self, x):
        """
        Forward pass through the SNN
        
        Args:
            x: Input tensor of shape [time_steps, batch_size, input_size]
            
        Returns:
            outputs: Final outputs for classification
            spike_recordings: Dictionary containing spike recordings for each layer
        """
        # Initialize membrane potentials
        mem_input = self.lif_input.init_leaky()
        mem_hidden = self.lif_hidden.init_leaky()
        mem_output = self.lif_output.init_leaky()
        
        # Record spikes and membrane potentials
        spk_input_rec = []
        spk_hidden_rec = []
        spk_output_rec = []
        mem_input_rec = []
        mem_hidden_rec = []
        mem_output_rec = []
        
        # Loop through time steps
        for step in range(x.size(0)):
            # Input layer
            cur_input = self.fc1(x[step])
            spk_input, mem_input = self.lif_input(cur_input, mem_input)
            
            # Hidden layer
            cur_hidden = self.fc2(spk_input)
            spk_hidden, mem_hidden = self.lif_hidden(cur_hidden, mem_hidden)
            
            # Output layer
            spk_output, mem_output = self.lif_output(spk_hidden, mem_output)
            
            # Record spikes and membrane potentials
            spk_input_rec.append(spk_input)
            spk_hidden_rec.append(spk_hidden)
            spk_output_rec.append(spk_output)
            mem_input_rec.append(mem_input)
            mem_hidden_rec.append(mem_hidden)
            mem_output_rec.append(mem_output)
        
        # Stack recordings
        spk_input_rec = torch.stack(spk_input_rec, dim=0)
        spk_hidden_rec = torch.stack(spk_hidden_rec, dim=0)
        spk_output_rec = torch.stack(spk_output_rec, dim=0)
        mem_input_rec = torch.stack(mem_input_rec, dim=0)
        mem_hidden_rec = torch.stack(mem_hidden_rec, dim=0)
        mem_output_rec = torch.stack(mem_output_rec, dim=0)
        
        # Store spike recordings
        self.spike_recordings = {
            'input_spikes': spk_input_rec,
            'hidden_spikes': spk_hidden_rec,
            'output_spikes': spk_output_rec,
            'input_membrane': mem_input_rec,
            'hidden_membrane': mem_hidden_rec,
            'output_membrane': mem_output_rec
        }
        
        # Prepare outputs based on output type
        if self.output_type == "spk_count":
            # Use spike counts over time
            outputs = spk_output_rec.sum(dim=0)  # [batch_size, output_size]
        elif self.output_type == "mem_potential":
            # Use final membrane potential
            outputs = mem_output_rec[-1]  # [batch_size, output_size]
        else:
            raise ValueError(f"Unknown output_type: {self.output_type}")
        
        return outputs, self.spike_recordings
    
    def get_spike_statistics(self):
        """Get statistics about spike activity"""
        if not self.spike_recordings:
            return None
        
        stats = {}
        for layer_name, spikes in [
            ('input', self.spike_recordings['input_spikes']),
            ('hidden', self.spike_recordings['hidden_spikes']),
            ('output', self.spike_recordings['output_spikes'])
        ]:
            stats[layer_name] = {
                'total_spikes': spikes.sum().item(),
                'avg_spikes_per_neuron': spikes.mean().item(),
                'spike_rate': spikes.mean().item(),  # spikes per neuron per timestep
                'max_spikes': spikes.max().item(),
                'min_spikes': spikes.min().item()
            }
        
        return stats


class SNNTrainer:
    """
    Trainer for SNN Speech Recognition model with learning rate scheduler and early stopping
    """
    
    def __init__(self, model, loss_type="MSE", learning_rate=0.001, 
                 scheduler_step_size=10, scheduler_gamma=0.7, 
                 early_stopping_patience=15, device='cpu'):
        self.model = model
        self.device = device
        self.early_stopping_patience = early_stopping_patience
        
        # Set up loss function
        if loss_type == "MSE":
            self.criterion = nn.MSELoss()
        elif loss_type == "CE":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        
        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=scheduler_step_size, 
            gamma=scheduler_gamma
        )
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs, _ = self.model(data)
            
            # Calculate loss
            if self.criterion.__class__.__name__ == 'CrossEntropyLoss':
                # For CrossEntropy, targets should be class indices
                target_indices = torch.argmax(targets, dim=1)
                loss = self.criterion(outputs, target_indices)
            else:
                # For MSE, use one-hot encoded targets
                loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            if self.model.output_type == "spk_count":
                predictions = outputs.argmax(dim=1)
            else:
                predictions = outputs.argmax(dim=1)
            
            target_indices = torch.argmax(targets, dim=1)
            correct += (predictions == target_indices).sum().item()
            total += targets.size(0)
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs, _ = self.model(data)
                
                # Calculate loss
                if self.criterion.__class__.__name__ == 'CrossEntropyLoss':
                    target_indices = torch.argmax(targets, dim=1)
                    loss = self.criterion(outputs, target_indices)
                else:
                    loss = self.criterion(outputs, targets)
                
                # Calculate accuracy
                if self.model.output_type == "spk_count":
                    predictions = outputs.argmax(dim=1)
                else:
                    predictions = outputs.argmax(dim=1)
                
                target_indices = torch.argmax(targets, dim=1)
                correct += (predictions == target_indices).sum().item()
                total += targets.size(0)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, num_epochs=100):
        """Train the model with early stopping and learning rate scheduling"""
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Output type: {self.model.output_type}")
        print(f"Loss function: {self.criterion.__class__.__name__}")
        print(f"Early stopping patience: {self.early_stopping_patience}")
        print("-" * 60)
        
        for epoch in range(num_epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_accuracy'].append(train_acc)
            history['val_accuracy'].append(val_acc)
            history['learning_rate'].append(current_lr)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                self.patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}]')
                print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                print(f'LR: {current_lr:.6f}, Patience: {self.patience_counter}/{self.early_stopping_patience}')
                print("-" * 60)
            
            # Check early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Loaded best model with validation loss: {self.best_val_loss:.4f}")
        
        return history
    
    def test(self, test_loader):
        """Test the model and return accuracy"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in test_loader:
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                outputs, _ = self.model(data)
                
                if self.model.output_type == "spk_count":
                    predictions = outputs.argmax(dim=1)
                else:
                    predictions = outputs.argmax(dim=1)
                
                target_indices = torch.argmax(targets, dim=1)
                correct += (predictions == target_indices).sum().item()
                total += targets.size(0)
        
        accuracy = 100. * correct / total
        return accuracy 