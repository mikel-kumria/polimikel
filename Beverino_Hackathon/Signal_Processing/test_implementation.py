"""
Test script for SNN Speech Recognition System

This script tests the implementation with synthetic data to ensure everything works correctly.
"""

import torch
import numpy as np
from snn_speech_model import SNNSpeechClassifier, SNNTrainer
from speech_preprocessing import SpeechCommandsDataset, create_speech_dataloaders
import os

def create_synthetic_dataset(num_samples=100, seq_len=50, num_features=16, num_classes=4):
    """Create synthetic dataset for testing"""
    
    class SyntheticDataset:
        def __init__(self, num_samples, seq_len, num_features, num_classes):
            self.num_samples = num_samples
            self.seq_len = seq_len
            self.num_features = num_features
            self.num_classes = num_classes
            
            # Generate synthetic data
            self.data = []
            self.labels = []
            
            for _ in range(num_samples):
                # Generate random features
                features = torch.randn(seq_len, num_features)
                
                # Generate random label
                label = torch.zeros(num_classes)
                label[torch.randint(0, num_classes, (1,))] = 1.0
                
                self.data.append(features)
                self.labels.append(label)
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]
    
    return SyntheticDataset(num_samples, seq_len, num_features, num_classes)

def test_model_creation():
    """Test model creation and forward pass"""
    print("Testing model creation...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test different configurations
    configs = [
        {"output_type": "mem_potential", "loss_type": "MSE"},
        {"output_type": "spk_count", "loss_type": "CE"},
    ]
    
    for config in configs:
        print(f"\nTesting {config['output_type']} + {config['loss_type']}...")
        
        # Create model
        model = SNNSpeechClassifier(
            input_size=16,
            hidden_size=32,
            output_size=4,
            output_type=config['output_type'],
            device=device
        )
        
        # Test forward pass
        batch_size = 8
        seq_len = 50
        x = torch.randn(seq_len, batch_size, 16).to(device)
        
        outputs, spike_recordings = model(x)
        
        # Check output shapes
        expected_shape = (batch_size, 4)
        assert outputs.shape == expected_shape, f"Expected {expected_shape}, got {outputs.shape}"
        
        # Check spike recordings
        assert 'input_spikes' in spike_recordings
        assert 'hidden_spikes' in spike_recordings
        assert 'output_spikes' in spike_recordings
        
        print(f"✓ {config['output_type']} + {config['loss_type']} works correctly")
        
        # Test spike statistics
        stats = model.get_spike_statistics()
        assert stats is not None
        assert 'input' in stats
        assert 'hidden' in stats
        assert 'output' in stats
        
        print(f"✓ Spike statistics work correctly")

def test_training():
    """Test training with synthetic data"""
    print("\nTesting training...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create synthetic dataset
    train_dataset = create_synthetic_dataset(100, 50, 16, 4)
    val_dataset = create_synthetic_dataset(20, 50, 16, 4)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Create model
    model = SNNSpeechClassifier(
        input_size=16,
        hidden_size=32,
        output_size=4,
        output_type="mem_potential",
        device=device
    )
    
    # Create trainer
    trainer = SNNTrainer(
        model=model,
        loss_type="MSE",
        learning_rate=0.001,
        early_stopping_patience=5,
        device=device
    )
    
    # Train for a few epochs
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=5
    )
    
    # Check history
    assert 'train_loss' in history
    assert 'val_loss' in history
    assert 'train_accuracy' in history
    assert 'val_accuracy' in history
    assert len(history['train_loss']) > 0
    
    print("✓ Training works correctly")

def test_beta_values():
    """Test custom beta values"""
    print("\nTesting custom beta values...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model with custom beta values
    model = SNNSpeechClassifier(
        input_size=16,
        hidden_size=32,
        output_size=4,
        beta_input_min=0.3,
        beta_input_max=0.9,
        beta_hidden=0.8,
        beta_output=0.85,
        output_type="mem_potential",
        device=device
    )
    
    # Check beta values
    assert len(model.beta_input) == 16
    assert model.beta_input[0] < model.beta_input[-1]  # Should be increasing
    assert model.beta_input[0] >= 0.3
    assert model.beta_input[-1] <= 0.9
    
    print(f"✓ Beta values: min={model.beta_input[0]:.4f}, max={model.beta_input[-1]:.4f}")

def test_preprocessing_info():
    """Test preprocessing information"""
    print("\nTesting preprocessing information...")
    
    from speech_preprocessing import get_preprocessing_info
    
    for method in ["mel", "mfcc", "gammatone", "cochleagram"]:
        info = get_preprocessing_info(method)
        assert 'description' in info
        print(f"✓ {method.upper()}: {info['description']}")

def main():
    """Run all tests"""
    print("="*60)
    print("Testing SNN Speech Recognition System")
    print("="*60)
    
    try:
        test_model_creation()
        test_training()
        test_beta_values()
        test_preprocessing_info()
        
        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        raise

if __name__ == "__main__":
    main() 