"""
Standalone Visualization Demo for SNN Speech Recognition System

This script demonstrates the comprehensive visualization capabilities
without requiring the full Google Speech Commands dataset.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from snn_visualization import SNNVisualizer
from snn_speech_model import SNNSpeechClassifier
import warnings
warnings.filterwarnings('ignore')

def create_synthetic_data():
    """Create synthetic data for visualization demonstration"""
    
    # Create synthetic audio
    sample_rate = 16000
    duration = 1.0  # 1 second
    t = torch.linspace(0, duration, int(sample_rate * duration))
    
    # Create a complex audio signal with multiple frequencies
    audio = (torch.sin(2 * torch.pi * 440 * t) +  # A4 note
             0.5 * torch.sin(2 * torch.pi * 880 * t) +  # A5 note
             0.3 * torch.sin(2 * torch.pi * 660 * t) +  # E5 note
             0.2 * torch.randn(len(t)))  # Add some noise
    
    audio = audio.unsqueeze(0)  # Add channel dimension [1, samples]
    
    # Create synthetic preprocessed features (simulating MEL spectrogram)
    n_features = 16
    time_steps = 32
    features = torch.randn(n_features, time_steps)
    
    # Add some temporal structure to make it more realistic
    for i in range(n_features):
        # Create frequency-dependent patterns
        freq_factor = i / n_features
        features[i] = torch.sin(2 * torch.pi * freq_factor * torch.arange(time_steps) / time_steps)
        features[i] += 0.3 * torch.randn(time_steps)
    
    # Normalize features
    features = (features - features.mean()) / (features.std() + 1e-9)
    
    return audio, features, sample_rate

def create_synthetic_spike_recordings():
    """Create synthetic spike recordings for visualization"""
    
    # Parameters
    time_steps = 50
    n_input_neurons = 16
    n_hidden_neurons = 32
    n_output_neurons = 4
    
    # Create synthetic spike recordings
    spike_recordings = {}
    membrane_recordings = {}
    
    # Input layer spikes and membrane potentials
    input_spikes = torch.zeros(time_steps, n_input_neurons)
    input_membrane = torch.zeros(time_steps, n_input_neurons)
    
    # Add some realistic spike patterns
    for i in range(n_input_neurons):
        # Different neurons have different firing rates
        firing_rate = 0.1 + 0.2 * (i / n_input_neurons)
        spike_times = torch.rand(time_steps) < firing_rate
        input_spikes[:, i] = spike_times.float()
        
        # Create membrane potential evolution
        membrane = torch.zeros(time_steps)
        for t in range(1, time_steps):
            if spike_times[t-1]:
                membrane[t] = 0.0  # Reset after spike
            else:
                membrane[t] = 0.9 * membrane[t-1] + 0.1 * torch.randn(1)  # Decay + noise
        input_membrane[:, i] = membrane
    
    # Hidden layer
    hidden_spikes = torch.zeros(time_steps, n_hidden_neurons)
    hidden_membrane = torch.zeros(time_steps, n_hidden_neurons)
    
    for i in range(n_hidden_neurons):
        firing_rate = 0.05 + 0.15 * (i / n_hidden_neurons)
        spike_times = torch.rand(time_steps) < firing_rate
        hidden_spikes[:, i] = spike_times.float()
        
        membrane = torch.zeros(time_steps)
        for t in range(1, time_steps):
            if spike_times[t-1]:
                membrane[t] = 0.0
            else:
                membrane[t] = 0.85 * membrane[t-1] + 0.1 * torch.randn(1)
        hidden_membrane[:, i] = membrane
    
    # Output layer
    output_spikes = torch.zeros(time_steps, n_output_neurons)
    output_membrane = torch.zeros(time_steps, n_output_neurons)
    
    for i in range(n_output_neurons):
        firing_rate = 0.02 + 0.08 * (i / n_output_neurons)
        spike_times = torch.rand(time_steps) < firing_rate
        output_spikes[:, i] = spike_times.float()
        
        membrane = torch.zeros(time_steps)
        for t in range(1, time_steps):
            if spike_times[t-1]:
                membrane[t] = 0.0
            else:
                membrane[t] = 0.9 * membrane[t-1] + 0.1 * torch.randn(1)
        output_membrane[:, i] = membrane
    
    # Store in dictionaries
    spike_recordings = {
        'input_spikes': input_spikes,
        'hidden_spikes': hidden_spikes,
        'output_spikes': output_spikes
    }
    
    membrane_recordings = {
        'input_membrane': input_membrane,
        'hidden_membrane': hidden_membrane,
        'output_membrane': output_membrane
    }
    
    return spike_recordings, membrane_recordings

def create_synthetic_training_history():
    """Create synthetic training history for visualization"""
    
    epochs = 20
    train_loss = [1.0 - 0.04 * i + 0.01 * torch.randn(1).item() for i in range(epochs)]
    val_loss = [0.95 - 0.035 * i + 0.02 * torch.randn(1).item() for i in range(epochs)]
    train_acc = [0.25 + 0.035 * i + 0.01 * torch.randn(1).item() for i in range(epochs)]
    val_acc = [0.23 + 0.032 * i + 0.015 * torch.randn(1).item() for i in range(epochs)]
    
    # Ensure values are reasonable
    train_loss = [max(0.1, min(1.0, x)) for x in train_loss]
    val_loss = [max(0.1, min(1.0, x)) for x in val_loss]
    train_acc = [max(0.0, min(1.0, x)) for x in train_acc]
    val_acc = [max(0.0, min(1.0, x)) for x in val_acc]
    
    return {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc
    }

def demo_audio_preprocessing():
    """Demonstrate audio preprocessing visualization"""
    print("1. Audio Preprocessing Visualization")
    print("-" * 40)
    
    # Create synthetic data
    audio, features, sample_rate = create_synthetic_data()
    
    # Create visualizer
    visualizer = SNNVisualizer()
    
    # Plot audio preprocessing
    visualizer.plot_audio_preprocessing(
        audio=audio,
        preprocessed_features=features,
        sample_rate=sample_rate,
        preprocessing_type="mel",
        title="Audio Preprocessing Pipeline Demo"
    )
    
    print("✓ Audio preprocessing visualization completed")

def demo_lif_neurons():
    """Demonstrate LIF neuron firing visualization"""
    print("\n2. LIF Neuron Firing Patterns")
    print("-" * 40)
    
    # Create synthetic spike recordings
    spike_recordings, membrane_recordings = create_synthetic_spike_recordings()
    
    # Create visualizer
    visualizer = SNNVisualizer()
    
    # Plot LIF neuron firing
    visualizer.plot_lif_neuron_firing(
        spike_recordings=spike_recordings,
        membrane_recordings=membrane_recordings,
        layer_names=['input', 'hidden', 'output'],
        max_neurons_per_layer=20,
        time_range=(0, 50)
    )
    
    print("✓ LIF neuron firing visualization completed")

def demo_layer_activity():
    """Demonstrate layer activity summary"""
    print("\n3. Layer Activity Summary")
    print("-" * 40)
    
    # Create synthetic data
    spike_recordings, membrane_recordings = create_synthetic_spike_recordings()
    
    # Create visualizer
    visualizer = SNNVisualizer()
    
    # Plot layer activity summary
    visualizer.plot_layer_activity_summary(
        spike_recordings=spike_recordings,
        membrane_recordings=membrane_recordings
    )
    
    print("✓ Layer activity summary completed")

def demo_network_dynamics():
    """Demonstrate network dynamics visualization"""
    print("\n4. Network Dynamics")
    print("-" * 40)
    
    # Create synthetic data
    spike_recordings, membrane_recordings = create_synthetic_spike_recordings()
    training_history = create_synthetic_training_history()
    
    # Create visualizer
    visualizer = SNNVisualizer()
    
    # Plot network dynamics
    visualizer.plot_network_dynamics(
        spike_recordings=spike_recordings,
        membrane_recordings=membrane_recordings,
        training_history=training_history
    )
    
    print("✓ Network dynamics visualization completed")

def demo_comprehensive_analysis():
    """Demonstrate comprehensive analysis"""
    print("\n5. Comprehensive Analysis")
    print("-" * 40)
    
    # Create synthetic data
    audio, features, sample_rate = create_synthetic_data()
    spike_recordings, membrane_recordings = create_synthetic_spike_recordings()
    training_history = create_synthetic_training_history()
    
    # Create visualizer
    visualizer = SNNVisualizer()
    
    # Plot comprehensive analysis
    visualizer.plot_comprehensive_analysis(
        audio=audio,
        preprocessed_features=features,
        spike_recordings=spike_recordings,
        membrane_recordings=membrane_recordings,
        sample_rate=sample_rate,
        preprocessing_type="mel",
        training_history=training_history,
        title="Complete SNN Speech Processing Analysis Demo"
    )
    
    print("✓ Comprehensive analysis completed")

def demo_model_visualization():
    """Demonstrate visualization with actual SNN model"""
    print("\n6. Model-based Visualization")
    print("-" * 40)
    
    # Create a simple SNN model
    device = 'cpu'
    model = SNNSpeechClassifier(
        input_size=16,
        hidden_size=32,
        output_size=4,
        output_type="mem_potential",
        device=device
    )
    
    # Create synthetic input data
    batch_size = 1
    time_steps = 32
    input_data = torch.randn(time_steps, batch_size, 16)
    
    # Forward pass to get spike recordings
    model.eval()
    with torch.no_grad():
        outputs, spike_recordings = model(input_data)
    
    # Create visualizer
    visualizer = SNNVisualizer()
    
    # Plot LIF neuron firing with real model data
    visualizer.plot_lif_neuron_firing(
        spike_recordings=spike_recordings,
        membrane_recordings=spike_recordings,
        layer_names=['input', 'hidden', 'output'],
        max_neurons_per_layer=15,
        time_range=(0, 32)
    )
    
    print("✓ Model-based visualization completed")
    
    # Print model statistics
    stats = model.get_spike_statistics()
    if stats:
        print("\nModel Spike Statistics:")
        for layer, layer_stats in stats.items():
            print(f"  {layer.title()} Layer:")
            print(f"    Total spikes: {layer_stats['total_spikes']:.0f}")
            print(f"    Avg spikes per neuron: {layer_stats['avg_spikes_per_neuron']:.3f}")
            print(f"    Spike rate: {layer_stats['spike_rate']:.3f}")

def main():
    """Run all visualization demos"""
    print("SNN Speech Recognition - Visualization Demo")
    print("=" * 60)
    print("This demo shows comprehensive visualizations of:")
    print("- Audio preprocessing pipeline")
    print("- LIF neuron firing patterns (neuron by neuron)")
    print("- Layer-by-layer activity")
    print("- Network dynamics and statistics")
    print("- Complete system analysis")
    print("=" * 60)
    
    try:
        # Run all demos
        demo_audio_preprocessing()
        demo_lif_neurons()
        demo_layer_activity()
        demo_network_dynamics()
        demo_comprehensive_analysis()
        demo_model_visualization()
        
        print("\n" + "=" * 60)
        print("All visualization demos completed successfully!")
        print("Check the generated plots to inspect the SNN system.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during visualization: {e}")
        print("Make sure all required packages are installed:")
        print("pip install matplotlib seaborn librosa scipy")

if __name__ == "__main__":
    main() 