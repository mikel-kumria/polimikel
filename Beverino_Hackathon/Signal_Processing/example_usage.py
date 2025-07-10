"""
Example Usage of SNN Speech Recognition System

This script demonstrates how to use the complete SNN speech recognition system
with different preprocessing methods, output types, and loss functions.
"""

import torch
from snn_speech_model import SNNSpeechClassifier, SNNTrainer
from speech_preprocessing import create_speech_dataloaders, get_preprocessing_info
from snn_visualization import SNNVisualizer

# Configuration
DATA_PATH = "/path/to/speech_commands_dataset"  # Update this path
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def example_1_basic_training():
    """Basic example: MEL preprocessing with membrane potential output and MSE loss"""
    print("="*60)
    print("Example 1: Basic Training")
    print("MEL preprocessing + Membrane potential + MSE loss")
    print("="*60)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_speech_dataloaders(
        root_dir=DATA_PATH,
        preprocessing="mel",
        spike_encoding=True,
        batch_size=32,
        commands=["yes", "no", "up", "down"]  # 4 classes
    )
    
    # Create model
    model = SNNSpeechClassifier(
        input_size=16,
        hidden_size=32,
        output_size=4,
        output_type="mem_potential",
        device=DEVICE
    )
    
    # Create trainer
    trainer = SNNTrainer(
        model=model,
        loss_type="MSE",
        learning_rate=0.001,
        device=DEVICE
    )
    
    # Train for a few epochs
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20
    )
    
    # Test
    test_accuracy = trainer.test(test_loader)
    print(f"\nTest Accuracy: {test_accuracy:.2f}%")
    
    return model, history


def example_2_spike_count_classification():
    """Example 2: MFCC preprocessing with spike count output and CrossEntropy loss"""
    print("\n" + "="*60)
    print("Example 2: Spike Count Classification")
    print("MFCC preprocessing + Spike count + CrossEntropy loss")
    print("="*60)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_speech_dataloaders(
        root_dir=DATA_PATH,
        preprocessing="mfcc",
        spike_encoding=True,
        batch_size=64,
        commands=["yes", "no", "up", "down"]
    )
    
    # Create model
    model = SNNSpeechClassifier(
        input_size=16,
        hidden_size=32,
        output_size=4,
        output_type="spk_count",
        device=DEVICE
    )
    
    # Create trainer with different hyperparameters
    trainer = SNNTrainer(
        model=model,
        loss_type="CE",
        learning_rate=0.002,
        scheduler_step_size=5,
        scheduler_gamma=0.7,
        device=DEVICE
    )
    
    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=30
    )
    
    # Test
    test_accuracy = trainer.test(test_loader)
    print(f"\nTest Accuracy: {test_accuracy:.2f}%")
    
    return model, history


def example_3_custom_beta_values():
    """Example 3: Custom beta values for frequency sensitivity"""
    print("\n" + "="*60)
    print("Example 3: Custom Beta Values")
    print("Gammatone preprocessing with custom frequency-sensitive beta values")
    print("="*60)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_speech_dataloaders(
        root_dir=DATA_PATH,
        preprocessing="gammatone",
        spike_encoding=True,
        batch_size=32,
        commands=["yes", "no", "up", "down"]
    )
    
    # Create model with custom beta values
    # Lower beta = more sensitive to low frequencies (longer time constant)
    model = SNNSpeechClassifier(
        input_size=16,
        hidden_size=32,
        output_size=4,
        beta_input_min=0.5,  # Very sensitive to low frequencies
        beta_input_max=0.95,  # Less sensitive to high frequencies
        beta_hidden=0.85,
        beta_output=0.9,
        output_type="mem_potential",
        device=DEVICE
    )
    
    print("\nInput layer beta values (logarithmically spaced):")
    print(f"  Min (low freq): {model.beta_input[0]:.4f}")
    print(f"  Max (high freq): {model.beta_input[-1]:.4f}")
    print(f"  All values: {model.beta_input}")
    
    # Train
    trainer = SNNTrainer(
        model=model,
        loss_type="MSE",
        learning_rate=0.001,
        device=DEVICE
    )
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=25
    )
    
    # Test
    test_accuracy = trainer.test(test_loader)
    print(f"\nTest Accuracy: {test_accuracy:.2f}%")
    
    return model, history


def example_4_inference():
    """Example 4: How to use a trained model for inference"""
    print("\n" + "="*60)
    print("Example 4: Inference with Trained Model")
    print("="*60)
    
    # Create a simple model and train briefly
    train_loader, val_loader, test_loader = create_speech_dataloaders(
        root_dir=DATA_PATH,
        preprocessing="mel",
        spike_encoding=True,
        batch_size=32,
        commands=["yes", "no", "up", "down"]
    )
    
    model = SNNSpeechClassifier(
        input_size=16,
        hidden_size=32,
        output_size=4,
        output_type="mem_potential",
        device=DEVICE
    )
    
    # Quick training
    trainer = SNNTrainer(model=model, device=DEVICE)
    trainer.train(train_loader, val_loader, num_epochs=5)
    
    # Inference mode
    model.eval()
    
    # Get a batch of test samples
    with torch.no_grad():
        test_batch, test_labels = next(iter(test_loader))
        test_batch = test_batch.to(DEVICE)
        
        # Forward pass
        outputs, spike_recordings = model(test_batch)
        
        # Get predictions
        if model.output_type == "mem_potential":
            # Use final membrane potential
            predictions = outputs.argmax(dim=1)
        else:
            # Use spike counts
            predictions = outputs.argmax(dim=1)
        
        # Get true labels
        true_labels = test_labels.argmax(dim=1)
        
        # Print some results
        print("\nSample predictions:")
        commands = ["yes", "no", "up", "down"]
        for i in range(min(5, len(predictions))):
            pred_cmd = commands[predictions[i]]
            true_cmd = commands[true_labels[i]]
            correct = "✓" if predictions[i] == true_labels[i] else "✗"
            print(f"  Sample {i}: Predicted={pred_cmd}, True={true_cmd} {correct}")
        
        # Calculate batch accuracy
        accuracy = (predictions == true_labels).float().mean() * 100
        print(f"\nBatch accuracy: {accuracy:.2f}%")
    
    return model


def example_5_compare_preprocessing():
    """Example 5: Compare different preprocessing methods"""
    print("\n" + "="*60)
    print("Example 5: Comparing Preprocessing Methods")
    print("="*60)
    
    results = {}
    
    for preprocessing in ["mel", "mfcc", "gammatone", "cochleagram"]:
        print(f"\nTesting {preprocessing.upper()} preprocessing...")
        
        try:
            # Create data loaders
            train_loader, val_loader, test_loader = create_speech_dataloaders(
                root_dir=DATA_PATH,
                preprocessing=preprocessing,
                spike_encoding=True,
                batch_size=32,
                commands=["yes", "no", "up", "down"]
            )
            
            # Create and train model
            model = SNNSpeechClassifier(
                input_size=16,
                hidden_size=32,
                output_size=4,
                output_type="mem_potential",
                device=DEVICE
            )
            
            trainer = SNNTrainer(
                model=model,
                loss_type="MSE",
                learning_rate=0.001,
                early_stopping_patience=5,
                device=DEVICE
            )
            
            # Train for a few epochs
            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=10
            )
            
            # Test
            test_accuracy = trainer.test(test_loader)
            
            results[preprocessing] = {
                'test_accuracy': test_accuracy,
                'best_val_accuracy': max(history['val_accuracy']),
                'epochs_trained': len(history['train_loss'])
            }
            
        except Exception as e:
            print(f"  Error with {preprocessing}: {e}")
            results[preprocessing] = {'error': str(e)}
    
    # Print comparison
    print("\n" + "="*60)
    print("RESULTS COMPARISON:")
    print("="*60)
    for method, result in results.items():
        if 'error' in result:
            print(f"{method.upper()}: Error - {result['error']}")
        else:
            print(f"{method.upper()}:")
            print(f"  Test Accuracy: {result['test_accuracy']:.2f}%")
            print(f"  Best Val Accuracy: {result['best_val_accuracy']:.2f}%")
            print(f"  Epochs Trained: {result['epochs_trained']}")


def example_6_comprehensive_visualization():
    """Example 6: Comprehensive visualization of the entire SNN pipeline"""
    print("\n" + "="*60)
    print("Example 6: Comprehensive Visualization")
    print("Visualizing audio preprocessing, LIF neurons, and network dynamics")
    print("="*60)
    
    # Create visualizer
    visualizer = SNNVisualizer()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_speech_dataloaders(
        root_dir=DATA_PATH,
        preprocessing="mel",
        spike_encoding=True,
        batch_size=1,  # Use batch size 1 for detailed visualization
        commands=["yes", "no", "up", "down"]
    )
    
    # Create model
    model = SNNSpeechClassifier(
        input_size=16,
        hidden_size=32,
        output_size=4,
        output_type="mem_potential",
        device=DEVICE
    )
    
    # Quick training
    trainer = SNNTrainer(
        model=model,
        loss_type="MSE",
        learning_rate=0.001,
        device=DEVICE
    )
    
    print("Training model for visualization...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=5
    )
    
    # Get a sample for visualization
    model.eval()
    with torch.no_grad():
        sample_batch, sample_labels = next(iter(test_loader))
        sample_batch = sample_batch.to(DEVICE)
        
        # Forward pass to get spike recordings
        outputs, spike_recordings = model(sample_batch)
        
        # Get original audio and preprocessed features for the first sample
        # Note: This requires access to the original audio data
        # For demonstration, we'll create synthetic data
        
        print("\nCreating visualizations...")
        
        # 1. Audio preprocessing visualization
        print("1. Plotting audio preprocessing...")
        # Create synthetic audio data for demonstration
        sample_rate = 16000
        duration = 1.0  # 1 second
        t = torch.linspace(0, duration, int(sample_rate * duration))
        synthetic_audio = torch.sin(2 * torch.pi * 440 * t) + 0.5 * torch.sin(2 * torch.pi * 880 * t)
        synthetic_audio = synthetic_audio.unsqueeze(0)  # Add channel dimension
        
        # Get preprocessed features for this sample
        preprocessed_features = sample_batch[0]  # [n_features, time_steps]
        
        visualizer.plot_audio_preprocessing(
            audio=synthetic_audio,
            preprocessed_features=preprocessed_features,
            sample_rate=sample_rate,
            preprocessing_type="mel",
            title="Audio Preprocessing Pipeline"
        )
        
        # 2. LIF neuron firing patterns
        print("2. Plotting LIF neuron firing patterns...")
        visualizer.plot_lif_neuron_firing(
            spike_recordings=spike_recordings,
            membrane_recordings=spike_recordings,  # Using same dict for both
            layer_names=['input', 'hidden', 'output'],
            max_neurons_per_layer=20,
            time_range=(0, 50)  # First 50 time steps
        )
        
        # 3. Layer activity summary
        print("3. Plotting layer activity summary...")
        visualizer.plot_layer_activity_summary(
            spike_recordings=spike_recordings,
            membrane_recordings=spike_recordings
        )
        
        # 4. Network dynamics
        print("4. Plotting network dynamics...")
        visualizer.plot_network_dynamics(
            spike_recordings=spike_recordings,
            membrane_recordings=spike_recordings,
            training_history=history
        )
        
        # 5. Comprehensive analysis
        print("5. Creating comprehensive analysis...")
        visualizer.plot_comprehensive_analysis(
            audio=synthetic_audio,
            preprocessed_features=preprocessed_features,
            spike_recordings=spike_recordings,
            membrane_recordings=spike_recordings,
            sample_rate=sample_rate,
            preprocessing_type="mel",
            training_history=history,
            title="Complete SNN Speech Processing Analysis"
        )
        
        print("\nAll visualizations completed!")
        print("Check the generated plots to inspect:")
        print("- Audio preprocessing pipeline")
        print("- LIF neuron firing patterns (neuron by neuron)")
        print("- Layer-by-layer activity")
        print("- Network dynamics and statistics")
        print("- Complete system analysis")
    
    return model, history, visualizer


if __name__ == "__main__":
    print("SNN Speech Recognition Examples")
    print("="*60)
    
    # Run examples
    # Note: Update DATA_PATH before running!
    
    # Example 1: Basic training
    model1, history1 = example_1_basic_training()
    
    # Example 2: Spike count classification
    model2, history2 = example_2_spike_count_classification()
    
    # Example 3: Custom beta values
    model3, history3 = example_3_custom_beta_values()
    
    # Example 4: Inference
    model4 = example_4_inference()
    
    # Example 5: Compare preprocessing methods
    example_5_compare_preprocessing()
    
    # Example 6: Comprehensive visualization
    model6, history6, visualizer = example_6_comprehensive_visualization()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60) 