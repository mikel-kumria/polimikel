# SNN Speech Recognition System

A comprehensive Spiking Neural Network (SNN) implementation for speech recognition using snnTorch, inspired by the human auditory system.

## Features

### 🧠 SNN Architecture
- **Input Layer**: 16 LIF neurons with logarithmically spaced beta values (like human ear frequency sensitivity)
- **Hidden Layer**: 32 LIF neurons
- **Output Layer**: 4 LIF neurons (for 4 speech commands)
- **Frequency Sensitivity**: Input neurons are more sensitive to low frequencies, mimicking human auditory perception

### 🎵 Preprocessing Methods
1. **MEL Spectrogram** - Standard audio feature extraction
2. **MFCC** - Mel-frequency cepstral coefficients (compact representation)
3. **Gammatone Filterbank** - Auditory-inspired frequency analysis
4. **Cochleagram** - Cochlea-inspired representation (most biologically realistic)

### 🔧 Output Types & Loss Functions
- **Spike Count Classification**: Uses spike counts over time
- **Membrane Potential Classification**: Uses final membrane potential
- **Cross Entropy Loss**: For spike count outputs
- **Mean Square Error Loss**: For membrane potential outputs

### 🚀 Training Features
- **Learning Rate Scheduler**: StepLR with configurable step size and gamma
- **Early Stopping**: Prevents overfitting based on validation loss
- **Spike Statistics**: Monitor spike activity during training
- **Flexible Configuration**: Easy parameter tuning

### 📊 Comprehensive Visualization
- **Audio Preprocessing**: Visualize original audio, spectrograms, and processed features
- **LIF Neuron Firing**: Raster plots showing when each neuron fires over time
- **Layer-by-Layer Activity**: Membrane potentials, spike rates, and activity heatmaps
- **Network Dynamics**: Overall network statistics and performance metrics
- **Complete Analysis**: Comprehensive view of the entire SNN pipeline

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd snn-speech-recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the Google Speech Commands dataset:
```bash
# Download from: https://www.tensorflow.org/datasets/catalog/speech_commands
# Extract to a directory and update DATA_PATH in example_usage.py
```

## Quick Start

```python
import torch
from snn_speech_model import SNNSpeechClassifier, SNNTrainer
from speech_preprocessing import create_speech_dataloaders

# Configuration
DATA_PATH = "/path/to/speech_commands_dataset"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create data loaders
train_loader, val_loader, test_loader = create_speech_dataloaders(
    root_dir=DATA_PATH,
    preprocessing="mel",
    spike_encoding=True,
    batch_size=32,
    commands=["yes", "no", "up", "down"]
)

# Create model
model = SNNSpeechClassifier(
    input_size=16,
    hidden_size=32,
    output_size=4,
    output_type="mem_potential",  # or "spk_count"
    device=DEVICE
)

# Create trainer
trainer = SNNTrainer(
    model=model,
    loss_type="MSE",  # or "CE"
    learning_rate=0.001,
    device=DEVICE
)

# Train
history = trainer.train(train_loader, val_loader, num_epochs=50)

# Test
test_accuracy = trainer.test(test_loader)
print(f"Test Accuracy: {test_accuracy:.2f}%")
```

## Usage Examples

### Example 1: Basic Training
```python
# MEL preprocessing + Membrane potential + MSE loss
model = SNNSpeechClassifier(output_type="mem_potential")
trainer = SNNTrainer(loss_type="MSE")
```

### Example 2: Spike Count Classification
```python
# MFCC preprocessing + Spike count + CrossEntropy loss
model = SNNSpeechClassifier(output_type="spk_count")
trainer = SNNTrainer(loss_type="CE")
```

### Example 3: Custom Beta Values
```python
# Custom frequency sensitivity
model = SNNSpeechClassifier(
    beta_input_min=0.5,  # Very sensitive to low frequencies
    beta_input_max=0.95,  # Less sensitive to high frequencies
    output_type="mem_potential"
)
```

### Example 4: Compare Preprocessing Methods
```python
# Test all preprocessing methods
for preprocessing in ["mel", "mfcc", "gammatone", "cochleagram"]:
    train_loader, val_loader, test_loader = create_speech_dataloaders(
        preprocessing=preprocessing
    )
    # Train and evaluate...
```

### Example 5: Comprehensive Visualization
```python
from snn_visualization import SNNVisualizer

# Create visualizer
visualizer = SNNVisualizer()

# Plot audio preprocessing
visualizer.plot_audio_preprocessing(audio, preprocessed_features, preprocessing_type="mel")

# Plot LIF neuron firing patterns
visualizer.plot_lif_neuron_firing(spike_recordings, membrane_recordings)

# Plot layer activity summary
visualizer.plot_layer_activity_summary(spike_recordings, membrane_recordings)

# Plot network dynamics
visualizer.plot_network_dynamics(spike_recordings, membrane_recordings, training_history)

# Comprehensive analysis
visualizer.plot_comprehensive_analysis(audio, preprocessed_features, spike_recordings, 
                                     membrane_recordings, training_history=training_history)
```

### Example 6: Standalone Visualization Demo
```bash
# Run the standalone visualization demo (no dataset required)
python visualization_demo.py
```

## Model Architecture Details

### Input Layer (16 LIF Neurons)
- **Beta Values**: Logarithmically spaced from 0.5 to 0.95
- **Purpose**: Mimic human ear frequency sensitivity
- **Lower Beta**: More sensitive to low frequencies (longer time constant)
- **Higher Beta**: Less sensitive to high frequencies (shorter time constant)

### Hidden Layer (32 LIF Neurons)
- **Beta Value**: 0.85 (configurable)
- **Purpose**: Feature extraction and temporal processing

### Output Layer (4 LIF Neurons)
- **Beta Value**: 0.9 (configurable)
- **Purpose**: Classification for 4 speech commands

## Preprocessing Methods

### 1. MEL Spectrogram
- **Description**: Standard audio feature extraction
- **Advantages**: Widely used, good frequency resolution, fast computation
- **Disadvantages**: Linear frequency scale, not biologically inspired

### 2. MFCC
- **Description**: Mel-frequency cepstral coefficients
- **Advantages**: Compact features, good for classification, standard in speech recognition
- **Disadvantages**: Loses phase information, not biologically inspired

### 3. Gammatone Filterbank
- **Description**: Auditory-inspired frequency analysis
- **Advantages**: Biologically inspired, good frequency resolution, matches human hearing
- **Disadvantages**: Computationally expensive, complex implementation

### 4. Cochleagram
- **Description**: Cochlea-inspired representation
- **Advantages**: Most biologically realistic, includes hair cell dynamics, good temporal resolution
- **Disadvantages**: Very computationally expensive, complex parameters

## Training Features

### Learning Rate Scheduler
- **Type**: StepLR
- **Step Size**: Configurable (default: 10 epochs)
- **Gamma**: Configurable (default: 0.7)
- **Purpose**: Extract maximum performance even in late training stages

### Early Stopping
- **Criterion**: Validation loss
- **Patience**: Configurable (default: 15 epochs)
- **Purpose**: Prevent overfitting on training data

### Spike Statistics
Monitor spike activity during training:
```python
stats = model.get_spike_statistics()
print(f"Input layer spike rate: {stats['input']['spike_rate']:.4f}")
print(f"Hidden layer spike rate: {stats['hidden']['spike_rate']:.4f}")
print(f"Output layer spike rate: {stats['output']['spike_rate']:.4f}")
```

## Visualization Features

### Audio Preprocessing Visualization
- **Original Audio**: Waveform and spectrogram
- **Processed Features**: Heatmap of preprocessed features (MEL, MFCC, etc.)
- **Feature Statistics**: Distribution and time evolution of features
- **Preprocessing Pipeline**: Complete view of audio transformation

### LIF Neuron Firing Patterns
- **Raster Plots**: Show when each neuron fires over time
- **Membrane Potentials**: Heatmap of membrane potential evolution
- **Neuron-by-Neuron**: Individual neuron activity patterns
- **Layer Comparison**: Compare firing patterns across layers

### Layer Activity Summary
- **Spike Rates**: Average spike rate over time for each layer
- **Activity Distribution**: Histogram of spike counts per neuron
- **Membrane Evolution**: Mean membrane potential with standard deviation
- **Layer Statistics**: Comprehensive layer-wise statistics

### Network Dynamics
- **Spike Statistics**: Total spikes and average rates per layer
- **Membrane Evolution**: Average membrane potential across all layers
- **Layer Correlation**: Correlation between layer spike rates
- **Training History**: Loss and accuracy curves
- **Activity Heatmap**: Complete network activity overview

### Comprehensive Analysis
- **Complete Pipeline**: End-to-end visualization of the entire system
- **Multi-panel View**: All visualizations in a single comprehensive plot
- **Synthetic Data Demo**: Standalone demo with synthetic data
- **Real Model Integration**: Works with actual trained models

### Usage Examples
```python
# Basic visualization
visualizer = SNNVisualizer()

# Audio preprocessing
visualizer.plot_audio_preprocessing(audio, features, preprocessing_type="mel")

# Neuron firing patterns
visualizer.plot_lif_neuron_firing(spike_recordings, membrane_recordings)

# Layer activity
visualizer.plot_layer_activity_summary(spike_recordings, membrane_recordings)

# Network dynamics
visualizer.plot_network_dynamics(spike_recordings, membrane_recordings, history)

# Complete analysis
visualizer.plot_comprehensive_analysis(audio, features, spike_recordings, 
                                     membrane_recordings, training_history=history)
```

## Configuration Options

### Model Parameters
- `input_size`: Number of input neurons (default: 16)
- `hidden_size`: Number of hidden neurons (default: 32)
- `output_size`: Number of output neurons (default: 4)
- `beta_input_min/max`: Input layer beta range (default: 0.5-0.95)
- `beta_hidden`: Hidden layer beta (default: 0.85)
- `beta_output`: Output layer beta (default: 0.9)
- `threshold`: Spiking threshold (default: 1.0)
- `spike_grad_slope`: Surrogate gradient slope (default: 25)
- `output_type`: "mem_potential" or "spk_count"

### Training Parameters
- `loss_type`: "MSE" or "CE"
- `learning_rate`: Initial learning rate (default: 0.001)
- `scheduler_step_size`: LR scheduler step size (default: 10)
- `scheduler_gamma`: LR scheduler gamma (default: 0.7)
- `early_stopping_patience`: Early stopping patience (default: 15)

### Preprocessing Parameters
- `preprocessing`: "mel", "mfcc", "gammatone", or "cochleagram"
- `spike_encoding`: Enable spike encoding (default: True)
- `batch_size`: Training batch size (default: 32)
- `commands`: List of speech commands to classify

## Performance Tips

1. **GPU Usage**: Use CUDA for faster training
2. **Batch Size**: Adjust based on available memory
3. **Preprocessing**: Start with MEL for quick experiments, use Gammatone/Cochleagram for best performance
4. **Beta Values**: Tune based on your specific audio characteristics
5. **Early Stopping**: Monitor validation loss to prevent overfitting

## Troubleshooting

### Common Issues
1. **Memory Issues**: Reduce batch size or use smaller model
2. **Slow Training**: Use GPU, reduce preprocessing complexity
3. **Poor Accuracy**: Try different preprocessing methods or adjust beta values
4. **Import Errors**: Ensure all dependencies are installed

### Dependencies
- PyTorch >= 1.12.0
- snnTorch >= 0.6.0
- torchaudio >= 0.12.0
- librosa >= 0.9.0
- scipy >= 1.7.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{snn_speech_recognition,
  title={SNN Speech Recognition System with Biologically-Inspired Preprocessing},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/snn-speech-recognition}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 