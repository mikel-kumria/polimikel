# Wave Order Recognition functions for plots.
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
import random

# Seeds for reproducibility
seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

def plot_wave(data_loader, save_path, num_plots=6):
    """
    Plots the waves from the data loader.

    Args:
    - data_loader (DataLoader): DataLoader containing the wave data.
    - save_path (str): Path to save the plot.
    - num_plots (int): Number of plots to display.
    """
    data_iter = iter(data_loader)
    waves, labels = next(data_iter)

    plt.figure(figsize=(18, 6))
    for i in range(num_plots):
        plt.subplot(2, 3, i + 1)
        plt.plot(waves[i].numpy(), label=f"Label {labels[i]}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=1200)
    plt.show()

def plot_accuracies(num_epochs, train_accuracy, validation_accuracy, save_path):
    """
    Plots the accuracies over epochs.

    Args:
    - num_epochs (int): Number of epochs.
    - test_accuracies (list): List of test accuracies over epochs.
    - validation_accuracies (list): List of validation accuracies over epochs.
    - save_path (str): Path to save the plot.
    """
    sns.set_style('whitegrid')
    plt.figure(figsize=(14, 6))
    cmap = plt.get_cmap('bwr')(np.linspace(0.1, 0.9, 2))
    plt.plot(range(num_epochs), train_accuracy, label='Test Accuracy', color=cmap[1])
    plt.plot(range(num_epochs), validation_accuracy, label='Validation Accuracy', color=cmap[0])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Train and Validation Accuracy over Epochs')
    plt.ylim(1, 100)
    plt.xlim(0, num_epochs)
    plt.legend(shadow=True)
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path + "_matplotlib.png", dpi=600)
    plt.close()

    fig_plotly = go.Figure()

    fig_plotly.add_trace(go.Scatter(
        x=list(range(num_epochs)),
        y=train_accuracy,
        mode='lines',
        name='Train Accuracy',
        line=dict(color='red')
    ))

    fig_plotly.add_trace(go.Scatter(
        x=list(range(num_epochs)),
        y=validation_accuracy,
        mode='lines',
        name='Validation Accuracy',
        line=dict(color='blue')
    ))

    fig_plotly.update_layout(
        title='Train and Validation Accuracy over Epochs',
        xaxis_title='Epochs',
        xaxis=dict(range=[0, num_epochs]),
        yaxis_title='Accuracy (%)',
        yaxis=dict(range=[1, 100]),
        legend=dict(x=0, y=1, traceorder='normal'),
        template='plotly_white'
    )
    fig_plotly.show()
    pio.write_html(fig_plotly, file=save_path + "_plotly.html")

def plot_metrics(metrics, save_path):
    """
    Prints the evaluation metrics and plots the confusion matrix.

    Args:
    - metrics (dict): A dictionary containing the evaluation metrics:
        - 'Test Loss': Average test loss.
        - 'Accuracy': Accuracy of the model on the test data.           Accuracy = (TP + TN) / (TP + TN + FP + FN)
        - 'Precision': Precision score.                                 Precision = TP / (TP + FP)
        - 'Recall': Recall score.                                       Recall = TP / (TP + FN)
        - 'F1 Score': F1 score.                                         F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
        - 'Confusion Matrix': Confusion matrix.                         Rows: True labels, Columns: Predicted labels.
        - 'Equal Prediction': Count of equal spike predictions.
    - save_path (str): Path to save the confusion matrix plot.
    """
    
    # Print metrics
    print(f"Test Loss: {metrics['Test Loss']:.4f}")
    print(f"Accuracy = (TP + TN) \ (TP + TN + FP + FN) = {metrics['Accuracy']:.2f}%")
    print(f"Precision = TP \ (TP + FP) = {metrics['Precision']:.2f}")
    print(f"Recall = TP \ (TP + FN) = {metrics['Recall']:.2f}")
    print(f"F1 Score = 2 * (Precision * Recall) \ (Precision + Recall) = {metrics['F1 Score']:.2f}")
    equal_pred_percentage = metrics['Equal Prediction'] / metrics['Total Predictions'] * 100
    print(f"Equal Predictions: {metrics['Equal Prediction']} ({equal_pred_percentage:.2f}%)")

    # Plot confusion matrix using Matplotlib
    sns.set_style('whitegrid')
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics['Confusion Matrix'], annot=True, fmt="d", cmap='Blues', cbar=False, xticklabels=['Sinusoidal', 'Square'], yticklabels=['Sinusoidal', 'Square'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path + "_confusion_matrix_matplotlib.png", dpi=600)
    plt.close()

    # Plot confusion matrix using Plotly
    fig_plotly = go.Figure(data=go.Heatmap(
        z=metrics['Confusion Matrix'],
        x=['Sinusoidal', 'Square'],
        y=['Sinusoidal', 'Square'],
        colorscale='Blues',
        showscale=False,
        text=metrics['Confusion Matrix'],
        texttemplate="%{text}",
        textfont={"size":20}
    ))

    fig_plotly.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        template='plotly_white'
    )

    fig_plotly.show()
    pio.write_html(fig_plotly, file=save_path + "_confusion_matrix_plotly.html")

def plot_loss_curve(train_losses, validation_losses, num_epochs, save_path):
    """
    Plots the training and validation loss curves over epochs.

    Args:
    - train_losses (list): List of training losses over epochs.
    - validation_losses (list): List of validation losses over epochs.
    - num_epochs (int): Number of epochs.
    - save_path (str): Path to save the plot.
    """
    epochs = np.arange(num_epochs)
    plt.figure(figsize=(14, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, validation_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path + "_matplotlib.png", dpi=600)
    plt.close()

    fig_plotly = go.Figure()

    fig_plotly.add_trace(go.Scatter(
        x=list(range(num_epochs)),
        y=train_losses,
        mode='lines',
        name='Training Loss',
        line=dict(color='blue')
    ))

    fig_plotly.add_trace(go.Scatter(
        x=list(range(num_epochs)),
        y=validation_losses,
        mode='lines',
        name='Validation Loss',
        line=dict(color='red')
    ))

    fig_plotly.update_layout(
        title='Training and Validation Loss over Epochs',
        xaxis_title='Epochs',
        xaxis=dict(range=[0, num_epochs]),
        yaxis_title='Loss',
        template='plotly_white'
    )

    fig_plotly.show()
    pio.write_html(fig_plotly, file=save_path + "_plotly.html")

def plot_equal_prediction_values(equal_prediction_values, num_epochs, save_path):
    """
    Plots the counts of no prediction and equal prediction values over epochs.

    Args:
    - equal_prediction_values (list): List of equal prediction counts over epochs.
    - num_epochs (int): Number of epochs.
    - save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(14, 6))
    plt.plot(range(num_epochs), equal_prediction_values, label='Equal Prediction Count', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Count')
    plt.title('Equal Prediction Count over Epochs')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path + "_matplotlib.png", dpi=600)
    plt.close()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(num_epochs)),
        y=equal_prediction_values,
        mode='lines',
        name='Equal Prediction Count',
        line=dict(color='blue')
    ))

    fig.update_layout(
        title='Equal Prediction Count over Epochs',
        xaxis_title='Epochs',
        yaxis_title='Count',
        template='plotly_white',
        legend=dict(x=0, y=1, traceorder='normal')
    )

    fig.show()
    pio.write_html(fig, file=save_path + "_plotly.html")

def plot_beta_values(beta_values, num_epochs, save_path, layer_name):
    """
    Plots the beta values of a certain layer over epochs.

    Args:
    - beta_values (list): List of beta values over epochs.
    - num_epochs (int): Number of epochs.
    - save_path (str): Path to save the plot.
    - layer_name (str): Name of the layer (e.g., 'Hidden' or 'Output').
    """
    beta_values = np.array(beta_values)
    plt.figure(figsize=(14, 6))
    for i in range(beta_values.shape[1]):
        plt.plot(range(num_epochs), beta_values[:, i])
    plt.xlabel('Epochs')
    plt.ylabel(f'beta {layer_name} layer')
    plt.title(f'beta values ({layer_name} layer) over Epochs')
    plt.ylim(0.001, 1)
    plt.xlim(0, num_epochs)
    plt.grid(True)
    plt.savefig(save_path + f"_{layer_name}_matplotlib.png", dpi=600)
    plt.close()

    fig_beta = go.Figure()
    for i in range(beta_values.shape[1]):
        fig_beta.add_trace(go.Scatter(
            x=list(range(num_epochs)),
            y=beta_values[:, i],
            mode='lines',
            name=f'Neuron {i}'
        ))
    fig_beta.update_layout(
        title=f'beta values ({layer_name} layer) over Epochs',
        xaxis_title='Epochs',
        xaxis=dict(range=[0, num_epochs]),
        yaxis=dict(range=[0.001, 1]),
        yaxis_title=f'beta {layer_name} layer',
        template='plotly_white'
    )
    pio.write_html(fig_beta, file=save_path + f"_{layer_name}_plotly.html")
    fig_beta.show()

def plot_histograms(model, save_path, layer_name):
    """
    Plots the histogram of beta values for a certain layer.

    Args:
    - model (nn.Module): The model containing the beta values.
    - save_path (str): Path to save the plot.
    - layer_name (str): Name of the layer (e.g., 'Hidden' or 'Output').
    """
    beta = model.lif1.beta.detach().numpy() if layer_name == 'Hidden' else model.lif2.beta.detach().numpy()

    max_beta_value = max(beta.max(), beta.max())
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    axs[0].hist(beta, alpha=0.8, color='blue')
    axs[0].set_xlabel(r'$\beta$ values')
    axs[0].set_xlim(0.001, 1 if max_beta_value > 1 else max_beta_value)
    axs[0].set_ylabel('Frequency')
    axs[0].set_title(f'Histogram of $\beta$ values for {layer_name} layer neurons')

    plt.tight_layout()
    plt.savefig(save_path + f"_{layer_name}_beta_histogram.png", dpi=600)
    plt.show()
    plt.close()

def plot_tau_values(beta_values, num_epochs, deltaT, save_path, layer_name):
    """
    Plots the tau values of a certain layer over epochs.

    Args:
    - beta_values (list): List of beta values over epochs.
    - num_epochs (int): Number of epochs.
    - deltaT (float): Time step value.
    - save_path (str): Path to save the plot.
    - layer_name (str): Name of the layer (e.g., 'Hidden' or 'Output').
    """
    beta_values = np.array(beta_values)
    plt.figure(figsize=(14, 6))
    for i in range(beta_values.shape[1]):
        plt.plot(range(num_epochs), -1000 * deltaT / np.log(beta_values[:, i]))
    plt.xlabel('Epochs')
    plt.ylabel(f'tau {layer_name} layer [ms]')
    plt.xlim(0, num_epochs)
    plt.ylim(0.001, None)
    plt.title(f'tau values ({layer_name} layer) over Epochs (deltaT = {1000 * deltaT}ms)')
    plt.grid(True)
    plt.savefig(save_path + f"_{layer_name}_tau_matplotlib.png", dpi=600)
    plt.close()

    fig_tau = go.Figure()
    for i in range(beta_values.shape[1]):
        fig_tau.add_trace(go.Scatter(
            x=list(range(num_epochs)),
            y=-1000 * deltaT / np.log(beta_values[:, i]),
            mode='lines',
            name=f'Neuron {i}'
        ))
    fig_tau.update_layout(
        title=f'tau values ({layer_name} layer) over Epochs (deltaT = {1000 * deltaT}ms)',
        xaxis_title='Epochs',
        xaxis=dict(range=[0, num_epochs]),
        yaxis=dict(range=[0.1, None]),
        yaxis_title=f'tau {layer_name} layer [ms]',
        template='plotly_white'
    )
    pio.write_html(fig_tau, file=save_path + f"_{layer_name}_plotly.html")
    fig_tau.show()

def plot_weight_distribution(weights_before, weights_after, save_path, layer_name):
    """
    Plots the weight distribution before and after training.

    Args:
    - weights_before (list): List of weights before training.
    - weights_after (list): List of weights after training.
    - save_path (str): Path to save the plot.
    - layer_name (str): Name of the layer (e.g., 'Hidden' or 'Output').
    """
    fig, axs = plt.subplots(3, 2, figsize=(14, 18))

    hidden_x_max = max(np.abs(weights_before[0]).max(), np.abs(weights_after[0]).max())
    hidden_y_max = max(np.histogram(weights_before[0].flatten(), bins=50)[0].max(),
                       np.histogram(weights_after[0].flatten(), bins=50)[0].max())
    hidden_x_min = min(np.abs(weights_before[0]).min(), np.abs(weights_after[0]).min())
    hidden_y_min = min(np.histogram(weights_before[0].flatten(), bins=50)[0].min(),
                       np.histogram(weights_after[0].flatten(), bins=50)[0].min())

    output_x_max = max(np.abs(weights_before[1]).max(), np.abs(weights_after[1]).max())
    output_y_max = max(np.histogram(weights_before[1].flatten(), bins=50)[0].max(),
                       np.histogram(weights_after[1].flatten(), bins=50)[0].max())
    output_x_min = min(np.abs(weights_before[1]).min(), np.abs(weights_after[1]).min())
    output_y_min = min(np.histogram(weights_before[1].flatten(), bins=50)[0].min(),
                       np.histogram(weights_after[1].flatten(), bins=50)[0].min())

    hidden_bins = np.linspace(hidden_x_min, hidden_x_max, 51)
    output_bins = np.linspace(output_x_min, output_x_max, 51)

    axs[0, 0].hist(weights_before[0].flatten(), bins=hidden_bins, alpha=0.8, color='blue', label='Before')
    axs[0, 0].set_xlabel('Weight Value')
    axs[0, 0].set_ylabel('Frequency')
    axs[0, 0].set_title('Hidden Layer Weight Distribution Before Training')
    axs[0, 0].set_xlim(hidden_x_min, hidden_x_max)
    axs[0, 0].set_ylim(hidden_y_min, hidden_y_max)
    axs[0, 0].legend()

    axs[0, 1].hist(weights_before[1].flatten(), bins=output_bins, alpha=0.8, color='blue', label='Before')
    axs[0, 1].set_xlabel('Weight Value')
    axs[0, 1].set_ylabel('Frequency')
    axs[0, 1].set_title('Output Layer Weight Distribution Before Training')
    axs[0, 1].set_xlim(output_x_min, output_x_max)
    axs[0, 1].set_ylim(output_y_min, output_y_max)
    axs[0, 1].legend()

    axs[1, 0].hist(weights_after[0].flatten(), bins=hidden_bins, alpha=0.9, color='orange', label='After')
    axs[1, 0].set_xlabel('Weight Value')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].set_title('Hidden Layer Weight Distribution After Training')
    axs[1, 0].set_xlim(hidden_x_min, hidden_x_max)
    axs[1, 0].set_ylim(hidden_y_min, hidden_y_max)
    axs[1, 0].legend()

    axs[1, 1].hist(weights_after[1].flatten(), bins=output_bins, alpha=0.9, color='orange', label='After')
    axs[1, 1].set_xlabel('Weight Value')
    axs[1, 1].set_ylabel('Frequency')
    axs[1, 1].set_title('Output Layer Weight Distribution After Training')
    axs[1, 1].set_xlim(output_x_min, output_x_max)
    axs[1, 1].set_ylim(output_y_min, output_y_max)
    axs[1, 1].legend()

    axs[2, 0].hist(weights_before[0].flatten(), bins=hidden_bins, alpha=0.5, color='blue', label='Before')
    axs[2, 0].hist(weights_after[0].flatten(), bins=hidden_bins, alpha=0.5, color='orange', label='After')
    axs[2, 0].set_xlabel('Weight Value')
    axs[2, 0].set_ylabel('Frequency')
    axs[2, 0].set_title('Hidden Layer Weight Distribution Before and After Training')
    axs[2, 0].set_xlim(hidden_x_min, hidden_x_max)
    axs[2, 0].set_ylim(hidden_y_min, hidden_y_max)
    axs[2, 0].legend()

    axs[2, 1].hist(weights_before[1].flatten(), bins=output_bins, alpha=0.5, color='blue', label='Before')
    axs[2, 1].hist(weights_after[1].flatten(), bins=output_bins, alpha=0.5, color='orange', label='After')
    axs[2, 1].set_xlabel('Weight Value')
    axs[2, 1].set_ylabel('Frequency')
    axs[2, 1].set_title('Output Layer Weight Distribution Before and After Training')
    axs[2, 1].set_xlim(output_x_min, output_x_max)
    axs[2, 1].set_ylim(output_y_min, output_y_max)
    axs[2, 1].legend()

    plt.savefig(save_path + f"_{layer_name}_matplotlib.png", dpi=600)
    plt.show()

    print("Input to hidden: \t Hidden to output 0: \t Hidden to output 1:")

    for i in range(weights_hidden_layer_last.shape[0]):
        input_to_hidden = f"Input [{i}]: {weights_hidden_layer_last[i][0]:.3f}"
        hidden_to_output_0 = f"Hidden_0 [{i}]: {weights_output_layer_last[0][i]:.3f}" if weights_output_layer_last[0][i] != 0 else f"Hidden_0 [{i}]: -"
        hidden_to_output_1 = f"Hidden_1 [{i}]: {weights_output_layer_last[1][i]:.3f}" if weights_output_layer_last[1][i] != 0 else f"Hidden_1 [{i}]: -"
        print(f"{input_to_hidden} \t {hidden_to_output_0} \t {hidden_to_output_1}")

def plot_layer_weights(weights_layer, num_epochs, save_path, layer_name):
    """
    Plots the weights of a certain layer over epochs.

    Args:
    - weights_layer (list): List of weights over epochs.
    - num_epochs (int): Number of epochs.
    - save_path (str): Path to save the plot.
    - layer_name (str): Name of the layer (e.g., 'Hidden' or 'Output').
    """
    plt.figure(figsize=(14, 6))
    weights_layer_np = np.array(weights_layer)
    num_neurons = weights_layer_np.shape[2]  # Use the correct dimension for neurons
    for i in range(num_neurons):
        plt.plot(range(num_epochs), weights_layer_np[:, :, i].mean(axis=1), label=f'Neuron {i}')  # Plot average weights over batches
    plt.xlabel('Epochs')
    plt.ylabel(f'{layer_name} Layer Weights')
    plt.xlim(0, num_epochs)
    plt.title(f'{layer_name} Layer Weights over Epochs')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path + f"_{layer_name}_matplotlib.png", dpi=600)
    plt.close()

    fig_weights = go.Figure()
    for i in range(num_neurons):
        fig_weights.add_trace(go.Scatter(
            x=list(range(num_epochs)),
            y=weights_layer_np[:, :, i].mean(axis=1),  # Plot average weights over batches
            mode='lines',
            name=f'Neuron {i}'
        ))
    fig_weights.update_layout(
        title=f'{layer_name} Layer Weights over Epochs',
        xaxis_title='Epochs',
        xaxis=dict(range=[0, num_epochs]),
        yaxis_title=f'{layer_name} Layer Weights',
        template='plotly_white'
    )
    pio.write_html(fig_weights, file=save_path + f"_{layer_name}_plotly.html")
    fig_weights.show()

def plot_spike_counts(hidden_spike_count, output_spike_count, output_spike_counts_neuron0, output_spike_counts_neuron1, num_epochs, save_path):
    """
    Plots the spike counts over epochs.

    Args:
    - hidden_spike_count (list): List of hidden layer spike counts over epochs.
    - output_spike_count (list): List of output layer spike counts over epochs.
    - output_spike_counts_neuron0 (list): List of output layer neuron 0 spike counts over epochs.
    - output_spike_counts_neuron1 (list): List of output layer neuron 1 spike counts over epochs.
    - num_epochs (int): Number of epochs.
    - save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(14, 6))
    plt.plot(range(num_epochs), hidden_spike_count, label='Hidden Layer Spike Count', color='blue')
    plt.plot(range(num_epochs), output_spike_count, label='Output Layer Spike Count (average)', color='red')
    plt.plot(range(num_epochs), output_spike_counts_neuron0, label='Output Layer Neuron 0 Spike Count', color='orange', linestyle='--')
    plt.plot(range(num_epochs), output_spike_counts_neuron1, label='Output Layer Neuron 1 Spike Count', color='violet', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Average Spike Count')
    plt.title('Spike Counts over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path + "_matplotlib.png", dpi=600)
    plt.close()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(num_epochs)),
        y=hidden_spike_count,
        mode='lines',
        name='Hidden Layer Spike Count',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=list(range(num_epochs)),
        y=output_spike_count,
        mode='lines',
        name='Output Layer Spike Count (Average)',
        line=dict(color='red')
    ))
    fig.add_trace(go.Scatter(
        x=list(range(num_epochs)),
        y=output_spike_counts_neuron0,
        mode='lines',
        name='Output Layer Neuron 0 Spike Count',
        line=dict(color='orange', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=list(range(num_epochs)),
        y=output_spike_counts_neuron1,
        mode='lines',
        name='Output Layer Neuron 1 Spike Count',
        line=dict(color='violet', dash='dash')
    ))
    fig.update_layout(
        title='Spike Counts over Epochs',
        xaxis_title='Epochs',
        yaxis_title='Average Spike Count',
        template='plotly_white',
        legend=dict(x=0, y=1, traceorder='normal')
    )
    fig.show()
    pio.write_html(fig, file=save_path + "_plotly.html")

def plot_snn_spikes(model, test_loader, device, save_path, layer_name, layer_size, num_steps):
    """
    Plots the spikes of a spiking neural network layer.

    Args:
    - model (nn.Module): The trained SNN model.
    - test_loader (DataLoader): DataLoader for the test data.
    - device (str): Device to run the model on ('cuda' or 'cpu').
    - save_path (str): Path to save the plot.
    - layer_name (str): Name of the layer (e.g., 'Hidden' or 'Output').
    - layer_size (int): Number of neurons in the layer.
    - num_steps (int): Number of time steps.
    """
    waves, labels = next(iter(test_loader))
    waves, labels = waves.to(device).float(), labels.to(device).long()
    spk1_rec, mem1_rec, spk2_rec, mem2_rec, hidden_spike_count, output_spike_count = model(waves)

    spikes = spk1_rec if layer_name == 'Hidden' else spk2_rec

    fig = go.Figure()
    def raster_with_dots(spikes, fig, label_colors):
        sizes = 6
        for i in range(spikes.shape[1]):
            spk_times = np.where(spikes[:, i].detach().cpu().numpy())[0]
            for t in spk_times:
                fig.add_trace(go.Scatter(
                    x=[t],
                    y=[i],
                    mode='markers',
                    marker=dict(color=label_colors[i], size=sizes, symbol='circle'),
                    name=f'{layer_name} Layer Neuron {i}' if t == spk_times[0] else ''
                ))
    label_colors = ['blue' if label == 0 else 'orange' for label in labels]
    spike_color_map = []
    for t in range(spikes.shape[0]):
        spike_color_map.append(label_colors[t % len(labels)])
    raster_with_dots(spikes, fig, spike_color_map)
    fig.update_layout(
        title=f'{layer_name} Layer Spikes',
        xaxis_title='Time Steps',
        yaxis_title='Batch Samples',
        template='plotly_white',
        showlegend=False
    )
    fig.add_annotation(
        text='- Sine-Square wave: blue <br>- Square-Sine wave: orange',
        xref='paper', yref='paper',
        x=0.5, y=1.1,
        showarrow=False,
        font=dict(size=12)
    )
    fig.show()
    pio.write_html(fig, file=save_path + f"_{layer_name}_spikes.html")

def plot_membrane_potentials(model, test_loader, device, layer_name, layer_size, num_steps, save_path):
    """
    Plots the membrane potentials of randomly selected neurons from a specified layer over time steps.

    Args:
    - model (torch.nn.Module): The trained spiking neural network model.
    - test_loader (DataLoader): DataLoader for the test dataset.
    - device (str): Device to run the model on ('cpu' or 'cuda').
    - layer_name (str): Name of the layer to plot ('Hidden' or 'Output').
    - layer_size (int): Size of the layer.
    - num_steps (int): Number of time steps.
    - save_path (str): Path to save the plot.
    """
    num_neurons_to_plot = min(8, layer_size)
    neuron_indices = random.sample(range(layer_size), num_neurons_to_plot)
    waves, labels = next(iter(test_loader))
    waves = waves.to(device).float()
    labels = labels.to(device).long()
    spk1_rec, mem1_rec, spk2_rec, mem2_rec, hidden_spike_count, output_spike_count = model(waves)

    potentials = mem1_rec if layer_name == 'Hidden' else mem2_rec
    spikes = spk1_rec if layer_name == 'Hidden' else spk2_rec

    time_steps = np.arange(num_steps)
    fig, axs = plt.subplots((num_neurons_to_plot + 1) // 2, 2, figsize=(24, 8) if num_neurons_to_plot == 2 else (24, 14))
    axs = axs.flatten()

    sine_wave_indices = [i for i, label in enumerate(labels) if label == 0]
    square_wave_indices = [i for i, label in enumerate(labels) if label == 1]
    selected_indices = random.sample(sine_wave_indices, num_neurons_to_plot // 2) + random.sample(square_wave_indices, num_neurons_to_plot // 2)

    for i, neuron_idx in enumerate(neuron_indices):
        mem_potential = potentials[:, selected_indices[i], neuron_idx].detach().cpu().numpy()
        input_wave = waves[selected_indices[i]].cpu().numpy()
        neuron_spikes = spikes[:, selected_indices[i], neuron_idx].detach().cpu().numpy()
        spike_times = np.where(neuron_spikes == 1)[0]
        spike_count = len(spike_times)

        axs[i].plot(time_steps, mem_potential, label=f'Neuron {neuron_idx} $V_{{mem}}$ (spike count: {spike_count})', color='blue')
        axs[i].plot(time_steps, input_wave, label=f'{"Sine" if labels[selected_indices[i]] == 0 else "Square"} Wave', color='red', linestyle='dashed')
        axs[i].scatter(spike_times, mem_potential[spike_times], color='green', s=10, label='Spike Times')
        axs[i].set_xlabel('Time Steps', fontsize=10)
        axs[i].set_title(f'Neuron {neuron_idx} $V_{{mem}}$', fontsize=10)
        axs[i].grid(True)
        axs[i].legend(fontsize=7)

        # Add vertical lines
        for step in range(0, num_steps, 1):
            linewidth = 1.5 if step % 10 == 0 else 0.5
            axs[i].axvline(x=step, color='lightgrey', linewidth=linewidth)

    fig.text(0.04, 0.5, f'{layer_name} Neuron Membrane Potential + Input Wave', va='center', rotation='vertical', fontsize=12)
    plt.tight_layout(rect=[0.04, 0, 1, 1])
    plt.savefig(save_path + f"_{layer_name}_membrane_potentials.png", dpi=1200)
    plt.show()
    plt.close()

def plot_threshold_potentials(threshold_values, num_epochs, save_path, layer_name):
    """
    Plots the distribution of threshold potentials of a certain layer vs Epochs.

    Args:
    threshold_values (list): List of threshold potentials of the layer over epochs. Example: model.lif1.threshold
    num_epochs (int): Number of epochs.
    save_path (str): Path to save the plot.
    layer_name (str): Name of the layer. Example: 'Hidden' or 'Output'.
    """
    threshold_values = np.array(threshold_values)
    plt.figure(figsize=(14, 6))
    for i in range(threshold_values.shape[1]):
        plt.plot(range(num_epochs), threshold_values[:, i])
    plt.xlabel('Epochs')
    plt.ylabel(f'Threshold Potentials {layer_name} layer')
    plt.title(f'Threshold Potentials ({layer_name} layer) over Epochs')
    plt.ylim(0.001, max(threshold_values.max(), 1.1))
    plt.xlim(0, num_epochs)
    plt.grid(True)
    plt.savefig(save_path + f"_{layer_name}_threshold_potentials_matplotlib.png", dpi=600)
    plt.close()

    fig_threshold = go.Figure()
    for i in range(threshold_values.shape[1]):
        fig_threshold.add_trace(go.Scatter(
            x=list(range(num_epochs)),
            y=threshold_values[:, i],
            mode='lines',
            name=f'Neuron {i}'
        ))
    fig_threshold.update_layout(
        title=f'Threshold Potentials ({layer_name} layer) over Epochs',
        xaxis_title='Epochs',
        xaxis=dict(range=[0, num_epochs]),
        yaxis=dict(range=[0.001, max(threshold_values.max(), 1.1)]),
        yaxis_title=f'Threshold Potentials {layer_name} layer',
        template='plotly_white'
    )
    pio.write_html(fig_threshold, file=save_path + f"_{layer_name}_threshold_potentials_plotly.html")
    fig_threshold.show()
