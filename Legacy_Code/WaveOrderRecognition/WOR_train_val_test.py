# Wave Order Recognition functions for training, validation, and testing.
import torch
import random
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def train_model(model, train_loader, criterion, optimizer, epoch, num_epochs, batch_size, hidden_size, output_size, weights_hidden_min_clamped, weights_hidden_max_clamped, weights_output_min_clamped, weights_output_max_clamped, penalty_weight, L1_lambda, device):
    """
    Trains the model for one epoch.

    Args:
    - model (nn.Module): The model to be trained.
    - train_loader (DataLoader): DataLoader for the training data.
    - criterion (nn.Module): Loss function.
    - optimizer (torch.optim.Optimizer): Optimizer for the model.
    - epoch (int): Current epoch.
    - num_epochs (int): Total number of epochs.
    - batch_size (int): Batch size.
    - hidden_size (int): Number of hidden units.
    - output_size (int): Number of output units.
    - weights_hidden_min_clamped (float): Minimum clamped value for hidden layer weights.
    - weights_hidden_max_clamped (float): Maximum clamped value for hidden layer weights.
    - weights_output_min_clamped (float): Minimum clamped value for output layer weights.
    - weights_output_max_clamped (float): Maximum clamped value for output layer weights.
    - device (str): Device to run the training on ('cuda' or 'cpu').

    Returns:
    - tuple: (avg_train_loss, avg_hidden_spike_count, avg_output_spike_count, avg_output_spike_counts_neuron0, avg_output_spike_counts_neuron1)
    """
    model.train()
    running_loss = 0.0
    hidden_total_spike_count = 0
    output_total_spike_count = 0
    output_spike_counts_neuron0 = 0
    output_spike_counts_neuron1 = 0
    equal_total = 0  # to accumulate equal predictions
    correct_train = 0
    total_train = 0



    def L1_regularization(model, L1_lambda):
        L1_norm = 0
        for name, p in model.named_parameters():
            if 'weight' in name:
                L1_norm += p.abs().sum()
        return L1_lambda * L1_norm


    for waves, labels in train_loader:
        waves, labels = waves.to(device).float(), labels.to(device).long()
        optimizer.zero_grad()
        spk1_rec, mem1_rec, spk2_rec, mem2_rec, hidden_spike_count, output_spike_count = model(waves)

        # Original loss
        loss = criterion(spk2_rec.sum(dim=0), labels)

        # Penalty for equal predictions and no predictions
        spike_counts = spk2_rec.sum(dim=0)
        equal_prediction = (spike_counts[:, 0] == spike_counts[:, 1]).sum().item()
        no_prediction = ((spike_counts[:, 0] == 0) & (spike_counts[:, 1] == 0)).sum().item()
        both_spike_penalty = ((spk2_rec[:, :, 0] == 1) & (spk2_rec[:, :, 1] == 1)).sum().item()

        # Add penalty to loss
        #loss += penalty_weight * (equal_prediction + no_prediction + both_spike_penalty)
        loss += penalty_weight * (equal_prediction + no_prediction)


        # L1 regularization
        loss += L1_regularization(model, L1_lambda)

        loss.backward()
        optimizer.step()
        
        # Clamping parameters
        model.lif1.beta.data.clamp_(0.01, 0.99)
        model.lif2.beta.data.clamp_(0.01, 0.99)
        #model.fc1.weight.data.clamp_(weights_hidden_min_clamped, weights_hidden_max_clamped)
        #model.fc2.weight.data.clamp_(weights_output_min_clamped, weights_output_max_clamped)
        
        running_loss += loss.item()
        hidden_total_spike_count += hidden_spike_count
        output_total_spike_count += output_spike_count
        output_spike_counts_neuron0 += spk2_rec[:, :, 0].sum().item()
        output_spike_counts_neuron1 += spk2_rec[:, :, 1].sum().item()
        equal_total += equal_prediction  # accumulate equal predictions


        predicted = []
        for count in spike_counts:
            if count[0] == count[1]:
                predicted.append(random.choice([0, 1]))
            else:
                predicted.append(count.argmax().item())
        predicted = torch.tensor(predicted, device=device)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()


    avg_train_loss = running_loss / len(train_loader)
    avg_hidden_spike_count = hidden_total_spike_count / (len(train_loader) * batch_size * hidden_size)
    avg_output_spike_count = output_total_spike_count / (len(train_loader) * batch_size * output_size)
    avg_output_spike_counts_neuron0 = output_spike_counts_neuron0 / (len(train_loader) * batch_size)
    avg_output_spike_counts_neuron1 = output_spike_counts_neuron1 / (len(train_loader) * batch_size)
    avg_equal_prediction = equal_total / len(train_loader)
    avg_train_accuracy = 100 * correct_train / total_train  # training accuracy
    return avg_train_loss, avg_hidden_spike_count, avg_output_spike_count, avg_output_spike_counts_neuron0, avg_output_spike_counts_neuron1, avg_equal_prediction, avg_train_accuracy

def validate_model(model, validation_loader, criterion, device):
    """
    Validates the model on the validation set.

    Args:
    - model (nn.Module): The model to be validated.
    - validation_loader (DataLoader): DataLoader for the validation data.
    - criterion (nn.Module): Loss function.
    - device (str): Device to run the validation on ('cuda' or 'cpu').

    Returns:
    - tuple: (accuracy, avg_validation_loss)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    equal_prediction = 0

    with torch.no_grad():
        for waves, labels in validation_loader:
            waves, labels = waves.to(device).float(), labels.to(device).long()
            spk1_rec, mem1_rec, spk2_rec, mem2_rec, _, _ = model(waves)
            loss = criterion(spk2_rec.sum(dim=0), labels)
            running_loss += loss.item()
            
            spike_counts = spk2_rec.sum(dim=0)
            predicted = []

            for count in spike_counts:
                if count[0] == count[1]:
                    equal_prediction += 1
                    predicted.append(random.choice([0, 1]))  # Randomly choose if all spike counts are equal
                else:
                    predicted.append(count.argmax().item())

            predicted = torch.tensor(predicted, device=device)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_validation_loss = running_loss / len(validation_loader)
    return accuracy, avg_validation_loss

def test_model(model, test_loader, criterion, device):
    """
    Evaluates the model on the test data and computes various metrics.

    Args:
    - model (nn.Module): The trained model to be evaluated.
    - test_loader (DataLoader): DataLoader for the test data.
    - criterion (nn.Module): Loss function.
    - device (str): Device to run the evaluation on ('cuda' or 'cpu').

    Returns:
    - dict: A dictionary containing the evaluation metrics:
        - 'Test Loss': Average test loss.
        - 'Accuracy': Accuracy of the model on the test data.
        - 'Precision': Precision score.
        - 'Recall': Recall score.
        - 'F1 Score': F1 score.
        - 'Confusion Matrix': Confusion matrix.
        - 'Equal Prediction': Count of equal spike predictions.
        - 'Total Predictions': Total number of predictions.
    """
    
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    equal_prediction = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for waves, labels in test_loader:
            waves, labels = waves.to(device).float(), labels.to(device).long()
            spk1_rec, mem1_rec, spk2_rec, mem2_rec, _, _ = model(waves)
            loss = criterion(spk2_rec.sum(dim=0), labels)
            running_loss += loss.item()
            
            spike_counts = spk2_rec.sum(dim=0)
            predicted = []

            for count in spike_counts:
                if count[0] == count[1]:
                    equal_prediction += 1
                    predicted.append(random.choice([0, 1]))  # Randomly choose if all spike counts are equal
                else:
                    predicted.append(count.argmax().item())

            predicted = torch.tensor(predicted, device=device)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_test_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    confusion = confusion_matrix(all_labels, all_predictions)

    metrics = {
        'Test Loss': avg_test_loss,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Confusion Matrix': confusion,
        'Equal Prediction': equal_prediction,
        'Total Predictions': total
    }

    return metrics