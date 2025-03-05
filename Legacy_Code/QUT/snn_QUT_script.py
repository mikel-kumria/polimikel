# Standard library imports
import os
import random
import json
from datetime import datetime
import logging

# Third-party imports
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import snntorch as snn
import snntorch.functional as SF
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import nni
import matplotlib.pyplot as plt
# Local Imports
from QUT_DataLoader import QUTDataset, get_dataloader



# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device-agnostic code
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
logger.info(f"Using device: {device}")
0.5
# Set random seeds for reproducibility
seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)



class Config:
    """Configuration parameters for the model, training, and paths."""

    def __init__(self):
        # Model parameters0.5
        self.input_size = 16
        self.hidden_size = 24
        self.output_size = 4

        # Training parameters
        self.num_epochs = 30
        self.batch_size = 32
        self.optimizer_betas = (0.9, 0.999)  # Adam optimizer betas
        self.learning_rate = 0.005
        self.scheduler_T_max = 10  # Cosine Annealing scheduler restart period
        self.scheduler_eta_min = 0.0001  # Cosine Annealing scheduler minimum learning rate
        self.weight_decay = 0.0  # L2 regularization
        self.patience = 500  # Early stopping patience
        self.delta = 0.001  # Early stopping minimum delta
        self.l1_lambda = 0.0  # L1 regularization

        # Initial weights and standard deviations
        self.weights_1 = 2.0
        self.weights_2 = 1.0
        self.weights_3 = 0.7
        self.weights_4 = 0.6

        self.std_w1 = 1.0
        self.std_w2 = 1.0
        self.std_w3 = 0.5
        self.std_w4 = 0.5

        # Quantization parameters
        self.min_weight = 0.0
        self.max_weight = 20.0
        self.min_vmem = 0.0
        self.max_vmem = 15.0
        self.output_threshold = 1.0  # Initial threshold for output neurons

        # To-Do
        self.quantize = False
        self.tau_bits = 8
        self.weights_bits = 5
        self.tau_noise = 0.0
        self.threshold_noise = 0.0
        # to add: Bernoulli distribution for spike generation

        # Paths
        self.results_dir = "/home/kumria/Documents/simulations"
        self.data_dir = "/home/kumria/Documents/Offline_Datasets/4_one_second_samples"




# Tau and beta values
def create_power_vector(n, size):
    """Create a vector of powers of two for tau values."""
    powers = [2 ** i for i in range(1, n + 1)]
    repeat_count = size // n
    vector = np.repeat(powers, repeat_count)
    return vector



class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid
    as this was done in Zenke & Ganguli (2018).
    """

    scale = 20.0  # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use the
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return (input > 0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the
        surrogate gradient of the loss with respect to the input.
        Here we use the normalized negative part of a fast sigmoid
        as this was done in Zenke & Ganguli (2018).
        """
        input_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        alpha = 2
        grad = (
            alpha
            / 2
            / (1 + (torch.pi / 2 * alpha * input_).pow_(2))
            * grad_input
        )
        return grad


activation = SurrGradSpike.apply

class Leaky(torch.nn.Module):
    def __init__(self, beta, n=1, bs=1, thr=1.0,device='cpu'):
        super(Leaky, self).__init__()
        self.beta = beta.to(device)
        self.thr = thr
        self.n = n
        self.mem = torch.zeros((bs,n)).to(device)
    def reset_mem(self,bs):
        self.mem = torch.zeros((bs,self.n)).to(self.mem.device)
    def forward(self, x):
        self.mem = self.beta * self.mem + x
        spk = activation(self.mem - self.thr)
        self.mem = self.mem * (1 - spk)
        return spk, self.mem

class SNNQUT(nn.Module):
    """Spiking Neural Network model without quantization."""
    def __init__(self, config, beta_hidden, beta_output, time_steps):
        super().__init__()
        self.time_steps = time_steps

        # Define the network layers using standard nn.Linear
        self.fc1 = nn.Linear(config.input_size, config.hidden_size, bias=False)
        #self.lif1 = snn.Leaky(beta=beta_hidden[0], reset_mechanism='zero',init_hidden=True,output=True)
        self.lif1 = Leaky(beta=beta_hidden[0], n=config.hidden_size, bs=config.batch_size,device=device)
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        #self.lif2 = snn.Leaky(beta=beta_hidden[1], reset_mechanism='zero',init_hidden=True,output=True)
        self.lif2 = Leaky(beta=beta_hidden[1], n=config.hidden_size, bs=config.batch_size,device=device)
        self.fc3 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        #self.lif3 = snn.Leaky(beta=beta_hidden[2], reset_mechanism='zero',init_hidden=True,output=True)
        self.lif3 = Leaky(beta=beta_hidden[2], n=config.hidden_size, bs=config.batch_size,device=device)
        self.fc4 = nn.Linear(config.hidden_size, config.output_size, bias=False)
        #self.lif4 = snn.Leaky(beta=beta_output, reset_mechanism='zero', threshold=10e7,init_hidden=True,output=True)
        self.lif4 = Leaky(beta=beta_output, n=config.output_size, bs=config.batch_size, thr=10e7,device=device)
        # self.seq = nn.Sequential(self.fc1, self.lif1, self.fc2, self.lif2, self.fc3, self.lif3, self.fc4, self.lif4)


        # Clamping parameters
        self.min_vmem = config.min_vmem
        self.max_vmem = config.max_vmem
        self.min_weight = config.min_weight
        self.max_weight = config.max_weight

        # Weight initialization
        self.weights_1 = config.weights_1
        self.weights_2 = config.weights_2
        self.weights_3 = config.weights_3
        self.weights_4 = config.weights_4

        # Standard deviations for weight initialization
        self.std_w1 = config.std_w1
        self.std_w2 = config.std_w2
        self.std_w3 = config.std_w3
        self.std_w4 = config.std_w4
        
        # Initialize weights
        self.normal_init()

    def normal_init(self):
        """Initialize weights using Kaiming uniform distribution."""
        nn.init.normal_(self.fc1.weight, mean=self.weights_1, std=self.std_w1)
        nn.init.normal_(self.fc2.weight, mean=self.weights_2, std=self.std_w2)
        nn.init.normal_(self.fc3.weight, mean=self.weights_3, std=self.std_w3)
        nn.init.normal_(self.fc4.weight, mean=self.weights_4, std=self.std_w4)


    def clamp_weights(self):
        """Clamp weights if necessary (optional)."""
        with torch.no_grad():
            self.fc1.weight.data.clamp_(min=self.min_weight, max=self.max_weight)
            self.fc2.weight.data.clamp_(min=self.min_weight, max=self.max_weight)
            self.fc3.weight.data.clamp_(min=self.min_weight, max=self.max_weight)
            self.fc4.weight.data.clamp_(min=self.min_weight, max=self.max_weight)
    def reset(self, bs):
        self.lif1.reset_mem(bs)
        self.lif2.reset_mem(bs)
        self.lif3.reset_mem(bs)
        self.lif4.reset_mem(bs)

    def forward(self, x):
            # Extract current input
            # Pass through fc1
            cur1 = self.fc1(x)
            spk1, _= self.lif1(cur1)
            #mem1_rec.append(mem1)
            #spk1_rec.append(spk1)

            # Pass through fc2
            cur2 = self.fc2(spk1)
            spk2, _ = self.lif2(cur2)
            #mem2_rec.append(mem2)
            #spk2_rec.append(spk2)

            # Pass through fc3
            cur3 = self.fc3(spk2)
            spk3, _ = self.lif3(cur3)
            #mem3_rec.append(mem3)
            #spk3_rec.append(spk3)

            # Pass through fc4
            cur4 = self.fc4(spk3)
            spk4, mem4 = self.lif4(cur4)
            #spk4_rec.append(spk4)
            #spk4, mem4 = self.seq(x)
            #Clamp membrane potentials
            #mem1 = torch.clamp(mem1, min=self.min_vmem, max=self.max_vmem)
            #mem2 = torch.clamp(mem2, min=self.min_vmem, max=self.max_vmem)
            #mem3 = torch.clamp(mem3, min=self.min_vmem, max=self.max_vmem)
            #mem4 = torch.clamp(mem4, min=self.min_vmem, max=self.max_vmem)
            return mem4


def train_model(model, train_loader, optimizer, device, l1_lambda, writer, epoch, args):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0

    # Define the loss function
    loss_fn = SF.mse_membrane_loss(on_target=1.0, off_target=0.0, reduction='mean')

    for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False, disable=args.stop_tqdm)):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        inputs = inputs.to(torch.float32)
        mem4_rec = []
        model.reset(inputs.shape[0])
        # Forward pass
        for step in range(inputs.shape[1]):
            mem4 = model(inputs[:,step,:])
            # Stack recordings
            #spk1_rec = torch.stack(spk1_rec, dim=0)
            #spk2_rec = torch.stack(spk2_rec, dim=0)
            #spk3_rec = torch.stack(spk3_rec, dim=0)
            #spk4_rec = torch.stack(spk4_rec, dim=0)

            #mem1_rec = torch.stack(mem1_rec, dim=0)
            #mem2_rec = torch.stack(mem2_rec, dim=0)
            #mem3_rec = torch.stack(mem3_rec, dim=0)
            mem4_rec.append(mem4)
        mem4_rec = torch.stack(mem4_rec, dim=0).to(device)

        # Compute the loss
        true_labels = labels.argmax(dim=1)
        loss = loss_fn(mem4_rec, true_labels)


        # L1 regularization
        #if l1_lambda > 0:
        #    l1_norm = sum(p.abs().sum() for p in model.parameters())
        #    loss = loss + l1_lambda * l1_norm

        # Backward pass and optimizer
        loss.backward()
        optimizer.step()

        # model.clamp_weights()
        running_loss += loss.item()

        # For accuracy
        final_membrane_potentials = mem4_rec[-1]
        predicted = final_membrane_potentials.argmax(dim=1)
        total += labels.size(0)
        correct += predicted.eq(true_labels).sum().item()
        
        if (batch_idx > 20): #& (args.nni_opt):
            break
    accuracy = 100.0 * correct / total
    avg_loss = running_loss / len(train_loader)

    return avg_loss, accuracy


def validate_model(model, val_loader, device, writer, epoch, args):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0
    all_labels = []
    all_predictions = []

    # Define the loss function
    loss_fn = SF.mse_membrane_loss(on_target=1.0, off_target=0.0, reduction='mean')

    with torch.no_grad():
        for batch_idx,(inputs, labels) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", leave=False, disable=args.stop_tqdm)):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.to(torch.float32)
            model.reset(inputs.shape[0])
            mem4_rec = []

            #x = x.to(torch.float32)
            #batch_size, time_steps, _ = inputs.shape

            # Ensure time_steps matches
            #if time_steps != self.time_steps:
            #    raise ValueError(f"Expected time_steps={self.time_steps}, but got {time_steps}")

            # Initialize membrane potentials
            #mem1 = torch.zeros(batch_size, self.fc1.out_features, device=x.device)
            #mem2 = torch.zeros(batch_size, self.fc2.out_features, device=x.device)
            #mem3 = torch.zeros(batch_size, self.fc3.out_features, device=x.device)
            #mem4 = torch.zeros(batch_size, self.fc4.out_features, device=x.device)


            for step in range(inputs.shape[1]):
                mem4 = model(inputs[:,step,:])
                # Stack recordings
                #spk1_rec = torch.stack(spk1_rec, dim=0)
                #spk2_rec = torch.stack(spk2_rec, dim=0)
                #spk3_rec = torch.stack(spk3_rec, dim=0)
                #spk4_rec = torch.stack(spk4_rec, dim=0)

                #mem1_rec = torch.stack(mem1_rec, dim=0)
                #mem2_rec = torch.stack(mem2_rec, dim=0)
                #mem3_rec = torch.stack(mem3_rec, dim=0)
                mem4_rec.append(mem4)
            mem4_rec = torch.stack(mem4_rec, dim=0).to(device)

            # Recordings for spikes and membrane potentials
            #spk1_rec, spk2_rec, spk3_rec, spk4_rec = [], [], [], []
            #mem1_rec, mem2_rec, mem3_rec, mem4_rec = [], [], [], []
            # Forward pass
            #mem4_rec = model(inputs)

            # Compute the loss
            true_labels = labels.argmax(dim=1)
            loss = loss_fn(mem4_rec, true_labels)
            running_loss += loss.item()

            # For accuracy, compare the final membrane potentials
            final_membrane_potentials = mem4_rec[-1]

            # Predicted labels
            predicted = final_membrane_potentials.argmax(dim=1)

            total += labels.size(0)
            correct += predicted.eq(true_labels).sum().item()

            all_labels.extend(true_labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            # if (batch_idx > 20):# & (args.nni_opt):
            #     break
    accuracy = 100.0 * correct / total
    avg_loss = running_loss / len(val_loader)

    # Compute additional metrics
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_predictions)

    return avg_loss, accuracy, precision, recall, f1, cm


def test_model(model, test_loader, device, args):
    """Test the model."""
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing", leave=False, disable=args.stop_tqdm):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.to(torch.float32)
            mem4_rec = []
            model.reset(inputs.shape[0])

            for step in range(inputs.shape[1]):
                mem4 = model(inputs[:,step,:])
                mem4_rec.append(mem4)
            mem4_rec = torch.stack(mem4_rec, dim=0)

            # For accuracy, compare the final membrane potentials
            final_membrane_potentials = mem4_rec[-1]

            # Predicted labels
            predicted = final_membrane_potentials.argmax(dim=1)

            total += labels.size(0)
            true_labels = labels.argmax(dim=1)
            correct += predicted.eq(true_labels).sum().item()
            all_labels.extend(true_labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100.0 * correct / total

    # Compute additional metrics
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_predictions)

    return accuracy, precision, recall, f1, cm


def main():
    config = Config()

    

    ap = argparse.ArgumentParser(description="Spiking Neural Network for QUT dataset")
    ap.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    ap.add_argument("--weights_1", type=float, default=config.weights_1, help="Initial weights for layer 1")
    ap.add_argument("--weights_2", type=float, default=config.weights_2, help="Initial weights for layer 2")
    ap.add_argument("--weights_3", type=float, default=config.weights_3, help="Initial weights for layer 3")
    ap.add_argument("--weights_4", type=float, default=config.weights_4, help="Initial weights for layer 4")
    ap.add_argument("--nni_opt", action='store_true', help="Use NNI optimization")
    #ap.add_argument("--batch_size", type=int, default=16, help="Batch size")
    ap.add_argument("--stop_tqdm", action='store_true', help="Stop tqdm progress bar")

    args = ap.parse_args()

    if args.nni_opt:
        #nni.get_next_parameter()
        PARAMS = nni.get_next_parameters()
    ## convert PARAMS to args
        args.lr = PARAMS['lr']
        args.weights_1 = PARAMS['weights_1']
        args.weights_2 = PARAMS['weights_2']
        args.weights_3 = PARAMS['weights_3']
        args.weights_4 = PARAMS['weights_4']
    # Create a unique folder for the experiment
    experiment_name = "SNNQUT_experiment"
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
    folder_name = os.path.join(config.results_dir, f"{experiment_name}_{current_time}")
    os.makedirs(folder_name, exist_ok=True)
    logger.info(f"Results will be saved to {folder_name}")

    # Save hyperparameters to a JSON file
    hyperparams_file = os.path.join(folder_name, 'hyperparameters.json')
    with open(hyperparams_file, 'w') as f:
        json.dump(config.__dict__, f, indent=4)
    logger.info(f"Hyperparameters saved to {hyperparams_file}")



    # Load the dataset
    dataset = QUTDataset(config.data_dir)

    # Split the dataset into training, validation, and testing sets
    train_idx, temp_idx = train_test_split(
        range(len(dataset)), test_size=0.3, random_state=seed, shuffle=True
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=seed, shuffle=True
    )

    # Create datasets
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    # Create data loaders
    CPU_workers = os.cpu_count() // 2
    train_loader, val_loader, test_loader = get_dataloader(train_dataset, val_dataset, test_dataset, batch_size=config.batch_size, num_workers=CPU_workers)

    size = config.hidden_size
    n_values = [2, 4, 8]
    tau_hidden = [create_power_vector(n, size) for n in n_values]
    tau_output = np.repeat(1e-3, config.output_size)

    delta_t = 1  # 1ms time step

    beta_hidden = [
        torch.exp(-torch.tensor(delta_t, dtype=torch.float32) / torch.tensor(tau, dtype=torch.float32))
        for tau in tau_hidden
    ]
    beta_output = torch.exp(
        -torch.tensor(delta_t, dtype=torch.float32) / torch.tensor(tau_output, dtype=torch.float32)
    )

    print('args.weights_1:',args.weights_1)
    print('args.weights_2:',args.weights_2)
    print('args.weights_3:',args.weights_3)
    print('args.weights_4:',args.weights_4)

    config.learning_rate = args.lr
    config.weights_1 = args.weights_1
    config.weights_2 = args.weights_2
    config.weights_3 = args.weights_3
    config.weights_4 = args.weights_4
    #config.batch_size = args.batch_size
    


    """Main function to train and evaluate the model."""

    # Initialize lists to store metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1s = []

    # Create a TensorBoard writer
    writer = SummaryWriter(log_dir=folder_name)

    # Get a sample batch to determine time_steps
    sample_inputs, _ = next(iter(train_loader))
    _, time_steps, _ = sample_inputs.shape

    # Initialize the model
    model = SNNQUT(
        config=config,
        beta_hidden=beta_hidden,
        beta_output=beta_output,
        time_steps=time_steps
    ).to(device)

    print('model.weights_1:',model.weights_1)
    print('model.weights_2:',model.weights_2)
    print('model.weights_3:',model.weights_3)
    print('model.weights_4:',model.weights_4)


    #model = torch.compile(model)
    #model = torch.compile(model, backend='aot_eager')


    # Log the model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(model)
    logger.info(f"Total parameters: {total_params}")
    logger.info(f"Trainable parameters: {trainable_params}")
    logger.info(f"Time steps: {time_steps}")

    # Define optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, betas=config.optimizer_betas, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.scheduler_T_max, eta_min=config.scheduler_eta_min
    )

    # Early stopping parameters
    best_val_loss = np.inf
    patience_counter = 0

    # Training loop
    for epoch in range(config.num_epochs):
        # Training
        train_loss, train_acc = train_model(
            model, train_loader, optimizer, device, config.l1_lambda, writer, epoch, args
        )
        # Validation
        val_loss, val_acc, val_prec, val_rec, val_f1, _ = validate_model(
            model, val_loader, device, writer, epoch, args
        )

        if args.nni_opt:
            nni.report_intermediate_result(val_loss)

        # Scheduler step
        scheduler.step()

        # Early stopping
        if val_loss < best_val_loss - config.delta:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info(f'Early stopping at epoch {epoch + 1}')
                break

        logger.info(f'Epoch [{epoch + 1}/{config.num_epochs}], '
                    f'Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.2f}%, '
                    f'Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.2f}%')

        # Record metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_precisions.append(val_prec)
        val_recalls.append(val_rec)
        val_f1s.append(val_f1)
    if args.nni_opt:
        nni.report_final_result(val_loss)

    # Load the best model
    model.load_state_dict(best_model_state)

    if not args.nni_opt:
        # Evaluation on test set
        test_accuracy, test_precision, test_recall, test_f1, test_cm = test_model(model, test_loader, device)
        logger.info(f'Test Accuracy: {test_accuracy:.2f}%')
        logger.info(f'Precision: {test_precision:.2f}, Recall: {test_recall:.2f}, F1 Score: {test_f1:.2f}')
        logger.info(f'Confusion Matrix:\n{test_cm}')


    # Save the model's state_dict instead of the entire model
    model_file = os.path.join(folder_name, 'trained_model.pth')
    torch.save(model.state_dict(), model_file)
    logger.info(f"Model saved to {model_file}")

    # Close the TensorBoard writer
    writer.close()


if __name__ == "__main__":
    main()
