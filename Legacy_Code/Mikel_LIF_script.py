import os
import random

#import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
#import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
#from pytorch_lightning.loggers import TensorBoardLogger
from snntorch import surrogate
import snntorch as snn
from torch.utils.data import DataLoader, Dataset, random_split

import nni
from QUT_DataLoader import QUTDataset

# Hyperparameters and Constants
input_size = 16
hidden_size = 24
output_size = 4
num_epochs = 2
batch_size = 32
scheduler_step_size = 20
scheduler_gamma = 0.2
num_workers = max(1, os.cpu_count() - os.cpu_count()//4)
cuba_tau = 2  # 2ms of tau_Vmem of the CUBA neuron
hidden_reset_mechanism = 'subtract'
output_reset_mechanism = 'none'
output_threshold = 10000
hid_threshold = 1

# Function to Get Hyperparameters from NNI
def get_nni_params():
    # Default hyperparameters
    params = {
        'learning_rate': 0.00001,
        'optimizer_betas': (0.9, 0.95),
        'fast_sigmoid_slope': 20,
    }
    # Update with parameters from NNI
    tuner_params = nni.get_next_parameter()
    params.update(tuner_params)
    return params

# Spiking Neural Network Model
class SNNQUT(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        beta_hidden_1,
        beta_hidden_2,
        beta_hidden_3,
        beta_output,
        cuba_beta,
        hidden_reset_mechanism,
        output_reset_mechanism,
        output_threshold,
        hid_threshold,
        fast_sigmoid_slope,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.lif1 = snn.Leaky(
            beta=beta_hidden_1, reset_mechanism=hidden_reset_mechanism
        )
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lif2 = snn.Leaky(
            beta=beta_hidden_2, reset_mechanism=hidden_reset_mechanism
        )
        self.fc3 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lif3 = snn.Leaky(
            beta=beta_hidden_3, reset_mechanism=hidden_reset_mechanism
        )
        self.fc4 = nn.Linear(hidden_size, output_size, bias=False)
        self.lif4 = snn.Leaky(
            beta=beta_output,
            reset_mechanism=output_reset_mechanism,
            threshold=1e7,
        )

        self._initialize_weights()


    def _initialize_weights(self):
        nn.init.uniform_(self.fc1.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.fc2.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.fc3.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.fc4.weight, a=-0.1, b=0.1)


    def forward(self, x):
        x = x.to(torch.float32)  # Convert input to float32
        batch_size, time_steps, _ = x.shape

        # Initialization of membrane potentials
        mem1 = torch.zeros(batch_size, self.fc1.out_features, device=x.device)
        mem2 = torch.zeros(batch_size, self.fc2.out_features, device=x.device)
        mem3 = torch.zeros(batch_size, self.fc3.out_features, device=x.device)
        mem4 = torch.zeros(batch_size, self.fc4.out_features, device=x.device)

        mem4_rec = []
        spk3_rec = []

        for step in range(time_steps):
            cur1 = self.fc1(x[:, step, :])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            cur4 = self.fc4(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)

            # Record at every time step
            mem4_rec.append(mem4)
            spk3_rec.append(spk3)

        # Stack along the time axis (first dimension)
        return torch.stack(mem4_rec, dim=0), torch.stack(spk3_rec, dim=0)

# Lightning Module
class Lightning_SNNQUT(pl.LightningModule):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        beta_hidden_1,
        beta_hidden_2,
        beta_hidden_3,
        beta_output,
        hidden_reset_mechanism,
        output_reset_mechanism,
        hid_threshold,
        output_threshold,
        cuba_beta,
        learning_rate,
        optimizer_betas,
        fast_sigmoid_slope,
    ):
        super().__init__()
        self.save_hyperparameters(
            'input_size',
            'hidden_size',
            'output_size',
            'hidden_reset_mechanism',
            'output_reset_mechanism',
            'hid_threshold',
            'output_threshold',
            'learning_rate',
            'optimizer_betas',
            'fast_sigmoid_slope',
        )
        # Assign tensors directly since using self.save_hyperparameters() gives error because beta tensors are not JSON serializable:
        # - they are torch tensors, can't convert to JSON
        self.beta_hidden_1 = beta_hidden_1
        self.beta_hidden_2 = beta_hidden_2
        self.beta_hidden_3 = beta_hidden_3
        self.beta_output = beta_output
        self.cuba_beta = cuba_beta

        # Initialize the SNN model
        self.model = SNNQUT(
            input_size=self.hparams.input_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.output_size,
            beta_hidden_1=self.beta_hidden_1,
            beta_hidden_2=self.beta_hidden_2,
            beta_hidden_3=self.beta_hidden_3,
            beta_output=self.beta_output,
            cuba_beta=self.cuba_beta,
            hidden_reset_mechanism=self.hparams.hidden_reset_mechanism,
            output_reset_mechanism=self.hparams.output_reset_mechanism,
            output_threshold=self.hparams.output_threshold,
            hid_threshold=self.hparams.hid_threshold,
            fast_sigmoid_slope=self.hparams.fast_sigmoid_slope,
        )

        # Initialize the loss function
        self.loss_function = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        mem4_rec, spk3_rec = self(inputs)

        # print spk3_rec
        print(f"Sum of spk3_rec: {spk3_rec.sum()}")


        # Expanding labels to match mem4_rec's shape
        labels_expanded = labels.unsqueeze(0).expand(mem4_rec.size(0), -1, -1)

        # Calculate loss
        loss = self.loss_function(mem4_rec, (labels_expanded * 5) + 5)

        # Use the final membrane potential for prediction
        final_mem4 = mem4_rec.sum(0)

        # Predicted class is the one with the highest membrane potential
        _, predicted = final_mem4.max(-1)
        _, targets = labels.max(-1)

        # Calculate accuracy
        correct = predicted.eq(targets).sum().item()
        total = targets.numel()
        accuracy = correct / total

        # Log training loss and accuracy
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', accuracy * 100, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        mem4_rec, _ = self(inputs)

        # Expanding labels to match mem4_rec's shape
        labels_expanded = labels.unsqueeze(0).expand(mem4_rec.size(0), -1, -1)

        # Calculate loss
        loss = self.loss_function(mem4_rec, (labels_expanded * 5) + 5)

        # Use the final membrane potential for prediction
        final_mem4 = mem4_rec.sum(0)

        # Predicted class is the one with the highest membrane potential
        _, predicted = final_mem4.max(-1)
        _, targets = labels.max(-1)

        # Calculate accuracy
        correct = predicted.eq(targets).sum().item()
        total = targets.numel()
        accuracy = correct / total

        # Log validation loss and accuracy
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy * 100, on_step=False, on_epoch=True, prog_bar=True)

        return {'val_loss': loss, 'val_accuracy': accuracy}

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        mem4_rec, _ = self(inputs)

        # Expanding labels to match mem4_rec's shape
        labels_expanded = labels.unsqueeze(0).expand(mem4_rec.size(0), -1, -1)

        # Calculate loss
        loss = self.loss_function(mem4_rec, (labels_expanded * 5) + 5)

        # Use the final membrane potential for prediction
        final_mem4 = mem4_rec.sum(0)

        # Predicted class is the one with the highest membrane potential
        _, predicted = final_mem4.max(-1)
        _, targets = labels.max(-1)

        # Calculate accuracy
        correct = predicted.eq(targets).sum().item()
        total = targets.numel()
        accuracy = correct / total

        # Log test loss and accuracy
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('test_accuracy', accuracy * 100, on_step=True, on_epoch=True, prog_bar=True)

        return {'test_loss': loss, 'test_accuracy': accuracy}

    def configure_optimizers(self):
        # optimizer = optim.Adam(
        #     self.parameters(),
        #     lr=self.hparams.learning_rate,
        #     betas=self.hparams.optimizer_betas,
        # )

        optimizer = optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
        )

        # Uncomment the scheduler if you wish to use it
        # scheduler = optim.lr_scheduler.StepLR(
        #     optimizer,
        #     step_size=self.hparams.scheduler_step_size,
        #     gamma=self.hparams.scheduler_gamma,
        # )
        return [optimizer]  # , [scheduler]

# Data Module
class QUTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        dataset = QUTDataset(self.data_dir)
        train_size = int(0.65 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

# Function to Generate Tau and Beta Values
def generate_tau_beta_values(hidden_size, output_size, cuba_tau):

    def create_power_vector(n, size):
        # Generate the powers of 2 up to 2^n
        powers = [2 ** i for i in range(1, n + 1)]
        # Calculate how many times each power should be repeated
        repeat_count = size // n
        # Create the final vector by repeating each power equally
        power_vector = np.repeat(powers, repeat_count)
        return power_vector

    # Generate Tau Values
    size = hidden_size
    tau_hidden_1 = create_power_vector(n=2, size=size)
    tau_hidden_2 = create_power_vector(n=4, size=size)
    tau_hidden_3 = create_power_vector(n=8, size=size)

    # Generate Beta Values from Tau
    delta_t = 1  # 1ms time step

    beta_hidden_1 = torch.exp(-torch.tensor(delta_t) / torch.tensor(tau_hidden_1, dtype=torch.float32))
    beta_hidden_2 = torch.exp(-torch.tensor(delta_t) / torch.tensor(tau_hidden_2, dtype=torch.float32))
    beta_hidden_3 = torch.exp(-torch.tensor(delta_t) / torch.tensor(tau_hidden_3, dtype=torch.float32))

    tau_output = np.repeat(10, output_size)
    beta_output = torch.exp(-torch.tensor(delta_t) / torch.tensor(tau_output, dtype=torch.float32))

    cuba_beta = torch.exp(-torch.tensor(delta_t) / torch.tensor(cuba_tau, dtype=torch.float32))

    # Return all beta values
    return beta_hidden_1, beta_hidden_2, beta_hidden_3, beta_output, cuba_beta

# Main Function
def main():
    # Get hyperparameters from NNI
    params = get_nni_params()
    beta_hidden_1, beta_hidden_2, beta_hidden_3, beta_output, cuba_beta = generate_tau_beta_values(hidden_size, output_size, cuba_tau)


    # Set random seeds for reproducibility
    pl.seed_everything(42)

    # Initialize the data module
    data_dir = 'data/TEST'
    data_module = QUTDataModule(
        data_dir, batch_size=batch_size, num_workers=num_workers
    )

    # Initialize the Lightning model with hyperparameters from NNI
    model = Lightning_SNNQUT(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        beta_hidden_1=beta_hidden_1,
        beta_hidden_2=beta_hidden_2,
        beta_hidden_3=beta_hidden_3,
        beta_output=beta_output,
        hidden_reset_mechanism=hidden_reset_mechanism,
        output_reset_mechanism=output_reset_mechanism,
        learning_rate=params['learning_rate'],
        #optimizer_betas=tuple(params['optimizer_betas']),
        optimizer_betas=params['optimizer_betas'],
        #scheduler_step_size=scheduler_step_size,
        #scheduler_gamma=scheduler_gamma,
        output_threshold=output_threshold,
        cuba_beta=cuba_beta,
        hid_threshold=hid_threshold,
        fast_sigmoid_slope=params['fast_sigmoid_slope'],
    )

    # Initialize the Trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        # devices and accelerator settings
        #accelerator='gpu' if torch.cuda.is_available() else 'cpu', devices='auto',
    )

    # Start training
    trainer.fit(model, datamodule=data_module)

    # Validate the model
    trainer.validate(model, datamodule=data_module)
    # Access the overall validation accuracy
    val_accuracy = trainer.callback_metrics['val_accuracy'].item()

    # Report the result to NNI
    nni.report_final_result(val_accuracy)

if __name__ == '__main__':
    main()