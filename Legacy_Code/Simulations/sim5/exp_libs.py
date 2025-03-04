import os, torch, sys
import numpy as np
from ray import tune, train
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import snntorch as snn
from torch.utils.data import Dataset


def compute_loss_and_metrics(
    spk_rec, mem_rec, labels, loss_function, loss_mode="Vmem", loss_type="MSE"
):
    """
    Compute loss and accuracy metrics for the network.

    Parameters:
    spk_rec: spike recordings from output layer
    mem_rec: membrane potential recordings from output layer
    labels: target labels
    loss_function: pytorch loss function to apply to spk or mem.
    loss_mode: "Vmem" for membrane potential or "Spk" for spikes
    loss_type: "MSE" or "CrossEntropy"
    """

    if loss_type == "MSE":
        # Original MSE loss computation
        labels_expanded = labels.unsqueeze(0).expand(mem_rec.size(0), -1, -1)
        loss = loss_function(mem_rec, (labels_expanded + 0.2))
    else:  # CrossEntropy
        # Sum activity over time steps
        if loss_mode == "Vmem":
            activity = mem_rec.sum(0)
        else:
            activity = spk_rec.sum(0)

        # Get target indices
        _, targets = labels.max(-1) # Convert one-hot to indices

        # Use the predefined CrossEntropyLoss
        main_loss = loss_function(activity, targets)

        # Add firing rate regularization
        time_steps = spk_rec.size(0)
        avg_firing_rate = activity / time_steps
        rate_reg = ((avg_firing_rate - 0.5).abs()).mean()  # Encourage ~50% firing rate

        # Scale regularization based on main loss magnitude
        with torch.no_grad():
            reg_scale = main_loss.detach() / (rate_reg.detach() + 1e-8)

        loss = main_loss + 0.1 * reg_scale * rate_reg  # Adjust coefficient as needed

    # Compute accuracy metrics
    if loss_mode == "Vmem":
        final_out = mem_rec.sum(0)
    else:
        final_out = spk_rec.sum(0)

    _, predicted = final_out.max(-1)
    _, targets = labels.max(-1)

    # Calculate accuracy
    correct = predicted.eq(targets).sum().item()
    total = targets.numel()
    accuracy = correct / total if total > 0 else 0.0

    return loss, accuracy, predicted, targets


class AudioDataset_Old_Filename(Dataset):
    """
    Label extraction differs from AudioDataset_Old_Filename to AudioDataset.
    For filenames like "CAFE-segments_1s", "CAR-segments_1s", "HOME-segments_1s", "STREET-segments_1s" use this.
    For filenames like "0_CAFE-segments_1s", "1_CAR-segments_1s", "2_HOME-segments_1s", "3_STREET-segments_1s" use AudioDataset.

    Dataset class for loading audio data organized in class-specific folders.
    Dataset Directory Structure:
    root_folder/
        class1/
            file1.csv
            file2.csv
        class2/
            file3.csv
            file4.csv
    """

    def __init__(self, root_folder, split="train", val_split=0.2, random_state=42):
        assert split in [
            "train",
            "val",
            "test",
        ], f"Split {split} not recognized. Use 'train', 'val', or 'test'"
        self.files = []
        self.labels = []
        self.class_to_idx = {}
        self.split = split

        # Get all class folders
        class_folders = [
            d
            for d in os.listdir(root_folder)
            if os.path.isdir(os.path.join(root_folder, d))
        ]

        # Save classes number
        self.n_classes = len(class_folders)

        # Create class mapping
        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(sorted(class_folders))
        }

        # Collect all files and labels
        all_files = []
        all_labels = []
        for class_name in class_folders:
            class_path = os.path.join(root_folder, class_name)
            class_files = [
                os.path.join(class_path, f)
                for f in os.listdir(class_path)
                if f.endswith(".csv") and not f.startswith(".")
            ]
            all_files.extend(class_files)
            all_labels.extend([self.class_to_idx[class_name]] * len(class_files))

        # Create a deterministic split
        generator = torch.Generator().manual_seed(random_state)
        indices = torch.randperm(len(all_files), generator=generator)

        # Calculate split sizes
        val_size = int(len(indices) * val_split)
        train_size = len(indices) - val_size

        # Select appropriate indices based on split
        if split == "train":
            split_indices = indices[:train_size]
        elif split == "val":
            split_indices = indices[train_size:]
        elif split == "test":
            split_indices = indices

        # Use the selected indices to populate the dataset
        self.files = [all_files[i] for i in split_indices]
        self.labels = [all_labels[i] for i in split_indices]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        label = torch.tensor(label)
        label_one_hot = torch.nn.functional.one_hot(label, num_classes=self.n_classes)

        data = np.loadtxt(file_path, delimiter=",", dtype=np.float32, encoding="latin1")
        data = data[:, np.newaxis]  # Add channel dimension
        data = torch.tensor(data, dtype=torch.float32)
        return data, label_one_hot, file_path


class AudioDataset(Dataset):
    """
    Label extraction differs from AudioDataset_Old_Filename to AudioDataset.
    For filenames like "CAFE-segments_1s", "CAR-segments_1s", "HOME-segments_1s", "STREET-segments_1s" use this.
    For filenames like "0_CAFE-segments_1s", "1_CAR-segments_1s", "2_HOME-segments_1s", "3_STREET-segments_1s" use AudioDataset.

    Dataset for loading data organized in folders named:
      "0_CAFE-segments_1s", "1_CAR-segments_1s", "2_HOME-segments_1s", "3_STREET-segments_1s"
    The class label is the integer before underscore (0 -> CAFE, 1 -> CAR, 2 -> HOME, 3 -> STREET).
    """

    def __init__(self, root_folder, split="train", val_split=0.2, random_state=42):
        assert split in ["train", "val", "test"], f"Split {split} not recognized."
        self.split = split

        class_folders = [
            d
            for d in os.listdir(root_folder)
            if os.path.isdir(os.path.join(root_folder, d))
        ]

        def get_int_label(folder_name):
            return int(folder_name.split("_")[0])

        all_files = []
        all_labels_int = []
        for folder_name in class_folders:
            label_int = get_int_label(folder_name)
            folder_path = os.path.join(root_folder, folder_name)
            csv_files = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.endswith(".csv") and not f.startswith(".")
            ]
            all_files.extend(csv_files)
            all_labels_int.extend([label_int] * len(csv_files))

        self.n_classes = 4
        generator = torch.Generator().manual_seed(random_state)
        indices = torch.randperm(len(all_files), generator=generator)

        val_size = int(len(indices) * val_split)
        train_size = len(indices) - val_size

        if split == "train":
            split_indices = indices[:train_size]
        elif split == "val":
            split_indices = indices[train_size:]
        else:  # 'test'
            split_indices = indices

        self.files = [all_files[i] for i in split_indices]
        self.labels_int = [all_labels_int[i] for i in split_indices]

        # Verify no overlap between splits
        if split == "val":
            train_files = {all_files[i] for i in indices[:train_size]}
            val_files = set(self.files)
            assert (
                len(train_files.intersection(val_files)) == 0
            ), "Data leakage detected between train and val sets!"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        int_label = self.labels_int[idx]  # e.g. 0, 1, 2, 3
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(int_label), num_classes=self.n_classes
        )

        # The CSV has 1000 rows and each row has 16 floats separated by commas
        data = np.loadtxt(file_path, delimiter=",", dtype=np.float32, encoding="latin1")

        # e.g. if it was a single column with 16 numbers per row, ensure shape is (1000, 16).
        data = data.reshape(-1, 16)
        data_tensor = torch.tensor(data, dtype=torch.float32)

        return data_tensor, one_hot_label, file_path

class CochleaDataset(Dataset):
    """
    Dataset class for loading neural response data organized in class-specific folders.
    Similar structure to AudioDataset but handles multi-neuron responses.

    EXAMPLE USAGE
    response_folder = "saved_nets/feedforward1/responses_1ms"
    neural_dataset = NeuralResponseDataset(response_folder, split="train")
    neural_loader = DataLoader(
        neural_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers_loader
    )

    # The loader will return:
    # - data of shape (batch_size, time_bins, n_neurons)
    # - labels
    # - file paths
    """

    def __init__(self, root_folder, split="train", val_split=0.2, random_state=42):
        assert split in [
            "train",
            "val",
            "test",
        ], f"Split {split} not recognized. Use 'train', 'val', or 'test' "
        self.files = []
        self.labels = []
        self.class_to_idx = {}
        self.split = split

        # Get all class folders
        class_folders = [
            d
            for d in os.listdir(root_folder)
            if os.path.isdir(os.path.join(root_folder, d))
        ]

        # Save classes number
        self.n_classes = len(class_folders)

        # Create class mapping
        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(sorted(class_folders))
        }

        # Collect all files and labels
        all_files = []
        all_labels = []
        for class_name in class_folders:
            class_path = os.path.join(root_folder, class_name)
            class_files = [
                os.path.join(class_path, f)
                for f in os.listdir(class_path)
                if f.endswith(".csv") and not f.startswith(".")
            ]
            all_files.extend(class_files)
            all_labels.extend([self.class_to_idx[class_name]] * len(class_files))

        # Create a deterministic split
        generator = torch.Generator().manual_seed(random_state)
        indices = torch.randperm(len(all_files), generator=generator)

        # Calculate split sizes
        val_size = int(len(indices) * val_split)
        train_size = len(indices) - val_size

        # Select appropriate indices based on split
        if split == "train":
            split_indices = indices[:train_size]
        elif split == "val":
            split_indices = indices[train_size:]
        elif split == "test":
            split_indices = indices

        # Use the selected indices to populate the dataset
        self.files = [all_files[i] for i in split_indices]
        self.labels = [all_labels[i] for i in split_indices]

        # Verify no overlap between splits
        if split == "val":
            train_files = {all_files[i] for i in indices[:train_size]}
            val_files = set(self.files)
            assert (
                len(train_files.intersection(val_files)) == 0
            ), "Data leakage detected between train and val sets!"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        label = torch.tensor(label)
        label_one_hot = torch.nn.functional.one_hot(label, num_classes=self.n_classes)

        # Load neural response data
        # Each row is a time bin, each column is a neuron
        data = np.loadtxt(file_path, delimiter=",", dtype=np.float32, encoding="latin1")

        # Reshape to (time_bins, n_neurons)
        # No need to add extra dimension since we already have multiple neurons
        data = data.reshape(data.shape[0], -1)  # Force 2D shape in case single neuron

        # Convert to tensor with shape (time_bins, n_neurons)
        data_tensor = torch.tensor(data, dtype=torch.float32)

        return data_tensor, label_one_hot, file_path


class FakeQuantize(nn.Module):
    def __init__(self, bits, w_min=0.001, w_max=1.0, quantize=True):
        super().__init__()
        self.bits = bits
        self.levels = 2**bits

        self.register_buffer("w_min", torch.tensor(w_min))
        self.register_buffer("w_max", torch.tensor(w_max))
        self.quantize = quantize

        if w_min >= w_max:
            raise ValueError(f"w_min ({w_min}) must be < w_max ({w_max})")
        if bits <= 0:
            raise ValueError(f"bits must be positive, got {bits}")

    def forward(self, input):
        if not torch.is_tensor(input):
            raise TypeError(f"Input must be a tensor, got {type(input)}")

        if self.quantize is False:
            return input

        input_clamped = torch.clamp(input, self.w_min, self.w_max)
        scale = (self.w_max - self.w_min) / (self.levels - 1)
        quant_indices = torch.round((input_clamped - self.w_min) / scale)
        quant_values = quant_indices * scale + self.w_min

        if self.training:
            # Straight-through estimator style
            return input_clamped + (quant_values - input_clamped).detach()
        return quant_values

    def extra_repr(self) -> str:
        return f"bits={self.bits}, w_min={self.w_min.item():.4f}, w_max={self.w_max.item():.4f}"


class QuantLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bits,
        bias=False,
        w_min=0.001,
        w_max=1.0,
        quantize=True,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.fake_quant = FakeQuantize(bits=bits, w_min=w_min, w_max=w_max, quantize=quantize)
        torch.nn.init.uniform_(self.weight.data, a=w_min, b=w_max)

    def forward(self, input):
        quant_weight = self.fake_quant(self.weight)
        return F.linear(input, quant_weight, self.bias)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, bits={self.fake_quant.bits}, quantize={self.fake_quant.quantize}"


class SNNQUT_with_clochlea(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        cochlea_betas,
        beta_hidden_1,
        beta_hidden_2,
        beta_hidden_3,
        beta_output,
        hidden_reset_mechanism,
        output_reset_mechanism,
        hidden_threshold,
        output_threshold,
        fast_sigmoid_slope,
    ):
        super().__init__()

        self.register_buffer("cochlea_betas", cochlea_betas)
        self.register_buffer("beta_hidden_1", beta_hidden_1)
        self.register_buffer("beta_hidden_2", beta_hidden_2)
        self.register_buffer("beta_hidden_3", beta_hidden_3)
        self.register_buffer("beta_output", beta_output)
        self.register_buffer("hidden_threshold", torch.tensor(hidden_threshold))
        self.register_buffer("output_threshold", torch.tensor(output_threshold))
        self.register_buffer("fast_sigmoid_slope", torch.tensor(fast_sigmoid_slope))

        # Cochlea layer
        self.ch_fc = nn.Linear(1, input_size, bias=False)
        self.ch_lif = snn.Leaky(
            beta=cochlea_betas,
            reset_mechanism=hidden_reset_mechanism,
            threshold=1,
            spike_grad=snn.surrogate.fast_sigmoid(slope=fast_sigmoid_slope),
        )

        self.fc1 = QuantLinear(
            input_size,
            hidden_size,
            bias=False,
            bits=5,
            w_min=0.001,
            w_max=1.0,
            quantize=True,
        )
        self.lif1 = snn.Leaky(
            beta=beta_hidden_1,
            reset_mechanism=hidden_reset_mechanism,
            threshold=hidden_threshold,
            spike_grad=snn.surrogate.fast_sigmoid(slope=fast_sigmoid_slope),
        )

        self.fc2 = QuantLinear(
            hidden_size,
            hidden_size,
            bias=False,
            bits=5,
            w_min=0.001,
            w_max=1.0,
            quantize=True,
        )
        self.lif2 = snn.Leaky(
            beta=beta_hidden_2,
            reset_mechanism=hidden_reset_mechanism,
            threshold=hidden_threshold,
            spike_grad=snn.surrogate.fast_sigmoid(slope=fast_sigmoid_slope),
        )

        self.fc3 = QuantLinear(
            hidden_size,
            hidden_size,
            bias=False,
            bits=5,
            w_min=0.001,
            w_max=1.0,
            quantize=True,
        )
        self.lif3 = snn.Leaky(
            beta=beta_hidden_3,
            reset_mechanism=hidden_reset_mechanism,
            threshold=hidden_threshold,
            spike_grad=snn.surrogate.fast_sigmoid(slope=fast_sigmoid_slope),
        )

        self.fc4 = QuantLinear(
            hidden_size,
            output_size,
            bias=False,
            bits=5,
            w_min=0.001,
            w_max=1.0,
            quantize=True,
        )
        self.lif4 = snn.Leaky(
            beta=beta_output,
            reset_mechanism=output_reset_mechanism,
            threshold=output_threshold,
            spike_grad=snn.surrogate.fast_sigmoid(slope=fast_sigmoid_slope),
        )

    def forward(self, x):
        batch_size, time_steps, _ = x.shape
        device = x.device

        ch_mem = torch.zeros(batch_size, self.ch_fc.out_features, device=device)
        mem1 = torch.zeros(batch_size, self.fc1.out_features, device=device)
        mem2 = torch.zeros(batch_size, self.fc2.out_features, device=device)
        mem3 = torch.zeros(batch_size, self.fc3.out_features, device=device)
        mem4 = torch.zeros(batch_size, self.fc4.out_features, device=device)

        spk1_rec, spk2_rec, spk3_rec, spk4_rec, mem4_rec = [], [], [], [], []

        for step in range(time_steps):

            input_t = x[:, step, :]
            ch_cur = self.ch_fc(input_t)
            ch_spk, ch_mem = self.ch_lif(ch_cur, ch_mem)
            cur1 = self.fc1(ch_spk)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            cur4 = self.fc4(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)

            spk1_rec.append(spk1)
            spk2_rec.append(spk2)
            spk3_rec.append(spk3)
            spk4_rec.append(spk4)
            mem4_rec.append(mem4)

        return (
            torch.stack(spk1_rec, dim=0),
            torch.stack(spk2_rec, dim=0),
            torch.stack(spk3_rec, dim=0),
            torch.stack(spk4_rec, dim=0),
            torch.stack(mem4_rec, dim=0),
        )


class SNNQUT_filter_bank(nn.Module):
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
        hidden_threshold,
        output_threshold,
        fast_sigmoid_slope,
        bits,
    ):
        super().__init__()

        # Store the betas and thresholds as buffers
        self.register_buffer("beta_hidden_1", beta_hidden_1)
        self.register_buffer("beta_hidden_2", beta_hidden_2)
        self.register_buffer("beta_hidden_3", beta_hidden_3)
        self.register_buffer("beta_output", beta_output)
        self.register_buffer("hidden_threshold", torch.tensor(hidden_threshold))
        self.register_buffer("output_threshold", torch.tensor(output_threshold))
        self.register_buffer("fast_sigmoid_slope", torch.tensor(fast_sigmoid_slope))

        self.fc1 = QuantLinear(input_size, hidden_size, w_min=-0.5, w_max=0.5, bits=bits, bias=False)
        self.lif1 = snn.Leaky(
            beta=beta_hidden_1,
            reset_mechanism=hidden_reset_mechanism,
            threshold=hidden_threshold,
            spike_grad=snn.surrogate.fast_sigmoid(slope=fast_sigmoid_slope),
        )

        self.fc2 = QuantLinear(hidden_size, hidden_size, bits=bits, bias=False)
        self.lif2 = snn.Leaky(
            beta=beta_hidden_2,
            reset_mechanism=hidden_reset_mechanism,
            threshold=hidden_threshold,
            spike_grad=snn.surrogate.fast_sigmoid(slope=fast_sigmoid_slope),
        )

        self.fc3 = QuantLinear(hidden_size, hidden_size, bits=bits, bias=False)
        self.lif3 = snn.Leaky(
            beta=beta_hidden_3,
            reset_mechanism=hidden_reset_mechanism,
            threshold=hidden_threshold,
            spike_grad=snn.surrogate.fast_sigmoid(slope=fast_sigmoid_slope),
        )

        self.fc4 = QuantLinear(hidden_size, output_size, bits=bits, bias=False)
        self.lif4 = snn.Leaky(
            beta=beta_output,
            reset_mechanism=output_reset_mechanism,
            threshold=output_threshold,
            spike_grad=snn.surrogate.fast_sigmoid(slope=fast_sigmoid_slope),
        )

    def forward(self, x):
        # x is shape (batch_size, time_steps, 16)
        batch_size, time_steps, _ = x.shape
        device = x.device

        mem1 = torch.zeros(batch_size, self.fc1.out_features, device=device)
        mem2 = torch.zeros(batch_size, self.fc2.out_features, device=device)
        mem3 = torch.zeros(batch_size, self.fc3.out_features, device=device)
        mem4 = torch.zeros(batch_size, self.fc4.out_features, device=device)

        spk1_rec, spk2_rec, spk3_rec, spk4_rec, mem4_rec = [], [], [], [], []

        for step in range(time_steps):
            input_t = x[:, step, :]
            spk1, mem1 = self.lif1(self.fc1(input_t/15), mem1) # Normalize input to 0-1 range, before it was 0-15
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            spk3, mem3 = self.lif3(self.fc3(spk2), mem3)
            spk4, mem4 = self.lif4(self.fc4(spk3), mem4)

            spk1_rec.append(spk1)
            spk2_rec.append(spk2)
            spk3_rec.append(spk3)
            spk4_rec.append(spk4)
            mem4_rec.append(mem4)

        # Stack them into (time_steps, batch_size, feature_dim)
        return (
            torch.stack(spk1_rec, dim=0),
            torch.stack(spk2_rec, dim=0),
            torch.stack(spk3_rec, dim=0),
            torch.stack(spk4_rec, dim=0),
            torch.stack(mem4_rec, dim=0),
        )


class Lightning_SNNQUT_with_cochlea(pl.LightningModule):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        cochlea_betas,
        beta_hidden_1,
        beta_hidden_2,
        beta_hidden_3,
        beta_output,
        hidden_reset_mechanism,
        output_reset_mechanism,
        hidden_threshold,
        output_threshold,
        learning_rate,
        scheduler_step_size,
        scheduler_gamma,
        optimizer_betas,
        fast_sigmoid_slope,
    ):
        super().__init__()
        self.save_hyperparameters(
            "input_size",
            "hidden_size",
            "output_size",
            "hidden_reset_mechanism",
            "output_reset_mechanism",
            "hidden_threshold",
            "output_threshold",
            "learning_rate",
            "scheduler_step_size",
            "scheduler_gamma",
            "optimizer_betas",
            "fast_sigmoid_slope",
        )
        self.cochlea_betas = cochlea_betas
        self.beta_hidden_1 = beta_hidden_1
        self.beta_hidden_2 = beta_hidden_2
        self.beta_hidden_3 = beta_hidden_3
        self.beta_output = beta_output

        self.model = SNNQUT_with_clochlea(
            input_size=self.hparams.input_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.output_size,
            cochlea_betas=self.cochlea_betas,
            beta_hidden_1=self.beta_hidden_1,
            beta_hidden_2=self.beta_hidden_2,
            beta_hidden_3=self.beta_hidden_3,
            beta_output=self.beta_output,
            hidden_reset_mechanism=self.hparams.hidden_reset_mechanism,
            output_reset_mechanism=self.hparams.output_reset_mechanism,
            output_threshold=self.hparams.output_threshold,
            hidden_threshold=self.hparams.hidden_threshold,
            fast_sigmoid_slope=self.hparams.fast_sigmoid_slope,
        )

        # DEFINE LOSS
        self.loss_type = "CrossEntropy"
        self.loss_mode = "Spk"

        if self.loss_type == "MSE":
            self.loss_function = nn.MSELoss()
            self.Vmem_shift_for_MSELoss = 0.2

        else:  # CrossEntropy
            self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels, _ = batch
        spk1_rec, spk2_rec, spk3_rec, spk4_rec, mem4_rec = self(inputs)

        # Use 'mem' mode during training for consistency with training approach
        loss, accuracy, predicted, targets = compute_loss_and_metrics(
            spk4_rec,
            mem4_rec,
            labels,
            loss_function=self.loss_function,
            loss_mode=self.loss_mode,
            loss_type=self.loss_type,
        )

        # Training Interrupt for when the net is not active
        if spk4_rec.sum() == 0:
            raise RuntimeError("Boss, we dead")

        with torch.no_grad():
            print(
                f"Mem4 stats: min={mem4_rec.min():.2f}, max={mem4_rec.max():.2f}, mean={mem4_rec.mean():.2f}"
            )
            print(f"spk4 stats: mean={spk4_rec.mean():.2f}")
            print(f"spk3 stats: mean={spk3_rec.mean():.2f}")
            print(f"spk2 stats: mean={spk2_rec.mean():.2f}")
            print(f"spk1 stats: mean={spk1_rec.mean():.2f}")

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_accuracy",
            accuracy * 100,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        # Report to NNI
        # nni.report_intermediate_result(
        #     {
        #         "default": accuracy * 100,
        #         "train_loss": loss.item(),
        #         "train_accuracy": accuracy * 100,
        #     }
        # )
        train.report({"loss": loss.item(), "accuracy": accuracy * 100})
        # Monitor gradients
        total_norm = 0
        for name, p in self.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                if param_norm > 1:  # Or any threshold you choose
                    print(f"Large gradient in {name}: {param_norm}")
        total_norm = total_norm**0.5

        if total_norm > 10:  # Adjust threshold as needed
            print(f"Warning: Total gradient norm: {total_norm}")

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, _ = batch
        spk1_rec, spk2_rec, spk3_rec, spk4_rec, mem4_rec = self(inputs)

        # For validation, also use 'mem' mode to stay consistent
        loss, accuracy, predicted, targets = compute_loss_and_metrics(
            spk4_rec,
            mem4_rec,
            labels,
            loss_function=self.loss_function,
            loss_mode=self.loss_mode,
            loss_type=self.loss_type,
        )

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", accuracy * 100, on_epoch=True, prog_bar=True)

        # # Report to NNI
        # nni.report_intermediate_result(
        #     {"val_loss": loss.item(), "val_accuracy": accuracy * 100}
        # )

        return {"val_loss": loss, "val_accuracy": accuracy}

    def test_step(self, batch, batch_idx):
        inputs, labels, _ = batch
        spk1_rec, spk2_rec, spk3_rec, spk4_rec, mem4_rec = self(inputs)

        # For testing, use 'spike' mode if you want to classify based on spikes
        loss, accuracy, predicted, targets = compute_loss_and_metrics(
            spk4_rec,
            mem4_rec,
            labels,
            loss_function=self.loss_function,
            loss_mode="spk",
            loss_type=self.loss_type,
        )

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "test_accuracy", accuracy * 100, on_step=False, on_epoch=True, prog_bar=True
        )

        return {"test_loss": loss, "test_accuracy": accuracy}

    def on_train_epoch_end(self):
        # Report epoch-level metrics to NNI
        metrics = self.trainer.callback_metrics
        # epoch_metrics = {
        #     "epoch": self.current_epoch,
        #     "train_loss_epoch": metrics.get("train_loss_epoch", 0).item(),
        #     "train_accuracy_epoch": metrics.get("train_accuracy_epoch", 0).item(),
        #     # "val_loss": metrics.get("val_loss", 0).item(),
        #     # "val_accuracy": metrics.get("val_accuracy", 0).item(),
        # }
        # nni.report_intermediate_result(epoch_metrics)

    def configure_optimizers(self):
        optimizer = optim.Adamax(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=self.hparams.optimizer_betas,
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.hparams.scheduler_step_size,
            gamma=self.hparams.scheduler_gamma,
        )

        # Add gradient clipping
        self.trainer.gradient_clip_val = 1.0
        self.trainer.gradient_clip_algorithm = "norm"

        return [optimizer], [scheduler]


class Lightning_SNNQUT_filter_bank(pl.LightningModule):
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
        hidden_threshold,
        output_threshold,
        learning_rate,
        scheduler_step_size,
        scheduler_gamma,
        optimizer_betas,
        fast_sigmoid_slope,
        bits,
    ):
        super().__init__()
        self.save_hyperparameters(
            "input_size",
            "hidden_size",
            "output_size",
            "hidden_reset_mechanism",
            "output_reset_mechanism",
            "hidden_threshold",
            "output_threshold",
            "learning_rate",
            "scheduler_step_size",
            "scheduler_gamma",
            "optimizer_betas",
            "fast_sigmoid_slope",
            "bits",
        )
        self.beta_hidden_1 = beta_hidden_1
        self.beta_hidden_2 = beta_hidden_2
        self.beta_hidden_3 = beta_hidden_3
        self.beta_output = beta_output

        self.model = SNNQUT_filter_bank(
            input_size=self.hparams.input_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.output_size,
            beta_hidden_1=self.beta_hidden_1,
            beta_hidden_2=self.beta_hidden_2,
            beta_hidden_3=self.beta_hidden_3,
            beta_output=self.beta_output,
            hidden_reset_mechanism=self.hparams.hidden_reset_mechanism,
            output_reset_mechanism=self.hparams.output_reset_mechanism,
            output_threshold=self.hparams.output_threshold,
            hidden_threshold=self.hparams.hidden_threshold,
            fast_sigmoid_slope=self.hparams.fast_sigmoid_slope,
            bits=self.hparams.bits,
        )

        # DEFINE LOSS
        self.loss_type = "CrossEntropy"
        self.loss_mode = "Spk"

        if self.loss_type == "MSE":
            self.loss_function = nn.MSELoss()
            self.Vmem_shift_for_MSELoss = 0.2

        else:  # CrossEntropy
            self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels, _ = batch
        spk1_rec, spk2_rec, spk3_rec, spk4_rec, mem4_rec = self(inputs)

        # Use 'mem' mode during training for consistency with training approach
        loss, accuracy, predicted, targets = compute_loss_and_metrics(
            spk4_rec,
            mem4_rec,
            labels,
            loss_function=self.loss_function,
            loss_mode=self.loss_mode,
            loss_type=self.loss_type,
        )

        # Training Interrupt for when the net is not active
        if spk4_rec.sum() == 0:

            exit("Boss, we dead")

        with torch.no_grad():
            print(
                f"Mem4 stats: min={mem4_rec.min():.2f}, max={mem4_rec.max():.2f}, mean={mem4_rec.mean():.2f}"
            )
            print(f"spk4 stats: mean={spk4_rec.mean():.2f}")
            print(f"spk3 stats: mean={spk3_rec.mean():.2f}")
            print(f"spk2 stats: mean={spk2_rec.mean():.2f}")
            print(f"spk1 stats: mean={spk1_rec.mean():.2f}")

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_accuracy",
            accuracy * 100,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        total_norm = 0
        for name, p in self.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                if param_norm > 1:  # Or any threshold you choose
                    print(f"Large gradient in {name}: {param_norm}")
        total_norm = total_norm**0.5

        if total_norm > 10:  # Adjust threshold as needed
            print(f"Warning: Total gradient norm: {total_norm}")

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, _ = batch
        spk1_rec, spk2_rec, spk3_rec, spk4_rec, mem4_rec = self(inputs)

        # For validation, also use 'mem' mode to stay consistent
        loss, accuracy, predicted, targets = compute_loss_and_metrics(
            spk4_rec,
            mem4_rec,
            labels,
            loss_function=self.loss_function,
            loss_mode=self.loss_mode,
            loss_type=self.loss_type,
        )

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", accuracy * 100, on_epoch=True, prog_bar=True)

        return {"val_loss": loss, "val_accuracy": accuracy}

    def test_step(self, batch, batch_idx):
        inputs, labels, _ = batch
        spk1_rec, spk2_rec, spk3_rec, spk4_rec, mem4_rec = self(inputs)

        # For testing, use 'spike' mode if you want to classify based on spikes
        loss, accuracy, predicted, targets = compute_loss_and_metrics(
            spk4_rec,
            mem4_rec,
            labels,
            loss_function=self.loss_function,
            loss_mode="spk",
            loss_type=self.loss_type,
        )

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "test_accuracy", accuracy * 100, on_step=False, on_epoch=True, prog_bar=True
        )

        return {"test_loss": loss, "test_accuracy": accuracy}

    def on_train_epoch_end(self):

        metrics = self.trainer.callback_metrics

        train.report(
            {
                "loss": metrics.get("train_loss_epoch", 0).item(),
                "accuracy": metrics.get("train_accuracy_epoch", 0).item(),
            }
        )

    def configure_optimizers(self):
        optimizer = optim.Adamax(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=self.hparams.optimizer_betas,
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.hparams.scheduler_step_size,
            gamma=self.hparams.scheduler_gamma,
        )

        # Add gradient clipping
        self.trainer.gradient_clip_val = 1.0
        self.trainer.gradient_clip_algorithm = "norm"

        return [optimizer], [scheduler]


def generate_tau_beta_values(hidden_size, output_size, delta_t=1):
    """
    Generate tau and beta values for a three-layer spiking neural network.
    Reference: "A Temporal Dimension in Spiking Neural Networks Through Multiple Timescales"
    (https://arxiv.org/pdf/2208.12991)

    Parameters:
    hidden_size: number of neurons in each hidden layer
    output_size: number of neurons in output layer
    delta_t: time step in milliseconds (default=1ms)
    """

    def create_power_vector(n, size):
        # Creates vector of time constants following tau_i = 2^i ms where i starts from 1
        powers = [2**i for i in range(1, n + 1)]  # Powers of 2 in milliseconds
        repeat_count = size // n
        power_vector = np.repeat(powers, repeat_count)
        return power_vector

    size = hidden_size

    # Generate tau values (in milliseconds) for each hidden layer
    # Following the rule tau_i = 2^i ms where i is the power
    tau_hidden_1 = create_power_vector(
        n=2, size=size
    )  # Uses 2 time constants: 2ms and 4ms
    tau_hidden_2 = create_power_vector(
        n=4, size=size
    )  # Uses 4 time constants: 2ms, 4ms, 8ms, 16ms
    tau_hidden_3 = create_power_vector(
        n=8, size=size
    )  # Uses 8 time constants: 2ms, 4ms, 8ms, ..., 256ms

    # Calculate beta values using beta = exp(-delta_t/tau)
    # delta_t must be in milliseconds to match tau units
    beta_hidden_1 = torch.exp(
        -torch.tensor(delta_t * 1e-3) / torch.tensor(tau_hidden_1 * 1e-3, dtype=torch.float32)
    )
    beta_hidden_2 = torch.exp(
        -torch.tensor(delta_t * 1e-3) / torch.tensor(tau_hidden_2 * 1e-3, dtype=torch.float32)
    )
    beta_hidden_3 = torch.exp(
        -torch.tensor(delta_t * 1e-3) / torch.tensor(tau_hidden_3 * 1e-3, dtype=torch.float32)
    )

    # Output layer uses fixed 10ms time constant for all neurons
    tau_output = np.repeat(10, output_size)  # 10ms time constant
    beta_output = torch.exp(
        -torch.tensor(delta_t * 1e-3) / torch.tensor(tau_output * 1e-3, dtype=torch.float32)
    )

    return beta_hidden_1, beta_hidden_2, beta_hidden_3, beta_output


def finalize_quantization(model, remove_fake_quant=True):
    """
    Finalize quantization by replacing weights with their quantized versions.

    Args:
        model (nn.Module): Model to quantize
        remove_fake_quant (bool): If True, removes fake quantization modules after finalizing
    """
    with torch.no_grad():
        model.eval()

        for name, module in model.named_modules():
            if hasattr(module, "fake_quant") and hasattr(module, "weight"):
                # Replace floating point weights with their quantized versions
                quantized_weight = module.fake_quant(module.weight)
                module.weight.data.copy_(quantized_weight)

                if remove_fake_quant:
                    # Remove quantization module to prevent double quantization
                    # during inference and reduce model size
                    module.fake_quant = None

    return model


def analyze_quantization(tensor, fake_quant):
    """
    Analyze the effects of quantization on a tensor.

    Returns:
        dict: Statistics about the quantization
    """
    with torch.no_grad():
        quantized = fake_quant(tensor)

        # Calculate various metrics to understand quantization effects
        stats = {
            "original_range": (tensor.min().item(), tensor.max().item()),
            "quantized_range": (quantized.min().item(), quantized.max().item()),
            "unique_values": len(torch.unique(quantized)),  # Should match levels
            "mse": F.mse_loss(tensor, quantized).item(),
            "max_abs_error": (tensor - quantized).abs().max().item(),
        }

    return stats


def get_tau_values(min_freq=50, max_freq=8000, num_neurons=16):
    """
    Generate logarithmically spaced tau values for frequency-selective neurons.

    Args:
        min_freq (float): Minimum frequency in Hz
        max_freq (float): Maximum frequency in Hz
        num_neurons (int): Number of frequency-selective neurons

    Returns:
        numpy.ndarray: Array of tau values in milliseconds
    """
    log_linspace = lambda x, y, num: np.exp(np.linspace(np.log(x), np.log(y), num))
    tau_values = log_linspace(min_freq, max_freq, num_neurons)
    return tau_values


def get_cochlea_beta_values(min_freq=50, max_freq=8000, num_neurons=16, delta_t=1):
    """ """
    # Generate tau values for frequency-selective neurons
    cochlea_tau_values = get_tau_values(min_freq, max_freq, num_neurons)

    cochlea_tau_values = torch.tensor(cochlea_tau_values, dtype=torch.float32)
    cochlea_beta_values = torch.exp(
        -torch.tensor(delta_t) / (cochlea_tau_values * 1e-3)
    )

    return cochlea_beta_values
