import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List
import random
from functools import partial
import pytorch_lightning as pl
import math

CLASSES = ["CAR", "STREET", "HOME", "CAFE"]

def get_label_from_folder(folder_name):
    base = os.path.basename(folder_name)
    label = base.split('-')[0]
    return label

def one_hot_encode(label):
    idx = CLASSES.index(label)
    vec = np.zeros(len(CLASSES), dtype=np.float32)
    vec[idx] = 1.0
    return vec

class CustomSNNTrainValDataset(Dataset):
    def __init__(self, root_dir, class_list=CLASSES):
        self.samples = []
        self.class_list = class_list

        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            label = get_label_from_folder(folder_path)
            if label not in self.class_list:
                continue
            label_vec = one_hot_encode(label)
            csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
            for csv_file in csv_files:
                data_array = self.load_csv(csv_file, time_steps=1000, input_dim=16)
                self.samples.append((data_array, label_vec))

    def load_csv(self, csv_path, time_steps=1000, input_dim=16):
        data = np.loadtxt(csv_path, delimiter=',', dtype=np.float32)
        if data.shape != (time_steps, input_dim):
            raise ValueError(f"CSV {csv_path} shape mismatch: {data.shape}")
        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data, label = self.samples[idx]
        return torch.tensor(data), torch.tensor(label)

def stratified_split(dataset, train_ratio=0.8, seed=42):
    # Use a deterministic split
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    labels = []
    for i in range(len(dataset)):
        _, l = dataset[i]
        c = torch.argmax(l).item()
        labels.append(c)
    labels = np.array(labels)

    train_indices = []
    val_indices = []

    for c in np.unique(labels):
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)
        split_point = int(len(class_indices)*train_ratio)
        train_part = class_indices[:split_point]
        val_part = class_indices[split_point:]
        train_indices.extend(train_part)
        val_indices.extend(val_part)

    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    return train_subset, val_subset

class CustomSNNTestDataset(Dataset):
    def __init__(self, test_dir, class_list=CLASSES):
        self.samples = []
        for folder in os.listdir(test_dir):
            folder_path = os.path.join(test_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            label = get_label_from_folder(folder_path)
            if label not in class_list:
                continue
            label_vec = one_hot_encode(label)
            csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
            if len(csv_files) == 0:
                raise ValueError(f"No test csv found in {folder_path}")
            # Large files, handle individually in test script
            # We'll just store folder and label here, not load now.
            self.samples.append((folder_path, label_vec))

    def load_csv(self, csv_path, time_steps=60000, input_dim=16):
        data = np.loadtxt(csv_path, delimiter=',', dtype=np.float32)
        if data.shape != (time_steps, input_dim):
            raise ValueError(f"Test CSV {csv_path} shape mismatch: {data.shape}")
        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Return folder path and label here
        folder_path, label = self.samples[idx]
        return folder_path, label

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, train_val_dir, test_dir, batch_size=32, num_workers=1, seed=42):
        super().__init__()
        self.train_val_dir = train_val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            full_dataset = CustomSNNTrainValDataset(self.train_val_dir)
            self.train_dataset, self.val_dataset = stratified_split(full_dataset, train_ratio=0.8, seed=self.seed)

        if stage == 'test' or stage is None:
            self.test_dataset = CustomSNNTestDataset(self.test_dir)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        # For testing, we will not return a loader here since the test script will handle loading large test files individually.
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True)
