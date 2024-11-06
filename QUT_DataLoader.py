import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class QUTDataset(Dataset):
    def __init__(self, data_dir):
        self.data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
        self.labels = ['CAR', 'STREET', 'HOME', 'CAFE']
        self.label_to_index = {label: index for index, label in enumerate(self.labels)}
        print(f"Loaded {len(self.data_files)} files from {data_dir}")

    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        df = pd.read_csv(self.data_files[idx])
        data = df.iloc[:, :-1].values
        labels = df['Label'].unique() # Get unique values in the 'Label' column
        
        if len(labels) != 1:
            raise ValueError(f"Inconsistent labels in file: {self.data_files[idx]}") # If there are multiple unique values, raise an error
        
        label = labels[0] # Extract the label from the unique value in the 'Label' column, which should be the same for all rows (that's why we use raise ValueError above)
        
        if data.shape != (1000, 16):
            raise ValueError(f"Unexpected data shape in file: {self.data_files[idx]}. Expected (1000, 16), got {data.shape}")
        
        data_tensor = torch.tensor(data, dtype=torch.uint8)
        
        label_index = self.label_to_index.get(label)
        if label_index is None:
            raise ValueError(f"Unknown label '{label}' in file: {self.data_files[idx]}")
        
        label_one_hot = torch.zeros(len(self.labels), dtype=torch.float32)
        label_one_hot[label_index] = 1.0
        #print(f"One hot encoded label: {label_one_hot} corresponding to label index: {label_index}, label: {label}")
        # Label is one-hot encoded with 5.0 for the correct class and 1.0 for all other classes
       # print(f"Data tensor shape: {data_tensor.shape}, Label one-hot shape: {label_one_hot.shape}")
        return data_tensor, label_one_hot

    def get_num_classes(self):
        return len(self.labels)

def split_dataset(dataset, train_ratio=0.65, val_ratio=0.15, test_ratio=0.2, seed=42):
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    print(f"Dataset split: Train {len(train_dataset)}, Validation {len(val_dataset)}, Test {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset, train_indices, val_indices, test_indices

def get_dataloader(train_dataset, val_dataset, test_dataset, batch_size=None, num_workers=1):
    assert batch_size is not None, "Batch size must be specified"
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Get a single batch for inspection
    sample_batch, sample_labels = next(iter(train_loader))
    print(f"Sample batch shape: {sample_batch.shape}")
    print(f"Sample labels shape: {sample_labels.shape}")
    
    return train_loader, val_loader, test_loader
