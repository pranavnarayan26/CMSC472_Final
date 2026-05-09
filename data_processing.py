import torch
import torch.nn as nn
import numpy as np
import glob
from torch.utils.data import Dataset
import pandas as pd

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # Shape: (1, max_len, d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class TelemetryDataset(Dataset):
    def __init__(self, data, labels=None, window_size=64, step_size=1):
        """
        data: numpy array of shape (total_timesteps, num_features)
        labels: numpy array of shape (total_timesteps,) indicating anomalies
        """
        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.step_size = step_size
        
        # Calculate valid window indices
        self.indices = np.arange(0, len(data) - window_size + 1, step_size)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.window_size
        
        window_data = torch.tensor(self.data[start_idx:end_idx], dtype=torch.float32)
        
        if self.labels is not None:
            # A window is considered anomalous if it contains any anomalous timestamps
            # Alternatively, you can just return the label of the last timestep.
            window_label = torch.tensor(max(self.labels[start_idx:end_idx]), dtype=torch.float32)
            return window_data, window_label
            
        return window_data


def load_and_combine_parquet(file_pattern, sort_col='id'):
    """
    Reads multiple parquet files based on a glob pattern, concatenates them,
    and ensures chronological order.
    """
    # 1. Find all files matching the pattern
    file_list = glob.glob(file_pattern)
        
    # 2. Read and concatenate
    dfs = [pd.read_parquet(f) for f in file_list]
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # 3. Sort chronologically (Crucial for time-series sliding windows!)
    combined_df = combined_df.sort_values(sort_col).reset_index(drop=True)
    # Drop the timestamp column so the model only sees telemetry features
    combined_df = combined_df.drop(columns=[sort_col])
        
    return combined_df
