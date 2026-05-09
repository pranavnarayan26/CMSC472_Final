import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.optim as optim
import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score, classification_report
import sys
from data_processing import PositionalEncoding



class TransformerAutoencoder(nn.Module):
    def __init__(self, num_features, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.1):
        super(TransformerAutoencoder, self).__init__()
        
        # Project input features to latent dimension
        self.input_projection = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True # Ensures input shape is (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Project latent dimension back to original feature space
        self.output_projection = nn.Linear(d_model, num_features)

    def forward(self, src):
        # src shape: (batch_size, seq_len, num_features)
        x = self.input_projection(src)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        out = self.output_projection(x)
        return out

def train_autoencoder(model, train_loader, num_epochs=10, learning_rate=1e-4, device='cuda'):
    print("training!")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            # Handle dataset returning (data, labels) or just data
            if isinstance(batch, list) or isinstance(batch, tuple):
                x = batch[0].to(device)
            else:
                x = batch.to(device)
            
            optimizer.zero_grad()
            reconstructed = model(x)
            
            # Loss is Mean Squared Error between input and reconstruction
            loss = criterion(reconstructed, x)
            loss.backward()
            
            # Gradient clipping to stabilize Transformer training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.6f}")
        
    return model


class AnomalyClassifier(nn.Module):
    def __init__(self, num_features):
        super(AnomalyClassifier, self).__init__()
        # Input is the per-channel MSE vector (size: num_features)
        self.network = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64), 
            nn.ReLU(), 
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid() # Outputs probability of anomaly (0 to 1)
        )

    def forward(self, original_x, reconstructed_x):
        # Calculate MSE per channel, preserving the feature dimension
        # Shape: (batch_size, num_features)
        per_channel_error = torch.mean((original_x - reconstructed_x) ** 2, dim=1)
        
        # Pass the error vector through the FFNN
        probability = self.network(per_channel_error)
        return probability

class CNNAnomalyClassifier(nn.Module):
    def __init__(self, num_features, seq_len):
        super(CNNAnomalyClassifier, self).__init__()
        
        # Input shape: (batch, num_features, seq_len)
        # Note: PyTorch Conv1d expects (batch, channels, length)
        self.network = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), # Squeezes time dimension to 1
            
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, original_x, reconstructed_x):
        # 1. Calculate absolute error sequence: (batch, seq_len, num_features)
        error_seq = torch.abs(original_x - reconstructed_x)
        
        # 2. Permute for Conv1d: (batch, num_features, seq_len)
        error_seq = error_seq.permute(0, 2, 1)
        
        # 3. Forward pass
        logits = self.network(error_seq)
        return logits.squeeze()

class BCEWithFBetaLoss(nn.Module):
    def __init__(self, beta=0.5, bce_weight=0.1, pos_weight_val=0.99, eps=1e-7):
        """
        beta: 0.5 to prioritize precision over recall (matches ESA F0.5 metric).
        bce_weight: How much standard BCE to mix in for gradient stability (0.0 to 1.0).
        """
        super(BCEWithFBetaLoss, self).__init__()
        self.beta = beta
        self.bce_weight = bce_weight
        self.eps = eps
        # Use standard PyTorch BCE internally for the BCE portion
        pos_weight_tensor = torch.tensor([pos_weight_val]).cuda()
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    def forward(self, logits, targets):
        # 1. Calculate standard BCE loss
        bce = self.bce_loss(logits, targets)
        
        # 2. Calculate Soft F-Beta / Dice Loss
        # Apply sigmoid to logits to get continuous probabilities (0 to 1)
        probs = torch.sigmoid(logits)
        
        # Flatten tensors
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        # Calculate 'Soft' True Positives, False Positives, and False Negatives
        # We use multiplication instead of boolean logic so it remains differentiable
        TP = (probs * targets).sum()
        FP = (probs * (1 - targets)).sum()
        FN = ((1 - probs) * targets).sum()
        
        beta_sq = self.beta ** 2
        
        # Calculate the F-beta score (adding epsilon to prevent division by zero)
        f_beta = ((1 + beta_sq) * TP + self.eps) / \
                 ((1 + beta_sq) * TP + FP + beta_sq * FN + self.eps)
                 
        # Since we want to MINIMIZE loss, and a perfect F-beta is 1.0:
        f_beta_loss = 1.0 - f_beta
        
        # 3. Combine them
        total_loss = (self.bce_weight * bce) + ((1.0 - self.bce_weight) * f_beta_loss)
        
        return total_loss




class Chsome_padding(nn.Module):
    """Slices the output to ensure causality (output length = input length)"""
    def __init__(self, padding):
        super(Chsome_padding, self).__init__()
        self.padding = padding

    def forward(self, x):
        return x[:, :, :-self.padding].contiguous()

class TCNBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TCNBlock, self).__init__()
        # First conv layer
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chsome_padding(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second conv layer
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chsome_padding(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # Residual connection (downsample if input/output channels differ)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNAnomalyClassifier(nn.Module):
    def __init__(self, num_features, num_channels=[32, 64], kernel_size=3, dropout=0.2):
        super(TCNAnomalyClassifier, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_features if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TCNBlock(in_channels, out_channels, kernel_size, stride=1, 
                                dilation=dilation_size,
                                padding=(kernel_size-1) * dilation_size)]

        self.tcn = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], 1)

    def forward(self, original_x, reconstructed_x):
        # 1. Compute per-channel error: (batch, seq_len, num_features)
        error_seq = torch.abs(original_x - reconstructed_x)
        
        # 2. Reshape for TCN: (batch, num_features, seq_len)
        x = error_seq.transpose(1, 2)
        
        # 3. Pass through TCN blocks
        y = self.tcn(x)
        
        # 4. Use the LAST timestep's output for classification
        logits = self.linear(y[:, :, -1])
        return logits.squeeze()

def evaluate_and_find_threshold(model, test_loader, device='cuda'):
    model.to(device)
    model.eval()
    
    reconstruction_errors = []
    true_labels = []
    
    print("testing!")
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            reconstructed = model(x)
            
            # Calculate MSE per sequence in the batch
            # Shape before mean: (batch_size, seq_len, num_features)
            # Shape after mean: (batch_size,)
            mse_per_sequence = torch.mean((x - reconstructed) ** 2, dim=[1, 2])
            
            reconstruction_errors.extend(mse_per_sequence.cpu().numpy())
            true_labels.extend(y.numpy())
            
    reconstruction_errors = np.array(reconstruction_errors)
    true_labels = np.array(true_labels)
    
    # Test thresholds between the min and max reconstruction errors
    thresholds = np.linspace(np.min(reconstruction_errors), np.max(reconstruction_errors), 100)
    
    best_f05 = 0.0
    best_threshold = 0.0
    best_metrics = {}
    print("threshold optimization!")
    for thresh in thresholds:
        # Predict 1 (anomaly) if error > threshold, else 0
        predictions = (reconstruction_errors > thresh).astype(int)
        
        # beta=0.5 weights precision higher than recall
        f05 = fbeta_score(true_labels, predictions, beta=0.5, zero_division=0)
        
        if f05 > best_f05:
            best_f05 = f05
            best_threshold = thresh
            best_metrics = {
                'precision': precision_score(true_labels, predictions, zero_division=0),
                'recall': recall_score(true_labels, predictions, zero_division=0),
                'f0.5': f05
            }
            
    print(f"Best Threshold: {best_threshold:.6f}")
    print(f"F0.5 Score: {best_metrics['f0.5']:.4f}")
    print(f"Precision:  {best_metrics['precision']:.4f}")
    print(f"Recall:     {best_metrics['recall']:.4f}")
    
    return best_threshold, best_metrics