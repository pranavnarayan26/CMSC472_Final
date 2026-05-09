import torch
import numpy as np
import random
import sys
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)
random.seed(42)
from torch.utils.data import DataLoader
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from data_processing import load_and_combine_parquet, TelemetryDataset
from models import TransformerAutoencoder, AnomalyClassifier, CNNAnomalyClassifier, TCNAnomalyClassifier, BCEWithFBetaLoss, evaluate_and_find_threshold, train_autoencoder



def generate_predictions(autoencoder, classifier, test_loader, threshold=0.5, device='cuda'):
    """
    Given an autoencoder and classifier, generates anomaly predictions for a test set. 
    """
    # Ensure both models are in evaluation mode (disables dropout, etc.)
    autoencoder.eval()
    classifier.eval()
    
    all_predictions = []
    
    print("Running inference on test data:")
    with torch.no_grad():
        for batch in test_loader:
            # Our test dataset might just return x, so handle accordingly
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            
            # 1. Unsupervised: Get reconstructions
            reconstructed = autoencoder(x)
            
            # 2. Supervised: Get anomaly logits from the classifier
            logits = classifier(x, reconstructed)
            
            # 3. Convert logits to probabilities (0.0 to 1.0)
            probs = torch.sigmoid(logits)
            
            # 4. Apply threshold to get binary predictions (0 or 1)
            preds = (probs > threshold).int()
            
            all_predictions.extend(preds.cpu().numpy())
            
    return np.array(all_predictions)



def generate_training_data():
    
    file = "train.parquet"

    dataset = load_and_combine_parquet(file)
    print("done loading data!")


    y = dataset['is_anomaly'].values
    X = dataset.drop(columns=['is_anomaly']).values

    return X, y


def train_two_stage_pipeline(autoencoder, classifier, train_loader_normal, train_loader_mixed, num_epochs_ae=10, num_epochs_clf=15, device='cpu'):
    autoencoder.to(device)
    classifier.to(device)
    
    # --- STAGE 1: Train the Autoencoder (Unsupervised) ---
    print("--- STAGE 1: Training Transformer Autoencoder ---")
    ae_criterion = nn.MSELoss()
    ae_optimizer = optim.AdamW(autoencoder.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs_ae):
        autoencoder.train()
        total_loss = 0.0
        for batch in train_loader_normal:
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            
            ae_optimizer.zero_grad()
            reconstructed = autoencoder(x)
            loss = ae_criterion(reconstructed, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
            ae_optimizer.step()
            total_loss += loss.item()
            
        print(f"AE Epoch [{epoch+1}/{num_epochs_ae}] | Loss: {total_loss/len(train_loader_normal):.6f}")

    # --- FREEZE AUTOENCODER ---
    print("\nFreezing Autoencoder weights...")
    for param in autoencoder.parameters():
        param.requires_grad = False
    autoencoder.eval()

    # --- STAGE 2: Train the Classifier (Supervised) ---
    print("\n--- STAGE 2: Training FFNN Classifier ---")
    clf_criterion = BCEWithFBetaLoss()
    clf_optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs_clf):
        classifier.train()
        total_clf_loss = 0.0
        
        for x, y in train_loader_mixed:
            x, y = x.to(device), y.float().to(device)
            
            clf_optimizer.zero_grad()
            
            # 1. Forward pass through frozen Autoencoder (no gradients needed here)
            with torch.no_grad():
                reconstructed = autoencoder(x)
                
            # 2. Forward pass through Classifier (requires gradients)
            logits = classifier(x, reconstructed)
            
            # 3. Calculate Binary Cross Entropy Loss
            loss = clf_criterion(logits[:, 0], y)
            loss.backward()
            clf_optimizer.step()
            total_clf_loss += loss.item()
            
        print(f"Classifier Epoch [{epoch+1}/{num_epochs_clf}] | Loss: {total_clf_loss/len(train_loader_mixed):.6f}")

    return autoencoder, classifier


def fold_loop(X, y, window, batch, autoencoder, classifier, classifier_type):
    sets = model_selection.TimeSeriesSplit(n_splits=3)
    folds = sets.split(X)

    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"Fold #{fold + 1}:")
        X_train_split, y_train_split = X[train_idx], y[train_idx]
        X_val_split, y_val_split = X[val_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_split)
        X_val_scaled = scaler.transform(X_val_split)

        normal_indices = (y_train_split == 0)
        X_train_normal = X_train_scaled[normal_indices]
        train_dataset = TelemetryDataset(X_train_normal, window_size=window)
        train_dataset_mixed = TelemetryDataset(X_train_scaled, labels=y_train_split, window_size=window)
        val_dataset = TelemetryDataset(X_val_scaled, labels=y_val_split, window_size=window)
                                        
        train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
        train_loader_mixed = DataLoader(train_dataset_mixed, batch_size=batch, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

        autoencoder = TransformerAutoencoder(num_features=X.shape[1], d_model=64, nhead=4, num_layers=3, dropout=0.1)
        
        if classifier_type == "AT":
            pass
        elif classifier_type == "NN":
            classifier = AnomalyClassifier(num_features=X.shape[1])
        elif classifier_type == "CNN":
            classifier = CNNAnomalyClassifier(num_features=X.shape[1], seq_len=window)
        elif classifier_type == "TCN":
            classifier = TCNAnomalyClassifier(num_features=X.shape[1], num_channels=[32, 64, 128], kernel_size=3).to('cuda')
        else:
            raise ValueError("INCORRECT CLASSIFIER TYPE!")



        # Training stage
        autoencoder, classifier = train_two_stage_pipeline(
            autoencoder, 
            classifier, 
            train_loader, 
            train_loader_mixed, 
            num_epochs_ae=5, 
            num_epochs_clf=5,
            device='cuda'
        )

        preds = generate_predictions(autoencoder, classifier, val_loader)

        y_val_aligned = np.array([
            max(y_val_split[i : i + window]) 
            for i in range(len(y_val_split) - window + 1)
        ])

        # 4. Calculate F0.5 Score
        #f05 = fbeta_score(y_val_aligned, preds, beta=0.5, zero_division=0)
        prec, rec, f05, _ = precision_recall_fscore_support(
            y_val_aligned, 
            preds, 
            beta=0.5, 
            average='binary', 
            zero_division=0
        )

        print(f"--- FOLD RESULTS ---")
        print(f"F0.5 Score: {f05:.4f}")
        print(f"Precision:  {prec:.4f} (Out of all flagged anomalies, {prec*100:.1f}% were real)")
        print(f"Recall:     {rec:.4f} (Out of all real anomalies, we caught {rec*100:.1f}%)")
        print(f"Total Actual Anomalies in this fold: {sum(y_val_aligned)}")
        print(classification_report(y_val_aligned, preds, zero_division=0))


def downsampling(X, y, factor=0.1):
    anomaly_indices = np.where(y == 1)[0]
    target_rows = int(len(X) * factor)
    center_idx = anomaly_indices[len(anomaly_indices) // 2]
    start_idx = max(0, center_idx - (target_rows // 2))
    end_idx = min(len(X), start_idx + target_rows)
    if (end_idx - start_idx) < target_rows:
        start_idx = max(0, end_idx - target_rows)
    X_small = X[start_idx:end_idx]
    y_small = y[start_idx:end_idx]  
    return X_small, y_small




def train_adaptive_thresholding():
    X, y = generate_training_data()
    X, y = downsampling(X, y)
    X_train = X[:np.int32(0.8 * np.shape(X)[0])]
    y_train = y[:np.int32(0.8 * np.shape(X)[0])]
    X_val = X[np.int32(0.8 * np.shape(X)[0]):]
    y_val = y[np.int32(0.8 * np.shape(X)[0]):]
    autoencoder = TransformerAutoencoder(num_features=X.shape[1], d_model=64, nhead=4, num_layers=3, dropout=0.1)
    window = 64
    batch = 1024
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    normal_indices = (y_train == 0)
    X_train_normal = X_train_scaled[normal_indices]
    train_dataset = TelemetryDataset(X_train_normal, window_size=window)
    val_dataset = TelemetryDataset(X_val_scaled, labels=y_val, window_size=window)
                                    
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)
    
    
    autoencoder = train_autoencoder(autoencoder, train_loader, num_epochs=5)
    
    classifier = evaluate_and_find_threshold(autoencoder, val_loader)
    fold_loop(X, y, window, batch, autoencoder, classifier, "AT")
    

def train_nn_detection():
    X, y = generate_training_data()
    X, y = downsampling(X, y)
    autoencoder = TransformerAutoencoder(num_features=X.shape[1], d_model=64, nhead=4, num_layers=3, dropout=0.1)
    window = 64
    batch = 1024
    classifier = AnomalyClassifier(num_features=X.shape[1])
    fold_loop(X, y, window, batch, autoencoder, classifier, "NN")
    

def train_cnn_detection():
    X, y = generate_training_data()
    X, y = downsampling(X, y)
    autoencoder = TransformerAutoencoder(num_features=X.shape[1], d_model=64, nhead=4, num_layers=3, dropout=0.1)
    window = 64
    batch = 1024
    classifier = CNNAnomalyClassifier(num_features=X.shape[1], seq_len=window)
    fold_loop(X, y, window, batch, autoencoder, classifier, "CNN")
    

def train_tcn_detection():
    X, y = generate_training_data()
    X, y = downsampling(X, y)
    autoencoder = TransformerAutoencoder(num_features=X.shape[1], d_model=64, nhead=4, num_layers=3, dropout=0.1)
    window = 64
    batch = 1024
    classifier = TCNAnomalyClassifier(num_features=X.shape[1], num_channels=[32, 64, 128], kernel_size=3).to('cuda')
    fold_loop(X, y, window, batch, autoencoder, classifier, "TCN")
    



train_adaptive_thresholding()
