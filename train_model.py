import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
DATA_FOLDER = "/Users/arsal/Downloads/CIC IDS 2017 Network Intrusion Dataset"
MODEL_SAVE_PATH = "cnn_lstm_sentinel.pth"
BATCH_SIZE = 128
EPOCHS = 15          # Increased epochs for deep learning convergence
LEARNING_RATE = 0.001
SEQUENCE_LENGTH = 10 # Temporal step size for LSTM
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ==========================================
# 2. Data Loading, Cleaning & Balancing
# ==========================================
def load_and_preprocess_data(folder_path):
    print(f"Scanning folder: {folder_path}...")
    
    # Find all CSV files (handles lowercase and uppercase extensions)
    all_files = glob.glob(os.path.join(folder_path, "*.csv")) + glob.glob(os.path.join(folder_path, "*.CSV"))
    
    if not all_files:
        raise ValueError(f"No CSV files found in '{folder_path}'. Please check your folder path.")
    
    print(f"Found {len(all_files)} files. Loading and merging...")
    
    df_list = []
    for filename in all_files:
        print(f"Loading -> {os.path.basename(filename)}")
        df_temp = pd.read_csv(filename, low_memory=False)
        df_list.append(df_temp)
        
    df = pd.concat(df_list, ignore_index=True)
    
    # Strip whitespace from column headers (crucial for CICIDS2017)
    df.columns = df.columns.str.strip()
    print(f"Initial merged shape: {df.shape}")
    
    # Drop non-numeric identifier columns if present
    drop_cols = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp']
    existing_drop_cols = [col for col in drop_cols if col in df.columns]
    df = df.drop(columns=existing_drop_cols)
    
    # Handle infinite and missing values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Map Target: Benign = 0, Attacks = 1
    if 'Label' in df.columns:
        df['Target'] = df['Label'].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1)
    else:
        raise ValueError("Dataset must contain a 'Label' column.")
    
    # Balance classes to fix poor accuracy/F1-score caused by overwhelming Benign traffic
    print(f"Class distribution before balancing:\n{df['Target'].value_counts()}")
    
    benign_df = df[df['Target'] == 0]
    attack_df = df[df['Target'] == 1]
    
    # Downsample benign traffic to a balanced, manageable sample size (e.g., 400,000 rows)
    # Adjust this number depending on your available RAM
    max_benign_samples = min(len(benign_df), 400000)
    if len(benign_df) > max_benign_samples:
        benign_df = resample(benign_df, replace=False, n_samples=max_benign_samples, random_state=42)
        
    df_balanced = pd.concat([benign_df, attack_df])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Class distribution after balancing:\n{df_balanced['Target'].value_counts()}")
    
    y = df_balanced['Target'].values
    X = df_balanced.drop(columns=['Label', 'Target'])
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    
    # Reshape into temporal sequences for CNN-LSTM: (Samples, Sequence_Length, Features)
    num_samples = X_scaled.shape[0]
    num_features = X_scaled.shape[1]
    
    valid_len = (num_samples // SEQUENCE_LENGTH) * SEQUENCE_LENGTH
    X_scaled = X_scaled[:valid_len]
    y = y[:valid_len]
    
    X_reshaped = X_scaled.reshape(-1, SEQUENCE_LENGTH, num_features)
    y_reshaped = y[::SEQUENCE_LENGTH]
    
    return train_test_split(X_reshaped, y_reshaped, test_size=0.2, random_state=42)

# ==========================================
# 3. Hybrid CNN-LSTM Model Architecture
# ==========================================
class HybridCNNLSTM(nn.Module):
    def __init__(self, num_features, hidden_dim=64, num_layers=2):
        super(HybridCNNLSTM, self).__init__()
        
        # Spatial Feature Extraction via 1D Convolution
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        
        # Temporal Dependency Tracking via LSTM
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, num_layers=num_layers, 
                            batch_first=True, dropout=0.3)
        
        # Classification Head
        self.fc = nn.Linear(hidden_dim, 2)  # Binary output: Benign (0) vs Malicious (1)
        
    def forward(self, x):
        # x shape: (Batch, Seq_Len, Features) -> Transpose for Conv1d: (Batch, Features, Seq_Len)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Transpose back for LSTM: (Batch, Seq_Len_Reduced, Channels)
        x = x.permute(0, 2, 1)
        out, (hn, cn) = self.lstm(x)
        
        # Take the output of the last time step
        out = self.fc(out[:, -1, :])
        return out

# ==========================================
# 4. Rigorous Training Loop Execution
# ==========================================
if __name__ == "__main__":
    # Load and preprocess the actual dataset from folder
    X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_FOLDER)
    
    num_features = X_train.shape[2]
    print(f"Feature count after preprocessing: {num_features}")

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = HybridCNNLSTM(num_features=num_features).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    
    # Using AdamW optimizer with weight decay for better regularization
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Learning rate scheduler to lower LR automatically if validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    print("Starting hard training process...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping to stabilize LSTM training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        scheduler.step(epoch_loss)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss:.4f} - Training Accuracy: {epoch_acc*100:.2f}%")

    # Save trained model weights
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Training finished! Model weights successfully saved to {MODEL_SAVE_PATH}.")