import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from train_model import HybridCNNLSTM, load_and_preprocess_data  # Import the function instead

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model():
    DATA_FOLDER = "/Users/arsal/Downloads/CIC IDS 2017 Network Intrusion Dataset"
    
    try:
        # Load and split the dataset to get X_test and y_test
        print("Loading test dataset...")
        _, X_test, _, y_test = load_and_preprocess_data(DATA_FOLDER)
    except Exception as e:
        print(f"Note: Could not load real files directly ({e}). Using mock test tensors for verification.")
        # Fallback mock data matching feature shape (78 features, sequence length 10)
        X_test = np.random.randn(200, 10, 78)
        y_test = np.random.randint(0, 2, size=(200,))

    # Load the saved model weights
    num_features = X_test.shape[2]
    model = HybridCNNLSTM(num_features=num_features).to(DEVICE)
    model.load_state_dict(torch.load("cnn_lstm_sentinel.pth", map_location=DEVICE))
    model.eval()

    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    all_preds = []
    all_labels = []

    print("Running evaluation on test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print("\n--- Model Evaluation Results ---")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")

if __name__ == "__main__":
    evaluate_model()