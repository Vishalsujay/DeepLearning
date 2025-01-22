import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# General Training Function with Early Stopping
def train_model_with_early_stopping(model, train_loader, test_loader, criterion, optimizer, epochs, patience=3):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        y_train_predict = []
        y_train_true = []

        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            y_train_predict.extend(predicted.numpy())
            y_train_true.extend(labels.numpy())

        train_losses.append(running_loss / len(train_loader))
        train_accuracy = accuracy_score(y_train_true, y_train_predict)
        train_accuracies.append(train_accuracy)

        model.eval()
        val_loss = 0.0
        y_val_predict = []
        y_val_true = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                y_val_predict.extend(predicted.numpy())
                y_val_true.extend(labels.numpy())

        val_losses.append(val_loss / len(test_loader))
        val_accuracy = accuracy_score(y_val_true, y_val_predict)
        val_accuracies.append(val_accuracy)

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            torch.save(model.state_dict(), "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {train_losses[-1]:.4f}, "
              f"Train Accuracy: {train_accuracy * 100:.2f}%, "
              f"Val Loss: {val_losses[-1]:.4f}, "
              f"Val Accuracy: {val_accuracy * 100:.2f}%")

    return train_losses, train_accuracies, val_losses, val_accuracies

# Experiment 1: Baseline Model
def experiment_baseline(train_loader, test_loader, input_size, output_size, epochs=50, patience=5):
    model = nn.Sequential(
        nn.Linear(input_size, 64),
        nn.ReLU(),
        nn.Linear(64, output_size)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    return train_model_with_early_stopping(model, train_loader, test_loader, criterion, optimizer, epochs, patience)

# Experiment 2: Model with Dropout
def experiment_with_dropout(train_loader, test_loader, input_size, output_size, epochs=50, patience=5):
    model = nn.Sequential(
        nn.Linear(input_size, 64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, output_size)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    return train_model_with_early_stopping(model, train_loader, test_loader, criterion, optimizer, epochs, patience)

# Experiment 3: Model with L2 Regularization
def experiment_with_l2(train_loader, test_loader, input_size, output_size, epochs=50, patience=5, weight_decay=0.01):
    model = nn.Sequential(
        nn.Linear(input_size, 64),
        nn.ReLU(),
        nn.Linear(64, output_size)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)

    return train_model_with_early_stopping(model, train_loader, test_loader, criterion, optimizer, epochs, patience)

# Experiment 4: Model with Batch Normalization
def experiment_with_batch_norm(train_loader, test_loader, input_size, output_size, epochs=50, patience=5):
    model = nn.Sequential(
        nn.Linear(input_size, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64, output_size)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    return train_model_with_early_stopping(model, train_loader, test_loader, criterion, optimizer, epochs, patience)

# Experiment 5: Combined Regularization Techniques
def experiment_combined(train_loader, test_loader, input_size, output_size, epochs=50, patience=5, weight_decay=0.01):
    model = nn.Sequential(
        nn.Linear(input_size, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, output_size)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)

    return train_model_with_early_stopping(model, train_loader, test_loader, criterion, optimizer, epochs, patience)

# Main Execution
if __name__ == "__main__":
    
    # Load data frame from the csv file
    data = pd.read_csv("sonar.all-data", header=None)
    data[60] = data[60].replace({'R':1,'M':0})
    X = data.drop(60, axis=1)
    Y = data[60]

    # Splitting the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    #Normalizing the input data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    #Conver the datas to the pytorch tensor format
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train.to_numpy(), dtype=torch.long)
    Y_test_tensor = torch.tensor(Y_test.to_numpy(), dtype=torch.long)
    
    #Pass it to the TensorDataset and DataLoader
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    # Define your dataset, DataLoader, and parameters here.
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True) # Replace with actual DataLoader
    test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)  # Replace with actual DataLoader
    input_size = 60  # Example input size for Sonar dataset
    output_size = 2  # Example output size for binary classification

    print("Running Experiment 1: Baseline Model")
    experiment_baseline(train_loader, test_loader, input_size, output_size)

    print("Running Experiment 2: Model with Dropout")
    experiment_with_dropout(train_loader, test_loader, input_size, output_size)

    print("Running Experiment 3: Model with L2 Regularization")
    experiment_with_l2(train_loader, test_loader, input_size, output_size)

    print("Running Experiment 4: Model with Batch Normalization")
    experiment_with_batch_norm(train_loader, test_loader, input_size, output_size)

    print("Running Experiment 5: Combined Regularization Techniques")
    experiment_combined(train_loader, test_loader, input_size, output_size)
