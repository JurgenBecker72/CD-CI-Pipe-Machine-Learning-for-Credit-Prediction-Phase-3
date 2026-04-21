# src/models/train.py

# -------------------------------
# Step 1: Imports
# -------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd


# -------------------------------
# Step 2: Model definition
# -------------------------------
class CreditModel(nn.Module):
    def __init__(self, input_dim):
        super(CreditModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.model(x)


# -------------------------------
# Step 3: Training function
# -------------------------------
def train_model(X_train, y_train, X_val, y_val, epochs=10, lr=0.001):

    # -------------------------------
    # 🔥 FIX: Force all data to numeric
    # -------------------------------
    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    X_val = X_val.apply(pd.to_numeric, errors="coerce")

    # Fill NaNs created by coercion
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)

    # -------------------------------
    # Convert to PyTorch tensors
    # -------------------------------
    X_train = torch.tensor(X_train.values.astype("float32"))
    y_train = torch.tensor(y_train.values.astype("float32")).view(-1, 1)

    X_val = torch.tensor(X_val.values.astype("float32"))
    y_val = torch.tensor(y_val.values.astype("float32")).view(-1, 1)

    # -------------------------------
    # Initialize model
    # -------------------------------
    model = CreditModel(X_train.shape[1])

    # -------------------------------
    # Loss + optimizer
    # -------------------------------
    criterion = nn.BCEWithLogitsLoss()  # better than BCELoss
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # -------------------------------
    # Training loop
    # -------------------------------
    for epoch in range(epochs):

        model.train()

        # Forward pass
        outputs = model(X_train)

        # Loss
        loss = criterion(outputs, y_train)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # -------------------------------
        # Validation
        # -------------------------------
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {loss.item():.4f} | "
            f"Val Loss: {val_loss.item():.4f}"
        )

    return model
