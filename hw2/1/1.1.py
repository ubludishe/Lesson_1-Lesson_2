import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


class RegularizedLinearRegression(nn.Module):
    def __init__(self, input_size, lambda_l1=0.01, lambda_l2=0.01):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.reset_parameters()

    def forward(self, x):  # ? ???? ????? ??? ?????????????!
        return self.linear(x)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.linear.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.linear.bias, 0)

    def l1_penalty(self):
        return torch.abs(self.linear.weight).sum()

    def l2_penalty(self):
        return (self.linear.weight ** 2).sum()

    def get_regularization_loss(self):
        return self.lambda_l1 * self.l1_penalty() + self.lambda_l2 * self.l2_penalty()


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_weights = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_state_dict(self.best_weights)


def generate_data(n_samples=1000, noise=0.1, n_features=5):
    np.random.seed(42)
    torch.manual_seed(42)

    X = np.random.randn(n_samples, n_features)
    true_weights = np.array([0.5, 1.2, -0.8, 0.3, -0.1])
    y = X @ true_weights + np.random.normal(0, noise, n_samples)

    return torch.FloatTensor(X), torch.FloatTensor(y).view(-1, 1)


def train_model(X_train, X_val, y_train, y_val,
                lambda_l1=0.01, lambda_l2=0.01,
                lr=0.01, epochs=1000, patience=20):
    model = RegularizedLinearRegression(X_train.shape[1], lambda_l1, lambda_l2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    early_stopping = EarlyStopping(patience=patience, min_delta=0.001)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass - ?????? ????????!
        train_pred = model(X_train)
        train_loss = criterion(train_pred, y_train)

        reg_loss = model.get_regularization_loss()
        total_train_loss = train_loss + reg_loss

        total_train_loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val)

        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        early_stopping(val_loss.item(), model)
        if early_stopping.early_stop:
            print(f"Early stopping ?? ????? {epoch + 1}")
            break

        if epoch % 100 == 0:
            print(f"????? {epoch}, Train: {train_loss:.4f}, Val: {val_loss:.4f}, Reg: {reg_loss:.4f}")

    return model, train_losses, val_losses


if __name__ == "__main__":
    X, y = generate_data(n_samples=1000, n_features=5)

    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    print("???????? ?????? ? L1+L2 ?????????????? ? Early Stopping...")

    model, train_losses, val_losses = train_model(
        X_train, X_val, y_train, y_val,
        lambda_l1=0.01, lambda_l2=0.01,
        lr=0.01, epochs=1000, patience=20
    )

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('?????')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('???????? ??????')

    plt.subplot(1, 2, 2)
    plt.bar(range(model.linear.weight.shape[1]), model.linear.weight.detach().numpy().flatten())
    plt.xlabel('????????')
    plt.ylabel('????')
    plt.title('???? ?????? ????? L1/L2 ?????????????')

    plt.tight_layout()
    plt.show()

    model.eval()
    with torch.no_grad():
        train_pred = model(X_train)
        val_pred = model(X_val)

        train_mse = nn.MSELoss()(train_pred, y_train).item()
        val_mse = nn.MSELoss()(val_pred, y_val).item()

        print(f"\n???????? ???????:")
        print(f"Train MSE: {train_mse:.4f}")
        print(f"Val MSE: {val_mse:.4f}")