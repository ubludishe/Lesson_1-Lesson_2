import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from copy import deepcopy
import warnings

warnings.filterwarnings('ignore')


class MulticlassLogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes, lambda_l1=0.01, lambda_l2=0.01):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.reset_parameters()

    def forward(self, x):
        # ???????? ?????????????? + Softmax ??? ?????????????? ?????????????
        logits = self.linear(x)
        return F.softmax(logits, dim=1)

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


def generate_multiclass_data(n_samples=2000, n_features=10, n_classes=4):
    """????????? ????????????? ?????? ??? ?????????????? ?????????????"""
    np.random.seed(42)
    torch.manual_seed(42)

    X = np.random.randn(n_samples, n_features)

    # ???????? ???? ??? ??????? ??????
    true_weights = np.random.randn(n_features, n_classes) * 0.5
    logits = X @ true_weights
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    y = np.array([np.random.choice(n_classes, p=prob) for prob in probabilities])

    # ????????? ???
    noise = np.random.normal(0, 0.1, n_samples)
    y = (y + noise).astype(int) % n_classes

    return torch.FloatTensor(X), torch.LongTensor(y)


def calculate_metrics(y_true, y_pred, y_pred_proba, num_classes):
    """?????? ???? ????????? ??????"""
    # Precision, Recall, F1 ??? ??????? ??????
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)

    # Macro average (??????? ?? ???????)
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()

    # ROC-AUC ??? ?????????????? (one-vs-rest)
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    roc_auc_macro = roc_auc_score(y_true_bin, y_pred_proba, multi_class='ovr', average='macro')

    return {
        'precision_macro': macro_precision,
        'recall_macro': macro_recall,
        'f1_macro': macro_f1,
        'roc_auc_macro': roc_auc_macro,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1
    }


def plot_confusion_matrix(y_true, y_pred, num_classes, class_names=None):
    """???????????? confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names or range(num_classes),
                yticklabels=class_names or range(num_classes))
    plt.xlabel('????????????? ??????')
    plt.ylabel('???????? ??????')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()


def train_multiclass_model(X_train, X_val, y_train, y_val,
                           num_classes, lr=0.01, epochs=1000, patience=20):
    model = MulticlassLogisticRegression(X_train.shape[1], num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    early_stopping = EarlyStopping(patience=patience)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train)
        train_loss = criterion(outputs, y_train)
        reg_loss = model.get_regularization_loss()
        total_loss = train_loss + reg_loss

        total_loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        early_stopping(val_loss.item(), model)
        if early_stopping.early_stop:
            print(f"Early stopping ?? ????? {epoch + 1}")
            break

        if epoch % 100 == 0:
            print(f"????? {epoch}: Train {train_loss:.4f}, Val {val_loss:.4f}, Reg {reg_loss:.4f}")

    return model, train_losses, val_losses


# ????????????
if __name__ == "__main__":
    # ????????? ?????? (4 ??????)
    X, y = generate_multiclass_data(n_samples=2000, n_features=10, n_classes=4)

    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    print("???????? ?????????????? ????????????? ?????????...")
    print(f"??????: 0-{y.max().item()}, ?????????: {X.shape[1]}")

    # ????????
    model, train_losses, val_losses = train_multiclass_model(
        X_train, X_val, y_train, y_val, num_classes=4
    )

    # ????????????
    model.eval()
    with torch.no_grad():
        y_pred_proba = model(X_val)
        y_pred = torch.argmax(y_pred_proba, dim=1)

    # ???????
    metrics = calculate_metrics(y_val.numpy(), y_pred.numpy(),
                                y_pred_proba.numpy(), num_classes=4)

    print("\n" + "=" * 50)
    print("? ???????? ??????? (Macro Average):")
    print(f"Precision: {metrics['precision_macro']:.4f}")
    print(f"Recall:    {metrics['recall_macro']:.4f}")
    print(f"F1-Score:  {metrics['f1_macro']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc_macro']:.4f}")

    print("\n? ??????? ?? ???????:")
    for i in range(4):
        print(f"????? {i}: P={metrics['precision_per_class'][i]:.3f}, "
              f"R={metrics['recall_per_class'][i]:.3f}, F1={metrics['f1_per_class'][i]:.3f}")

    # ????????????
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('?????')
    plt.ylabel('CrossEntropy Loss')
    plt.legend()
    plt.title('???????? ????????')

    plt.subplot(1, 3, 2)
    plt.bar(range(4), metrics['f1_per_class'])
    plt.xlabel('??????')
    plt.ylabel('F1-Score')
    plt.title('F1 ?? ???????')

    plt.subplot(1, 3, 3)
    plt.imshow(y_pred_proba[:10].numpy(), aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.ylabel('???????')
    plt.xlabel('??????')
    plt.title('Softmax ??????????? (?????? 10)')

    plt.tight_layout()
    plt.show()

    # Confusion Matrix
    plot_confusion_matrix(y_val.numpy(), y_pred.numpy(), 4,
                          ['????? 0', '????? 1', '????? 2', '????? 3'])