import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from itertools import product
import warnings

warnings.filterwarnings('ignore')


# ????????? ????????????? ?????????
def create_regression_dataset(n_samples=2000):
    np.random.seed(42)
    X = np.random.randn(n_samples, 8)
    true_weights = np.array([1.2, -0.8, 0.5, 1.1, -0.3, 0.7, -0.9, 0.4])
    y = X @ true_weights + np.random.normal(0, 0.1, n_samples)

    n_train = int(0.8 * n_samples)
    return (torch.FloatTensor(X[:n_train]), torch.FloatTensor(y[:n_train]).view(-1, 1),
            torch.FloatTensor(X[n_train:]), torch.FloatTensor(y[n_train:]).view(-1, 1))


def create_classification_dataset(n_samples=2000):
    np.random.seed(42)
    X = np.random.randn(n_samples, 8)
    true_weights = np.random.randn(8) * 0.5
    logits = X @ true_weights
    y = (1 / (1 + np.exp(-logits)) > 0.5).astype(np.float32)

    n_train = int(0.8 * n_samples)
    return (torch.FloatTensor(X[:n_train]), torch.FloatTensor(y[:n_train]).view(-1, 1),
            torch.FloatTensor(X[n_train:]), torch.FloatTensor(y[n_train:]).view(-1, 1))


# ??????
class LinearRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


# ??????? ?????? ????????????
def run_experiment(X_train, y_train, X_val, y_val, model_class, criterion,
                   lr, batch_size, optimizer_name, max_epochs=300, task='regression'):
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = model_class(X_train.shape[1])

    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)

    train_losses = []
    start_time = time.time()

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)

        # Early stopping
        if epoch > 20 and len(train_losses) > 10:
            if train_losses[-1] > min(train_losses[-10:]) * 1.05:
                break

    training_time = time.time() - start_time

    # ?????? ?? ?????????
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred, y_val)

    if task == 'regression':
        rmse = np.sqrt(val_loss.item())
        final_metric = rmse
    else:
        accuracy = ((val_pred > 0.5).float() == y_val).float().mean().item()
        final_metric = accuracy

    return {
        'learning_rate': lr,
        'batch_size': batch_size,
        'optimizer': optimizer_name,
        'final_train_loss': train_losses[-1],
        'metric': final_metric,
        'training_time': training_time,
        'epochs_completed': len(train_losses)
    }


# ???????? ???????????
if __name__ == "__main__":
    print("???????????? ? ????????????????")
    print("=" * 70)

    # ???? ??????????
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [32, 128, 512]
    optimizers_list = ['SGD', 'Adam', 'RMSprop']

    # ???????? ?????????
    reg_data = create_regression_dataset()
    cls_data = create_classification_dataset()

    all_results = []

    print("\n1. ???????????? ??? ????????? (???????: RMSE)")
    print("-" * 50)

    # ?????????
    for lr, bs, opt in product(learning_rates, batch_sizes, optimizers_list):
        print(f"LR={lr}, BS={bs}, {opt}", end=" ... ")
        result = run_experiment(*reg_data, LinearRegression, nn.MSELoss(),
                                lr, bs, opt, task='regression')
        result['task'] = 'regression'
        all_results.append(result)
        print(f"RMSE={result['metric']:.3f}, ?????={result['training_time']:.1f}s")

    print("\n2. ???????????? ??? ????????????? (???????: Accuracy)")
    print("-" * 50)

    # ?????????????
    for lr, bs, opt in product(learning_rates, batch_sizes, optimizers_list):
        print(f"LR={lr}, BS={bs}, {opt}", end=" ... ")
        result = run_experiment(*cls_data, LogisticRegression, nn.BCELoss(),
                                lr, bs, opt, task='classification')
        result['task'] = 'classification'
        all_results.append(result)
        print(f"Acc={result['metric']:.3f}, ?????={result['training_time']:.1f}s")

    # ?????? ???????????
    results_df = pd.DataFrame(all_results)

    # ??????? ???????????
    print("\n" + "=" * 70)
    print("?????????? ????????? (RMSE, ?????? = ?????)")
    print("-" * 70)
    reg_results = results_df[results_df['task'] == 'regression']
    reg_pivot = reg_results.pivot_table(
        values='metric', index=['optimizer', 'learning_rate'],
        columns='batch_size', aggfunc='mean'
    ).round(3)
    print(reg_pivot)

    print("\n" + "=" * 70)
    print("?????????? ????????????? (Accuracy, ?????? = ?????)")
    print("-" * 70)
    cls_results = results_df[results_df['task'] == 'classification']
    cls_pivot = cls_results.pivot_table(
        values='metric', index=['optimizer', 'learning_rate'],
        columns='batch_size', aggfunc='mean'
    ).round(3)
    print(cls_pivot)

    # ?????? ????????????
    print("\n" + "=" * 70)
    print("?????? ????????????")
    print("-" * 70)
    best_regression = reg_results.loc[reg_results['metric'].idxmin()]
    best_classification = cls_results.loc[cls_results['metric'].idxmax()]

    print(f"????????? (?????? RMSE):")
    print(f"  ???????????: {best_regression['optimizer']}")
    print(f"  Learning Rate: {best_regression['learning_rate']}")
    print(f"  Batch Size: {best_regression['batch_size']}")
    print(f"  RMSE: {best_regression['metric']:.3f}")

    print(f"\n????????????? (?????? Accuracy):")
    print(f"  ???????????: {best_classification['optimizer']}")
    print(f"  Learning Rate: {best_classification['learning_rate']}")
    print(f"  Batch Size: {best_classification['batch_size']}")
    print(f"  Accuracy: {best_classification['metric']:.3f}")

    # ????????????
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Heatmap ?????????
    reg_pivot_heatmap = reg_results.pivot(index='optimizer', columns='learning_rate',
                                          values='metric')
    sns.heatmap(reg_pivot_heatmap, annot=True, cmap='RdYlGn_r', fmt='.3f', ax=axes[0, 0])
    axes[0, 0].set_title('?????????: RMSE ?? ????????????? ? LR')

    # Heatmap ?????????????
    cls_pivot_heatmap = cls_results.pivot(index='optimizer', columns='learning_rate',
                                          values='metric')
    sns.heatmap(cls_pivot_heatmap, annot=True, cmap='RdYlGn', fmt='.3f', ax=axes[0, 1])
    axes[0, 1].set_title('?????????????: Accuracy ?? ????????????? ? LR')

    # ??????? batch size
    for task, color in [('regression', 'blue'), ('classification', 'orange')]:
        task_data = results_df[results_df['task'] == task]
        axes[0, 2].scatter(task_data['batch_size'], task_data['metric'],
                           c=color, label=task, alpha=0.7, s=100)
    axes[0, 2].set_xlabel('Batch Size')
    axes[0, 2].set_ylabel('???????')
    axes[0, 2].set_title('??????? ??????? ?????')
    axes[0, 2].legend()

    # ????? ????????
    for task, color in [('regression', 'blue'), ('classification', 'orange')]:
        task_data = results_df[results_df['task'] == task]
        axes[1, 0].scatter(task_data['batch_size'], task_data['training_time'],
                           c=color, label=task, alpha=0.7, s=100)
    axes[1, 0].set_xlabel('Batch Size')
    axes[1, 0].set_ylabel('????? ???????? (?)')
    axes[1, 0].set_title('????? ????????')
    axes[1, 0].legend()

    # ?????????? ???? ?? ??????????
    for task, color in [('regression', 'blue'), ('classification', 'orange')]:
        task_data = results_df[results_df['task'] == task]
        axes[1, 1].scatter(task_data['learning_rate'], task_data['epochs_completed'],
                           c=color, label=task, alpha=0.7, s=100)
    axes[1, 1].set_xlabel('Learning Rate')
    axes[1, 1].set_ylabel('???? ?? ??????????')
    axes[1, 1].set_title('??????????')
    axes[1, 1].legend()

    # Boxplot ?? ?????????????
    reg_melted = reg_results.melt(id_vars=['optimizer'], value_vars=['metric'],
                                  var_name='metric', value_name='value')
    sns.boxplot(data=reg_melted, x='optimizer', y='value', ax=axes[1, 2])
    axes[1, 2].set_title('????????????? RMSE ?? ?????????????')

    plt.tight_layout()
    plt.show()

    print("\n??????????? ????????. ????? ?????????????? 54 ????????????.")