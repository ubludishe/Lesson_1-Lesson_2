import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, accuracy_score


def create_nonlinear_dataset(n_samples=2000, n_features=6):
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)

    # ?????????? ???????????
    f0, f1, f2 = X[:, 0], X[:, 1], X[:, 2]
    y_reg = (2 * f0 ** 2 - 1.5 * f1 * f2 + 0.5 * f0 * f1 + np.random.normal(0, 0.1, n_samples))

    logits = f0 ** 2 - f1 * f2 + 0.3 * f0
    y_cls = (1 / (1 + np.exp(-logits)) > 0.5).astype(np.float32)

    n_train = int(0.8 * n_samples)

    return {
        'regression': {
            'X_train': torch.FloatTensor(X[:n_train]),
            'X_val': torch.FloatTensor(X[n_train:]),
            'y_train': torch.FloatTensor(y_reg[:n_train]).view(-1, 1),
            'y_val': torch.FloatTensor(y_reg[n_train:]).view(-1, 1)
        },
        'classification': {
            'X_train': torch.FloatTensor(X[:n_train]),
            'X_val': torch.FloatTensor(X[n_train:]),
            'y_train': torch.FloatTensor(y_cls[:n_train]).view(-1, 1),
            'y_val': torch.FloatTensor(y_cls[n_train:]).view(-1, 1)
        }
    }


class LinearRegression(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)


class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def train_model(X_train, y_train, X_val, y_val, model_class, epochs=200, lr=0.01):
    model = model_class(X_train.shape[1])
    criterion = nn.MSELoss() if issubclass(model_class, LinearRegression) else nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        if issubclass(model_class, LinearRegression):
            return np.sqrt(mean_squared_error(y_val.numpy(), val_pred.numpy()))
        else:
            return accuracy_score(y_val.numpy(), (val_pred.numpy() > 0.5).astype(int))


class FeatureEngineer:
    def __init__(self, X_train, X_val):
        self.X_train = X_train.numpy()
        self.X_val = X_val.numpy()

    def polynomial_features(self):
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(self.X_train[:, :3])
        X_val_poly = poly.transform(self.X_val[:, :3])
        return torch.FloatTensor(X_train_poly), torch.FloatTensor(X_val_poly)

    def interaction_features(self):
        f0_tr, f1_tr = self.X_train[:, 0], self.X_train[:, 1]
        f0_val, f1_val = self.X_val[:, 0], self.X_val[:, 1]

        inter_tr = np.column_stack([f0_tr * f1_tr, f0_tr ** 2, f1_tr ** 2])
        inter_val = np.column_stack([f0_val * f1_val, f0_val ** 2, f1_val ** 2])

        return torch.FloatTensor(inter_tr), torch.FloatTensor(inter_val)

    def statistical_features(self):
        stats_train = []
        stats_val = []

        # ?????????? ???????!
        for i in range(0, 4, 2):
            mean_tr = (self.X_train[:, i] + self.X_train[:, i + 1]) / 2
            ratio_tr = self.X_train[:, i] / (self.X_train[:, i + 1] + 1e-8)
            stats_train.extend([mean_tr, ratio_tr])

            mean_val = (self.X_val[:, i] + self.X_val[:, i + 1]) / 2
            ratio_val = self.X_val[:, i] / (self.X_val[:, i + 1] + 1e-8)
            stats_val.extend([mean_val, ratio_val])

        return (torch.FloatTensor(np.column_stack(stats_train)),
                torch.FloatTensor(np.column_stack(stats_val)))

    def combined_features(self):
        poly_tr, poly_val = self.polynomial_features()
        inter_tr, inter_val = self.interaction_features()
        stat_tr, stat_val = self.statistical_features()

        base_tr = torch.FloatTensor(self.X_train)
        base_val = torch.FloatTensor(self.X_val)

        combined_tr = torch.cat([base_tr, poly_tr[:, 3:7], inter_tr, stat_tr], dim=1)
        combined_val = torch.cat([base_val, poly_val[:, 3:7], inter_val, stat_val], dim=1)

        return combined_tr, combined_val


if __name__ == "__main__":
    print("Feature Engineering ???????????")
    print("=" * 40)

    data = create_nonlinear_dataset()

    # ??????? ??????
    print("\n??????? ?????? (6 ?????????)")
    base_reg_rmse = train_model(data['regression']['X_train'],
                                data['regression']['y_train'],
                                data['regression']['X_val'],
                                data['regression']['y_val'],
                                LinearRegression)
    base_cls_acc = train_model(data['classification']['X_train'],
                               data['classification']['y_train'],
                               data['classification']['X_val'],
                               data['classification']['y_val'],
                               LogisticRegression)

    print(f"????????? RMSE: {base_reg_rmse:.4f}")
    print(f"????????????? Accuracy: {base_cls_acc:.4f}")

    # Feature Engineering
    reg_eng = FeatureEngineer(data['regression']['X_train'], data['regression']['X_val'])
    cls_eng = FeatureEngineer(data['classification']['X_train'], data['classification']['X_val'])

    methods = {
        'Polynomial': reg_eng.polynomial_features(),
        'Interactions': reg_eng.interaction_features(),
        'Statistical': reg_eng.statistical_features(),
        'Combined': reg_eng.combined_features()
    }

    reg_results = []
    cls_results = []

    print("\n?????? ? ?????? ??????????")
    print("-" * 30)

    for name, (X_tr, X_val) in methods.items():
        print(f"{name}:")

        rmse = train_model(X_tr, data['regression']['y_train'],
                           X_val, data['regression']['y_val'], LinearRegression)
        reg_results.append({'method': name, 'rmse': rmse, 'features': X_tr.shape[1]})

        acc = train_model(X_tr, data['classification']['y_train'],
                          X_val, data['classification']['y_val'], LogisticRegression)
        cls_results.append({'method': name, 'accuracy': acc, 'features': X_tr.shape[1]})

        print(f"  RMSE: {rmse:.4f}, Acc: {acc:.4f} ({X_tr.shape[1]} ?????????)")

    # ??????????
    reg_df = pd.DataFrame(reg_results)
    cls_df = pd.DataFrame(cls_results)

    reg_df['improvement'] = base_reg_rmse - reg_df['rmse']
    cls_df['improvement'] = cls_df['accuracy'] - base_cls_acc

    print("\n" + "=" * 40)
    print("????????? ??????????")
    print("=" * 40)
    print("?????????:")
    print(reg_df[['method', 'rmse', 'features', 'improvement']].round(4))
    print("\n?????????????:")
    print(cls_df[['method', 'accuracy', 'features', 'improvement']].round(4))

    # ???????
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    methods_all = ['Baseline'] + reg_df['method'].tolist()
    axes[0].bar(methods_all, [base_reg_rmse] + reg_df['rmse'].tolist())
    axes[0].set_title('????????? RMSE')
    axes[0].tick_params(axis='x', rotation=45)

    axes[1].bar(methods_all, [base_cls_acc] + cls_df['accuracy'].tolist(), color='green')
    axes[1].set_title('????????????? Accuracy')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()