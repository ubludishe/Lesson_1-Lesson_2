import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


#  CSVDataset (?????? ??????? ??????)
class CSVDataset(Dataset):
    def __init__(self, csv_file, target_col=None, categorical_ordinal=None,
                 drop_cols=None, test_size=0.2, random_state=42):
        self.target_col = target_col

        # ????????
        self.data = pd.read_csv(csv_file)
        print(f" {csv_file}: {self.data.shape}")

        if drop_cols:
            self.data = self.data.drop(columns=drop_cols)

        # ??????????? ????? ???????
        self._identify_column_types(categorical_ordinal)
        self._preprocess_data()
        self._split_data(test_size, random_state)

    def _identify_column_types(self, categorical_ordinal=None):
        if categorical_ordinal is None:
            categorical_ordinal = []

        all_cols = self.data.columns.tolist()
        if self.target_col:
            if self.target_col in all_cols:
                all_cols.remove(self.target_col)

        # ????????
        self.num_cols = self.data[all_cols].select_dtypes(include=[np.number]).columns.tolist()
        self.binary_cols = []

        # ???????? ?? ????????
        for col in self.num_cols[:]:
            unique = self.data[col].dropna().unique()
            if len(unique) == 2 and set(unique) <= {0, 1}:
                self.binary_cols.append(col)
                self.num_cols.remove(col)

        # ??????????????
        cat_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        self.ordinal_cols = [col for col in categorical_ordinal if col in cat_cols]
        self.nominal_cols = [col for col in cat_cols if col not in self.ordinal_cols]

    def _preprocess_data(self):
        # ?????????? ?????????
        for col in self.num_cols + self.binary_cols:
            self.data[col] = self.data[col].fillna(self.data[col].mean())
        for col in self.nominal_cols + self.ordinal_cols:
            self.data[col] = self.data[col].fillna(self.data[col].mode().iloc[0])

        # Label encoding ??? ordinal
        for col in self.ordinal_cols:
            self.data[col] = pd.Categorical(self.data[col]).codes

        # Preprocessing pipeline
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer

        feature_cols = self.num_cols + self.binary_cols + self.nominal_cols + self.ordinal_cols

        transformers = []
        if self.num_cols:
            transformers.append(('num', StandardScaler(), self.num_cols))
        if self.nominal_cols:
            transformers.append(('cat', OneHotEncoder(sparse_output=False, drop='first'), self.nominal_cols))

        if transformers:
            ct = ColumnTransformer(transformers, remainder='passthrough', verbose_feature_names_out=False)
            self.X_processed = ct.fit_transform(self.data[feature_cols])
            self.feature_names = ct.get_feature_names_out()
        else:
            self.X_processed = self.data[feature_cols].values
            self.feature_names = np.array(feature_cols)

    def _split_data(self, test_size, random_state):
        indices = np.arange(len(self.data))
        np.random.seed(random_state)
        np.random.shuffle(indices)

        split = int(len(indices) * (1 - test_size))
        self.train_idx, self.val_idx = indices[:split], indices[split:]

        self.X_train = self.X_processed[self.train_idx]
        self.X_val = self.X_processed[self.val_idx]
        self.y_train = self.data[self.target_col].iloc[self.train_idx].values
        self.y_val = self.data[self.target_col].iloc[self.val_idx].values

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X_train[idx])
        y = torch.FloatTensor([self.y_train[idx]])
        return x, y


#  ????????? California Housing Dataset (?????????)
def create_housing_dataset():
    np.random.seed(42)
    n = 2000
    data = {
        'income': np.clip(np.random.normal(3.5, 1.5, n), 0.5, 10),
        'rooms_per_household': np.clip(np.random.normal(6, 1, n), 1, 15),
        'bedrooms_per_room': np.clip(np.random.normal(0.15, 0.05, n), 0.1, 0.3),
        'population_per_household': np.clip(np.random.normal(2.8, 0.7, n), 1, 6),
        'ocean_proximity': np.random.choice([0, 1, 2, 3], n)
    }
    df = pd.DataFrame(data)
    # ???????????? ??????? ????
    df['price'] = (50000 * df['income'] +
                   20000 * df['rooms_per_household'] -
                   50000 * df['bedrooms_per_room'] +
                   np.random.normal(0, 30000, n))
    df.to_csv('housing.csv', index=False)
    print(" Housing dataset ?????? (2000 ?????)")
    return 'housing.csv'


#  ????????? Bank Marketing Dataset (?????????????)
def create_bank_dataset():
    np.random.seed(42)
    n = 3000
    data = {
        'age': np.random.randint(18, 70, n),
        'balance': np.random.normal(1500, 1000, n),
        'day': np.random.randint(1, 32, n),
        'duration': np.random.exponential(300, n).clip(0, 2000),
        'campaign': np.random.poisson(2.5, n).clip(1, 10),
        'job': np.random.choice(['admin', 'blue-collar', 'technician'], n),
        'marital': np.random.choice(['married', 'single', 'divorced'], n)
    }
    df = pd.DataFrame(data)
    # ???????????? ?????? ????????
    prob_subscribe = 0.1 + 0.001 * df['duration'] - 0.0001 * df['age'] + 0.01 * (df['balance'] > 2000)
    df['subscribe'] = (np.random.random(n) < np.clip(prob_subscribe, 0, 1)).astype(int)
    df.to_csv('bank.csv', index=False)
    print(" Bank dataset ?????? (3000 ?????)")
    return 'bank.csv'


# ??????
class LinearRegression(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)
        nn.init.kaiming_uniform_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)
        nn.init.kaiming_uniform_(self.linear.weight)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def train_model(dataset, model_class, criterion, epochs=300, lr=0.01, task='regression'):
    model = model_class(dataset.X_processed.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        x_batch = torch.FloatTensor(dataset.X_train)
        y_batch = torch.FloatTensor(dataset.y_train).view(-1, 1)

        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 100 == 0:
            print(f"????? {epoch}: loss={loss:.4f}")

    # ??????
    model.eval()
    with torch.no_grad():
        x_val = torch.FloatTensor(dataset.X_val)
        y_pred = model(x_val).numpy()

    if task == 'regression':
        rmse = np.sqrt(np.mean((dataset.y_val - y_pred.flatten()) ** 2))
        print(f" RMSE: {rmse:.1f}")
        return model, losses, rmse
    else:
        acc = np.mean((y_pred.flatten() > 0.5) == dataset.y_val)
        print(f" Accuracy: {acc:.3f}")
        return model, losses, acc


# ? ??????? ????
if __name__ == "__main__":
    print(" ???????? ???????: ???????? + ????????????? ?????????")
    print("=" * 60)

    # 1. ??????? ???????? ????????
    housing_file = create_housing_dataset()
    bank_file = create_bank_dataset()

    print("\n 1. ???????? ????????? (????: ???? ?????)")
    print("-" * 40)
    housing_ds = CSVDataset(housing_file, target_col='price',
                            categorical_ordinal=['ocean_proximity'])

    reg_model, reg_losses, reg_rmse = train_model(
        housing_ds, LinearRegression, nn.MSELoss(), task='regression'
    )

    print("\n? 2. ????????????? ????????? (????: ???????? ?? ???????)")
    print("-" * 40)
    bank_ds = CSVDataset(bank_file, target_col='subscribe',
                         categorical_ordinal=['job', 'marital'])

    cls_model, cls_losses, cls_acc = train_model(
        bank_ds, LogisticRegression, nn.BCEWithLogitsLoss(), task='classification'
    )

    # 3. ????????????
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # ????????
    axes[0, 0].plot(reg_losses, label='Regression')
    axes[0, 0].plot(cls_losses, label='Classification')
    axes[0, 0].set_title('???????? ????????')
    axes[0, 0].legend()

    # ???????????? ?????????
    with torch.no_grad():
        reg_pred = reg_model(torch.FloatTensor(housing_ds.X_val)).numpy()
    axes[0, 1].scatter(housing_ds.y_val[:100], reg_pred[:100].flatten(), alpha=0.6)
    min_val, max_val = min(housing_ds.y_val.min(), reg_pred.min()), max(housing_ds.y_val.max(), reg_pred.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--')
    axes[0, 1].set_xlabel('???????? ????')
    axes[0, 1].set_ylabel('?????????????')
    axes[0, 1].set_title(f'RMSE: {reg_rmse:.0f}')

    # ???????????? ?????????????
    with torch.no_grad():
        cls_pred = torch.sigmoid(cls_model(torch.FloatTensor(bank_ds.X_val))).numpy().flatten()
    axes[1, 0].hist(cls_pred[bank_ds.y_val == 0], bins=30, alpha=0.7, label='?? ??????????')
    axes[1, 0].hist(cls_pred[bank_ds.y_val == 1], bins=30, alpha=0.7, label='??????????')
    axes[1, 0].axvline(0.5, color='r', ls='--')
    axes[1, 0].legend()
    axes[1, 0].set_title(f'Accuracy: {cls_acc:.3f}')

    # ???????? ????????? (?????? ????)
    reg_weights = reg_model.linear.weight.detach().numpy().flatten()
    axes[1, 1].bar(range(min(10, len(reg_weights))), reg_weights[:10])
    axes[1, 1].set_title('???? ???????? ?????????')
    axes[1, 1].set_xticks(range(min(10, len(reg_weights))))
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    print(f"? ????????? RMSE: {reg_rmse:.1f}")
    print(f"? ????????????? Accuracy: {cls_acc:.3f}")