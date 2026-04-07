import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')


class CSVDataset(Dataset):
    def __init__(self, csv_file, target_col=None, categorical_ordinal=None,
                 drop_cols=None, test_size=0.2, random_state=42, transform=None):
        self.transform = transform
        self.target_col = target_col

        # 1. ???????? ??????
        print(f"? ???????? {csv_file}...")
        self.data = pd.read_csv(csv_file)
        print(f"   ??????: {self.data.shape}")

        # 2. ???????? ???????? ????????
        if drop_cols:
            self.data = self.data.drop(columns=drop_cols)

        # 3. ??????????????? ????? ???????? (??????????)
        self._identify_column_types(categorical_ordinal)

        # 4. ????????????? (??????????)
        self._preprocess_data()

        # 5. ?????????? train/val
        if self.target_col and test_size > 0:
            self._split_data(test_size, random_state)

    def _identify_column_types(self, categorical_ordinal=None):
        """? ???????????? ??????????????? - ??? ??????????"""
        if categorical_ordinal is None:
            categorical_ordinal = []

        all_cols = self.data.columns.tolist()

        # ????????? target
        if self.target_col and self.target_col in all_cols:
            all_cols.remove(self.target_col)

        # ???????? ??????? (?????? float/int)
        num_mask = self.data[all_cols].select_dtypes(include=[np.number]).columns
        self.num_cols = num_mask.tolist()

        # ???????? (0/1)
        self.binary_cols = []
        for col in self.num_cols:
            unique_vals = self.data[col].dropna().unique()
            if len(unique_vals) <= 2 and np.isin(unique_vals, [0, 1]).all():
                self.binary_cols.append(col)
                self.num_cols.remove(col)  # ? ??????? ?? num_cols

        # ?????????????? object/category
        cat_mask = self.data.select_dtypes(include=['object', 'category']).columns
        cat_cols = cat_mask.tolist()

        # ??????????
        self.ordinal_cols = [col for col in categorical_ordinal if col in cat_cols]

        # ??????????? (????????? ??????????????)
        self.nominal_cols = [col for col in cat_cols
                             if col not in self.ordinal_cols]

        print(f"   ????: num={len(self.num_cols)}, bin={len(self.binary_cols)}, "
              f"nom={len(self.nominal_cols)}, ord={len(self.ordinal_cols)}")

    def _preprocess_data(self):
        """? ???????????? ?????????????"""
        # ????????? ?????????
        for col in self.num_cols + self.binary_cols:
            self.data[col] = SimpleImputer(strategy='mean').fit_transform(
                self.data[[col]]).flatten()

        for col in self.nominal_cols + self.ordinal_cols:
            if col in self.data.columns:
                self.data[col] = SimpleImputer(strategy='most_frequent').fit_transform(
                    self.data[[col]]).flatten()

        # Label Encoding ??? ??????????
        if self.ordinal_cols:
            self.label_encoders = {}
            for col in self.ordinal_cols:
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col].astype(str))
                self.label_encoders[col] = le

        # ? ?????? ?????? ?????????? ????????
        feature_cols = (self.num_cols + self.binary_cols +
                        self.nominal_cols + self.ordinal_cols)

        # ??????? preprocessor ?????? ??? ?????? ????????
        transformers = []

        if self.num_cols:
            transformers.append(('num', StandardScaler(), self.num_cols))

        if self.nominal_cols:
            transformers.append(('nominal',
                                 OneHotEncoder(sparse_output=False, drop='first'),
                                 self.nominal_cols))

        # ?? ??????? binary_cols ? ordinal_cols - ??? ??? ??????

        if transformers:
            preprocessor = ColumnTransformer(
                transformers,
                remainder='passthrough',
                verbose_feature_names_out=False
            )
            X_temp = self.data[feature_cols]
            self.X_processed = preprocessor.fit_transform(X_temp)
            self.feature_names = preprocessor.get_feature_names_out()
        else:
            self.X_processed = self.data[feature_cols].values
            self.feature_names = feature_cols

        print(f"   ?????????: {self.X_processed.shape[1]}")

    def _split_data(self, test_size, random_state):
        np.random.seed(random_state)
        indices = np.arange(len(self.data))
        np.random.shuffle(indices)

        split_idx = int(len(indices) * (1 - test_size))
        self.train_indices = indices[:split_idx]
        self.val_indices = indices[split_idx:]

        self.X_train = self.X_processed[self.train_indices]
        self.X_val = self.X_processed[self.val_indices]

        if self.target_col:
            self.y_train = self.data[self.target_col].iloc[self.train_indices].values
            self.y_val = self.data[self.target_col].iloc[self.val_indices].values

    def __len__(self):
        return len(self.X_train) if self.target_col else len(self.X_processed)

    def __getitem__(self, idx):
        if self.target_col:
            if idx < len(self.X_train):
                X = torch.FloatTensor(self.X_train[idx])
                y = torch.tensor(self.y_train[idx],
                                 dtype=torch.float32 if np.issubdtype(self.y_train.dtype, np.floating) else torch.long)
            else:
                val_idx = idx - len(self.X_train)
                X = torch.FloatTensor(self.X_val[val_idx])
                y = torch.tensor(self.y_val[val_idx],
                                 dtype=torch.float32 if np.issubdtype(self.y_val.dtype, np.floating) else torch.long)
        else:
            X = torch.FloatTensor(self.X_processed[idx])
            y = torch.zeros(1)

        if self.transform:
            X = self.transform(X)

        return X, y


# ? ????
if __name__ == "__main__":
    np.random.seed(42)
    data = {
        'age': np.random.randint(18, 80, 1000),
        'salary': np.random.normal(50000, 15000, 1000),
        'city': np.random.choice(['Moscow', 'SPb', 'Ekaterinburg'], 1000),
        'education': np.random.choice(['School', 'College', 'Bachelor'], 1000),
        'has_car': np.random.choice([0, 1], 1000),
        'income_level': np.random.choice([0, 1], 1000)
    }
    df = pd.DataFrame(data)
    df.to_csv('demo_data.csv', index=False)

    dataset = CSVDataset(
        'demo_data.csv',
        target_col='income_level',
        categorical_ordinal=['education'],
        drop_cols=None
    )

    X, y = dataset[0]
    print(f"\n? ????????! X.shape={X.shape}, y={y}")