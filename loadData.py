from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch
import pandas as pd

class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length, target_column, feature_columns):
        self.data = data.reset_index(drop=True)
        self.sequence_length = sequence_length
        self.target_column = target_column
        self.feature_columns = feature_columns
        
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        x = self.data.iloc[idx:idx + self.sequence_length][self.feature_columns].values
        y = self.data.iloc[idx + self.sequence_length][self.target_column]
        return torch.FloatTensor(x), torch.FloatTensor([y])

def prepare_data(file_path, sequence_length=48, target_column='Lake water level (m)'):
    df_pl = pl.read_csv(file_path)
    df = df_pl.to_pandas()
    print(f"Data loaded {len(df)} rows, {len(df.columns)} columns")
    
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)
    
    feature_columns = [col for col in df.columns if col not in ['Time', target_column]]
    feature_columns.append(target_column)
    
    print(f"Feature columns: {feature_columns}")
    
    scaler_features = StandardScaler()
    scaler_target = StandardScaler()
    
    df_scaled = df.copy()
    df_scaled[feature_columns] = scaler_features.fit_transform(df[feature_columns])
    
    train_size = int(0.8 * len(df_scaled))
    print(f"Train size: {train_size}, Test size: {len(df_scaled) - train_size}")
    
    train_data = df_scaled[:train_size].reset_index(drop=True)
    test_data = df_scaled[train_size:].reset_index(drop=True)
    
    print("Creating datasets...")
    train_dataset = TimeSeriesDataset(train_data, sequence_length, target_column, feature_columns)
    test_dataset = TimeSeriesDataset(test_data, sequence_length, target_column, feature_columns)
    
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")
    
    return train_dataset, test_dataset, scaler_features, scaler_target