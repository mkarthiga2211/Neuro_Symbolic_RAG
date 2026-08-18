import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from src.phase2_model.neuro_symbolic import NeuroSymbolicForecaster

class TimeSeriesRAGDataset(Dataset):
    def __init__(self, parquet_file, window_size=24, horizon=24):
        print(f"Loading data from {parquet_file}...")
        self.df = pd.read_parquet(parquet_file)
        self.window_size = window_size
        self.horizon = horizon
        
        # Normalize Demand
        self.demand_mean = self.df['demand'].mean()
        self.demand_std = self.df['demand'].std()
        self.norm_demand = (self.df['demand'].values - self.demand_mean) / self.demand_std
        
        # Get RAG Vectors
        vector_cols = [c for c in self.df.columns if c.startswith('v_')]
        self.rag_vectors = self.df[vector_cols].values.astype(np.float32)

    def __len__(self):
        return len(self.df) - self.window_size - self.horizon

    def __getitem__(self, idx):
        # Input Window
        x_window = self.norm_demand[idx : idx + self.window_size]
        rag_vec = self.rag_vectors[idx + self.window_size] # Use the context of the prediction start time
        
        # Target Horizon
        y_target = self.norm_demand[idx + self.window_size : idx + self.window_size + self.horizon]
        
        return torch.FloatTensor(x_window), torch.FloatTensor(rag_vec), torch.FloatTensor(y_target)

class NeuroSymbolicTrainer:
    def __init__(self, data_path, model_save_path="results/models/nsrag_best.pt", epochs=5, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Training Device: {self.device}")
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.save_path = Path(model_save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Data
        self.dataset = TimeSeriesRAGDataset(data_path)
        train_size = int(0.8 * len(self.dataset))
        test_size = len(self.dataset) - train_size
        self.train_set, self.test_set = torch.utils.data.random_split(self.dataset, [train_size, test_size])
        
        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_set, batch_size=batch_size)
        
        # Model
        self.model = NeuroSymbolicForecaster().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()

    def train(self):
        print("🔥 Starting Training Loop...")
        best_loss = float('inf')
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            for x, rag, y in pbar:
                x, rag, y = x.to(self.device), rag.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(x, rag)
                
                # Loss on Final Forecast (Symbolic Guidance) + Loss on Raw Forecast (Auxiliary Task)
                loss_final = self.criterion(outputs['final_forecast'], y)
                loss_raw = self.criterion(outputs['raw_forecast'], y)
                loss = loss_final + 0.5 * loss_raw
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            # Validation
            val_loss = self.evaluate()
            print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(self.train_loader):.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), self.save_path)
                print(f"💾 Model Saved: {self.save_path}")

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, rag, y in self.test_loader:
                x, rag, y = x.to(self.device), rag.to(self.device), y.to(self.device)
                outputs = self.model(x, rag)
                loss = self.criterion(outputs['final_forecast'], y)
                total_loss += loss.item()
        return total_loss / len(self.test_loader)

if __name__ == "__main__":
    # Example usage
    trainer = NeuroSymbolicTrainer(data_path="data/processed/aligned_train_data.parquet", epochs=2)
    trainer.train()
