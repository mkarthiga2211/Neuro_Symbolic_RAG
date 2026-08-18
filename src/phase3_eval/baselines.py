import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, Tuple

class BaselineRunner:
    def __init__(self, forecast_horizon: int = 24):
        self.horizon = forecast_horizon

    def _calculate_metrics(self, y_true: np.array, y_pred: np.array) -> Dict[str, float]:
        """Calculates standard regression metrics: MAPE, RMSE, MAE."""
        # Avoid division by zero for MAPE
        epsilon = 1e-10
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        return {"MAPE": mape, "RMSE": rmse, "MAE": mae}

    def run_arima(self, train_series: pd.Series, test_series: pd.Series) -> Dict[str, float]:
        """Fits ARIMA(5,1,0) as a statistical baseline."""
        print("📉 Running ARIMA Baseline...")
        try:
            # Simple autoregressive model
            model = ARIMA(train_series, order=(5, 1, 0))
            model_fit = model.fit()
            # Forecast entire test range (step-by-step or one-shot)
            # For simplicity in this baseline, we do a one-shot forecast for the first window
            forecast = model_fit.forecast(steps=len(test_series))
            return self._calculate_metrics(test_series.values, forecast.values)
        except Exception as e:
            print(f"⚠️ ARIMA Failed: {e}")
            return {"MAPE": np.nan, "RMSE": np.nan}

    def run_prophet(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, float]:
        """Fits Facebook Prophet (Daily Seasonality)."""
        print("🔮 Running Prophet Baseline...")
        # Prepare DataFrames: ds, y
        p_train = train_df.rename(columns={'timestamp': 'ds', 'demand': 'y'})
        p_test = test_df.rename(columns={'timestamp': 'ds', 'demand': 'y'})
        
        m = Prophet(daily_seasonality=True, yearly_seasonality=True)
        m.fit(p_train)
        
        forecast = m.predict(p_test[['ds']])
        return self._calculate_metrics(p_test['y'].values, forecast['yhat'].values)

    def run_bilstm(self, train_loader, test_loader, epochs=5) -> Dict[str, float]:
        """Trains a pure Bi-LSTM on numerical data only (No RAG)."""
        print("🧠 Running Bi-LSTM Baseline...")
        
        class BiLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
                self.head = nn.Linear(64*2, 24) # Output 24h horizon
            def forward(self, x):
                out, _ = self.lstm(x) # x: [batch, 24, 1]
                return self.head(out[:, -1, :])

        model = BiLSTM()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Quick Training Loop
        model.train()
        for _ in range(epochs):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                output = model(batch_x.unsqueeze(-1))
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

        # Evaluation
        model.eval()
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                preds = model(batch_x.unsqueeze(-1))
                all_preds.extend(preds.numpy().flatten())
                all_trues.extend(batch_y.numpy().flatten())

        return self._calculate_metrics(np.array(all_trues), np.array(all_preds))
