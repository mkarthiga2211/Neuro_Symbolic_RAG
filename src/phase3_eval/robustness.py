import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Callable, Any

class RobustnessTester:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.max_capacity = 2000.0 # Physical Grid Limit (Example)
        self.min_load = 0.0        # Physical Base Load

    def _count_violations(self, predictions: torch.Tensor) -> float:
        """Returns the percentage of predictions violating physical constraints."""
        violations = ((predictions > self.max_capacity) | (predictions < self.min_load)).sum().item()
        total_points = predictions.numel()
        return (violations / total_points) * 100.0

    def test_noise_resilience(self, dataloader: Any) -> pd.DataFrame:
        """
        Runs the 'Noisy Context' Stress Test.
        Pass A: Clean RAG Context.
        Pass B: RAG Context corrupted with Gaussian Noise.
        """
        print("🛡️ Starting Robustness Stress Test (Clean vs. Noisy)...")
        results = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (ts_window, clean_rag) in enumerate(tqdm(dataloader)):
                # --- Pass A: Clean Context ---
                out_clean = self.model(ts_window, clean_rag, self.min_load, self.max_capacity)
                viol_clean_raw = self._count_violations(out_clean['raw_forecast'])
                viol_clean_final = self._count_violations(out_clean['final_forecast'])

                # --- Pass B: Noisy Context (Simulating Hallucination) ---
                # Inject Gaussian Noise matching the embedding stats (mean=0, std=1 typically)
                noise = torch.randn_like(clean_rag) * 2.0 # High variance noise
                noisy_rag = clean_rag + noise
                
                out_noisy = self.model(ts_window, noisy_rag, self.min_load, self.max_capacity)
                viol_noisy_raw = self._count_violations(out_noisy['raw_forecast'])
                viol_noisy_final = self._count_violations(out_noisy['final_forecast'])
                
                # --- Quantify Drift ---
                # How much did the prediction change due to noise?
                drift = torch.mean(torch.abs(out_clean['final_forecast'] - out_noisy['final_forecast'])).item()

                results.append({
                    "Batch": batch_idx,
                    "Violation_Clean_Raw": viol_clean_raw,
                    "Violation_Clean_Final": viol_clean_final,
                    "Violation_Noisy_Raw": viol_noisy_raw,
                    "Violation_Noisy_Final": viol_noisy_final, # THIS SHOULD BE 0.0
                    "Prediction_Drift": drift
                })

        df = pd.DataFrame(results)
        
        # Summary Statistics
        print("\n=== Robustness Test Results ===")
        print(f"Typical Violation Rate (Raw - Noisy): {df['Violation_Noisy_Raw'].mean():.2f}%")
        print(f"Safety Anchor Rate (Final - Noisy):   {df['Violation_Noisy_Final'].mean():.2f}% (Should be 0%)")
        
        if df['Violation_Noisy_Final'].mean() < 0.01:
            print("✅ HYPOTHESIS VALIDATED: Symbolic Layer successfully anchored predictions despite noisy context.")
        else:
            print("❌ HYPOTHESIS FAILED: Constraint barrier breached.")
            
        return df

if __name__ == "__main__":
    # Mock Test for the Module
    from src.phase2_model.neuro_symbolic import NeuroSymbolicForecaster
    
    # 1. Setup Mock Model
    model = NeuroSymbolicForecaster(forecast_horizon=24)
    
    # 2. Mock Dataloader (List of tuples)
    mock_loader = [
        (torch.randn(8, 24), torch.randn(8, 384)), # Batch 1
        (torch.randn(8, 24), torch.randn(8, 384))  # Batch 2
    ]
    
    tester = RobustnessTester(model)
    tester.test_noise_resilience(mock_loader)
