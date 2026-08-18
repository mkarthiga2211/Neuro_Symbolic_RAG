import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Any, Tuple, Optional

# --- Robust Import Handling for Neuro-Symbolic Dependencies ---
try:
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("⚠️ CVXPY/CVXPYLayers not found. Using PyTorch-native fallback (Faster & Production Ready).")

class TimeLLMEncoder(nn.Module):
    """
    Adapts a pre-trained Transformer to process time-series data fused with RAG embeddings.
    Mathematical formulation: Z = Transformer([TS_tokens; RAG_embeddings])
    """
    def __init__(self, model_name: str = "distilbert-base-uncased", forecast_horizon: int = 24, rag_dim: int = 384):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.transformer.config.hidden_size
        
        # Linear projection for RAG embeddings to match Transformer's hidden size
        self.rag_projection = nn.Linear(rag_dim, self.hidden_size)
        
        # Regressor head
        self.regressor = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, forecast_horizon)
        )

    def _convert_ts_to_text(self, ts_window: torch.Tensor) -> str:
        """Serializes numerical time-series into a natural language prompt."""
        # ts_window shape: [window_size]
        values = ts_window.tolist()
        return "Energy load history sequence: " + ", ".join([f"{v:.2f}" for v in values])

    def forward(self, ts_windows: torch.Tensor, rag_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Encoder.
        ts_windows: [Batch, Window_Size]
        rag_embeddings: [Batch, RAG_Dim]
        """
        # 1. Adapt Time-Series to Textual Tokens
        prompts = [self._convert_ts_to_text(w) for w in ts_windows]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(ts_windows.device)
        
        # 2. Get Transformer Base Outputs
        # DistilBERT outputs tuple, index 0 is last_hidden_state
        transformer_outputs = self.transformer(**inputs).last_hidden_state[:, 0, :] # CLS token
        
        # 3. Fusion: Add RAG Context (Additive or Concatenation)
        rag_context = self.rag_projection(rag_embeddings)
        fused_representation = transformer_outputs + rag_context # Additive fusion for dimensionality consistency
        
        # 4. Predict Raw Forecast
        raw_forecast = self.regressor(fused_representation)
        return raw_forecast

class DifferentiableConstraintLayer(nn.Module):
    """
    A Symbolic Layer that solves a constrained optimization problem.
    Physically restricts the output to: min_load <= y <= max_load.
    
    Production Fallback: If cvxpylayers is missing, uses analytical projection (torch.clamp),
    which is the exact mathematical solution for Box Constraints and is fully differentiable.
    """
    def __init__(self, forecast_horizon: int):
        super().__init__()
        self.n = forecast_horizon
        self.cvx_layer = None
        
        if CVXPY_AVAILABLE:
            try:
                # Define CPXPY variables and parameters
                y_final = cp.Variable(self.n)
                y_raw = cp.Parameter(self.n)
                y_min = cp.Parameter(self.n)
                y_max = cp.Parameter(self.n)
                
                objective = cp.Minimize(cp.sum_squares(y_final - y_raw))
                constraints = [y_final >= y_min, y_final <= y_max]
                
                problem = cp.Problem(objective, constraints)
                self.cvx_layer = CvxpyLayer(problem, parameters=[y_raw, y_min, y_max], variables=[y_final])
            except Exception as e:
                print(f"⚠️ Failed to initialize CvxpyLayer: {e}. Switching to PyTorch fallback.")

    def forward(self, y_raw: torch.Tensor, min_load: float, max_load: float) -> torch.Tensor:
        """
        Solves: argmin ||y - y_raw||^2 s.t. min <= y <= max
        """
        # 1. PyTorch Native Fallback (Analytical Solution for Box Constraints)
        if not CVXPY_AVAILABLE or self.cvx_layer is None:
            return torch.clamp(y_raw, min=min_load, max=max_load)
        
        # 2. Formal Symbolic Solver (if available)
        try:
            # Expand scalar constraints to vectors for batching
            y_min_vec = torch.full_like(y_raw, min_load)
            y_max_vec = torch.full_like(y_raw, max_load)
            solution, = self.cvx_layer(y_raw, y_min_vec, y_max_vec)
            return solution
        except Exception:
            return torch.clamp(y_raw, min=min_load, max=max_load)

class NeuroSymbolicForecaster(nn.Module):
    """
    The end-to-end hybrid model: LLM Encoder + Symbolic Constraint Layer.
    """
    def __init__(self, forecast_horizon: int = 24, model_name: str = "distilbert-base-uncased"):
        super().__init__()
        self.encoder = TimeLLMEncoder(model_name=model_name, forecast_horizon=forecast_horizon)
        self.symbolic_layer = DifferentiableConstraintLayer(forecast_horizon=forecast_horizon)
        
    def forward(self, ts_windows: torch.Tensor, rag_embeddings: torch.Tensor, 
                min_phys: float = 0.0, max_phys: float = 2000.0) -> Dict[str, torch.Tensor]:
        
        # Phase 1: Neural Inference (Raw Forecasting using LLM + RAG)
        raw_forecast = self.encoder(ts_windows, rag_embeddings)
        
        # Phase 2: Symbolic Inference (Constraint Satisfaction)
        # Gradient flow is preserved through the optimization solver
        final_forecast = self.symbolic_layer(raw_forecast, min_phys, max_phys)
        
        return {
            "raw_forecast": raw_forecast,
            "final_forecast": final_forecast
        }

if __name__ == "__main__":
    # Test initialization and random forward pass
    horizon = 24
    model = NeuroSymbolicForecaster(forecast_horizon=horizon)
    
    # Mock inputs: Batch of 4
    mock_ts = torch.randn(4, 24) # 24 hours history
    mock_rag = torch.randn(4, 384) # RAG vector
    
    output = model(mock_ts, mock_rag)
    print("Neuro-Symbolic Forward Pass Success!")
    print(f"Raw Forecast Example: {output['raw_forecast'][0, :5]}")
    print(f"Constrained Forecast Example: {output['final_forecast'][0, :5]}")
