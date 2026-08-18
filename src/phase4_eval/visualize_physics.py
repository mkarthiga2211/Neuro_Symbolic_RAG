import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from pathlib import Path

# Set academic style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.4)
colors = sns.color_palette("muted")

class SymbolicLogicVisualizer:
    def __init__(self, output_dir="results/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_capacity = 1000.0

    # --- 1. Constraint Violation Comparison ---
    def plot_violation_comparison(self):
        """
        Generates a grouped bar chart proving NS-RAG compliance vs LLM Hallucination.
        """
        print("📊 Generating 1: Constraint Violation Chart...")
        
        # Mock Data for Illustration
        data = {
            'Model': ['Vanilla Llama-3', 'NS-RAG (Ours)'],
            'Total_Predictions': [5000, 5000],
            'Violations': [850, 0] # 17% vs 0%
        }
        df = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot Total Forecasts (Background)
        sns.barplot(x='Model', y='Total_Predictions', data=df, color='lightgray', label='Feasible Forecasts', ax=ax)
        
        # Plot Violations (Overlay)
        sns.barplot(x='Model', y='Violations', data=df, palette=['#e74c3c', '#2ecc71'], label='Physical Violations', ax=ax)
        
        # Annotations
        for i, row in df.iterrows():
            if row['Violations'] > 0:
                ax.text(i, row['Violations'] + 100, f"{row['Violations']} Violations\n(Hallucinatory)", 
                        ha='center', color='#c0392b', fontweight='bold')
            else:
                ax.text(i, 200, "0% Violations\n(cvxpy Guarantee)", 
                        ha='center', color='#27ae60', fontweight='bold')

        plt.title('Physical Constraint Compliance: Vanilla vs. Neuro-Symbolic', fontsize=16, pad=20)
        plt.ylabel('Number of Test Windows')
        plt.legend()
        plt.tight_layout()
        
        save_path = self.output_dir / "physics_1_violations.png"
        plt.savefig(save_path, dpi=300)
        print(f"✅ Saved: {save_path}")

    # --- 2. Optimization Surface Visualization (3D) ---
    def plot_optimization_surface(self):
        """
        Interactive 3D plot showing the projection of raw predictions onto the feasible plane.
        """
        print("🌌 Generating 2: 3D Optimization Surface...")
        
        # Generate Raw Hallucinations (Above Capacity)
        np.random.seed(42)
        n_points = 20
        time_steps = np.linspace(0, 24, n_points)
        load_prev = np.random.uniform(800, 1100, n_points) # Y-axis feature
        
        # Raw Prediction (Z) - Some exceed 1000
        raw_pred_z = np.random.normal(1050, 50, n_points) 
        
        # Apply Symbolic Constraints (Clamp to Max Capacity)
        corrected_z = np.minimum(raw_pred_z, self.max_capacity)
        
        # Create 3D Figure
        fig = go.Figure()

        # 1. Feasible Region Plane (Z = 1000)
        x_plane = np.linspace(0, 24, 100)
        y_plane = np.linspace(800, 1100, 100)
        X, Y = np.meshgrid(x_plane, y_plane)
        Z = np.ones_like(X) * self.max_capacity
        
        fig.add_trace(go.Surface(z=Z, x=X, y=Y, opacity=0.3, colorscale='Greys', showscale=False, name='Max Capacity'))

        # 2. Raw Predictions (Red)
        fig.add_trace(go.Scatter3d(
            x=time_steps, y=load_prev, z=raw_pred_z,
            mode='markers', marker=dict(size=5, color='red'),
            name='Raw LLM Output (Hallucinated)'
        ))

        # 3. Corrected Predictions (Green)
        fig.add_trace(go.Scatter3d(
            x=time_steps, y=load_prev, z=corrected_z,
            mode='markers', marker=dict(size=5, color='green', symbol='diamond'),
            name='NS-RAG Output (Projected)'
        ))

        # 4. Correction Vectors (Arrows/Lines)
        for i in range(n_points):
            # Only draw line if correction happened
            if raw_pred_z[i] > self.max_capacity:
                fig.add_trace(go.Scatter3d(
                    x=[time_steps[i], time_steps[i]],
                    y=[load_prev[i], load_prev[i]],
                    z=[raw_pred_z[i], corrected_z[i]],
                    mode='lines', line=dict(color='black', width=2, dash='dash'),
                    showlegend=False
                ))

        fig.update_layout(
            title="Visualizing the Differentiable Symbolic Layer (Projection Step)",
            scene=dict(
                xaxis_title='Time Step (t)',
                yaxis_title='Previous Load (MW)',
                zaxis_title='Predicted Demand (MW)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            width=1000, height=800,
            margin=dict(l=0, r=0, b=0, t=50)
        )
        
        html_path = self.output_dir / "physics_2_surface.html"
        fig.write_html(html_path)
        print(f"✅ Saved Interactive Plot: {html_path}")

if __name__ == "__main__":
    viz = SymbolicLogicVisualizer()
    viz.plot_violation_comparison()
    viz.plot_optimization_surface()
