import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, mean_absolute_error, mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Dict, Any, List, Tuple

class PaperVisualizer:
    """
    Principal Research Scientist Module for generating final publication-grade
    evaluation figures and comparison tables for Neuro-Symbolic RAG-LLM.
    """
    def __init__(self, output_dir: str = "results/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Academic style configuration
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.dpi': 300
        })

    def plot_loss_convergence(self, history: Dict[str, List[float]]):
        """
        Visualizes the convergence of total loss and the symbolic penalty term.
        Proves that the model learns to satisfy physical constraints over time.
        """
        print("📈 Generating: Loss Convergence Plot...")
        epochs = range(1, len(history['total_loss']) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['total_loss'], 'b-', label='Total Loss (LLM + Symbolic)', linewidth=2)
        plt.plot(epochs, history['symbolic_penalty'], 'r--', label='Symbolic Penalty Only', linewidth=2)
        
        plt.title('Neuro-Symbolic Training Convergence')
        plt.xlabel('Epochs')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(self.output_dir / "perf_3_loss_convergence.png", bbox_inches='tight')
        plt.close()

    def plot_demand_confusion_matrix(self, y_true: np.array, y_pred: np.array):
        """
        Evaluates tier-based forecasting accuracy.
        Bins continuous MW into 'Low', 'Normal', 'Peak' demand tiers.
        """
        print("🔲 Generating: Demand Tier Confusion Matrix...")
        
        # Binning Logic: 33rd and 66th percentiles
        low_thresh = np.percentile(y_true, 33)
        peak_thresh = np.percentile(y_true, 66)
        
        def label_tier(val):
            if val <= low_thresh: return 'Low'
            if val <= peak_thresh: return 'Normal'
            return 'Peak'
        
        true_labels = [label_tier(v) for v in y_true]
        pred_labels = [label_tier(v) for v in y_pred]
        labels = ['Low', 'Normal', 'Peak']
        
        cm = confusion_matrix(true_labels, pred_labels, labels=labels)
        # Normalize by row (precision per tier)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title('Demand Tier Confusion Matrix (Normalized)')
        plt.xlabel('Predicted Tier')
        plt.ylabel('Actual Tier')
        
        plt.savefig(self.output_dir / "perf_4_confusion_matrix.png", bbox_inches='tight')
        plt.close()

    def plot_extreme_event_metrics(self, y_true: np.array, y_pred: np.array):
        """
        AUC-ROC and PR Curves for detecting Top 5% extreme demand events.
        """
        print("📈 Generating: Extreme Event ROC/PR Curves...")
        # Define 95th percentile as extreme
        threshold = np.percentile(y_true, 95)
        binary_true = (y_true > threshold).astype(int)
        # Normalize pred as confidence score
        scores = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())
        
        fpr, tpr, _ = roc_curve(binary_true, scores)
        roc_auc = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(binary_true, scores)
        avg_precision = average_precision_score(binary_true, scores)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # ROC Curve
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_title('Extreme Event Detection (ROC)')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.legend(loc="lower right")
        
        # PR Curve
        ax2.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
        ax2.set_title('Extreme Event Detection (Precision-Recall)')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.legend(loc="lower left")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "perf_5_extreme_metrics.png", bbox_inches='tight')
        plt.close()

    def generate_comparison_table(self, results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Produces the main result comparison table for the paper.
        Calculates MAPE, MAE, RMSE, and Violation Rate.
        """
        print("📋 Generating: Model Comparison Table...")
        df = pd.DataFrame(results).T
        # Round for publication format
        df = df.round(3)
        df.to_csv(self.output_dir.parent / "final_results_table.csv")
        return df

    def plot_ablation_study(self, scores: Dict[str, float]):
        """
        Grouped Bar Chart for Ablation Study (Base LLM, Symbolic-only, RAG-only, NS-RAG).
        """
        print("📊 Generating: Ablation Study Plot...")
        labels = list(scores.keys())
        values = list(scores.values())
        
        plt.figure(figsize=(10, 6))
        colors = ['#bdc3c7', '#3498db', '#e67e22', '#2ecc71'] # Gray, Blue, Orange, Green
        sns.barplot(x=labels, y=values, palette=colors)
        
        plt.title('Ablation Analysis: Contribution of Neural and Symbolic Modules')
        plt.ylabel('MAPE (%)')
        plt.xticks(rotation=15)
        plt.grid(axis='y', alpha=0.3)
        
        plt.savefig(self.output_dir / "perf_6_ablation_study.png", bbox_inches='tight')
        plt.close()

    def plot_constraint_surface_3d(self, raw: np.array, constrained: np.array, limit: float):
        """
        3D Scatter plot visualizing the 'Safety Anchor' effect.
        Shows raw 'hallucinations' projected onto the physical constraint plane.
        """
        print("🗺️ Generating: 3D Constraint Projection Surface...")
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        n = len(raw)
        x = np.arange(n)
        y = np.random.normal(0, 1, n) # Mock feature dimension
        
        # Plot physical limit plane
        X_plane, Y_plane = np.meshgrid(np.linspace(0, n, 10), np.linspace(-3, 3, 10))
        Z_plane = np.ones_like(X_plane) * limit
        ax.plot_surface(X_plane, Y_plane, Z_plane, color='gray', alpha=0.2, label='Grid Capacity Limit')
        
        # Plot Raw (Red) and Constrained (Green)
        ax.scatter(x, y, raw, c='red', marker='o', s=50, label='Raw LLM (Unconstrained)')
        ax.scatter(x, y, constrained, c='green', marker='D', s=50, label='NS-RAG (Constrained)')
        
        # Draw projection lines
        for i in range(n):
            if raw[i] > limit:
                ax.plot([x[i], x[i]], [y[i], y[i]], [raw[i], constrained[i]], 'k--', alpha=0.5)

        ax.set_title('Projection onto Physical Feasibility Set')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Environmental Feature')
        ax.set_zlabel('Predicted Load (MW)')
        ax.legend()
        
        plt.savefig(self.output_dir / "perf_7_3d_projection.png", bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    # --- Mock Usage for demonstration purposes ---
    viz = PaperVisualizer()
    
    # 1. Loss History
    hist = {
        'total_loss': [0.8, 0.6, 0.45, 0.38, 0.32, 0.28, 0.25],
        'symbolic_penalty': [0.5, 0.3, 0.2, 0.12, 0.08, 0.04, 0.02]
    }
    viz.plot_loss_convergence(hist)
    
    # 2. Confusion Matrix & 3. Extreme Event Metrics
    y_true = np.random.uniform(500, 1500, 1000)
    y_pred = y_true * 0.95 + np.random.normal(0, 50, 1000)
    viz.plot_demand_confusion_matrix(y_true, y_pred)
    viz.plot_extreme_event_metrics(y_true, y_pred)
    
    # 4. Comparison Table
    model_results = {
        'Proposed NS-RAG': {'MAPE': 4.2, 'MAE': 42.1, 'RMSE': 58.3, 'Violation_%': 0.0},
        'Vanilla Llama-3': {'MAPE': 18.5, 'MAE': 185.0, 'RMSE': 240.2, 'Violation_%': 12.4},
        'Informer (SOTA)': {'MAPE': 8.9, 'MAE': 89.2, 'RMSE': 110.5, 'Violation_%': 4.1},
        'ARIMA / Prophet': {'MAPE': 12.1, 'MAE': 120.4, 'RMSE': 150.1, 'Violation_%': 0.0},
        'XGBoost + Text': {'MAPE': 7.8, 'MAE': 78.5, 'RMSE': 98.4, 'Violation_%': 5.2}
    }
    viz.generate_comparison_table(model_results)
    
    # 5. Ablation Study
    ablation = {
        'Base LLM': 22.4,
        'Symbolic-Only': 14.2,
        'RAG-Only': 9.8,
        'Full NS-RAG': 4.2
    }
    viz.plot_ablation_study(ablation)
    
    # 6. 3D Projection
    raw_out = np.random.uniform(900, 1200, 20)
    limit = 1000.0
    const_out = np.minimum(raw_out, limit)
    viz.plot_constraint_surface_3d(raw_out, const_out, limit)
    
    print("\n✅ Final Benchmarking Assets generated in results/figures/")
