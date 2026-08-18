import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta

# Set academic styling
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)

class PerformanceVisualizer:
    def __init__(self, output_dir="results/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_season(self, dt):
        """Helper to map dates to seasons."""
        month = dt.month
        if month in [12, 1, 2]: return 'Winter'
        if month in [6, 7, 8]: return 'Summer'
        return 'Transition'

    def find_max_peak_week(self, df):
        """Automatically identifies the 7-day window with the highest energy demand."""
        # Calculate 7-day rolling sum of ground truth
        df = df.sort_values('Timestamp')
        # We look for the day with the highest peak and take 3 days before and 3 days after
        max_idx = df['Ground_Truth'].idxmax()
        peak_date = df.loc[max_idx, 'Timestamp']
        
        start_date = peak_date - timedelta(days=3)
        end_date = peak_date + timedelta(days=4)
        
        return df[(df['Timestamp'] >= start_date) & (df['Timestamp'] < end_date)]

    def plot_forecast_overlap(self, df):
        """
        Chart 1: Zoomed time-series showing how RAG context fixes peak under-prediction.
        """
        print("📈 Generating 1: Forecast Overlap Plot...")
        plot_df = self.find_max_peak_week(df)
        
        plt.figure(figsize=(14, 7))
        
        # Plot Lines
        plt.plot(plot_df['Timestamp'], plot_df['Ground_Truth'], color='black', linewidth=2.5, label='Ground Truth (Actual)')
        plt.plot(plot_df['Timestamp'], plot_df['Vanilla_LLM_Pred'], color='#e74c3c', linestyle='--', linewidth=2, label='Vanilla LLM (No Context)')
        plt.plot(plot_df['Timestamp'], plot_df['NSRAG_Pred'], color='#27ae60', linewidth=2, label='NS-RAG (Ours)')
        
        # Highlight RAG Context Gain
        max_idx = plot_df['Ground_Truth'].idxmax()
        peak_time = plot_df.loc[max_idx, 'Timestamp']
        gt_val = plot_df.loc[max_idx, 'Ground_Truth']
        vanilla_val = plot_df.loc[max_idx, 'Vanilla_LLM_Pred']
        
        plt.annotate('RAG Context Gain\n(Fixing Peak Under-prediction)', 
                     xy=(peak_time, (gt_val + vanilla_val)/2),
                     xytext=(peak_time + timedelta(hours=12), (gt_val + vanilla_val)/2 + 100),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                     fontsize=12, fontweight='bold', bbox=dict(boxstyle="round", fc="0.9"))

        # Formatting
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.title('Predictive Performance During Extreme Peak Demand', fontsize=18, pad=15)
        plt.ylabel('Energy Demand (MW)')
        plt.xlabel('Timestamp')
        plt.legend(loc='upper right', frameon=True)
        plt.grid(True, alpha=0.3)
        
        save_path = self.output_dir / "perf_1_forecast_overlap.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")

    def plot_mape_improvement_heatmap(self, df):
        """
        Chart 2: Heatmap showing MAPE improvement delta across event types and seasons.
        """
        print("🔥 Generating 2: MAPE Improvement Heatmap...")
        
        # Pre-process: Add Season and calculate Errors
        df['Season'] = df['Timestamp'].apply(self._get_season)
        df['Error_Vanilla'] = np.abs((df['Ground_Truth'] - df['Vanilla_LLM_Pred']) / df['Ground_Truth']) * 100
        df['Error_NSRAG'] = np.abs((df['Ground_Truth'] - df['NSRAG_Pred']) / df['Ground_Truth']) * 100
        
        # Pivot table for improvement
        mape_v = df.groupby(['Event_Type', 'Season'])['Error_Vanilla'].mean().unstack()
        mape_n = df.groupby(['Event_Type', 'Season'])['Error_NSRAG'].mean().unstack()
        
        improvement_delta = mape_v - mape_n
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(improvement_delta, annot=True, fmt=".2f", cmap='Greens', 
                    cbar_kws={'label': 'MAPE Improvement Delta (%)'})
        
        plt.title('MAPE Improvement: NS-RAG vs Vanilla LLM\n(Context-Specific Gains)', fontsize=16, pad=20)
        plt.xlabel('Season')
        plt.ylabel('Event Type')
        
        save_path = self.output_dir / "perf_2_mape_heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")

# --- Generate Mock Results for Visualization Demo ---
def generate_mock_results():
    dates = pd.date_range(start="2014-01-01", end="2014-12-31", freq='H')
    n = len(dates)
    
    # Base pattern
    gt = 1000 + 400 * np.sin(2 * np.pi * np.arange(n) / 24) + np.random.normal(0, 50, n)
    
    # Create extreme event in Summer (e.g., August 10th)
    heatwave_mask = (dates.month == 8) & (dates.day >= 7) & (dates.day <= 14)
    gt[heatwave_mask] += 600 # Massive peak
    
    # Vanilla LLM (Under-predicts the peak)
    vanilla = gt * 0.9 + np.random.normal(0, 30, n)
    vanilla[heatwave_mask] = gt[heatwave_mask] * 0.75 # Deep failure at peak
    
    # NS-RAG (Corrects peak via context)
    nsrag = gt * 0.98 + np.random.normal(0, 15, n)
    
    # Event Types
    event_types = np.array(['Normal'] * n, dtype=object)
    event_types[heatwave_mask] = 'Heatwave'
    # Randomly assign others
    maint_mask = (dates.day == 5) & (dates.hour > 10) & (dates.hour < 16)
    event_types[maint_mask] = 'Grid_Maintenance'
    
    return pd.DataFrame({
        'Timestamp': dates,
        'Ground_Truth': gt,
        'Vanilla_LLM_Pred': vanilla,
        'NSRAG_Pred': nsrag,
        'Event_Type': event_types
    })

if __name__ == "__main__":
    results_df = generate_mock_results()
    
    viz = PerformanceVisualizer()
    viz.plot_forecast_overlap(results_df)
    viz.plot_mape_improvement_heatmap(results_df)
    
    print("\n✅ Performance Visualizations Complete!")
