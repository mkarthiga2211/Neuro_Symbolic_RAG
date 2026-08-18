import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from pathlib import Path
import zipfile

# Set aesthetic style
plt.style.use('ggplot')
sns.set_context("talk")

class EDAPipeline:
    def __init__(self, data_root="data"):
        self.data_root = Path(data_root)
        self.numerical_dir = self.data_root / "processed" / "numerical"
        self.gdelt_dir = self.data_root / "raw" / "gdelt"
        
    def load_numerical_data(self):
        print("📊 Loading Numerical Data...")
        self.gefcom_df = pd.read_csv(self.numerical_dir / "Load_history.csv")
        self.demand_df = pd.read_csv(self.numerical_dir / "demand_data.csv")
        self.demand_df['timestamp'] = pd.to_datetime(self.demand_df['timestamp'])
        
    def process_gdelt_sample(self, limit_days=30):
        """Processes a sample of GDELT days to generate the EDA data."""
        print(f"🌍 Processing GDELT Sample ({limit_days} days)...")
        gdelt_files = sorted(list(self.gdelt_dir.glob("*.zip")))[:limit_days]
        daily_stats = []
        
        for zf_path in gdelt_files:
            date_str = zf_path.name.split('.')[0]
            try:
                with zipfile.ZipFile(zf_path, 'r') as z:
                    with z.open(z.namelist()[0]) as f:
                        # GDELT v1 is tab-separated, no header. 
                        # Col 1: Day, Col 26: EventCode, Col 34: AvgTone
                        df = pd.read_csv(f, sep='\t', header=None, usecols=[1, 26, 34])
                        df.columns = ['Day', 'EventCode', 'AvgTone']
                        
                        stats = {
                            'date': pd.to_datetime(date_str, format='%Y%m%d'),
                            'event_count': len(df),
                            'avg_tone': df['AvgTone'].mean(),
                            'weather_events': len(df[df['EventCode'].astype(str).str.startswith('04')]), # 04 = Exhibit Force/Material
                            'econ_events': len(df[df['EventCode'].astype(str).str.startswith('03')])    # 03 = Express Intent
                        }
                        daily_stats.append(stats)
            except Exception as e:
                continue
        
        self.gdelt_df = pd.DataFrame(daily_stats)
        print(f"✅ GDELT processed: {len(self.gdelt_df)} days.")

    # --- 1. The 'News-Driven' Peak Analysis ---
    def plot_news_peak_analysis(self):
        print("📈 Generating Chart 1: News-Driven Peak Analysis...")
        # Merge for plotting
        merged = pd.merge(self.demand_df, self.gdelt_df, left_on=self.demand_df['timestamp'].dt.date, right_on=self.gdelt_df['date'].dt.date)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=merged['timestamp'], y=merged['demand'], name="Demand"), secondary_y=False)
        fig.add_trace(go.Bar(x=merged['timestamp'], y=merged['event_count'], name="GDELT Intensity", opacity=0.3), secondary_y=True)
        fig.update_layout(title="News Intensity vs Demand Peaks", template="plotly_dark")
        fig.write_html("results/figures/chart1_news_peaks.html")

    # --- 2. Seasonal Decomposition ---
    def plot_residual_analysis(self):
        print("📉 Generating Chart 2: Residual Anomaly Analysis...")
        result = seasonal_decompose(self.demand_df.set_index('timestamp')['demand'], model='additive', period=24)
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        result.observed.plot(ax=axes[0], title='Observed')
        result.trend.plot(ax=axes[1], title='Trend')
        result.seasonal.plot(ax=axes[2], title='Seasonal')
        result.resid.plot(ax=axes[3], title='Residuals (Potential News Signal)', color='red')
        plt.tight_layout()
        plt.savefig("results/figures/chart2_residuals.png")

    # --- 3. Multivariate Heatmap ---
    def plot_correlation_heatmap(self):
        print("🔥 Generating Chart 3: Correlation Heatmap...")
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.demand_df[['demand', 'temp']].corr(), annot=True, cmap='coolwarm')
        plt.title("Physical Feature Correlations")
        plt.savefig("results/figures/chart3_heatmap.png")

    # --- 4. Cluster Visualization (Mock/Simple) ---
    def plot_clusters(self):
        print("🌌 Generating Chart 4: Textual Cluster Visualization...")
        # Using event counts as simple features for clusters
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        
        features = self.gdelt_df[['event_count', 'weather_events', 'econ_events']]
        scaled = StandardScaler().fit_transform(features)
        pca = PCA(n_components=2).fit_transform(scaled)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(pca[:, 0], pca[:, 1], c=self.gdelt_df['avg_tone'], cmap='RdYlGn')
        plt.colorbar(label='Avg Tone')
        plt.title("GDELT Contextual Clusters (PCA)")
        plt.savefig("results/figures/chart4_clusters.png")

    # --- 5. Event Type Violins (Mock categories for demo) ---
    def plot_violins(self):
        print("🎻 Generating Chart 5: Demand by Event Type...")
        # Create dummy categories based on event counts
        self.gdelt_df['category'] = 'Normal'
        self.gdelt_df.loc[self.gdelt_df['weather_events'] > self.gdelt_df['weather_events'].median(), 'category'] = 'High Event'
        
        # Merge with daily demand
        daily_demand = self.demand_df.groupby(self.demand_df['timestamp'].dt.date)['demand'].max().reset_index()
        daily_demand.columns = ['date', 'peak_demand']
        daily_demand['date'] = pd.to_datetime(daily_demand['date'])
        
        violin_df = pd.merge(daily_demand, self.gdelt_df, on='date')
        
        plt.figure(figsize=(10, 6))
        sns.violinplot(x='category', y='peak_demand', data=violin_df)
        plt.title("Peak Demand Distribution by GDELT Activity")
        plt.savefig("results/figures/chart5_violins.png")

if __name__ == "__main__":
    # Ensure results dir exists
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    
    pipeline = EDAPipeline()
    pipeline.load_numerical_data()
    pipeline.process_gdelt_sample(limit_days=100) # Process 100 days for faster EDA
    
    pipeline.plot_news_peak_analysis()
    pipeline.plot_residual_analysis()
    pipeline.plot_correlation_heatmap()
    pipeline.plot_clusters()
    pipeline.plot_violins()
    
    print("\n✅ EDA Complete! All figures saved to results/figures/")
