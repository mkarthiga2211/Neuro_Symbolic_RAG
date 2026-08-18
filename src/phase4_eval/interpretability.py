import numpy as np
import pandas as pd
import torch
import shap
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Callable

# Set global aesthetics
plt.style.use('seaborn-v0_8-muted')
sns.set_context("talk")

class RAGInterpreter:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', output_dir: str = "results/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.encoder = SentenceTransformer(model_name)
        print(f"🧠 XAI Suite Initialized with {model_name}")

    # --- 1. Context-Relevance SHAP Analysis ---
    def plot_shap_text_contribution(self, texts: List[str], mock_regressor: Callable):
        """
        Visualizes keyword importance using SHAP. 
        Shows how specific tokens in the RAG context influence the forecast.
        """
        print("🔍 Running SHAP Text attribution...")
        
        def model_wrapper(text_inputs):
            # 1. Embed text
            embeddings = self.encoder.encode(text_inputs)
            # 2. Mock forward pass (in real scenario, this calls the NS-RAG model)
            return mock_regressor(embeddings)

        # Using KernelExplainer for flexibility with the text wrapper
        explainer = shap.Explainer(model_wrapper, masker=shap.maskers.Text(tokenizer=r"\W+"))
        shap_values = explainer(texts)

        # Visualization: Bar plot of top contributions for the first text
        plt.figure(figsize=(12, 6))
        # shap.plots.bar handles the display
        shap.plots.bar(shap_values[0], show=False)
        plt.title("RAG Context Token Attribution (SHAP)", pad=20)
        
        save_path = self.output_dir / "xai_1_shap_text.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {save_path}")

    # --- 2. Retrieval Similarity Map (UMAP) ---
    def plot_retrieval_manifold(self, query_date: str, query_vec: np.array, 
                                historical_df: pd.DataFrame, top_k_dates: List[str]):
        """
        Visualizes why specific 'Contextual Twins' were chosen using UMAP 2D Manifold.
        """
        print("🗺️ Generating UMAP Manifold Map...")
        
        all_vecs = np.vstack([query_vec] + historical_df['embedding'].tolist())
        
        # Fit UMAP
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding_2d = reducer.fit_transform(all_vecs)
        
        # Prepare Plotting DataFrame
        plot_df = pd.DataFrame({
            'x': embedding_2d[1:, 0],
            'y': embedding_2d[1:, 1],
            'Date': historical_df['Date'],
            'Load': historical_df['Load'],
            'News': historical_df['News_Snippet'],
            'Type': 'Background'
        })
        
        # Mark the Query and Twins
        plot_df.loc[plot_df['Date'].isin(top_k_dates), 'Type'] = 'Contextual Twin'
        
        # Add the Query point (the star)
        query_row = pd.DataFrame({
            'x': [embedding_2d[0, 0]], 'y': [embedding_2d[0, 1]],
            'Date': [query_date], 'Load': [np.nan], 
            'News': ['Current Target Day Context'], 'Type': ['Target Query']
        })
        
        full_df = pd.concat([plot_df, query_row], ignore_index=True)
        
        # Create Plotly Figure for Interaction
        fig = px.scatter(
            full_df, x='x', y='y', color='Load', 
            symbol='Type',
            hover_data=['Date', 'News'],
            color_continuous_scale='RdYlBu_r',
            title='Retrieval Similarity Map: Locating Contextual Twins',
            labels={'Load': 'Energy Demand (MW)'}
        )
        
        # Customize point sizes and symbols
        fig.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
        fig.update_traces(selector=dict(symbol='star'), marker=dict(size=18, color='black')) # Query
        fig.update_traces(selector=dict(symbol='diamond'), marker=dict(size=12)) # Twins

        html_path = self.output_dir / "xai_2_retrieval_map.html"
        fig.write_html(str(html_path))
        print(f"✅ Saved Interactive Map: {html_path}")

# --- Helper for Mock Simulation ---
def mock_ns_rag_regressor(embeddings):
    """Simulates a regressor that reacts strongly to 'hot' keywords."""
    # Assume we are predicting a scalar "Impact Factor"
    # Random base + high impact for certain 'simulated' text features
    return np.sum(embeddings, axis=1) * 5 + np.random.normal(0, 1, embeddings.shape[0])

if __name__ == "__main__":
    interpreter = RAGInterpreter()
    
    # 1. SHAP Demo
    sample_news = ["Heatwave warning: Solar intensity reaching record peaks, expect massive outage risk and high grid load."]
    interpreter.plot_shap_text_contribution(sample_news, mock_ns_rag_regressor)
    
    # 2. UMAP Demo
    # Create mock historical manifold
    n_days = 200
    mock_dates = pd.date_range(start="2012-01-01", periods=n_days).strftime("%Y-%m-%d").tolist()
    
    # Background embeddings + some 'hot' clusters
    embeddings = np.random.normal(0, 1, (n_days, 384))
    # Cluster for high load days
    embeddings[10:20] += 5 
    
    hist_df = pd.DataFrame({
        'Date': mock_dates,
        'embedding': list(embeddings),
        'Load': 500 + 500 * (np.sum(embeddings, axis=1) > 5) + np.random.normal(0, 50, n_days),
        'News_Snippet': [f"Report for day {d}" for d in mock_dates]
    })
    
    # Query point (near the hot cluster)
    query_vec = np.random.normal(0, 1, (384,)) + 5
    twins = mock_dates[10:15] # Assume these were retrieved
    
    interpreter.plot_retrieval_manifold("2014-08-10", query_vec, hist_df, twins)
    
    print("\n✅ Interpretability Analysis (Phase C) complete!")
