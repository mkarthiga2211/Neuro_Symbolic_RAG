import os
import pandas as pd
import numpy as np
import zipfile
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

class MultiModalSynchronizer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        print(f"🧠 Loading Embedding Model: {model_name}...")
        self.encoder = SentenceTransformer(model_name)
        self.keywords = ["grid", "heatwave", "storm", "outage", "electricity", "energy", "power", "weather"]
        
    def _clean_and_score_text(self, text_list: List[str]) -> str:
        """Filters text snippets for energy relevance and joins them."""
        relevant_texts = [
            t for t in text_list 
            if any(k in str(t).lower() for k in self.keywords)
        ]
        return " ".join(relevant_texts[:20]) # Limit to top 20 snippets to prevent overhead

    def process_gdelt_daily(self, gdelt_path: Path) -> pd.DataFrame:
        """Extracts and embeds text from GDELT zip files."""
        daily_embeddings = []
        zip_files = sorted(list(gdelt_path.glob("*.zip")))
        
        print(f"🌍 Embedding {len(zip_files)} days of GDELT data...")
        for zf_path in tqdm(zip_files):
            date_str = zf_path.name.split('.')[0]
            try:
                with zipfile.ZipFile(zf_path, 'r') as z:
                    with z.open(z.namelist()[0]) as f:
                        # Simplified GDELT extraction (Actor1 + EventCode + Actor2 approximation)
                        df = pd.read_csv(f, sep='\t', header=None, usecols=[1, 6, 16, 26])
                        df.columns = ['Date', 'Actor1', 'Actor2', 'EventCode']
                        
                        # Create a 'Text_Snippet' from metadata
                        df['Snippet'] = df['Actor1'].fillna('') + " " + df['EventCode'].astype(str) + " " + df['Actor2'].fillna('')
                        summary_text = self._clean_and_score_text(df['Snippet'].tolist())
                        
                        embedding = self.encoder.encode(summary_text) if summary_text else np.zeros(384)
                        
                        daily_embeddings.append({
                            'date': pd.to_datetime(date_str, format='%Y%m%d'),
                            'news_vector': embedding.tolist(),
                            'raw_text': summary_text
                        })
            except Exception:
                continue
                
        return pd.DataFrame(daily_embeddings)

    def synchronize(self, numerical_file: str, gdelt_dir: str, output_file: str):
        # 1. Load Numerical (Hourly)
        df_num = pd.read_csv(numerical_file)
        df_num['timestamp'] = pd.to_datetime(df_num['timestamp'])
        df_num['date_join'] = df_num['timestamp'].dt.normalize()

        # 2. Process GDELT (Daily)
        df_news = self.process_gdelt_daily(Path(gdelt_dir))
        
        # 3. Join & Forward Fill
        print("🔗 Synchronizing modalities...")
        merged = pd.merge(df_num, df_news, left_on='date_join', right_on='date', how='left')
        
        # Handle days with no news: Fill with zero vectors
        zero_vec = [0.0] * 384
        merged['news_vector'] = merged['news_vector'].apply(lambda x: x if isinstance(x, list) else zero_vec)
        
        # Expand vector columns for Parquet storage
        vec_cols = pd.DataFrame(merged['news_vector'].tolist(), index=merged.index)
        vec_cols.columns = [f"v_{i}" for i in range(384)]
        
        final_df = pd.concat([merged[['timestamp', 'demand', 'temp']], vec_cols], axis=1)
        
        print(f"💾 Saving to {output_file}...")
        final_df.to_parquet(output_file, index=False)
        print("✅ Synchronization Complete.")

if __name__ == "__main__":
    sync = MultiModalSynchronizer()
    sync.synchronize(
        numerical_file="data/processed/numerical/demand_data.csv",
        gdelt_dir="data/raw/gdelt",
        output_file="data/processed/aligned_train_data.parquet"
    )
