import pandas as pd
import numpy as np
import chromadb
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

class NewsRetriever:
    def __init__(self, db_path: str = "./data/vector_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name="historical_news")
        
    def build_index(self, parquet_path: str):
        """Indexes unique daily news vectors with associated demand levels."""
        print(f"🏗️ Building RAG Index from {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        
        # Get unique days
        df['date'] = df['timestamp'].dt.normalize()
        daily_df = df.groupby('date').agg({
            'demand': 'mean',
            'temp': 'mean',
            **{f'v_{i}': 'first' for i in range(384)}
        }).reset_index()

        ids = [d.strftime("%Y-%m-%d") for d in daily_df['date']]
        embeddings = daily_df[[f'v_{i}' for i in range(384)]].values.tolist()
        metadatas = [
            {"date": d.strftime("%Y-%m-%d"), "avg_load": float(l), "avg_temp": float(t)} 
            for d, l, t in zip(daily_df['date'], daily_df['demand'], daily_df['temp'])
        ]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"✅ Indexed {len(ids)} unique days into vector store.")

    def get_rag_context(self, current_date: datetime, query_vector: List[float], k: int = 5) -> Dict[str, Any]:
        """
        Retrieves context while strictly preventing data leakage.
        Only retrieves news from dates < current_date.
        """
        # Temporal filtering
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k + 10, # Fetch more to allow for temporal filtering
            include=["metadatas", "distances"]
        )

        # Filter strictly for past dates
        past_twins = []
        for meta, dist in zip(results['metadatas'][0], results['distances'][0]):
            meta_date = datetime.strptime(meta['date'], "%Y-%m-%d")
            if meta_date < current_date:
                past_twins.append({
                    "date": meta['date'],
                    "load": meta['avg_load'],
                    "distance": float(dist)
                })
            
            if len(past_twins) >= k:
                break
        
        return {
            "query_date": current_date.strftime("%Y-%m-%d"),
            "contextual_twins": past_twins,
            "mean_historical_load": np.mean([t['load'] for t in past_twins]) if past_twins else 0.0
        }

if __name__ == "__main__":
    # Internal test logic
    retriever = NewsRetriever()
    # To run this, synchronizer must be finished:
    # retriever.build_index("data/processed/aligned_train_data.parquet")
