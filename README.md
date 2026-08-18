# Neuro-Symbolic RAG-LLM for Demand Forecasting

## 📌 Project Overview
This repository contains the official implementation of the research paper **"Neuro-Symbolic RAG-LLM for Demand Forecasting"**. 

It introduces a novel hybrid architecture that combines:
1.  **Retrieval-Augmented Generation (RAG)**: To contextualize forecasts with external knowledge (News/Events) using Large Language Models.
2.  **Differentiable Symbolic Logic**: A constraint layer using `cvxpylayers` (or PyTorch fallback) to enforce hard physical guarantees (e.g., Grid Capacity).

### 🚀 Key Features
*   **Multi-Modal Fusion**: Aligns hourly numerical load data with daily textual GDELT news.
*   **Physics-Aware**: Guarantees 0% physical constraint violations via the Symbolic Layer.
*   **Interpretability**: Includes SHAP (Text Attribution) and UMAP (Retrieval Manifold) visualizations.
*   **Robustness**: Proven resilience against "Noisy Context" hallucinations.

---

## 🛠️ Installation

### 1. Prerequisites
*   Python 3.9+
*   (Optional) Microsoft Visual C++ Build Tools (for full `cvxpylayers` support on Windows).

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*Note: If `cvxpylayers` installation fails on Windows, the code automatically switches to a robust PyTorch-native fallback.*

---

## 🏃‍♂️ Usage (The "One-Click" Workflow)

This project is orchestrated by `main.py`. You can run specific phases or the entire pipeline.

### Step 1: Data Ingestion
Downloads GEFCom 2012 (or generates synthetic fallbacks) and GDELT news data.
```bash
python main.py ingest
```

### Step 2: Synchronization
embedding news text (using `all-MiniLM-L6-v2`) and aligns it with hourly demand data.
```bash
python main.py sync
```
*Output: `data/processed/aligned_train_data.parquet`*

### Step 3: Model Training
Trains the Neuro-Symbolic Forecaster with the differentiable constraint loss.
```bash
python main.py train
```
*Output: `results/models/nsrag_best.pt`*

### Step 4: Evaluation & Visualization
Runs benchmarking, generates comparison tables, and creates all publication figures.
```bash
python main.py eval
```

### ⚡ Run All
Execute the entire research pipeline from start to finish:
```bash
python main.py all
```

---

## 📊 Results & Artifacts

All research outputs are stored in `results/figures/`:

| Figure / Table | Description |
| :--- | :--- |
| **`final_results_table.png`** | Comparative Benchmarking (MAPE, RMSE, Violation Rate). |
| **`perf_1_forecast_overlap.png`** | Time-series zoom showing RAG fixing peak under-prediction. |
| **`perf_7_3d_projection.png`** | 3D Visualization of the Symbolic Constraint Projection. |
| **`xai_1_shap_text.png`** | SHAP Analysis showing which news keywords influence the forecast. |
| **`xai_2_retrieval_map.html`** | Interactive UMAP plot of the RAG retrieval manifold. |

---

## 📂 Project Structure
```
Neuro_Symbolic_RAG/
├── data/                       # Raw and Processed Datasets
├── results/
│   ├── figures/                # Generated Plots & Tables
│   └── models/                 # Checkpoints
├── src/
│   ├── phase1_data/            # Ingestion, Sync, RAG Engine
│   ├── phase2_model/           # Neuro-Symbolic Architecture
│   ├── phase3_training/        # Training Loop
│   └── phase4_eval/            # Benchmarking & Visualization
├── main.py                     # Master Entry Point
└── requirements.txt            # Dependencies
```

## 📜 Citation
If you use this code, please cite our corresponding research paper.
