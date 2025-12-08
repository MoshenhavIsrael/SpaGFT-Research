import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import scanpy as sc

# Adding the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_manager import ensure_directories, load_or_download_data, get_reference_svgs
from evaluation import calculate_jaccard_index, run_spagft_local

# --- Default Configuration (Constants) ---
DEFAULT_SLIDE_ID = "Human_Breast_Andersson_10142021_ST_A1"
DEFAULT_TECH = "ST"
DEFAULT_METHOD_NAME = "SpaGFT"
DEFAULT_TOP_N = 100

def run_stage_0_pipeline(slide_id=DEFAULT_SLIDE_ID, 
                         tech=DEFAULT_TECH, 
                         method_name=DEFAULT_METHOD_NAME, 
                         top_n=DEFAULT_TOP_N):
    """
    Executes the Stage 0 pipeline: Setup, Data Loading, and Baseline Comparison.
    
    Returns:
        dict: A dictionary containing 'adata', 'local_genes', 'reference_genes', and 'metrics'.
    """
    print(f"--- Stage 0 Pipeline | Slide: {slide_id} ---")
    
    # 1. Setup
    ensure_directories()
    
    # 2. Get Data
    adata = load_or_download_data(slide_id, tech)
    if adata is None:
        return None

    # 3. Get Reference Results
    reference_genes = get_reference_svgs(slide_id, method_name, top_n=top_n)
    if not reference_genes:
        print("[ERROR] Cannot proceed without reference data.")
        return None

    # 4. Run Local SpaGFT
    # We pass the full adata, and slicing happens here or inside the function
    local_genes = run_spagft_local(adata)[:top_n]
    
    # 5. Compare Results
    metrics = {}
    if local_genes:
        jaccard = calculate_jaccard_index(reference_genes, local_genes)
        overlap_count = len(set(reference_genes).intersection(set(local_genes)))
        
        metrics = {
            "jaccard": jaccard,
            "overlap_count": overlap_count
        }
        
        print("\n" + "="*40)
        print(f"RESULTS COMPARISON (Top {top_n} SVGs)")
        print("="*40)
        print(f"Jaccard Index: {jaccard:.4f}")
        print(f"Overlap: {overlap_count}")
        print("="*40)
    else:
        print("\n[INFO] Skipping comparison (local results empty).")

    # Return all valuable objects for Jupyter analysis
    return {
        "adata": adata,
        "reference_genes": reference_genes,
        "local_genes": local_genes,
        "metrics": metrics
    }

if __name__ == "__main__":
    # When running as a script, use defaults
    results = run_stage_0_pipeline()