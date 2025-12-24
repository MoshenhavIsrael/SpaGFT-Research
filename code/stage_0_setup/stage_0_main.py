from contextlib import redirect_stdout
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import scanpy as sc

# Adding the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_manager import ensure_directories, load_or_download_data, preprocess_adata, get_reference_svgs, set_local_svg_path
from evaluation import calculate_jaccard_index, run_spagft_local

# --- Default Configuration (Constants) ---
DEFAULT_SLIDE_ID = "Human_Breast_Andersson_10142021_ST_A1"
DEFAULT_TECH = "ST"
DEFAULT_METHOD_NAME = "SpaGFT"
DEFAULT_TOP_N = 100

def run_stage_0_pipeline(slide_id=DEFAULT_SLIDE_ID, 
                         tech=DEFAULT_TECH, 
                         method_name=DEFAULT_METHOD_NAME, 
                         top_n=DEFAULT_TOP_N,
                         save_SVGs=True,
                         restrict_logs=True):
    """
    Executes the Stage 0 pipeline: Setup, Data Loading, and Baseline Comparison.
    
    Returns:
        dict: A dictionary containing 'adata', 'local_genes', 'reference_genes', and 'metrics'.
    """
    if restrict_logs:
        # Open os.devnull and redirect stdout
        with open(os.devnull, 'w') as fnull:
            with redirect_stdout(fnull):
                try:
                    func_output = run_stage_0_pipeline(slide_id=slide_id, 
                             tech=tech,
                             method_name=method_name,
                             top_n=top_n,
                             save_SVGs=True,
                             restrict_logs=False)
                    return func_output
                except Exception as e:
                    raise e
        
    print(f"--- Stage 0 Pipeline | Slide: {slide_id} ---")
    
    # 1. Setup
    ensure_directories()
    
    # 2. Get Reference Results
    reference_genes = get_reference_svgs(slide_id, method_name, top_n=top_n)
    if not reference_genes:
        print("[ERROR] Cannot proceed without reference data.")
        return None

    # 3. Get Data, Construct AnnData and Preprocess
    if not (save_SVGs and set_local_svg_path(slide_id, method_name).exists()):
        adata = load_or_download_data(slide_id, tech)
        if adata is None:
            return None
        adata = preprocess_adata(adata)

    # 4. Run Local SVG identification method if needed
    if save_SVGs:
        local_svg_file = set_local_svg_path(slide_id, method_name)
        if not local_svg_file.exists():
            local_svg_df = run_spagft_local(adata)
            # Save local SVGs to file
            local_svg_df.to_csv(local_svg_file, index=True)
            print(f"[INFO] Local SVG results saved to {local_svg_file}.")
        else:
            # Load from file
            local_svg_df = pd.read_csv(local_svg_file, index_col=0)
            print(f"[INFO] Loaded local SVG results from {local_svg_file}.")
    else:
        local_svg_df = run_spagft_local(adata)
    
    # Cut off at top N genes
    local_genes = list(local_svg_df.index[:top_n]) if not local_svg_df.empty else []

    # 5. Compare Results
    metrics = {}
    if local_genes:
        jaccard, overlap = calculate_jaccard_index(reference_genes, local_genes)
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

    # Return metrics objects for analysis
    return metrics
    

if __name__ == "__main__":
    # When running as a script, use defaults
    results = run_stage_0_pipeline()