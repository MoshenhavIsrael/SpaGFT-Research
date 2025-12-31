from contextlib import redirect_stdout
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Adding the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_manager import (
    ensure_directories, 
    load_or_download_data, 
    preprocess_adata, 
    get_reference_svgs, 
    RESULTS_DIR 
)
from evaluation import run_svg_methods, calculate_jaccard_index

# --- Default Configuration (Constants) ---
DEFAULT_SLIDE_ID = "Human_Breast_Andersson_10142021_ST_A1"
DEFAULT_TECH = "ST"
DEFAULT_METHODS = ["SpaGFT", "MoranI"]
DEFAULT_TOP_N = 100

def run_stage_0_pipeline(slide_id=DEFAULT_SLIDE_ID, 
                         tech=DEFAULT_TECH, 
                         methods_to_run=DEFAULT_METHODS,
                         top_n=DEFAULT_TOP_N,
                         save_SVGs=True,
                         restrict_logs=False):
    """
    Executes the Stage 0 pipeline: Setup, Data Loading, and Baseline Comparison.
    Supports running multiple SVG detection methods.    
    Returns:
        dict: A dictionary containing 'metrics' and 'SVG_rank'.
    """
    if restrict_logs:
        # Open os.devnull and redirect stdout
        with open(os.devnull, 'w') as fnull:
            with redirect_stdout(fnull):
                try:
                    func_output = run_stage_0_pipeline(slide_id=slide_id, 
                             tech=tech,
                             methods_to_run=methods_to_run,
                             top_n=top_n,
                             save_SVGs=True,
                             restrict_logs=False)
                    return func_output
                except Exception as e:
                    raise e
        
    print(f"--- Stage 0 Pipeline | Slide: {slide_id} ---")
    
    # 1. Setup
    ensure_directories()
    
    # Define expected path for checking existing results
    # Assuming: results/svg_results/{method_name}_{tissue_id}_SVGs.csv
    svg_results_dir = RESULTS_DIR / "svg_results"

    # 2. Check which methods need calculation (Lazy Loading Logic)
    missing_methods = []
    if save_SVGs:
        for method in methods_to_run:
            if not (svg_results_dir / f"{method}_{slide_id}_SVGs.csv").exists():
                missing_methods.append(method)
    else:
        missing_methods = methods_to_run # If not saving, always run
    
    # 3. Get Data (Only if needed)
    adata = None
    if missing_methods:
        print(f"[INFO] Calculating needed for: {missing_methods}. Loading data...")
        adata = load_or_download_data(slide_id, tech)
        if adata is None:
            return None
        adata = preprocess_adata(adata)
    else:
        print("[INFO] All result files exist. Skipping data loading.")

    # 4. Get Reference SVGs for Comparison
    reference_dict = get_reference_svgs(slide_id, methods_to_run, top_n=top_n)

    # 5. Run SVG detection methods
    svg_results_dict = run_svg_methods(adata, methods_to_run, slide_id, svg_results_dir)
    
    # 6. Compare Results
    metrics = {}
    print("\n" + "="*60)
    print(f"RESULTS COMPARISON (Top {top_n} SVGs)")
    print("="*60)

    for method in methods_to_run:
        # Get results
        res_df = svg_results_dict.get(method, pd.DataFrame())
        local_genes = list(res_df.index[:top_n]) if not res_df.empty else []
        
        # Get reference
        ref_genes = reference_dict.get(method, [])
        
        if not local_genes:
            print(f"[{method}] No SVGs identified.")
            continue
            
        if not ref_genes:
            print(f"[{method}] No reference genes available for comparison.")
            continue

        # Calculate Metrics
        jaccard, overlap = calculate_jaccard_index(ref_genes, local_genes)
        
        metrics[method] = {
            "jaccard": jaccard,
            "overlap_count": overlap
        }
        
        print(f"[{method}] Jaccard: {jaccard:.4f} | Overlap: {overlap}/{top_n}")

    print("="*60)

    # Return metrics objects for analysis
    return {
        "metrics": metrics,
        "SVG_rank": svg_results_dict
    }    

if __name__ == "__main__":
    # When running as a script, use defaults
    results = run_stage_0_pipeline()