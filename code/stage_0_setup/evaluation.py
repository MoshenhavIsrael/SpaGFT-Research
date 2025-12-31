# --- Imports ---
import SpaGFT as spg
import numpy as np
import scanpy as sc
try:
    import squidpy as sq
except ImportError:
    sq = None
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
import os
from pathlib import Path
import pandas as pd
import sys
sys.path.append("../../SpaGFT")  # Adjust path as needed to locate SpaGFT package

N_JOBS = 1  # Number of parallel jobs for computations


# --- Utility Functions ---


def calculate_jaccard_index(list1, list2):
    """Calculates the Jaccard Index between two lists of genes. Returns Jaccard index and overlap count."""
    set1 = set(list1)
    set2 = set(list2)
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0.0
    
    return intersection / union, intersection


# --- SVG Method Implementations ---


def _run_spagft_local(adata):
    """
    Runs the local SpaGFT implementation on the provided AnnData object.
     Returns a DataFrame with SpaGFT results.
    """
    print("\n[INFO] Running local SpaGFT implementation...")
    
    # Using a copy to avoid modifying the original adata_local in main
    adata_local = adata.copy()
    
    # Define spatial info columns
    # Assuming data_manager put them in 'array_row'/'array_col'
    coord_columns = ['array_row', 'array_col']  
    # Check if Z exists
    if 'array_z' in adata_local.obs.columns:
        coord_columns.append('array_z') 
    
    # Open os.devnull and redirect stdout
    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull):
            try:
                # determine the number of low-frequency FMs and high-frequency FMs
                (ratio_low, ratio_high) = spg.gft.determine_frequency_ratio(adata_local,
                                                                            ratio_neighbors=1,
                                                                            spatial_info=coord_columns)
                
                # calculation
                gene_df = spg.detect_svg(adata_local,
                                        ratio_low_freq=ratio_low,
                                        ratio_high_freq=ratio_high,
                                        ratio_neighbors=1,
                                        filter_peaks=True,
                                        S=6)
                # S determines the  sensitivity of kneedle algorithm

            except Exception as e:
                raise e
    
    
    # extract spaitally variable genes
    svg_df = gene_df[gene_df.cutoff_gft_score][gene_df.fdr < 0.05]

    print("[INFO] Finished running local SpaGFT.")
    return svg_df 

def _run_scanpy_morans(adata):
    """
    Calculates Moran's I using Scanpy's built-in implementation.
    Requires spatial neighbors graph.
    """
    print("[INFO] Running Scanpy Moran's I...")
    adata_calc = adata.copy()

    # 1. Ensure 'spatial' key exists in obsm for neighbor calculation
    if 'spatial' not in adata_calc.obsm:
        # Assuming array_row/col exist as per data_manager
        if 'array_row' in adata_calc.obs.columns and 'array_col' in adata_calc.obs.columns:
            if 'array_z' in adata_calc.obs.columns:
                adata_calc.obsm['spatial'] = adata_calc.obs[['array_row', 'array_col', 'array_z']].values
            else:
                adata_calc.obsm['spatial'] = adata_calc.obs[['array_row', 'array_col']].values
        else:
            raise KeyError("Could not find 'array_row'/'array_col' or 'spatial' in adata.")

    # 2. Ensure spatial connectivities exist (Spatial Graph)
    if 'spatial_connectivities' not in adata_calc.obsp:
        # Use spatial coordinates to find neighbors
        sc.pp.neighbors(adata_calc, use_rep='spatial', n_neighbors=6)
    
    # 3. Run Moran's I
    # Scanpy stores results in adata.var['morans_i']
    moran_values = sc.metrics.morans_i(adata_calc)
    
    # 4. Format Output
    # Create a DataFrame consistent with other methods (Index=Genes, Columns=['score'])
    moran_df = pd.DataFrame(
        moran_values, 
        index=adata_calc.var_names, 
        columns=['score'] # Use 'score' as standard column for ranking
    )
    
    # Sort descending (Higher Moran's I = stronger spatial autocorrelation)
    moran_df = moran_df.sort_values('score', ascending=False)
    
    return moran_df

def _run_squidpy_morans(adata):
    """
    Calculates Moran's I using Squidpy to generate Z-scores.
    Falls back to simple Scanpy implementation if Squidpy is missing (with warning).
    """
    print("[INFO] Running Moran's I analysis...")
    adata_calc = adata.copy()
    
    # 1. Ensure spatial graph exists
    if 'spatial_connectivities' not in adata_calc.obsp:
        print("   -> Computing spatial neighbors...")
        # Squidpy requires spatial key in obsm, usually 'spatial'
        if 'spatial' not in adata_calc.obsm:
             if 'array_row' in adata_calc.obs and 'array_col' in adata_calc.obs:
                 if 'array_z' in adata_calc.obs.columns:
                     adata_calc.obsm['spatial'] = adata_calc.obs[['array_row', 'array_col', 'array_z']].values
                 else:
                     adata_calc.obsm['spatial'] = adata_calc.obs[['array_row', 'array_col']].values
        
        sc.pp.neighbors(adata_calc, use_rep='spatial', n_neighbors=6)

    # 2. Run with Squidpy (Preferred: gives Z-score and P-value)
    if sq is not None:
        print("   -> Using Squidpy for permutation-based Z-scores...")
        sq.gr.spatial_autocorr(
            adata_calc, 
            mode='moran', 
            connectivity_key='connectivities',
            n_perms=100, 
            n_jobs=N_JOBS,  
            genes=adata_calc.var_names
        )
        
        # Squidpy saves results in adata.uns['moranI']
        # Columns usually: 'I', 'pval_sim', 'z_score'
        res_df = adata_calc.uns['moranI'].copy()
        
        # Rank by z_score descending
        res_df = res_df.sort_values('pval_z_sim', ascending=True)
        res_df['SVG_rank'] = range(1, len(res_df) + 1)
        return res_df[['I', 'pval_z_sim']]

    else:
        # 3. Fallback to Scanpy (Not recommended for this specific comparison)
        print("[WARN] Squidpy not found. Using basic Scanpy Moran's I (Raw Index only).")
        print("       To get Z-scores matching the Atlas, please install squidpy: `pip install squidpy`")
        
        return _run_scanpy_morans(adata_calc)
    

# --- Registry & Wrapper ---


# Mapping method names (strings) to functions
SVG_METHODS_REGISTRY = {
    "SpaGFT": _run_spagft_local,
    "MoranI": _run_squidpy_morans,
}

def run_svg_methods(adata, methods_list, tissue_id, results_dir):
    """
    Wrapper function to run multiple SVG detection methods.
    
    Logic:
    1. Checks if result file exists in 'results_dir'.
    2. If yes -> Loads it.
    3. If no -> Runs the method (using 'adata') and saves it.
    
    Returns:
        dict: { method_name: DataFrame }
    """
    results = {}
    results_dir = Path(results_dir)
    
    # Create the svg_results subdirectory if passed directly, 
    # but usually passed path should already be complete.
    if not results_dir.exists():
        results_dir.mkdir(parents=True, exist_ok=True)
    
    for method_name in methods_list:
        if method_name not in SVG_METHODS_REGISTRY:
            print(f"[WARN] Method '{method_name}' is not registered. Skipping.")
            continue

        file_path = results_dir / f"{method_name}_{tissue_id}_SVGs.csv"

        if file_path.exists():
            print(f"[INFO] Loading existing results for {method_name}...")
            results[method_name] = pd.read_csv(file_path, index_col=0)
        else:
            if adata is None:
                print(f"[ERR] Result for {method_name} missing and adata is None. Cannot calculate.")
                continue

            try:
                func = SVG_METHODS_REGISTRY[method_name]
                df = func(adata)
                
                # Save results
                df.to_csv(file_path, index=True)
                print(f"[INFO] Saved {method_name} results to {file_path.name}")
                results[method_name] = df
                
            except NotImplementedError:
                print(f"[WARN] {method_name} is not implemented yet. Skipping.")
            except Exception as e:
                print(f"[ERR] Failed to run {method_name}: {e}")
                
    return results