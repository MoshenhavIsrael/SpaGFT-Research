import pandas as pd
import numpy as np
import scanpy as sc
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Adding the current directory and its parent to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../stage_0_setup")

# Import local modules
from transformations import transform_adata
# Importing from stage_0 to reuse loading logic 
from stage_0_setup.evaluation import run_spagft_local, calculate_jaccard_index
from stage_0_setup.data_manager import load_or_download_data, preprocess_adata

# --- Default Configuration (Constants) ---
DEFAULT_SLIDE_ID = "Human_Breast_Andersson_10142021_ST_A1"
DEFAULT_TECH = "ST"
DEFAULT_METHOD_NAME = "SpaGFT"
DEFAULT_TOP_N = 100

def run_rotation_stability_test(slide_id=DEFAULT_SLIDE_ID,
                                tech=DEFAULT_TECH,
                                angles=[30, 45, 60, 90, 180],
                                spatial_key="spatial", # Can be changed to ['array_row', 'array_col']
                                top_n=100):
     
    print(f"=== Stage 1: Rotation Stability Analysis | Slide: {slide_id} ===")

    # 1. Get Data, Construct AnnData and Preprocess
    adata_orig = load_or_download_data(slide_id, tech)
    if adata_orig is None:
        return None
    adata_orig = preprocess_adata(adata_orig)
    
    # 2. Run Baseline (0 degrees)
    if not 'gft_score' in adata_orig.var:
        print("\nRunning Baseline (0°)...")
        svg_orig_df = run_spagft_local(adata_orig)
    top_genes_orig = svg_orig_df.index.tolist()[:top_n]
    
    # Note: To calculate Spearman, we need the raw scores. 
    # In a real implementation, we would extract 'gft_score' from adata_orig.var after running SpaGFT.
    scores_orig = svg_orig_df['gft_score'][:top_n]

    results = []

    # 3. Run Rotations
    for angle in angles:
        print(f"\nProcessing Angle: {angle}°...")
        
        # A. Transform
        adata_rot = transform_adata(adata_orig, angle=angle, spatial_key=spatial_key)
        
        # B. Run SpaGFT on Rotated Data
        svg__rot_df = run_spagft_local(adata_rot)
        top_genes_rot = svg__rot_df.index.tolist()[:top_n]
        scores_rot = svg__rot_df['gft_score'][:top_n]
        
        # C. Calculate Metrics
        jaccard, overlap = calculate_jaccard_index(top_genes_orig, top_genes_rot)
        
        spearman = np.nan
        if scores_orig is not None and scores_rot is not None:
            # Align indices
            common_genes = scores_orig.index.intersection(scores_rot.index)
            spearman, _ = spearmanr(scores_orig[common_genes], scores_rot[common_genes])
        
        print(f"\n  -> Jaccard: {jaccard:.4f} | Spearman: {spearman:.4f}")
        
        results.append({
            'Angle': angle,
            'Jaccard': jaccard,
            'Overlap': overlap,
            'Spearman': spearman
        })

    # 4. Aggregate & Visualize
    df_results = pd.DataFrame(results)
    
    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Jaccard
    sns.lineplot(data=df_results, x='Angle', y='Jaccard', marker='o', ax=ax[0], color='tab:blue')
    ax[0].set_ylim(0, 1.05)
    ax[0].set_title(f"Top-{top_n} SVG Consistency (Jaccard)")
    ax[0].grid(True, linestyle='--', alpha=0.6)
    
    # Spearman
    sns.lineplot(data=df_results, x='Angle', y='Spearman', marker='s', ax=ax[1], color='tab:orange')
    ax[1].set_ylim(0, 1.05)
    ax[1].set_title("Global Score Correlation (Spearman)")
    ax[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()
    
    return df_results

if __name__ == "__main__":
    # Example usage
    # Ensure you are pointing to the correct spatial key. 
    # For Visium it is often 'spatial' in obsm.
    # For older ST data it might be columns ['array_row', 'array_col'].
    metrics = run_rotation_stability_test(spatial_key="spatial")