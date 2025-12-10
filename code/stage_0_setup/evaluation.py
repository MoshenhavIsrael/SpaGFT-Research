import SpaGFT as spg
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
import os

sc.settings.verbosity = 1
sc.settings.set_figure_params(dpi=80, facecolor='white')


def calculate_jaccard_index(list1, list2):
    """Calculates the Jaccard Index between two lists of genes. Returns Jaccard index and overlap count."""
    set1 = set(list1)
    set2 = set(list2)
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0.0
    
    return intersection / union, intersection

def run_spagft_local(adata):
    """
    PLACEHOLDER: This function will run your local SpaGFT implementation.
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
    
    # We open os.devnull (the "black hole" of the OS) and redirect stdout there
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
                # If error happens, we want to see it, so we re-raise
                # The 'with' block exits, so stdout is restored before raising
                raise e
    
    
    # extract spaitally variable genes
    svg_df = gene_df[gene_df.cutoff_gft_score][gene_df.fdr < 0.05]

    print("[INFO] Finished running local SpaGFT.")
    return svg_df 