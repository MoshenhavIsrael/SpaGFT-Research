import SpaGFT as spg
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt


sc.settings.verbosity = 1
sc.settings.set_figure_params(dpi=80, facecolor='white')


def calculate_jaccard_index(list1, list2):
    """Calculates the Jaccard Index between two lists of genes."""
    set1 = set(list1)
    set2 = set(list2)
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0.0
    
    return intersection / union

def run_spagft_local(adata):
    """
    PLACEHOLDER: This function will run your local SpaGFT implementation.
    """
    print("\n[INFO] Running local SpaGFT implementation...")
    
    coord_columns = adata.obs.columns.tolist()
    # QC
    sc.pp.filter_genes(adata, min_cells=10)
    # Normalization
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)

    # determine the number of low-frequency FMs and high-frequency FMs
    (ratio_low, ratio_high) = spg.gft.determine_frequency_ratio(adata,
                                                                ratio_neighbors=1,
                                                                spatial_info=coord_columns)
    
    # calculation
    gene_df = spg.detect_svg(adata,
                            spatial_info=coord_columns,
                            ratio_low_freq=ratio_low,
                            ratio_high_freq=ratio_high,
                            ratio_neighbors=1,
                            filter_peaks=True,
                            S=6)
    # S determines the  sensitivity of kneedle algorithm
    # extract spaitally variable genes
    svg_list = gene_df[gene_df.cutoff_gft_score][gene_df.fdr < 0.05].index.tolist()

    print("\n[INFO] Finished running local SpaGFT.")
    return svg_list 