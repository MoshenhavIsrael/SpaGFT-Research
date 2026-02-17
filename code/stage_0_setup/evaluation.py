# --- Imports ---
import SpaGFT as spg
import numpy as np
import scanpy as sc
from scipy.stats import binom
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
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import multipletests

N_JOBS = 1  # Number of parallel jobs for computations
RANK_MORAN_I_BY_Z = True  # Whether to rank Moran's I results by Z-score (if False, ranks by raw I value)

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


# --- Create Consensus Ground Truth by RRA ---


def impute_missing_ranks(rank_df):
    """
    Processes rank data to:
    1. Resolve ties using average ranking (e.g., ties for rank 1 become 1.5).
    2. Impute missing ranks for genes not found in a specific method.
    
    Formula for missing: ((n + 1) + N) / 2
    
    Args:
        rank_df (pd.DataFrame): DataFrame with genes as index, methods as columns. 
                                Values are raw ranks from Atlas. NaNs indicate missing genes.
    
    Returns:
        pd.DataFrame: DataFrame with corrected ties and filled missing values.
    """
    N = len(rank_df) # Total number of unique genes (Universe size)
    
    # We work on a copy to avoid SettingWithCopy warnings on the original df
    filled_df = rank_df.copy()
    
    for method in filled_df.columns:
        # 1. Extract valid ranks for this method (exclude NaNs)
        valid_data = filled_df[method].dropna()
        n = len(valid_data)
        
        # 2. Resolve Ties:
        # If input is [1, 1, 3], 'average' method converts it to [1.5, 1.5, 3.0]
        # This handles the case where Atlas gives '1' to multiple genes efficiently.
        corrected_ranks = valid_data.rank(method='average')
        
        # Update the non-NaN values in the dataframe with the averaged ranks
        filled_df.loc[corrected_ranks.index, method] = corrected_ranks
        
        # 3. Calculate Imputation Value for the tail:
        # The average rank of positions (n+1) to N
        imputed_value = (n + 1 + N) / 2
        
        # Fill NaNs
        filled_df[method] = filled_df[method].fillna(imputed_value)
        
    return filled_df

def robust_rank_aggregation(rank_df):
    """
    Implements the Robust Rank Aggregation (RRA) algorithm.
    Checks if genes are ranked consistently better than expected under a null hypothesis
    of uniform distribution.
    
    Ref: Kolde et al., "Robust rank aggregation for gene list integration and meta-analysis"
    
    Args:
        rank_df (pd.DataFrame): DataFrame (Genes x Methods) with FULL ranks (missing values already imputed).
        
    Returns:
        pd.DataFrame: A DataFrame containing the RRA score and final ranking.
    """
    # 1. Normalize ranks to [0, 1] (r = rank / N)
    N = len(rank_df)
    normalized_ranks = rank_df / N
    
    # 2. Sort the normalized rank vector for each gene
    # resulting shape: (n_genes, n_methods), values are sorted row-wise
    sorted_r = np.sort(normalized_ranks.values, axis=1)
    n_methods = sorted_r.shape[1]
    
    # 3. Calculate P-values for each order statistic
    # We test the probability that the k-th rank is <= x under uniform distribution.
    # This follows a Beta distribution or Cumulative Binomial Distribution.
    # P(U_(k) <= x) = sum_{j=k}^{M} (M choose j) * x^j * (1-x)^(M-j)
    # Using scipy.stats.binom.sf (Survival Function = 1 - CDF) to compute the tail sum efficiently.
    # sf(k-1, n, p) is equivalent to probability of getting >= k successes
    
    p_values = np.zeros_like(sorted_r)
    
    for k in range(1, n_methods + 1):
        # k is 1-based index (1st rank, 2nd rank...)
        # In Python array, this corresponds to column k-1
        x = sorted_r[:, k-1]
        
        # Calculate p-value for the k-th order statistic across all genes
        # "What is the probability of seeing a value <= x at the k-th position?"
        # logic: binom.sf(k-1, n_methods, x) gives prob of having >= k values <= x
        p_values[:, k-1] = binom.sf(k-1, n_methods, x)

    # 4. The RRA score for a gene is the minimum of these p-values
    # (Correction for multiple testing *within* the gene vector is implicitly handled by the rho-score definition, 
    # but a Bonferroni correction for the number of methods is often applied to the final score 
    # to maintain strict p-value interpretation. Here we return the raw score for ranking).
    rra_scores = np.min(p_values, axis=1)
    
    # Create result DataFrame
    results = pd.DataFrame({
        'gene': rank_df.index,
        'RRA_score': rra_scores
    })
    
    # Sort by score (lower is better/more significant)
    results = results.sort_values('RRA_score')
    results['consensus_rank'] = range(1, len(results) + 1)
    
    return results


# --- SVG Method Implementations ---


def _run_spagft_local(adata, graph_params=None):
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
    
    try:
        # construct graph builder
        if graph_params is None:
            graph_builder = None
        else:
            graph_builder = spg.utils.SpatialGraphBuilder(
                method=graph_params.get('method', 'knn'),
                n_neighbors=graph_params.get('n_neighbors', 6),
                radius=graph_params.get('radius', None),
                weighting_scheme = graph_params.get('weighting_scheme', 'binary'),
                kernel_width = graph_params.get('kernel_width', 1.0)
                    )
        
        # determine the number of low-frequency FMs and high-frequency FMs
        (ratio_low, ratio_high) = spg.gft.determine_frequency_ratio(adata_local,
                                                                    ratio_neighbors=1,
                                                                    spatial_info=coord_columns,
                                                                    graph_builder=graph_builder)
        
        # calculation
        gene_df = spg.detect_svg(adata_local,
                                ratio_low_freq=ratio_low,
                                ratio_high_freq=ratio_high,
                                ratio_neighbors=1,
                                filter_peaks=True,
                                S=6,
                                spatial_info=coord_columns,
                                graph_builder=graph_builder,
                                )
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

def _run_squidpy_morans(adata, graph_params=None):
    """
    Calculates Moran's I using Squidpy to generate Z-scores.
    Supports custom graph construction parameters to align with SpaGFT logic.
    Falls back to simple Scanpy implementation if Squidpy is missing (with warning).
    """
    print("[INFO] Running Moran's I analysis...")
    adata_calc = adata.copy()
    
    # 0. Ensure spatial coordinates exist in obsm['spatial']
    if 'spatial' not in adata_calc.obsm:
        if 'array_row' in adata_calc.obs and 'array_col' in adata_calc.obs:
            # Check for 3D or 2D
            cols = ['array_row', 'array_col']
            if 'array_z' in adata_calc.obs.columns:
                cols.append('array_z')
            adata_calc.obsm['spatial'] = adata_calc.obs[cols].values
        else:
            print("[WARN] Could not find spatial coordinates. Moran's I might fail.")

    # Variable to track which connectivity key to use for Moran's I
    # Defaulting to standard Scanpy output
    target_conn_key = 'connectivities' 

    # 1. Graph Construction Logic
    if graph_params is not None and sq is not None:
        print(f"[INFO] Building custom spatial graph using Squidpy: {graph_params['method']}")
        
        # Extract params with defaults
        method = graph_params.get('method', 'knn')
        n_neighs = graph_params.get('n_neighbors', 6)
        radius = graph_params.get('radius', 'auto')
        weighting = graph_params.get('weighting_scheme', 'binary') # 'binary' or 'gaussian_kernel'
        sigma = graph_params.get('kernel_width', 1.0) # Sigma for RBF
        
        # -- A. Auto K and Radius Calculation --
        if method == 'knn' and n_neighs == 'auto':
            print("   -> Calculating auto-K as sqrt(N) / 2...")
            n_neighs = int(np.ceil(np.sqrt(adata_calc.n_obs) / 2))
            n_neighs = max(4, n_neighs)  # Ensure at least 4 neighbors
            print(f"   -> Auto-K set to: {n_neighs}")
        if method == 'knn' and n_neighs == 'all':
            n_neighs = adata_calc.n_obs - 1  # All other points
            print(f"   -> Using all points as neighbors: K={n_neighs}")
        if method == 'radius' and radius == 'auto':
            print("   -> Calculating auto-radius based on median 6-NN distance...")
            coords = adata_calc.obsm['spatial']
            # Safety check for small datasets
            k_target = min(6, coords.shape[0] - 1)
            if k_target > 0:
                nbrs = NearestNeighbors(n_neighbors=k_target + 1).fit(coords)
                dists, _ = nbrs.kneighbors(coords)
                # Median distance to 6th neighbor * 1.5 buffer
                radius = np.median(dists[:, k_target]) * 1.5
                print(f"   -> Auto-radius set to: {radius:.4f}")
            else:
                radius = 1.0 # Fallback
        
        # -- B. Build Topology (Squidpy) --
        # Note: squidpy.gr.spatial_neighbors outputs to 'spatial_connectivities'
        if method == 'delaunay':
            sq.gr.spatial_neighbors(adata_calc, coord_type="generic", delaunay=True)
        elif method == 'radius':
            sq.gr.spatial_neighbors(adata_calc, coord_type="generic", radius=float(radius), n_neighs=1)
        else: # knn
            sq.gr.spatial_neighbors(adata_calc, coord_type="generic", n_neighs=n_neighs)
            
        # Update key to point to Squidpy's output
        target_conn_key = 'spatial_connectivities'

        # -- C. Apply Weighting --
        # If the user wants Gaussian weights, we override the binary connectivities
        # with values derived from spatial_distances.
        if weighting != 'binary' and 'spatial_distances' in adata_calc.obsp:
            print(f"   -> Applying {weighting} weighting...")
            
            # Get the sparse distance matrix
            dist_mtx = adata_calc.obsp['spatial_distances']
            
            # We copy the structure of distances to be our new weights
            weighted_mtx = dist_mtx.copy()
            
            if weighting == 'inverse_distance':
                # Apply Inverse Distance: W = 1 / d^2
                # Note: We only operate on .data (the non-zero edges) to keep sparsity
                # Avoid division by zero just in case
                weighted_mtx.data = 1.0 / np.maximum(weighted_mtx.data ** 2, 1e-10)
            
            elif weighting == 'gaussian_kernel':
                # Apply RBF kernel: W = exp(-d^2 / (2*sigma^2))
                # Note: We only operate on .data (the non-zero edges) to keep sparsity
                # Avoid division by zero just in case
                denom = 2 * (sigma ** 2)
                weighted_mtx.data = np.exp(-(weighted_mtx.data ** 2) / denom)
            
            # Overwrite the connectivity matrix used by Squidpy
            adata_calc.obsp[target_conn_key] = weighted_mtx

    else:
        # Fallback / Default Legacy Behavior
        if 'spatial_connectivities' not in adata_calc.obsp and 'connectivities' not in adata_calc.obsp:
            print("   -> Computing standard spatial neighbors (Scanpy KNN)...")
            sc.pp.neighbors(adata_calc, use_rep='spatial', n_neighbors=6)
            target_conn_key = 'connectivities'
        elif 'spatial_connectivities' in adata_calc.obsp:
             target_conn_key = 'spatial_connectivities'

    # 2. Run Moran's I with Squidpy
    if sq is not None:
        print(f"   -> Using Squidpy for permutation-based Z-scores (key='{target_conn_key}')...")
        
        try:
            sq.gr.spatial_autocorr(
                adata_calc, 
                mode='moran', 
                connectivity_key=target_conn_key,  # <--- DYNAMIC KEY HERE
                n_perms=100, 
                n_jobs=1, # Often safer to set 1 inside functions to avoid nesting issues
                genes=adata_calc.var_names
            )
        except KeyError as e:
            # Fallback if specific key is missing despite checks
            print(f"[ERR] Failed to find connectivity key {target_conn_key}. trying default.")
            sq.gr.spatial_autocorr(adata_calc, mode='moran', n_perms=100)

        # Squidpy saves results in adata.uns['moranI']
        if 'moranI' in adata_calc.uns:
            res_df = adata_calc.uns['moranI'].copy()
            # Calculate BH FDR on pval_z_sim
            pvals = res_df['pval_z_sim'].values
            _, fdrs, _, _ = multipletests(pvals, method='fdr_bh')
            res_df['fdr_bh'] = fdrs
            
            # Rank by z_score descending (pval_z_sim ascending). If RANK_MORAN_I_BY_Z is False, rank by raw I value instead.
            if RANK_MORAN_I_BY_Z:
                res_df = res_df.sort_values(['pval_z_sim'], ascending=[True])
            else:
                res_df = res_df.sort_values(['I'], ascending=[False])
                # Filter out non-significant genes 
                res_df = res_df[res_df['fdr_bh'] < 0.05]
            
            res_df['SVG_rank'] = range(1, len(res_df) + 1)
            return res_df[['I', 'pval_z_sim', 'fdr_bh', 'SVG_rank']]
        else:
            print("[WARN] Moran's I calculation produced no result.")
            return None

    else:
        # 3. Fallback to Scanpy (Raw Index only)
        print("[WARN] Squidpy not found. Using basic Scanpy Moran's I.")
        return _run_scanpy_morans(adata_calc)
    

# --- Registry & Wrapper ---


# Mapping method names (strings) to functions
SVG_METHODS_REGISTRY = {
    "SpaGFT": _run_spagft_local,
    "MoranI": _run_squidpy_morans,
}

def run_svg_methods(adata, methods_list, tissue_id, results_dir, graph_params=None):
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
    
    # Support single string input just in case
    if isinstance(methods_list, str):
        methods_list = [methods_list]

    # Create the svg_results subdirectory if passed directly, 
    # but usually passed path should already be complete.
    if results_dir is not None:
        results_dir = Path(results_dir)
        if not results_dir.exists():
            results_dir.mkdir(parents=True, exist_ok=True)
    
    for method_name in methods_list:
        if method_name not in SVG_METHODS_REGISTRY:
            print(f"[WARN] Method '{method_name}' is not registered. Skipping.")
            continue

        if results_dir is not None:
            file_path = results_dir / f"{method_name}_{tissue_id}_SVGs.csv"

        if results_dir and file_path.exists():
            print(f"[INFO] Loading existing results for {method_name}...")
            results[method_name] = pd.read_csv(file_path, index_col=0)
        else:
            if adata is None:
                print(f"[ERR] Result for {method_name} missing and adata is None. Cannot calculate.")
                continue

            try:
                func = SVG_METHODS_REGISTRY[method_name]
                df = func(adata, graph_params=graph_params)
                
                # Save results
                results[method_name] = df
                if results_dir is not None:
                    df.to_csv(file_path, index=True)
                    print(f"[INFO] Saved {method_name} results to {file_path.name}")
                
            except NotImplementedError:
                print(f"[WARN] {method_name} is not implemented yet. Skipping.")
            except Exception as e:
                print(f"[ERR] Failed to run {method_name}: {e}")
                
    return results