from contextlib import redirect_stdout
import shutil
import pandas as pd
import numpy as np
import scanpy as sc
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Adding the current directory and its parent to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../stage_0_setup")

# Import local modules
from transformations import transform_adata
from scenarios_manager import generate_default_scenarios, check_scenario_cache
# Importing from stage_0 to reuse loading logic 
from stage_0_setup.evaluation import run_svg_methods, calculate_jaccard_index
from stage_0_setup.data_manager import ensure_directories, load_or_download_data, preprocess_adata, STABILITY_RESULTS_DIR, SVG_RESULTS_DIR

# --- Default Configuration ---
DEFAULT_SLIDE_ID = "Human_Breast_Andersson_10142021_ST_A1"
DEFAULT_TECH = "ST"
DEFAULT_METHODS = ["SpaGFT", "MoranI"]
DEFAULT_TOP_N = 100



# --- Main Stability Test Function ---
def run_stability_test(slide_id=DEFAULT_SLIDE_ID,
                       tech=DEFAULT_TECH,
                       methods_to_test=DEFAULT_METHODS,
                       graph_params=None,
                       scenarios=None,
                       spatial_key="spatial",
                       top_n=DEFAULT_TOP_N,
                       use_saved_baseline=True,
                       save_results=False,
                       restrict_logs=True):
    """
    Runs a comprehensive stability test suite applying various linear transformations.
    """
    # Handle restricted logging
    if restrict_logs:
        # Open os.devnull and redirect stdout
        with open(os.devnull, 'w') as fnull:
            with redirect_stdout(fnull):
                try:
                    func_output = run_stability_test(slide_id=slide_id,
                                       tech=tech,
                                       methods_to_test=methods_to_test,
                                       scenarios=scenarios,
                                       graph_params=graph_params,
                                       spatial_key=spatial_key,
                                       top_n=top_n,
                                       use_saved_baseline=use_saved_baseline,
                                       save_results=save_results,
                                       restrict_logs=False)
                    return func_output
                except Exception as e:
                    raise e

        
    print(f"=== Stage 1: Transformation Stability Analysis | Slide: {slide_id} ===")

    # 0. Setup
    ensure_directories()
    svg_results_dir = SVG_RESULTS_DIR
    stability_results_dir = STABILITY_RESULTS_DIR
    results = []
    results_path = os.path.join(stability_results_dir, f"stability_{slide_id}.csv")
    former_results = pd.DataFrame()
    if save_results and os.path.exists(results_path):
        former_results = pd.read_csv(results_path)

    # Support single string input just in case
    if isinstance(methods_to_test, str):
        methods_to_test = [methods_to_test]

    # Define Test Scenarios
    if scenarios is None:
        scenarios = generate_default_scenarios()

    # Skip calculation if all scenarios and methods are cached
    if all(len(check_scenario_cache(methods_to_test, config, former_results)) == 0 for config in scenarios):
        print("[INFO] All scenarios and methods have cached results. Skipping stability test.")
        return former_results       

    # 1. Load Original Data
    adata_orig = load_or_download_data(slide_id, tech)
    if adata_orig is None:
        return None
    adata_orig = preprocess_adata(adata_orig)
    
    # 2. Load/Run Baseline Results (for all methods)
    baseline_dfs = {}
    baseline_scores = {}
    baseline_top_genes = {}
    
    # Batch run/load baseline methods
    baseline_results_dict = run_svg_methods(
        adata_orig, 
        methods_list=methods_to_test, 
        graph_params=graph_params,
        tissue_id=slide_id, 
        results_dir=svg_results_dir if use_saved_baseline else None,
        )
    
    for method in methods_to_test:
        df = baseline_results_dict.get(method)
        if df is None or df.empty:
            print(f"[WARN] No baseline results for {method}. Skipping.")
            # Remove from methods to test
            methods_to_test.remove(method)
            continue
            
        baseline_dfs[method] = df
        baseline_top_genes[method] = set(df.index[:top_n])
        # Handle different column names (SpaGFT uses 'gft_score', others might use 'score')
        score_col = 'gft_score' if method == 'SpaGFT' else 'pval_z_sim' if method == 'MoranI' else 'score'
        baseline_scores[method] = df[score_col]


    # 3. Run Test Scenarios
    # Iterate over Test Scenarios
    for config in scenarios:
        name = config.get('name', 'Unknown')
        print(f"\nProcessing Scenario: {name}")
        
        missing_methods = check_scenario_cache(methods_to_test, config, former_results)
        if len(missing_methods) == 0:
            print("  -> All methods have cached results. Skipping transformation.")
            continue

        # Use a temp directory for rotated results to avoid polluting the main results
        temp_trans_dir = stability_results_dir / "transformed_temp"
        # Perform Transform. We transform once per scenario, then run all methods 
        adata_trans = transform_adata(
                    adata_orig, 
                    angle=config.get('angle', 0),
                    translation=config.get('translation', 0),
                    scaling=config.get('scaling', 1.0),
                    flip=config.get('flip', False),
                    spatial_key=spatial_key
                )

        for method in missing_methods:
            # Run Method on Transformed Data
            trans_res_dict = run_svg_methods(adata_trans, 
                                             methods_list=[method],
                                             graph_params=graph_params,
                                             tissue_id=slide_id, 
                                             results_dir=temp_trans_dir)
            svg_trans_df = trans_res_dict[method]
                
            # Calculate Metrics        
            # 1. Jaccard
            top_genes_trans = list(svg_trans_df.index[:top_n])
            jaccard, overlap_count = calculate_jaccard_index(baseline_top_genes[method], top_genes_trans)

            # 2. Spearman
            score_col = 'gft_score' if method == 'SpaGFT' else 'pval_z_sim' if method == 'MoranI' else 'score'
            scores_trans = svg_trans_df[score_col]
            
            common_genes = baseline_scores[method].index.intersection(scores_trans.index)
            print(f"[INFO] Method: {method} | Common Genes for Spearman: {len(common_genes)}. Num of baseline Genes: {len(baseline_scores[method])}")
            spearman_val = np.nan
            if len(common_genes) > 0:
                spearman_val, _ = spearmanr(baseline_scores[method][common_genes], scores_trans[common_genes])
            
            print(f"  -> Jaccard: {jaccard:.4f} | Spearman: {spearman_val:.4f}")
            
            # Store Result
            res_entry = config.copy()
            # Convert translation list to tuple so it's hashable for pandas
            trans_val = config.get('translation', 0)
            if isinstance(trans_val, list):
                res_entry['translation'] = tuple(trans_val)
            res_entry['method'] = method
            res_entry['Jaccard'] = jaccard
            res_entry['Spearman'] = spearman_val
            res_entry['Overlap'] = overlap_count
            results.append(res_entry)

        # Cleanup temp folder
        if os.path.exists(temp_trans_dir):
            shutil.rmtree(temp_trans_dir)
        # Reset transformed adata
        adata_trans = None 
        

    # 4. Visualization & Reporting
    df_results = pd.DataFrame(results)
    
    if save_results:
        # Merge with former results (The new results in the head; Avoid duplicates)
        if 'former_results' in locals():
            df_results = pd.concat([df_results, former_results], ignore_index=True)
            df_results['translation'] = df_results['translation'].astype(str)
            df_results = df_results.drop_duplicates(
                subset=['method', 'name', 'angle', 'scaling', 'translation', 'flip'], 
                keep='first'
                )
        # Save to CSV
        df_results.to_csv(results_path, index=False)
        print(f"\n[INFO] Stability results saved to {results_path}")

    return df_results

def plot_results(df_results, top_n):
    """Visualizes the stability metrics across different scenarios."""
    # Melt dataframe for seaborn (easier to plot both metrics side by side or in separate panels)
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Jaccard Plot
    sns.barplot(data=df_results, x='name', y='Jaccard', ax=ax[0], palette="magma")
    ax[0].set_title(f"Top-{top_n} SVG Set Consistency (Jaccard Index)")
    ax[0].set_ylim(0, 1.1)
    ax[0].set_ylabel("Jaccard Index")
    ax[0].grid(axis='y', linestyle='--', alpha=0.5)
    
    # Spearman Plot
    sns.barplot(data=df_results, x='name', y='Spearman', ax=ax[1], palette="magma")
    ax[1].set_title("Global Score Ranking Correlation (Spearman)")
    ax[1].set_ylim(0, 1.1)
    ax[1].set_ylabel("Spearman Rho")
    ax[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.xlabel("Transformation Scenario")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    # You can define custom scenarios here or use the defaults
    metrics = run_stability_test(slide_id="fem3_5x_E7_A_left",
                      tech = "exseq",
                      save_results=True)