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
# Importing from stage_0 to reuse loading logic 
from stage_0_setup.evaluation import run_spagft_local, calculate_jaccard_index
from stage_0_setup.data_manager import load_or_download_data, preprocess_adata, RESULTS_DIR

# --- Default Configuration ---
DEFAULT_SLIDE_ID = "Human_Breast_Andersson_10142021_ST_A1"
DEFAULT_TECH = "ST"
DEFAULT_TOP_N = 100

def generate_default_scenarios():
    """Generates a standard suite of stability tests."""
    scenarios = []
    
    # 1. Rotation Sweep
    for angle in [30, 45, 90, 180]:
        scenarios.append({
            'name': f'Rotate {angle}°',
            'angle': angle, 'scaling': 1.0, 'translation': 0, 'flip': False
        })
        
    # 2. Scaling
    scenarios.append({'name': 'Scale x0.8', 'angle': 0, 'scaling': 0.8, 'translation': 0, 'flip': False})
    scenarios.append({'name': 'Scale x1.2', 'angle': 0, 'scaling': 1.2, 'translation': 0, 'flip': False})
    
    # 3. Translation (Shift)
    scenarios.append({'name': 'Shift X+10', 'angle': 0, 'scaling': 1.0, 'translation': (10, 0), 'flip': False})
    
    # 4. Flip
    scenarios.append({'name': 'Flip X', 'angle': 0, 'scaling': 1.0, 'translation': 0, 'flip': True})
    
    return scenarios

def run_stability_test(slide_id=DEFAULT_SLIDE_ID,
                       tech=DEFAULT_TECH,
                       scenarios=None,
                       spatial_key="spatial",
                       top_n=DEFAULT_TOP_N,
                       save_results=False):
    """
    Runs a comprehensive stability test suite applying various linear transformations.
    """
    if scenarios is None:
        scenarios = generate_default_scenarios()
        
    print(f"=== Stage 1: Transformation Stability Analysis | Slide: {slide_id} ===")

    # 1. Load & Preprocess Baseline Data
    adata_orig = load_or_download_data(slide_id, tech)
    if adata_orig is None:
        return None
    adata_orig = preprocess_adata(adata_orig)
    
    # 2. Run Baseline SpaGFT (0 degrees, no transform)
    print("\n--- Running Baseline (Original Data) ---")
    # Note: run_spagft_local returns a DataFrame sorted by rank/score
    svg_orig_df = run_spagft_local(adata_orig)
    
    # Extract baseline metrics
    top_genes_orig = set(svg_orig_df.index[:top_n])
    # We keep the full score series to calculate Spearman on all common genes later
    scores_orig = svg_orig_df['gft_score']

    # 3. Run Test Scenarios
    results = []
    # Store former results for uniting with new ones
    if save_results:
        results_path = os.path.join(RESULTS_DIR, f"stability_results_{slide_id}.csv")
        if os.path.exists(results_path):
            former_results = pd.read_csv(results_path)

    # Iterate over Test Scenarios
    for config in scenarios:
        name = config.get('name', 'Unknown')
        print(f"\nProcessing Scenario: {name}")
        print(f"  Params: Angle={config.get('angle')}, Scale={config.get('scaling')}, "
              f"Trans={config.get('translation')}, Flip={config.get('flip')}")
        
        # Use former result if available
        if save_results:
            if 'former_results' in locals():
                match = former_results[
                    (former_results['angle'] == config.get('angle')) &
                    (former_results['scaling'] == config.get('scaling')) &
                    (former_results['translation'] == str(config.get('translation'))) &
                    (former_results['flip'] == config.get('flip')) &
                    (former_results['Jaccard'].notnull()) &
                    (former_results['Spearman'].notnull())
                ]
                if not match.empty:
                    print("  -> Using cached results from previous run.")
                    results.append({
                        'name': name,
                        'Jaccard': match['Jaccard'].values[0],
                        'Spearman': match['Spearman'].values[0],
                        'Overlap': match['Overlap'].values[0],
                        'angle': config.get('angle'),
                        'scaling': config.get('scaling'),
                        'translation': config.get('translation'),
                        'flip': config.get('flip')
                    })
                    continue
        
        # A. Transform
        adata_trans = transform_adata(
            adata_orig, 
            angle=config.get('angle', 0),
            translation=config.get('translation', 0),
            scaling=config.get('scaling', 1.0),
            flip=config.get('flip', False),
            spatial_key=spatial_key
        )
        
        # B. Run SpaGFT on Transformed Data
        svg_trans_df = run_spagft_local(adata_trans)
        
        # C. Calculate Metrics
        
        # 1. Jaccard (Top N Overlap)
        top_genes_trans = list(svg_trans_df.index[:top_n])
        # Using the helper from stage_0 
        jaccard, overlap_count = calculate_jaccard_index(top_genes_orig, top_genes_trans)

        # 2. Spearman Correlation (Global Rank Stability)
        # We align the indices (genes) to ensure we compare Gene A to Gene A
        scores_trans = svg_trans_df['gft_score']
        common_genes = scores_orig.index.intersection(scores_trans.index)
        
        spearman_val = np.nan
        if len(common_genes) > 0:
            spearman_val, _ = spearmanr(scores_orig[common_genes], scores_trans[common_genes])
        
        print(f"  -> Jaccard: {jaccard:.4f} | Spearman: {spearman_val:.4f}")
        
        # Store Result
        res_entry = config.copy()
        # Convert translation list to tuple so it's hashable for pandas
        trans_val = config.get('translation', 0)
        if isinstance(trans_val, list):
            res_entry['translation'] = tuple(trans_val)
        res_entry['Jaccard'] = jaccard
        res_entry['Spearman'] = spearman_val
        res_entry['Overlap'] = overlap_count
        results.append(res_entry)

    # 4. Visualization & Reporting
    df_results = pd.DataFrame(results)
    
    # Plotting
    # plot_results(df_results, top_n)
    
    if save_results:
        # Merge with former results (The new results in the head; Avoid duplicates)
        if 'former_results' in locals():
            df_results = pd.concat([df_results, former_results], ignore_index=True)
            df_results['translation'] = df_results['translation'].astype(str)
            df_results = df_results.drop_duplicates(
                subset=['name', 'angle', 'scaling', 'translation', 'flip'], 
                keep='first'
                )
        # Save to CSV
        df_results.to_csv(results_path, index=False)

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