import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from evaluation import impute_missing_ranks, robust_rank_aggregation

# The constant pointing to the Atlas dataset repository
ATLAS_DATASET_REPO = "jiawennnn/STimage-benchmark-data"

def list_available_methods():
    """
    Scans the Hugging Face repository and returns a list of all methods
    for which SVG data exists.
    """
    print("Fetching available methods from Atlas...")
    try:
        # Retrieve list of all files in the repository
        all_files = list_repo_files(repo_id=ATLAS_DATASET_REPO, repo_type="dataset")
        
        # Filter: We are looking for folders inside 'SVG/'
        # The structure is SVG/{MethodName}/{FileName}
        methods = set()
        for file_path in all_files:
            if file_path.startswith("SVG/"):
                parts = file_path.split("/")
                if len(parts) >= 2:
                    methods.add(parts[1]) # The second part is the method name
        
        sorted_methods = sorted(list(methods))
        print(f"Found {len(sorted_methods)} available methods: {', '.join(sorted_methods)}")
        return sorted_methods
        
    except Exception as e:
        print(f"Error listing methods: {e}")
        return []

def get_svg_list(slide_name, method, output_dir=None):
    """
    Checks if SVG (Spatially Variable Genes) list for a specific slide and method 
    exist in 'results/atlas_results'. If not, downloads it.
    
    Args:
        slide_name (str): The slide identifier (e.g., '151673').
        method (str): The method name as it appears in the repo (e.g., 'SpaGFT').
        output_dir (str, optional): If specified, saves the file to the local disk.
        
    Returns:
        pd.DataFrame: A table containing the genes and their ranking, or None if failed.
    """
    # Check if the file exists locally first
    if output_dir:
        local_file_path = os.path.join(output_dir, f"{slide_name}_{method}_SVG.csv")
        if os.path.exists(local_file_path):
            print(f"Loading SVG list from local file: {local_file_path}")
            return pd.read_csv(local_file_path)
    
    # Construct the file path within the repository based on the pattern found in app.R
    # Pattern: SVG/{method}/{slide_name}_organized.csv
    repo_filename = f"SVG/{method}/{slide_name}_organized.csv"
    
    print(f"Attempting to fetch SVG list for Slide: '{slide_name}' using Method: '{method}'...")
    
    try:
        # Download the file to local cache
        file_path = hf_hub_download(
            repo_id=ATLAS_DATASET_REPO,
            repo_type="dataset",
            filename=repo_filename
        )
        
        # Load data into Pandas
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(df)} rows.")
        
        # Save locally if requested
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            df.to_csv(local_file_path, index=False)
            print(f"Saved local copy to: {local_file_path}")
            
        return df

    except Exception as e:
        print(f"Error fetching SVG list: {e}")
        print(f"Tip: Verify that slide ID '{slide_name}' exists for method '{method}'.")
        return None

def get_ground_truth_list(slide_name, methods_list=None, output_dir=None):
    """
    Generates a consensus 'Ground Truth' SVG list for a specific slide 
    by aggregating results from all available methods using RRA.
    
    Args:
        slide_name (str): Target slide ID.
        methods_list (list): Optional list of methods to use. If None, fetches all available.
        output_dir (str): Optional. Directory to save/load individual lists and final GT.
        
    Returns:
        pd.DataFrame: A table with columns ['gene', 'RRA_score', 'SVG_rank']
    """
    # 0. Check Cache / Output
    if output_dir:
        gt_file_path = os.path.join(output_dir, f"CONSENSUS_GT_{slide_name}.csv")
        if os.path.exists(gt_file_path):
            print(f"Loading cached Ground Truth from: {gt_file_path}")
            return pd.read_csv(gt_file_path)
        
    print(f"\n--- Generating Ground Truth for Slide: {slide_name} ---")
    
    # 1. Get methods if not provided
    if methods_list is None:
        methods_list = list_available_methods()
        
    # 2. Collect rankings from all methods
    valid_methods = []    
    data_frames = []
    
    for method in methods_list:
        df = get_svg_list(slide_name, method)
        
        if df is not None and not df.empty and '.rank_index' in df.columns:
            # We assume the index or a column holds the gene name. 
            # Looking at typical format, usually 'gene' or index.
            # Let's standardize column names if needed.
            if 'gene' not in df.columns:
                 # Attempt to find gene column or use index
                 if df.index.name == 'gene':
                     df = df.reset_index()
                 else:
                     # Fallback: assume first string column is gene or first column
                     df.rename(columns={df.columns[0]: 'gene'}, inplace=True)
            
            # Keep only essential data to save memory
            df = df[['gene', '.rank_index']].copy()
            df.set_index('gene', inplace=True)
            df.rename(columns={'.rank_index': method}, inplace=True)
            
            data_frames.append(df)
            valid_methods.append(method)
            print(f"  [+] Included {method} ({len(df)} genes)")
        else:
            print(f"  [-] Skipped {method} (Data not found or empty)")
            
    if not valid_methods:
        print("Error: No valid data found for any method.")
        return None

    # 3. Merge all dataframes (Outer Join to get Union of genes)
    print("Aggregating gene lists...")
    full_rank_df = pd.concat(data_frames, axis=1)
    
    print(f"Total unique genes across {len(valid_methods)} methods: {len(full_rank_df)}")
    
    # 4. Impute missing ranks
    # Rank for missing = (n + 1 + N) / 2
    imputed_df = impute_missing_ranks(full_rank_df)
    
    # 5. Run Robust Rank Aggregation
    print("Running Robust Rank Aggregation (RRA)...")
    gt_results = robust_rank_aggregation(imputed_df)
    
    print("Ground Truth generation complete.")
    
    if output_dir:
        save_path = os.path.join(output_dir, f"CONSENSUS_GT_{slide_name}.csv")
        gt_results.to_csv(save_path, index=False)
        print(f"Saved consensus list to: {save_path}")
        
    return gt_results

# --- Example Usage ---
if __name__ == "__main__":
    # Example flow
    target_slide = "Human_Breast_Andersson_10142021_ST_A1" 
    
    # Generate GT
    gt_df = get_ground_truth_list(target_slide, output_dir=r'C:\mnt\data\SpaGFT-Research\results\atlas_results')