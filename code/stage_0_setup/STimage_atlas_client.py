import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files
import os

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

# --- Example Usage ---
if __name__ == "__main__":
    # 1. First, check which methods are available to ensure correct spelling (e.g., SpaGFT)
    methods = list_available_methods()
    
    # 2. Suppose we want SpaGFT data for a specific slide
    target_slide = "Human_Breast_Andersson_10142021_ST_A1" 
    target_method = "SpaGFT" # Ensure this matches one of the printed methods
    
    if target_method in methods:
        df = get_svg_list(target_slide, target_method, output_dir="./atlas_results")
        if df is not None:
            print("\nTop 10 SVGs identified:")
            print(df.head(10))
    else:
        print(f"Method '{target_method}' not found in the Atlas.")