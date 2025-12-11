import os
import sys
from pathlib import Path
import pandas as pd
import scanpy as sc
import anndata as ad

# Import helper functions from the same directory
from get_STimage_data import download_slide_data
from STimage_atlas_client import get_svg_list

# --- Path Configurations ---
# Assuming structure: SpaGFT-Research/code/stage_0_setup/data_manager.py
# So parent is 'stage_0_setup', parent.parent.parent is 'SpaGFT-Research'
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data Directories 
DATA_DIR = PROJECT_ROOT / "data"
GENE_EXP_DIR = DATA_DIR / "gene_exp"
COORD_DIR = DATA_DIR / "coord"
IMAGE_DIR = DATA_DIR / "image"

# Results Directories 
RESULTS_DIR = PROJECT_ROOT / "results"
ATLAS_RESULTS_DIR = RESULTS_DIR / "atlas_results"

def ensure_directories():
    """
    Creates the required directory structure if it doesn't exist.
    Structure:
    SpaGFT-Research/
    ├── data/
    │   ├── gene_exp/
    │   ├── coord/
    │   └── image/
    └── results/
        └── atlas_results/
    """
    dirs_to_create = [
        DATA_DIR, 
        GENE_EXP_DIR, 
        COORD_DIR, 
        IMAGE_DIR, 
        RESULTS_DIR, 
        ATLAS_RESULTS_DIR
    ]
    
    for d in dirs_to_create:
        if not d.exists():
            print(f"[SETUP] Creating directory: {d}")
            d.mkdir(parents=True, exist_ok=True)

def load_or_download_data(slide_id, tech, include_z_axis=False):
    """
    Checks if raw data (count, coord, image) exists. 
    If not, downloads it using 'download_slide_data'.
    Then, constructs and returns an AnnData object.
    
    Args:
        slide_id (str): The ID of the slide.
        tech (str): Technology (e.g., 'ST', 'Visium').

    Returns:
        AnnData: An anndata object containing expression matrix and spatial coords.
    """
    # Define expected file paths based on naming conventions
    count_file = GENE_EXP_DIR / f"{slide_id}_count.csv"
    coord_file = COORD_DIR / f"{slide_id}_coord.csv"
    image_file = IMAGE_DIR / f"{slide_id}_image.png"

    # Check existence of all required files
    data_missing = not (count_file.exists() and coord_file.exists() and image_file.exists())

    if data_missing:
        print(f"\n[INFO] Data missing for {slide_id}. Downloading...")
        download_slide_data(slide_name=slide_id, tech=tech, output_dir=str(DATA_DIR))

    # --- Construct AnnData from CSVs ---
    # SpaGFT requires specific spatial info in obs/obsm
    try:
        print("[INFO] Loading CSVs and constructing AnnData...")
        
        # 1. Load Counts (Gene Expression)
        # Assuming index is barcodes/spots and columns are genes
        counts_df = pd.read_csv(count_file, index_col=0)
        
        # 2. Load Coordinates
        # Assuming format with columns like 'imagerow', 'imagecol' or similar
        coords_df = pd.read_csv(coord_file, index_col=0)

        # 3. Align Indices (Ensure spots match)
        # Keep only intersection of spots found in both files
        common_spots = counts_df.index.intersection(coords_df.index)
        counts_df = counts_df.loc[common_spots]
        coords_df = coords_df.loc[common_spots]
        if counts_df.empty or coords_df.empty:
            print(f"[ERROR] No common spots found between counts and coordinates for {slide_id}.")
            return None
        if len(common_spots) < len(counts_df.index) or len(common_spots) < len(coords_df.index):
            print(f"[WARNING] Reduced to {len(common_spots)} common spots for {slide_id}.")
            print("Ensure that the data files are correct and correspond to the same spots.")

        # 4. Create AnnData
        adata = sc.AnnData(X=counts_df)
        
        # 5. Add spatial coordinates to .obs and .obsm
        adata.obs['array_row'] = coords_df.iloc[:, 0]
        adata.obs['array_col'] = coords_df.iloc[:, 1]
        if coords_df.shape[1] > 2: adata.obs['array_z'] = coords_df.iloc[:, 2]        
        # Also standard Scanpy location
        # Assuming the coord file has columns 0 and 1 as x, y or similar
        if coords_df.shape[1] >= 2:
            adata.obsm['spatial'] = coords_df.iloc[:, :2 + include_z_axis].values
            
        adata.var_names_make_unique()
        adata.raw = adata

        print(f"[SUCCESS] AnnData object created: {adata.n_obs} spots x {adata.n_vars} genes.")
        return adata

    except Exception as e:
        print(f"[ERROR] Failed to construct AnnData for {slide_id}: {e}")
        return None

def preprocess_adata(adata, min_cells=10, normalize=True, log_transform=True):
    """
    Preprocesses the AnnData object with filtering, normalization, and log transformation.
    
    Args:
        adata (AnnData): The input AnnData object.
        min_cells (int): Minimum number of cells a gene must be expressed in to be kept.
        normalize (bool): Whether to normalize total counts per cell.
        log_transform (bool): Whether to apply log1p transformation.
    
    Returns:
        AnnData: The preprocessed AnnData object.
    """
    print("\n[INFO] Preprocessing AnnData...")
    # 1. Filter Genes
    if min_cells > 0:
        sc.pp.filter_genes(adata, min_cells=min_cells)
    
    # 2. Normalize
    if normalize:
        sc.pp.normalize_total(adata, inplace=True)
    
    # 3. Log Transform
    if log_transform:
        sc.pp.log1p(adata)
    
    print("[INFO] Preprocessing complete.")

    return adata
  

def get_reference_svgs(slide_id, method_name, output_dir=ATLAS_RESULTS_DIR, top_n=100):
    """
    Fetches the SVG list from the Atlas (reference).
    Returns a list of the top 100 genes.
    """
    print(f"\n[INFO] Fetching reference SVG list for {method_name}...")
    
    # We pass the directory path as string as per your example usage
    df = get_svg_list(slide_name=slide_id, method=method_name, output_dir=str(output_dir))
    
    if df is not None and not df.empty:
        # Assuming the dataframe has the gene names in the index or a specific column.
        
        # If there is a rank column, sort by it, otherwise assume order is rank.
        if 'rank' in df.columns:
            df = df.sort_values('rank')
            
        top_genes = df.index.tolist()[:top_n] if df.index.name else df.iloc[:top_n, 0].tolist()
        
        print(f"[SUCCESS] Retrieved {len(top_genes)} reference SVGs.")
        return top_genes
    else:
        print(f"[WARNING] Could not retrieve reference SVGs for {slide_id}.")
        return []
