import os
import pandas as pd

# --- Conventions ---
COORDS_COLUMNS = ['global_x', 'global_y', 'Z'] # name of coordinates columns (e.g. ['x', 'y', 'z'], ['X', 'Y'])
CELL_COLUMN = 'cell' # name of cell ids column
GENE_COLUMN = 'gene' # name of gene columns
SUFFIX = '_regions_genes_with_cell_types' # suffix for puncta files

# --- Load Puncta Data ---
def load_exseq_puncta_data(tissue_name, data_path, suffix=SUFFIX, coords_columns=COORDS_COLUMNS, cell_column=CELL_COLUMN, gene_column=GENE_COLUMN,
                           convert_pixel_to_micron=False, expansion_factor=1.0):
    """
    Loads ExSeq puncta data from a CSV file, applies scaling if necessary,
    and generates count and coordinate matrices.

    Save files of DataFrames:
        count_mtx: Gene counts per cell.
        coord_mtx: Cell centroids.
    """

    # Ensure puncta file exists
    puncta_file = os.path.join(data_path, 'puncta', f"{tissue_name}{suffix}.csv")
    if not os.path.exists(puncta_file):
        raise FileNotFoundError(f"Puncta file not found: {puncta_file}")
    puncta_mtx = pd.read_csv(puncta_file)
    
    # Ensure column names
    if cell_column not in puncta_mtx.columns or gene_column not in puncta_mtx.columns:
        raise ValueError(f"Expected columns '{cell_column}' and '{gene_column}' not found in puncta data.")
        
    puncta_mtx[cell_column] = puncta_mtx[cell_column].astype(str)
    if convert_pixel_to_micron:
        puncta_mtx[coords_columns[:2]] = puncta_mtx[coords_columns[:2]] * 0.171 / expansion_factor
        if len(coords_columns) > 2:
            puncta_mtx[coords_columns[2]] = puncta_mtx[coords_columns[2]] * 0.4 / expansion_factor
    elif expansion_factor != 1:
        puncta_mtx[coords_columns[:2]] = puncta_mtx[coords_columns[:2]] / expansion_factor
        if len(coords_columns) > 2:
            puncta_mtx[coords_columns[2]] = puncta_mtx[coords_columns[2]] / expansion_factor

    # Generate count_mtx: Gene Counts per Cell 
    # Group by cell and gene, count the number of puncta in each group (using .size()),
    # and then use unstack() to pivot the gene names into columns.
    # fill_value=0 ensures cells without a specific gene show a count of 0 instead of NaN.
    count_mtx = (
        puncta_mtx
        .groupby([cell_column, gene_column])[gene_column]
        .size()
        .unstack(fill_value=0)
    )
    
    # Generate coord_mtx: Cell Centroids 
    # Group by cell and calculate the mean (average) of the coordinate columns.
    coord_mtx = puncta_mtx.groupby(cell_column)[coords_columns].mean()

    # Save DataFrames to CSV files
    count_mtx.to_csv(os.path.join(data_path, "gene_exp", f"{tissue_name}_count.csv"))
    coord_mtx.to_csv(os.path.join(data_path, "coord", f"{tissue_name}_coord.csv"))

def load_full_table_data(tissue_name, data_path, suffix=SUFFIX, coords_columns=COORDS_COLUMNS, cell_column=CELL_COLUMN, needles_columns=None,
                         convert_pixel_to_micron=False, expansion_factor=1.0):
    """
    Loads single cell data (exseq/xenium/etc.) from a CSV file, with one-table format, 
    where each row corresponds to a single cell, and columns include both gene expression (many columns) 
    and spatial coordinates (2 / 3 columns).
    It is important to define the needles_columns to separate them from gene expressions columns.

    Generates count and coordinate matrices.

    Save files of DataFrames:
        count_mtx: Gene counts per cell.
        coord_mtx: Cell centroids.
    """
    # Ensure data file exists
    data_file = os.path.join(data_path, 'full_table', f"{tissue_name}{suffix}.csv")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    data_mtx = pd.read_csv(data_file, engine='python')

    # Ensure column names
    if cell_column not in data_mtx.columns:
        raise ValueError(f"Expected column '{cell_column}' not found in data.")
    data_mtx[cell_column] = data_mtx[cell_column].astype(str)
    
    if convert_pixel_to_micron:
        data_mtx[coords_columns[:2]] = data_mtx[coords_columns[:2]] * 0.171 / expansion_factor
        if len(coords_columns) > 2:
            data_mtx[coords_columns[2]] = data_mtx[coords_columns[2]] * 0.4 / expansion_factor
    elif expansion_factor != 1:
        data_mtx[coords_columns[:2]] = data_mtx[coords_columns[:2]] / expansion_factor
        if len(coords_columns) > 2:
            data_mtx[coords_columns[2]] = data_mtx[coords_columns[2]] / expansion_factor

    # Generate count_mtx: Gene Counts per Cell
    # We assume that all columns that are not in needles_columns or coords_columns are gene expression
    if needles_columns is None:
        needles_columns = []
    gene_columns = [col for col in data_mtx.columns if col not in needles_columns + coords_columns + [cell_column]]
    count_mtx = data_mtx[[cell_column] + gene_columns].set_index(cell_column)

    # Generate coord_mtx: Cell Centroids
    coord_mtx = data_mtx[[cell_column] + coords_columns].set_index(cell_column)

    # Save DataFrames to CSV files
    count_mtx.to_csv(os.path.join(data_path, "gene_exp", f"{tissue_name}_count.csv"))
    coord_mtx.to_csv(os.path.join(data_path, "coord", f"{tissue_name}_coord.csv"))


if __name__ == "__main__":
    # Example usage
    tissue_name = "Xenium_breast_summary_S1R2"
    data_path = r"C:\mnt\data\SpaGFT-Research\data"
    suffix = ""
    cell_column = 'Var1'
    coords_columns = ['X_space', 'Y_space']
    
    # Load and process full table data
    needles_columns = ['tissue', 'cell_type']  # Example needle columns, adjust as needed
    load_full_table_data(tissue_name, data_path, suffix=suffix, cell_column=cell_column, coords_columns=coords_columns, needles_columns=needles_columns)