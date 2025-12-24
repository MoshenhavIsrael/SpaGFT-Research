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

