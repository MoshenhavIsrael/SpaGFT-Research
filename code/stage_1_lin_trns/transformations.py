import numpy as np
import pandas as pd
import scanpy as sc
from typing import Union, List

def apply_rotation_matrix(coords: np.ndarray, angle_degrees: float) -> np.ndarray:
    """
    Rotates 2D or 3D coordinates around the Z-axis (in the XY plane).
    Preserves Z coordinates if they exist.
    """
    # Convert to radians
    theta = np.radians(angle_degrees)
    
    # Rotation Matrix for XY plane
    # R = [[cos, -sin], [sin, cos]]
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array(((c, -s), (s, c)))
    
    # Separate XY from Z (if Z exists)
    xy_coords = coords[:, :2]
    
    # Apply rotation: v' = R . v^T
    rotated_xy = np.dot(xy_coords, rotation_matrix.T)
    
    # Re-assemble
    if coords.shape[1] > 2:
        z_coords = coords[:, 2:]
        return np.hstack([rotated_xy, z_coords])
    
    return rotated_xy

def transform_adata(adata: sc.AnnData, 
                    angle: float = 0, 
                    spatial_key: Union[str, List[str]] = "spatial") -> sc.AnnData:
    """
    Creates a new AnnData object with rotated spatial coordinates.
    
    Args:
        adata: Original AnnData object.
        angle: Rotation angle in degrees.
        spatial_key: Key in .obsm (e.g., 'spatial') OR list of columns in .obs (e.g., ['x', 'y']).
    
    Returns:
        Rotated AnnData copy.
    """
    adata_rotated = adata.copy()
    
    coords = None
    is_obsm = False
    
    # 1. Fetch Coordinates based on spatial_key type
    if isinstance(spatial_key, str):
        if spatial_key in adata_rotated.obsm:
            coords = adata_rotated.obsm[spatial_key].copy()
            is_obsm = True
        else:
            raise KeyError(f"Key '{spatial_key}' not found in adata.obsm")
            
    elif isinstance(spatial_key, list):
        # Assume these are columns in .obs
        try:
            coords = adata_rotated.obs[spatial_key].values.copy()
        except KeyError:
            raise KeyError(f"Columns {spatial_key} not found in adata.obs")
    
    # 2. Apply Rotation
    if coords is not None:
        rotated_coords = apply_rotation_matrix(coords, angle)
        
        # 3. Save back to AnnData
        if is_obsm:
            adata_rotated.obsm[spatial_key] = rotated_coords
            # Optional: If SpaGFT relies on specific obs columns, update them too based on the new obsm
            # This is a safety fallback for SpaGFT's specific utils
            if 'array_row' in adata_rotated.obs and 'array_col' in adata_rotated.obs:
                adata_rotated.obs['array_row'] = rotated_coords[:, 0]
                adata_rotated.obs['array_col'] = rotated_coords[:, 1]
        else:
            # Update specific obs columns
            for i, col_name in enumerate(spatial_key):
                if i < rotated_coords.shape[1]:
                    adata_rotated.obs[col_name] = rotated_coords[:, i]

    return adata_rotated