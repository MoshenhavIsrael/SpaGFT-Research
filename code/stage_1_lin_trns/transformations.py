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

def apply_scaling(coords: np.ndarray, scaling_factor: float) -> np.ndarray:
    """Applies uniform scaling to coordinates."""
    return coords * scaling_factor

def apply_translation(coords: np.ndarray, translation: Union[float, List[float], np.ndarray]) -> np.ndarray:
    """Applies linear translation (scalar or vector)."""
    # If scalar, numpy broadcasting handles it (adds to all axes)
    if isinstance(translation, (int, float)):
        return coords + translation
    
    # If vector, ensure numpy array
    t_vec = np.array(translation)
    
    # Validation: Allow 2D translation on 3D data (assume Z shift is 0), otherwise require strict dimension match
    if t_vec.ndim == 1 and len(t_vec) != coords.shape[1]:
        if len(t_vec) == 2 and coords.shape[1] == 3:
             t_vec = np.append(t_vec, 0) # Pad Z with 0
        else:
             raise ValueError(f"Translation dimension {len(t_vec)} does not match coordinate dimension {coords.shape[1]}")
             
    return coords + t_vec

def transform_adata(adata: sc.AnnData, 
                    angle: float = 0, 
                    translation: Union[float, List[float]] = 0,
                    scaling: float = 1.0,                      
                    flip: bool = False,                        
                    spatial_key: Union[str, List[str]] = "spatial") -> sc.AnnData:
    """
    Creates a new AnnData object with transformed spatial coordinates.
    
    Args:
        adata: Original AnnData object.
        angle: Rotation angle in degrees.
        translation: Translation factor. Scalar (all axes) or vector (per axis). 
        scaling: Uniform scaling factor. 
        flip: If True, flips the X-axis (reflection). 
        spatial_key: Key in .obsm (e.g., 'spatial') OR list of columns in .obs (e.g., ['x', 'y']).
    
    Returns:
        Transformed AnnData copy.
    """    
    
    adata_transformed = adata.copy()
    
    coords = None
    is_obsm = False
    
    # 1. Fetch Coordinates based on spatial_key type
    if isinstance(spatial_key, str):
        if spatial_key in adata_transformed.obsm:
            coords = adata_transformed.obsm[spatial_key].copy()
            is_obsm = True
        else:
            raise KeyError(f"Key '{spatial_key}' not found in adata.obsm")
            
    elif isinstance(spatial_key, list):
        # Assume these are columns in .obs
        try:
            coords = adata_transformed.obs[spatial_key].values.copy()
        except KeyError:
            raise KeyError(f"Columns {spatial_key} not found in adata.obs")
    
    # 2. Apply Transformations 
    if coords is not None:
        # A. Flip (X-axis reflection) 
        if flip:
            coords[:, 0] = -coords[:, 0]

        # B. Rotation
        if angle != 0:
            coords = apply_rotation_matrix(coords, angle)
        
        # C. Scaling
        if scaling != 1.0:
            coords = apply_scaling(coords, scaling)
            
        # D. Translation
        if np.any(np.array(translation) != 0):
            coords = apply_translation(coords, translation)
        
        # 3. Save back to AnnData
        if is_obsm:
            adata_transformed.obsm[spatial_key] = coords 
            # Optional: If SpaGFT relies on specific obs columns, update them too based on the new obsm
            if 'array_row' in adata_transformed.obs and 'array_col' in adata_transformed.obs:
                adata_transformed.obs['array_row'] = coords[:, 0]
                adata_transformed.obs['array_col'] = coords[:, 1]
                if coords.shape[1] > 2 and 'array_z' in adata_transformed.obs:
                    adata_transformed.obs['array_z'] = coords[:, 2]
        else:
            # Update specific obs columns
            for i, col_name in enumerate(spatial_key):
                if i < coords.shape[1]:
                    adata_transformed.obs[col_name] = coords[:, i]

    return adata_transformed