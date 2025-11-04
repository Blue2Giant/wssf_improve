import numpy as np
import cv2
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.image_utils import normalize_to_range, steerable_gaussians2

def wssf_gradient_feature(scale_space, e_scale_space, max_scale_space, min_scale_space, phase_scale_space, scale_invariance, n_octaves):
    """
    Calculate gradient features for WSSF algorithm
    
    Parameters:
    -----------
    scale_space : list
        Scale space images
    e_scale_space : list
        Edge space images
    max_scale_space : list
        Maximum space images
    min_scale_space : list
        Minimum space images
    phase_scale_space : list
        Phase space images
    scale_invariance : str
        'YES' or 'NO' for scale invariance
    n_octaves : int
        Number of octaves
        
    Returns:
    --------
    blob_space : list
        Blob space
    corner_space : list
        Corner space
    blob_gradient_cell : list
        Blob gradient features
    corner_gradient_cell : list
        Corner gradient features
    blob_angle_cell : list
        Blob angle features
    corner_angle_cell : list
        Corner angle features
    """
    # Determine number of layers
    if scale_invariance == 'YES':
        layers = n_octaves
    else:
        layers = 1
    
    # Get image dimensions from first layer
    if isinstance(scale_space[0], np.ndarray):
        m, n = scale_space[0].shape
    else:
        m, n = scale_space[0].shape[:2]
    
    # Initialize output cells
    blob_gradient_cell = []
    corner_gradient_cell = []
    blob_angle_cell = []
    corner_angle_cell = []
    blob_space = []
    corner_space = []
    
    # Sobel operators for gradient calculation
    h1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    h2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    # Process each layer
    for j in range(layers):
        # Convert to grayscale if needed
        if len(np.array(scale_space[j]).shape) > 2:
            current_scale = cv2.cvtColor(scale_space[j], cv2.COLOR_BGR2GRAY)
        else:
            current_scale = scale_space[j]
        
        # Calculate blob space
        blob_space_j = 1.0 * max_scale_space[j] + 2.0 * e_scale_space[j] + 0.0 * current_scale
        blob_space_j = steerable_gaussians2(blob_space_j, 5, 5)
        blob_space.append(max_scale_space[j])
        
        # Normalize current scale
        current_scale = normalize_to_range(current_scale, 0, 1)
        
        # Calculate corner space
        corner_space_j = 1.0 * min_scale_space[j] + 2.0 * e_scale_space[j] + 0.0 * current_scale
        corner_space_j = steerable_gaussians2(corner_space_j, 5, 5)
        corner_space.append(min_scale_space[j])
        
        # Calculate blob gradients
        blob_dx = cv2.filter2D(blob_space_j, -1, h1)
        blob_dy = cv2.filter2D(blob_space_j, -1, h2)
        blob_gradient = np.sqrt(blob_dx**2 + blob_dy**2)
        blob_gradient = normalize_to_range(blob_gradient)
        blob_gradient_cell.append(blob_gradient)
        
        # Calculate blob angles
        blob_angle = np.arctan2(blob_dy, blob_dx) * 180 / np.pi
        blob_angle[blob_angle < 0] += 360
        blob_angle_cell.append(blob_angle.astype(np.float32))
        
        # Calculate corner gradients
        corner_dx = cv2.filter2D(corner_space_j, -1, h1)
        corner_dy = cv2.filter2D(corner_space_j, -1, h2)
        corner_gradient = np.sqrt(corner_dx**2 + corner_dy**2)
        corner_gradient = normalize_to_range(corner_gradient)
        corner_gradient_cell.append(corner_gradient)
        
        # Calculate corner angles
        corner_angle = np.arctan2(corner_dy, corner_dx) * 180 / np.pi
        corner_angle[corner_angle < 0] += 360
        corner_angle_cell.append(corner_angle.astype(np.float32))
    
    return blob_space, corner_space, blob_gradient_cell, corner_gradient_cell, blob_angle_cell, corner_angle_cell