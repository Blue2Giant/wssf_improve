import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.gradient_feature import wssf_gradient_feature
from core.feature_detection import feature_detection
from core.nms import wssf_selectmax_nms

def kpts_orientation(keypoints, gradient_cell, angle_cell, nonlinear_space, sigma_1, ratio):
    """
    Calculate orientation for keypoints
    
    Parameters:
    -----------
    keypoints : ndarray
        Array of keypoints [x, y, scale, response]
    gradient_cell : list
        Gradient features
    angle_cell : list
        Angle features
    nonlinear_space : list
        Nonlinear space
    sigma_1 : float
        Initial sigma value
    ratio : float
        Ratio between scales
        
    Returns:
    --------
    keypoints_with_orientation : ndarray
        Keypoints with orientation
    """
    num_points = keypoints.shape[0]
    keypoints_with_orientation = np.zeros((num_points, 6))
    keypoints_with_orientation[:, :4] = keypoints
    
    # Process each keypoint
    for i in range(num_points):
        x, y, scale, _ = keypoints[i]
        
        # Determine which layer to use based on scale
        layer = int(np.round(np.log(scale/sigma_1) / np.log(ratio)))
        layer = max(0, min(len(gradient_cell)-1, layer))
        
        # Get coordinates in the layer
        x_layer = int(np.round(x / (ratio**layer)))
        y_layer = int(np.round(y / (ratio**layer)))
        
        # Ensure coordinates are within image bounds
        h, w = gradient_cell[layer].shape
        x_layer = max(0, min(w-1, x_layer))
        y_layer = max(0, min(h-1, y_layer))
        
        # Get orientation
        orientation = angle_cell[layer][y_layer, x_layer]
        
        # Store orientation and scale
        keypoints_with_orientation[i, 4] = orientation
        keypoints_with_orientation[i, 5] = scale
    
    return keypoints_with_orientation

def wssf_features(nonlinear_space, e_space, max_space, min_space, phase_space, sigma_1, ratio, scale_invariance, n_octaves):
    """
    Extract WSSF features
    
    Parameters:
    -----------
    nonlinear_space : list
        Nonlinear space
    e_space : list
        Edge space
    max_space : list
        Maximum space
    min_space : list
        Minimum space
    phase_space : list
        Phase space
    sigma_1 : float
        Initial sigma value
    ratio : float
        Ratio between scales
    scale_invariance : str
        'YES' or 'NO' for scale invariance
    n_octaves : int
        Number of octaves
        
    Returns:
    --------
    position_1 : ndarray
        Blob keypoints with orientation
    position_2 : ndarray
        Corner keypoints with orientation
    blob_gradient_cell : list
        Blob gradient features
    corner_gradient_cell : list
        Corner gradient features
    blob_angle_cell : list
        Blob angle features
    corner_angle_cell : list
        Corner angle features
    """
    # Calculate gradient features
    blob_space, corner_space, blob_gradient_cell, corner_gradient_cell, blob_angle_cell, corner_angle_cell = wssf_gradient_feature(
        nonlinear_space, e_space, max_space, min_space, phase_space, scale_invariance, n_octaves
    )
    
    # Detect features
    points_layer1 = 5000
    points_layer2 = 5000
    blob_key_point_array, corner_key_point_array = feature_detection(
        blob_space, corner_space, n_octaves, points_layer1, points_layer2, sigma_1, ratio
    )
    
    # Filter unique points for blob keypoints
    if blob_key_point_array.size > 0:
        _, unique_indices = np.unique(blob_key_point_array[:, :2], axis=0, return_index=True)
        blob_key_point_array = blob_key_point_array[np.sort(unique_indices)]
    
    # Filter unique points for corner keypoints
    if corner_key_point_array.size > 0:
        _, unique_indices = np.unique(corner_key_point_array[:, :2], axis=0, return_index=True)
        corner_key_point_array = corner_key_point_array[np.sort(unique_indices)]
    
    # Apply NMS
    window = 5
    if blob_key_point_array.size > 0:
        keypoints, _ = wssf_selectmax_nms(blob_key_point_array, window)
        blob_key_point_array = keypoints['kpts'][:, :4]
    
    if corner_key_point_array.size > 0:
        keypoints, _ = wssf_selectmax_nms(corner_key_point_array, window)
        corner_key_point_array = keypoints['kpts'][:, :4]
    
    # Calculate orientation
    position_1 = kpts_orientation(blob_key_point_array, blob_gradient_cell, blob_angle_cell, nonlinear_space, sigma_1, ratio)
    position_2 = kpts_orientation(corner_key_point_array, corner_gradient_cell, corner_angle_cell, nonlinear_space, sigma_1, ratio)
    
    return position_1, position_2, blob_gradient_cell, corner_gradient_cell, blob_angle_cell, corner_angle_cell