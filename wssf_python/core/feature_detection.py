import numpy as np
import cv2
from utils.image_utils import normalize_to_range

def detect_kaze_features(image, max_points=5000):
    """
    Detect KAZE features in an image
    
    Parameters:
    -----------
    image : ndarray
        Input image
    max_points : int
        Maximum number of points to detect
        
    Returns:
    --------
    keypoints : list
        List of keypoints
    """
    # Create KAZE detector
    kaze = cv2.KAZE_create()
    
    # Detect keypoints
    keypoints = kaze.detect(normalize_to_range(image, 0, 255).astype(np.uint8), None)
    
    # Sort keypoints by response
    keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)
    
    # Limit number of keypoints
    if len(keypoints) > max_points:
        keypoints = keypoints[:max_points]
    
    return keypoints

def feature_detection(blob_space, corner_space, n_octaves, points_layer1, points_layer2, sigma_1, ratio):
    """
    Detect features in blob and corner spaces
    
    Parameters:
    -----------
    blob_space : list
        Blob space images
    corner_space : list
        Corner space images
    n_octaves : int
        Number of octaves
    points_layer1 : int
        Maximum number of points for blob space
    points_layer2 : int
        Maximum number of points for corner space
    sigma_1 : float
        Initial sigma value
    ratio : float
        Ratio between scales
        
    Returns:
    --------
    blob_key_point_array : ndarray
        Array of blob keypoints
    corner_key_point_array : ndarray
        Array of corner keypoints
    """
    # Initialize arrays
    blob_key_point_array = np.empty((0, 4))
    corner_key_point_array = np.empty((0, 4))
    
    # Process each layer
    for j in range(n_octaves):
        # Normalize images
        blob_img = normalize_to_range(blob_space[j], 0, 1)
        corner_img = normalize_to_range(corner_space[j], 0, 1)
        
        # Detect keypoints
        blob_keypoints = detect_kaze_features(blob_img, points_layer1)
        corner_keypoints = detect_kaze_features(corner_img, points_layer2)
        
        # Convert keypoints to arrays
        if blob_keypoints:
            blob_points = np.array([[kp.pt[0], kp.pt[1], kp.size, kp.response] for kp in blob_keypoints])
            
            # Scale coordinates based on layer
            if j > 0:
                scale_factor = ratio ** j
                blob_points[:, 0] *= scale_factor
                blob_points[:, 1] *= scale_factor
                blob_points[:, 2] *= scale_factor
            
            # Normalize response values
            if len(blob_points) > 0:
                max_response = np.max(blob_points[:, 3])
                if max_response > 0:
                    blob_points[:, 3] /= max_response
            
            blob_key_point_array = np.vstack((blob_key_point_array, blob_points))
        
        if corner_keypoints:
            corner_points = np.array([[kp.pt[0], kp.pt[1], kp.size, kp.response] for kp in corner_keypoints])
            
            # Scale coordinates based on layer
            if j > 0:
                scale_factor = ratio ** j
                corner_points[:, 0] *= scale_factor
                corner_points[:, 1] *= scale_factor
                corner_points[:, 2] *= scale_factor
            
            # Normalize response values
            if len(corner_points) > 0:
                max_response = np.max(corner_points[:, 3])
                if max_response > 0:
                    corner_points[:, 3] /= max_response
            
            corner_key_point_array = np.vstack((corner_key_point_array, corner_points))
    
    return blob_key_point_array, corner_key_point_array