import numpy as np
import cv2
from scipy import ndimage
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.image_utils import normalize_to_range, steerable_gaussians2

def create_image_space(image, n_octaves, scale_invariance, scale_value, ratio, sigma_1, filter_size):
    """
    Create image space for WSSF algorithm
    
    Parameters:
    -----------
    image : ndarray
        Input image (grayscale, float64)
    n_octaves : int
        Number of octaves
    scale_invariance : str
        'YES' or 'NO' for scale invariance
    scale_value : float
        Scale value
    ratio : float
        Ratio between scales
    sigma_1 : float
        Initial sigma value
    filter_size : int
        Filter size
        
    Returns:
    --------
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
    """
    # Initialize spaces
    nonlinear_space = []
    e_space = []
    max_space = []
    min_space = []
    phase_space = []
    
    # Determine number of layers
    if scale_invariance == 'YES':
        layers = n_octaves
    else:
        layers = 1
    
    # Create spaces for each layer
    for j in range(layers):
        # Scale the image if needed
        if j > 0:
            scaled_image = cv2.resize(image, None, fx=1/(ratio**j), fy=1/(ratio**j), 
                                     interpolation=cv2.INTER_LINEAR)
        else:
            scaled_image = image.copy()
        
        # Apply Gaussian blur
        sigma = sigma_1 * (ratio**j)
        blurred = cv2.GaussianBlur(scaled_image, (0, 0), sigma)
        
        # Calculate gradient using Sobel operators
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=filter_size)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=filter_size)
        
        # Calculate gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x) * 180 / np.pi
        
        # Normalize magnitude
        magnitude_norm = normalize_to_range(magnitude)
        
        # Calculate edge space (E)
        edge_space = magnitude_norm
        
        # Calculate maximum and minimum spaces
        max_filter = ndimage.maximum_filter(blurred, size=filter_size)
        min_filter = ndimage.minimum_filter(blurred, size=filter_size)
        
        # Calculate nonlinear space
        nonlinear = steerable_gaussians2(blurred, sigma, sigma)
        
        # Append to spaces
        nonlinear_space.append(nonlinear)
        e_space.append(edge_space)
        max_space.append(max_filter)
        min_space.append(min_filter)
        phase_space.append(direction)
    
    return nonlinear_space, e_space, max_space, min_space, phase_space