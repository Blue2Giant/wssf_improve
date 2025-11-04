import numpy as np
import cv2
from skimage import exposure

def normalize_to_range(image, min_val=0, max_val=1):
    """Normalize image to specified range"""
    if image.max() == image.min():
        return np.zeros_like(image)
    return min_val + (max_val - min_val) * (image - image.min()) / (image.max() - image.min())

def preprocess_image(image):
    """Preprocess image for WSSF algorithm"""
    # Convert to grayscale if needed
    if len(image.shape) > 2 and image.shape[2] > 1:
        if image.shape[2] > 3:
            image = image[:, :, :3]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply adaptive histogram equalization
    gray_eq = exposure.equalize_adapthist(gray)
    
    # Convert to float
    return gray_eq.astype(np.float64)

def steerable_gaussians2(image, sigma_x, sigma_y):
    """Simplified implementation of steerable gaussians filter"""
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)
    return blurred / 6  # Matching the MATLAB division by 6