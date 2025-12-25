import numpy as np
from scipy import ndimage
import math

def gloh_descriptors(gradient, angle, key_point_array, path_block, ratio=None, sigma_1=None):
    """
    Compute GLOH descriptors for key points using gradient and angle information.
    
    Parameters:
    -----------
    gradient : list of numpy.ndarray
        List of gradient images for different layers
    angle : list of numpy.ndarray
        List of angle images for different layers
    key_point_array : numpy.ndarray
        Array of key points. Expected columns: [x, y, scale, response, orientation, scale]
    path_block : int
        Block size for descriptor calculation
    ratio : float, optional
        Not used in this implementation
    sigma_1 : float, optional
        Not used in this implementation
        
    Returns:
    --------
    dict
        Dictionary with 'locs' (key point locations) and 'des' (descriptors)
    """
    LOG_POLAR_WIDTH = 16
    LOG_POLAR_HIST_BINS = 12
    circle_count = 2
    
    M = key_point_array.shape[0]
    d = LOG_POLAR_WIDTH
    n = LOG_POLAR_HIST_BINS
    descriptors = np.zeros((M, (circle_count * d + 1) * n))
    locs = key_point_array.copy()
    
    for i in range(M):
        x = key_point_array[i, 0]
        y = key_point_array[i, 1]
        # Determine layer from scale when ratio and sigma_1 are provided; fallback to layer column if present
        layer = 0
        if ratio is not None and sigma_1 is not None:
            try:
                scale = float(key_point_array[i, 2])
                layer = int(np.round(np.log(scale/sigma_1) / np.log(ratio)))
                layer = max(0, min(len(gradient)-1, layer))
            except Exception:
                layer = int(key_point_array[i, 2]) if key_point_array.shape[1] > 2 else 0
        else:
            layer = int(key_point_array[i, 2]) if key_point_array.shape[1] > 2 else 0

        # Orientation is expected at column 4 if available; otherwise default to 0
        main_angle = float(key_point_array[i, 4]) if key_point_array.shape[1] > 4 else 0.0

        current_gradient = gradient[layer]
        current_angle = angle[layer]
        descriptors[i, :] = calc_log_polar_descriptor(
            current_gradient, current_angle, x, y, main_angle, 
            d, n, path_block, circle_count
        )
    
    return {'locs': locs, 'des': descriptors}

def calc_log_polar_descriptor(gradient, angle, x, y, main_angle, d, n, path_block, circle_count):
    """
    Calculate log-polar descriptor for a key point.
    
    Parameters:
    -----------
    gradient : numpy.ndarray
        Gradient image
    angle : numpy.ndarray
        Angle image
    x, y : float
        Key point coordinates
    main_angle : float
        Main orientation angle of the key point
    d : int
        Log-polar width
    n : int
        Number of histogram bins
    path_block : int
        Block size for descriptor calculation
    circle_count : int
        Number of circles in the log-polar grid
        
    Returns:
    --------
    numpy.ndarray
        Descriptor vector
    """
    cos_t = np.cos(-main_angle / 180 * np.pi)
    sin_t = np.sin(-main_angle / 180 * np.pi)
    # 强制使用整数坐标，避免边界切片为空
    x = int(np.round(x))
    y = int(np.round(y))
    
    M, N = gradient.shape
    radius = round(path_block)
    
    # Calculate region boundaries
    radius_x_left = max(int(x - radius), 0)
    radius_x_right = min(int(x + radius), N-1)
    radius_y_up = max(int(y - radius), 0)
    radius_y_down = min(int(y + radius), M-1)
    
    # Extract sub-regions
    sub_gradient = gradient[radius_y_up:radius_y_down+1, radius_x_left:radius_x_right+1]
    sub_angle = angle[radius_y_up:radius_y_down+1, radius_x_left:radius_x_right+1]
    if sub_gradient.size == 0 or sub_angle.size == 0:
        return np.zeros((circle_count * d + 1) * n, dtype=np.float64)
    
    # Adjust angles relative to main orientation
    sub_angle = np.round((sub_angle - main_angle) * n / 360)
    sub_angle[sub_angle <= 0] += n
    sub_angle[sub_angle == 0] = n
    
    # Create coordinate grid
    X = np.arange(-(x - radius_x_left), (radius_x_right - x) + 1, dtype=np.int64)
    Y = np.arange(-(y - radius_y_up), (radius_y_down - y) + 1, dtype=np.int64)
    if X.size == 0 or Y.size == 0:
        return np.zeros((circle_count * d + 1) * n, dtype=np.float64)
    XX, YY = np.meshgrid(X, Y)
    
    # Rotate coordinates
    c_rot = XX * cos_t - YY * sin_t
    r_rot = XX * sin_t + YY * cos_t
    
    # Calculate log-polar coordinates
    log_angle = np.arctan2(r_rot, c_rot)
    log_angle = log_angle / np.pi * 180
    log_angle[log_angle < 0] += 360
    log_amplitude = np.log2(np.sqrt(c_rot**2 + r_rot**2) + 1e-10)  # Add small value to avoid log(0)
    
    # Quantize angles
    log_angle = np.round(log_angle * d / 360)
    log_angle[log_angle <= 0] += d
    log_angle[log_angle > d] -= d
    
    # Quantize amplitudes
    if log_amplitude.size == 0:
        return np.zeros((circle_count * d + 1) * n, dtype=np.float64)
    radius = np.max(log_amplitude)
    if circle_count == 2:
        r1 = radius * 0.25 * 0.73
        r2 = radius * 0.73
        
        log_amplitude_bins = np.ones_like(log_amplitude)
        log_amplitude_bins[(log_amplitude > r1) & (log_amplitude <= r2)] = 2
        log_amplitude_bins[log_amplitude > r2] = 3
    else:
        r1 = np.log2(radius * 0.3006)
        r2 = np.log2(radius * 0.7071)
        r3 = np.log2(radius * 0.866)
        
        log_amplitude_bins = np.ones_like(log_amplitude)
        log_amplitude_bins[(log_amplitude > r1) & (log_amplitude <= r2)] = 2
        log_amplitude_bins[(log_amplitude > r2) & (log_amplitude <= r3)] = 3
        log_amplitude_bins[log_amplitude > r3] = 4
    
    # Initialize histogram
    temp_hist = np.zeros((circle_count * d + 1) * n)
    
    # Fill histogram
    row, col = log_angle.shape
    for i in range(row):
        for j in range(col):
            angle_bin = int(log_angle[i, j])
            amplitude_bin = int(log_amplitude_bins[i, j])
            bin_vertical = int(sub_angle[i, j])
            mag = sub_gradient[i, j]
            
            if amplitude_bin == 1:
                temp_hist[bin_vertical-1] += mag
            else:
                idx = ((amplitude_bin - 2) * d + angle_bin - 1) * n + bin_vertical - 1 + n
                if 0 <= idx < len(temp_hist):  # Ensure index is within bounds
                    temp_hist[idx] += mag
    
    # Normalize histogram
    norm = np.sqrt(np.sum(temp_hist**2))
    if norm > 0:
        temp_hist = temp_hist / norm
    
    # Threshold and renormalize
    temp_hist[temp_hist > 0.2] = 0.2
    norm = np.sqrt(np.sum(temp_hist**2))
    if norm > 0:
        temp_hist = temp_hist / norm
    
    return temp_hist
