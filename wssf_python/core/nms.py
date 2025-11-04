import numpy as np

def wssf_selectmax_nms(keypoints, window):
    """
    Non-Maximum Suppression for keypoints
    
    Parameters:
    -----------
    keypoints : ndarray
        Array of keypoints [x, y, scale, response]
    window : int
        Window size for NMS
        
    Returns:
    --------
    filtered_keypoints : ndarray
        Filtered keypoints after NMS
    rejected_keypoints : ndarray
        Rejected keypoints
    """
    r = window / 2
    num_points = keypoints.shape[0]
    
    # Add a column for marking (1 = keep, 0 = discard)
    key_points = np.ones((num_points, 5))
    key_points[:, :4] = keypoints
    
    if window != 0:
        # Compare each point with all other points
        for i in range(num_points):
            for j in range(i+1, num_points):
                # Calculate squared distance between points
                distance = (abs(key_points[i, 0] - key_points[j, 0]))**2 + (abs(key_points[i, 1] - key_points[j, 1]))**2
                
                # If points are within window, keep only the one with higher response
                if distance <= r**2:
                    if key_points[i, 3] < key_points[j, 3]:
                        key_points[i, 4] = 0
                    else:
                        key_points[j, 4] = 0
    
    # Separate kept and discarded keypoints
    kept_keypoints = key_points[key_points[:, 4] == 1]
    discarded_keypoints = key_points[key_points[:, 4] == 0]
    
    return {
        'kpts': kept_keypoints
    }, {
        'kpts': discarded_keypoints
    }