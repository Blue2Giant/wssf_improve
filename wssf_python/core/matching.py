import numpy as np
from scipy.spatial.distance import cdist

def match_features(des1, des2, max_ratio=1, match_threshold=50, unique=True):
    """
    Match feature descriptors between two images.
    
    Parameters:
    -----------
    des1, des2 : numpy.ndarray
        Feature descriptors from two images
    max_ratio : float, optional
        Maximum ratio for nearest neighbor matching
    match_threshold : float, optional
        Threshold for matching distance
    unique : bool, optional
        Whether to ensure unique matches
        
    Returns:
    --------
    numpy.ndarray
        Indices of matched features (N x 2)
    numpy.ndarray
        Distances between matched features
    """
    # Calculate pairwise distances between all descriptors
    distances = cdist(des1, des2)
    
    # Find the best and second-best matches for each descriptor in des1
    idx1 = np.arange(des1.shape[0])
    idx2 = np.argmin(distances, axis=1)
    
    # Get the minimum distances
    min_distances = distances[idx1, idx2]
    
    # Create a mask for matches that meet the threshold
    valid_matches = min_distances < match_threshold
    
    # Apply the threshold
    idx1 = idx1[valid_matches]
    idx2 = idx2[valid_matches]
    min_distances = min_distances[valid_matches]
    
    # If unique matching is required
    if unique:
        # Find unique matches (each point in des2 matches at most one point in des1)
        unique_idx2, unique_indices = np.unique(idx2, return_index=True)
        idx1 = idx1[unique_indices]
        idx2 = unique_idx2
        min_distances = min_distances[unique_indices]
    
    # Create the index pairs and distances
    index_pairs = np.column_stack((idx1 + 1, idx2 + 1))  # +1 to match MATLAB 1-based indexing
    
    return index_pairs, min_distances

def back_projection(interior_points1, interior_points2, scale_value):
    """
    Project key points from feature space back to image space.
    
    Parameters:
    -----------
    interior_points1, interior_points2 : numpy.ndarray
        Key points in feature space
    scale_value : float
        Scale factor between layers
        
    Returns:
    --------
    numpy.ndarray, numpy.ndarray
        Key points projected to image space
    """
    key_nums1 = interior_points1.shape[0]
    key_nums2 = interior_points2.shape[0]
    
    image_points1 = np.zeros((key_nums1, 2))
    image_points2 = np.zeros((key_nums2, 2))
    
    # Project points from first set
    for i in range(key_nums1):
        x = interior_points1[i, 0]
        y = interior_points1[i, 1]
        layer = int(interior_points1[i, 2])
        
        if layer == 1:
            pass  # Keep coordinates as they are
        else:
            x = int(x * (scale_value ** (layer - 1)))
            y = int(y * (scale_value ** (layer - 1)))
        
        image_points1[i, :] = [x, y]
    
    # Project points from second set
    for j in range(key_nums2):
        xx = interior_points2[j, 0]
        yy = interior_points2[j, 1]
        layer = int(interior_points2[j, 2])
        
        if layer == 1:
            pass  # Keep coordinates as they are
        else:
            xx = int(xx * (scale_value ** (layer - 1)))
            yy = int(yy * (scale_value ** (layer - 1)))
        
        image_points2[j, :] = [xx, yy]
    
    return image_points1, image_points2

def lsm(match1, match2, change_form='perspective'):
    """
    Least Squares Method for estimating transformation parameters.
    
    Parameters:
    -----------
    match1, match2 : numpy.ndarray
        Matched points from two images
    change_form : str, optional
        Transformation type: 'affine', 'perspective', or 'similarity'
        
    Returns:
    --------
    numpy.ndarray
        Transformation parameters
    float
        Root Mean Square Error
    """
    match1_xy = match1[:, :2]
    match2_xy = match2[:, :2]
    
    # Build the A matrix for the linear system Ax = b
    n = match1_xy.shape[0]
    A = np.zeros((2*n, 6))
    
    for i in range(n):
        A[2*i, 0] = match1_xy[i, 0]
        A[2*i, 1] = match1_xy[i, 1]
        A[2*i, 4] = 1
        A[2*i+1, 2] = match1_xy[i, 0]
        A[2*i+1, 3] = match1_xy[i, 1]
        A[2*i+1, 5] = 1
    
    # Build the b vector
    b = match2_xy.flatten()
    
    if change_form == 'affine':
        # Solve for affine parameters
        parameters = np.zeros(8)
        parameters[:6], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        # Calculate RMSE
        M = np.array([[parameters[0], parameters[1]], 
                      [parameters[2], parameters[3]]])
        t = np.array([parameters[4], parameters[5]])
        
        match1_test_trans = (M @ match1_xy.T).T + t
        rmse = np.sqrt(np.sum((match1_test_trans - match2_xy)**2) / n)
        
    elif change_form == 'perspective':
        # For perspective transformation, we need to add more columns to A
        A_perspective = np.zeros((2*n, 8))
        A_perspective[:, :6] = A
        
        for i in range(n):
            u = match2_xy[i, 0]
            v = match2_xy[i, 1]
            x = match1_xy[i, 0]
            y = match1_xy[i, 1]
            
            A_perspective[2*i, 6] = -u * x
            A_perspective[2*i, 7] = -u * y
            A_perspective[2*i+1, 6] = -v * x
            A_perspective[2*i+1, 7] = -v * y
        
        # Solve for perspective parameters
        parameters = np.zeros(8)
        parameters, _, _, _ = np.linalg.lstsq(A_perspective, b, rcond=None)
        
        # Calculate RMSE
        M = np.array([[parameters[0], parameters[1], parameters[4]],
                      [parameters[2], parameters[3], parameters[5]],
                      [parameters[6], parameters[7], 1]])
        
        match1_test = np.hstack((match1_xy, np.ones((n, 1)))).T
        match1_test_trans = M @ match1_test
        match1_test_trans = match1_test_trans[:2, :] / match1_test_trans[2, :]
        match1_test_trans = match1_test_trans.T
        
        rmse = np.sqrt(np.sum((match1_test_trans - match2_xy)**2) / n)
        
    elif change_form == 'similarity':
        # For similarity transformation - with normalization for numerical stability
        # Center the points
        mean1 = np.mean(match1_xy, axis=0)
        mean2 = np.mean(match2_xy, axis=0)
        match1_c = match1_xy - mean1
        match2_c = match2_xy - mean2
        
        # Compute average distance from origin
        dist1 = np.mean(np.sqrt(np.sum(match1_c**2, axis=1)))
        dist2 = np.mean(np.sqrt(np.sum(match2_c**2, axis=1)))
        dist1 = dist1 if dist1 > 0 else 1.0
        dist2 = dist2 if dist2 > 0 else 1.0
        
        # Scaling factors
        scale1 = np.sqrt(2) / dist1
        scale2 = np.sqrt(2) / dist2
        
        # Normalized points
        match1_n = match1_c * scale1
        match2_n = match2_c * scale2
        
        # Build A for normalized points
        A_similarity = np.zeros((2*n, 4))
        for i in range(n):
            x = match1_n[i, 0]
            y = match1_n[i, 1]
            A_similarity[2*i, :] = [x, y, 1, 0]
            A_similarity[2*i+1, :] = [y, -x, 0, 1]
        
        # b for normalized
        b_n = match2_n.flatten()
        
        # Solve using QR
        Q, R = np.linalg.qr(A_similarity)
        sim_params_n = np.linalg.solve(R, Q.T @ b_n)
        
        # Construct normalized matrix
        a_n = sim_params_n[0]
        b_n = sim_params_n[1]
        tx_n = sim_params_n[2]
        ty_n = sim_params_n[3]
        
        M_n = np.array([[a_n, b_n, tx_n],
                        [-b_n, a_n, ty_n],
                        [0, 0, 1]])
        
        # Denormalization matrices
        T_x = np.array([[scale1, 0, -scale1 * mean1[0]],
                        [0, scale1, -scale1 * mean1[1]],
                        [0, 0, 1]])
        T_y_inv = np.array([[1/scale2, 0, mean2[0]],
                            [0, 1/scale2, mean2[1]],
                            [0, 0, 1]])
        
        # Full matrix
        solution = T_y_inv @ M_n @ T_x
        
        # Extract parameters
        parameters = np.zeros(8)
        parameters[0] = solution[0,0]
        parameters[1] = solution[0,1]
        parameters[2] = solution[1,0]
        parameters[3] = solution[1,1]
        parameters[4] = solution[0,2]
        parameters[5] = solution[1,2]
        parameters[6] = solution[2,0]
        parameters[7] = solution[2,1]
        
        # Calculate RMSE with full parameters
        M = solution[:2, :2]
        t = solution[:2, 2]
        match1_test_trans = (M @ match1_xy.T).T + t
        rmse = np.sqrt(np.sum((match1_test_trans - match2_xy)**2) / n)
    
    return parameters, rmse

def fsc(cor1, cor2, change_form='perspective', error_t=3, angle_prior=None, angle_sigma=None, scale_prior=None, scale_sigma=None):
    """
    Fast Sample Consensus algorithm for robust feature matching.
    
    Parameters:
    -----------
    cor1, cor2 : numpy.ndarray
        Corresponding points from two images
    change_form : str, optional
        Transformation type: 'affine', 'perspective', or 'similarity'
    error_t : float, optional
        Error threshold for inlier detection
    angle_prior : float, optional
        Prior angle (degrees) for similarity transformation bias
    angle_sigma : float, optional
        Standard deviation (degrees) for angle prior
    scale_prior : float, optional
        Prior scale for similarity transformation bias
    scale_sigma : float, optional
        Standard deviation for scale prior
        
    Returns:
    --------
    numpy.ndarray
        Transformation matrix
    float
        Root Mean Square Error
    numpy.ndarray
        Filtered points from first set
    numpy.ndarray
        Filtered points from second set
    """
    M, N = cor1.shape
    
    # Determine number of points needed for model and max iterations
    if change_form == 'similarity':
        n = 2
        max_iteration = int(M * (M - 1) / 2)
    elif change_form == 'affine':
        n = 3
        max_iteration = int(M * (M - 1) * (M - 2) / (2 * 3))
    elif change_form == 'perspective':
        n = 4
        max_iteration = int(M * (M - 1) * (M - 2) / (2 * 3))
    
    # Limit iterations for large point sets
    iterations = min(max_iteration, 10000)
    
    most_consensus_number = 0
    cor1_new = np.zeros((M, N))
    cor2_new = np.zeros((M, N))
    
    # RANSAC loop
    for i in range(iterations):
        # Randomly select points
        while True:
            a = np.random.choice(M, n, replace=False)
            cor11 = cor1[a, :2]
            cor22 = cor2[a, :2]
            
            # Check if points are distinct
            if n == 2:
                if not np.array_equal(cor11[0], cor11[1]) and not np.array_equal(cor22[0], cor22[1]):
                    break
            elif n == 3:
                if (not np.array_equal(cor11[0], cor11[1]) and 
                    not np.array_equal(cor11[0], cor11[2]) and 
                    not np.array_equal(cor11[1], cor11[2]) and
                    not np.array_equal(cor22[0], cor22[1]) and 
                    not np.array_equal(cor22[0], cor22[2]) and 
                    not np.array_equal(cor22[1], cor22[2])):
                    break
            elif n == 4:
                if (len(set(tuple(p) for p in cor11)) == 4 and 
                    len(set(tuple(p) for p in cor22)) == 4):
                    break
        
        # Estimate transformation parameters
        # 修正：确保变换矩阵是从image1到image2的逆时针旋转90度
        parameters, _ = lsm(cor11, cor22, change_form)
        
        if change_form == 'perspective':
            solution = np.array([
                [parameters[0], parameters[1], parameters[4]],
                [parameters[2], parameters[3], parameters[5]],
                [parameters[6], parameters[7], 1]
            ])
        else:
            solution = np.array([
                [parameters[0], parameters[1], parameters[4]],
                [parameters[2], parameters[3], parameters[5]],
                [0, 0, 1]
            ])
        
        # Apply transformation to all points
        match1_xy = cor1[:, :2].T
        match1_xy = np.vstack((match1_xy, np.ones(M)))
        
        if change_form == 'perspective':
            # For perspective transformation
            match1_test_trans = solution @ match1_xy
            match1_test_trans = match1_test_trans[:2, :] / match1_test_trans[2, :]
            match1_test_trans = match1_test_trans.T
            
            match2_xy = cor2[:, :2]
            test = match1_test_trans - match2_xy
            diff_match2_xy = np.sqrt(np.sum(test**2, axis=1))
            index_in = np.where(diff_match2_xy < error_t)[0]
            consensus_num = len(index_in)
        else:
            # For affine and similarity transformations
            t_match1_xy = solution @ match1_xy
            match2_xy = cor2[:, :2].T
            match2_xy = np.vstack((match2_xy, np.ones(M)))
            
            diff_match2_xy = np.sqrt(np.sum((t_match1_xy - match2_xy)**2, axis=0))
            index_in = np.where(diff_match2_xy < error_t)[0]
            consensus_num = len(index_in)
        
        # Update best consensus set
        if consensus_num > most_consensus_number:
            most_consensus_number = consensus_num
            cor1_new = cor1[index_in, :]
            cor2_new = cor2[index_in, :]
    
    # Remove duplicate point pairs
    if len(cor1_new) > 0:
        # Remove duplicates in first set
        _, unique_indices = np.unique(cor1_new[:, :2], axis=0, return_index=True)
        cor1_new = cor1_new[np.sort(unique_indices)]
        cor2_new = cor2_new[np.sort(unique_indices)]
        
        # Remove duplicates in second set
        _, unique_indices = np.unique(cor2_new[:, :2], axis=0, return_index=True)
        cor1_new = cor1_new[np.sort(unique_indices)]
        cor2_new = cor2_new[np.sort(unique_indices)]
    
    # Final parameter estimation
    # 修正：确保变换矩阵是从image1到image2的逆时针旋转90度
    parameters, rmse = lsm(cor1_new[:, :2], cor2_new[:, :2], change_form)
    
    if change_form == 'perspective':
        solution = np.array([
            [parameters[0], parameters[1], parameters[4]],
            [parameters[2], parameters[3], parameters[5]],
            [parameters[6], parameters[7], 1]
        ])
    else:
        solution = np.array([
            [parameters[0], parameters[1], parameters[4]],
            [parameters[2], parameters[3], parameters[5]],
            [0, 0, 1]
        ])
    
    return solution, rmse, cor1_new, cor2_new