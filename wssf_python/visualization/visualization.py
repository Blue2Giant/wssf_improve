import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def cp_show_match(img1, img2, loc1, loc2, correct_pos=None, save_path=None):
    """
    Display matching points between two images.
    
    Parameters:
    -----------
    img1, img2 : numpy.ndarray
        Input images
    loc1, loc2 : numpy.ndarray
        Matching point coordinates in both images
    correct_pos : numpy.ndarray, optional
        Indices of correct matches
    save_path : str, optional
        Path to save the visualization
    """
    # Ensure images have 3 channels
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    elif img1.shape[2] > 3:
        img1 = img1[:, :, :3]
        
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    elif img2.shape[2] > 3:
        img2 = img2[:, :, :3]
    
    # Create a combined image (black canvas)
    cols = img1.shape[1]
    if img1.shape[0] < img2.shape[0]:
        img1_p = np.vstack([img1, np.zeros((img2.shape[0] - img1.shape[0], img1.shape[1], 3), dtype=img1.dtype)])
        combined_img = np.hstack([img1_p, img2])
    elif img1.shape[0] > img2.shape[0]:
        img2_p = np.vstack([img2, np.zeros((img1.shape[0] - img2.shape[0], img2.shape[1], 3), dtype=img2.dtype)])
        combined_img = np.hstack([img1, img2_p])
    else:
        combined_img = np.hstack([img1, img2])

    fig = plt.figure(figsize=(12, 8), facecolor='black')
    ax = fig.add_subplot(111)
    ax.set_facecolor('black')
    ax.imshow(combined_img)
    
    # Draw matching lines
    if correct_pos is not None:
        for i in range(loc1.shape[0]):
            ax.plot([loc1[i, 0], loc2[i, 0] + cols], [loc1[i, 1], loc2[i, 1]], color=(1,1,0), linewidth=0.8, alpha=0.9)
            ax.plot(loc1[i, 0], loc1[i, 1], marker='o', markersize=3, color=(0,1,1))
            ax.plot(loc2[i, 0] + cols, loc2[i, 1], marker='o', markersize=3, color=(1,0,1))
        
        for i in correct_pos:
            ax.plot([loc1[i, 0], loc2[i, 0] + cols], [loc1[i, 1], loc2[i, 1]], color=(0,0.7,1), linewidth=1.5, alpha=1.0)
            ax.plot(loc1[i, 0], loc1[i, 1], marker='*', markersize=6, color=(0,1,0))
            ax.plot(loc2[i, 0] + cols, loc2[i, 1], marker='*', markersize=6, color=(0,1,0))
    else:
        for i in range(loc1.shape[0]):
            ax.plot([loc1[i, 0], loc2[i, 0] + cols], [loc1[i, 1], loc2[i, 1]], color=(0.3,0.8,1), linewidth=1.0, alpha=0.9)
            ax.plot(loc1[i, 0], loc1[i, 1], marker='s', markersize=4, color=(1,0.2,0.2))
            ax.plot(loc2[i, 0] + cols, loc2[i, 1], marker='s', markersize=4, color=(0.2,1,0.2))
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        parent_path, _, ext = save_path.rpartition('.')
        if not ext:
            ext = 'png'
        plt.savefig(f"{parent_path}_matched_features.{ext}", facecolor=fig.get_facecolor())
        
    plt.close()

def image_fusion(image_1, image_2, solution, save_path=None):
    """
    Fuse two images using a transformation matrix.
    
    Parameters:
    -----------
    image_1, image_2 : numpy.ndarray
        Input images
    solution : numpy.ndarray
        Transformation matrix
    save_path : str, optional
        Path to save the fused image
    
    Returns:
    --------
    numpy.ndarray
        Fused image
    """
    # Get image dimensions
    M1, N1 = image_1.shape[:2]
    M2, N2 = image_2.shape[:2]
    
    # Handle different channel counts
    if len(image_1.shape) == 3 and len(image_2.shape) == 3:
        fusion_image = np.zeros((3*M1, 3*N1, 3), dtype=np.uint8)
    elif len(image_1.shape) == 3 and len(image_2.shape) == 2:
        fusion_image = np.zeros((3*M1, 3*N1), dtype=np.uint8)
        image_2 = cv2.cvtColor(image_2, cv2.COLOR_GRAY2RGB)
    elif len(image_1.shape) == 2 and len(image_2.shape) == 3:
        fusion_image = np.zeros((3*M1, 3*N1), dtype=np.uint8)
        image_1 = cv2.cvtColor(image_1, cv2.COLOR_GRAY2RGB)
    else:
        fusion_image = np.zeros((3*M1, 3*N1), dtype=np.uint8)
    
    # Define transformation matrices
    solution_1 = np.array([[1, 0, N1], [0, 1, M1], [0, 0, 1]], dtype=np.float32)
    
    # Apply transformations
    # Ensure solution is float matrix
    solution = solution.astype(np.float32)
    f_1 = cv2.warpPerspective(image_1, solution_1, (3*N1, 3*M1))
    f_2 = cv2.warpPerspective(image_2, (solution_1 @ solution).astype(np.float32), (3*N1, 3*M1))
    
    # Fuse images
    # Find regions where both images have content
    same_index = np.logical_and(f_1 > 0, f_2 > 0)
    index_1 = np.logical_and(f_1 > 0, f_2 == 0)
    index_2 = np.logical_and(f_1 == 0, f_2 > 0)
    
    # Combine images
    if len(fusion_image.shape) == 3:
        fusion_image[same_index] = f_1[same_index] // 2 + f_2[same_index] // 2
        fusion_image[index_1] = f_1[index_1]
        fusion_image[index_2] = f_2[index_2]
    else:
        fusion_image[same_index] = f_1[same_index] // 2 + f_2[same_index] // 2
        fusion_image[index_1] = f_1[index_1]
        fusion_image[index_2] = f_2[index_2]
    
    # Calculate boundaries of the transformed image
    corners = np.array([[1, 1, 1], [1, M2, 1], [N2, 1, 1], [N2, M2, 1]]).T
    transformed_corners = solution_1 @ solution @ corners
    
    X = transformed_corners[0, :] / transformed_corners[2, :]
    Y = transformed_corners[1, :] / transformed_corners[2, :]
    
    X_min = max(int(np.floor(np.min(X))), 1)
    X_max = min(int(np.ceil(np.max(X))), 3*N1)
    Y_min = max(int(np.floor(np.min(Y))), 1)
    Y_max = min(int(np.ceil(np.max(Y))), 3*M1)
    
    # Adjust boundaries
    X_min = N1 + 1 if X_min > N1 + 1 else X_min
    X_max = 2*N1 if X_max < 2*N1 else X_max
    Y_min = M1 + 1 if Y_min > M1 + 1 else Y_min
    Y_max = 2*M1 if Y_max < 2*M1 else Y_max
    
    # Crop the fusion image
    fusion_image = fusion_image[Y_min:Y_max, X_min:X_max]
    f_1_cropped = f_1[Y_min:Y_max, X_min:X_max]
    f_2_cropped = f_2[Y_min:Y_max, X_min:X_max]
    
    # Save the fusion image if a path is provided
    if save_path:
        parent_path, _, ext = save_path.rpartition('.')
        if not ext:
            ext = 'png'
        cv2.imwrite(f"{parent_path}_fusion.{ext}", fusion_image)
        
        # Create a chessboard pattern
        grid_num = 7
        grid_size = min(fusion_image.shape[0], fusion_image.shape[1]) // grid_num
        chessboard = np.zeros_like(fusion_image)
        
        for i in range(grid_num):
            for j in range(grid_num):
                y_start = i * grid_size
                y_end = (i + 1) * grid_size
                x_start = j * grid_size
                x_end = (j + 1) * grid_size
                
                if y_end <= chessboard.shape[0] and x_end <= chessboard.shape[1]:
                    if (i + j) % 2 == 0:
                        # 使用image1作为基础
                        chessboard[y_start:y_end, x_start:x_end] = f_1_cropped[y_start:y_end, x_start:x_end]
                    else:
                        # 隔一个方格绘制变换后的image2纹理
                        chessboard[y_start:y_end, x_start:x_end] = f_2_cropped[y_start:y_end, x_start:x_end]
        
        cv2.imwrite(f"{parent_path}_chessboard.{ext}", chessboard)
    
    return fusion_image

def image_fusion_overlapped(image_1, image_2, solution, save_path=None, alpha=0.5):
    M1, N1 = image_1.shape[:2]
    M2, N2 = image_2.shape[:2]

    if len(image_1.shape) == 3 and len(image_2.shape) == 3:
        fusion_image = np.zeros((3*M1, 3*N1, 3), dtype=np.uint8)
    elif len(image_1.shape) == 3 and len(image_2.shape) == 2:
        fusion_image = np.zeros((3*M1, 3*N1), dtype=np.uint8)
        image_2 = cv2.cvtColor(image_2, cv2.COLOR_GRAY2RGB)
    elif len(image_1.shape) == 2 and len(image_2.shape) == 3:
        fusion_image = np.zeros((3*M1, 3*N1), dtype=np.uint8)
        image_1 = cv2.cvtColor(image_1, cv2.COLOR_GRAY2RGB)
    else:
        fusion_image = np.zeros((3*M1, 3*N1), dtype=np.uint8)

    solution_1 = np.array([[1, 0, N1], [0, 1, M1], [0, 0, 1]], dtype=np.float32)
    solution = solution.astype(np.float32)
    f_1 = cv2.warpPerspective(image_1, solution_1, (3*N1, 3*M1))
    f_2 = cv2.warpPerspective(image_2, (solution_1 @ solution).astype(np.float32), (3*N1, 3*M1))

    same_index = np.logical_and(f_1 > 0, f_2 > 0)
    index_1 = np.logical_and(f_1 > 0, f_2 == 0)
    index_2 = np.logical_and(f_1 == 0, f_2 > 0)

    if len(fusion_image.shape) == 3:
        fusion_image[same_index] = (alpha * f_1[same_index] + (1 - alpha) * f_2[same_index]).astype(fusion_image.dtype)
        fusion_image[index_1] = f_1[index_1]
        fusion_image[index_2] = f_2[index_2]
    else:
        fusion_image[same_index] = (alpha * f_1[same_index] + (1 - alpha) * f_2[same_index]).astype(fusion_image.dtype)
        fusion_image[index_1] = f_1[index_1]
        fusion_image[index_2] = f_2[index_2]

    corners = np.array([[1, 1, 1], [1, M2, 1], [N2, 1, 1], [N2, M2, 1]]).T
    transformed_corners = solution_1 @ solution @ corners
    X = transformed_corners[0, :] / transformed_corners[2, :]
    Y = transformed_corners[1, :] / transformed_corners[2, :]
    X_min = max(int(np.floor(np.min(X))), 1)
    X_max = min(int(np.ceil(np.max(X))), 3*N1)
    Y_min = max(int(np.floor(np.min(Y))), 1)
    Y_max = min(int(np.ceil(np.max(Y))), 3*M1)
    X_min = N1 + 1 if X_min > N1 + 1 else X_min
    X_max = 2*N1 if X_max < 2*N1 else X_max
    Y_min = M1 + 1 if Y_min > M1 + 1 else Y_min
    Y_max = 2*M1 if Y_max < 2*M1 else Y_max

    fusion_image = fusion_image[Y_min:Y_max, X_min:X_max]

    if save_path:
        parent_path, _, ext = save_path.rpartition('.')
        if not ext:
            ext = 'png'
        cv2.imwrite(f"{parent_path}_overlapped.{ext}", fusion_image)

    return fusion_image
