#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Import WSSF modules
from core.image_space import create_image_space
from core.gradient_feature import wssf_gradient_feature
from core.feature_extraction import wssf_features
from core.descriptors import gloh_descriptors
from core.matching import match_features, back_projection, fsc
from visualization.visualization import cp_show_match, image_fusion

def main():
    """
    Main function to demonstrate the WSSF algorithm workflow.
    """
    print("Starting WSSF demo...")
    
    # Set paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.dirname(current_dir)
    
    # Load images
    # img1_path = os.path.join(img_dir, "pair1.jpg")
    # img2_path = os.path.join(img_dir, "pair2.jpg")
    img1_path = "/Users/leolan/Downloads/WorkSpace/wssf_improve/pair1.jpg"
    img2_path = "/Users/leolan/Downloads/WorkSpace/wssf_improve/pair2.jpg"
    
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print(f"Error: Image files not found at {img1_path} or {img2_path}")
        return
    
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print("Error: Failed to load images")
        return
    
    # Convert to RGB for visualization
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale for processing
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img1_gray = clahe.apply(img1_gray)
    img2_gray = clahe.apply(img2_gray)
    
    print("Images loaded and preprocessed")
    
    # Parameters
    scale_num = 3  # Number of scales
    scale_invariance = 'YES'  # 'YES' or 'NO' for scale invariance
    scale_val = 1.6  # Scale factor
    ratio = 0.5  # Ratio for image space
    sigma_1 = 1.0  # Gaussian sigma
    filter_size = 5  # Filter size
    
    # Create image spaces
    print("Creating image spaces...")
    start_time = time()
    
    spaces1 = create_image_space(img1_gray, scale_num, scale_invariance, scale_val, ratio, sigma_1, filter_size)
    spaces2 = create_image_space(img2_gray, scale_num, scale_invariance, scale_val, ratio, sigma_1, filter_size)
    
    print(f"Image spaces created in {time() - start_time:.2f} seconds")
    
    # Extract WSSF features (includes gradient and angle cells)
    print("Extracting WSSF features...")
    start_time = time()
    nonlinear1, e1, max1, min1, phase1 = spaces1
    nonlinear2, e2, max2, min2, phase2 = spaces2

    blob_kpts1, corner_kpts1, blob_grad_cell1, corner_grad_cell1, blob_angle_cell1, corner_angle_cell1 = \
        wssf_features(nonlinear1, e1, max1, min1, phase1, sigma_1, ratio, scale_invariance, scale_num)
    blob_kpts2, corner_kpts2, blob_grad_cell2, corner_grad_cell2, blob_angle_cell2, corner_angle_cell2 = \
        wssf_features(nonlinear2, e2, max2, min2, phase2, sigma_1, ratio, scale_invariance, scale_num)
    
    print(f"WSSF features extracted in {time() - start_time:.2f} seconds")
    print(f"Image 1: {len(blob_kpts1)} blob keypoints, {len(corner_kpts1)} corner keypoints")
    print(f"Image 2: {len(blob_kpts2)} blob keypoints, {len(corner_kpts2)} corner keypoints")
    
    # Compute GLOH descriptors
    print("Computing GLOH descriptors...")
    start_time = time()
    
    path_block = 10
    blob_desc1 = gloh_descriptors(blob_grad_cell1, blob_angle_cell1, blob_kpts1, path_block, ratio=ratio, sigma_1=sigma_1)
    blob_desc2 = gloh_descriptors(blob_grad_cell2, blob_angle_cell2, blob_kpts2, path_block, ratio=ratio, sigma_1=sigma_1)
    
    corner_desc1 = gloh_descriptors(corner_grad_cell1, corner_angle_cell1, corner_kpts1, path_block, ratio=ratio, sigma_1=sigma_1)
    corner_desc2 = gloh_descriptors(corner_grad_cell2, corner_angle_cell2, corner_kpts2, path_block, ratio=ratio, sigma_1=sigma_1)
    
    print(f"GLOH descriptors computed in {time() - start_time:.2f} seconds")
    
    # Match features with OpenCV BFMatcher：加入双向交叉检验 + ratio test
    print("Matching features with OpenCV...")
    start_time = time()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    ratio = 0.85

    # Ensure descriptors are float32 for OpenCV
    blob_des1 = blob_desc1['des'].astype(np.float32)
    blob_des2 = blob_desc2['des'].astype(np.float32)
    corner_des1 = corner_desc1['des'].astype(np.float32)
    corner_des2 = corner_desc2['des'].astype(np.float32)

    def cross_check_ratio(desA, desB, r=0.85):
        if desA is None or desB is None or len(desA) == 0 or len(desB) == 0:
            return []
        knnAB = bf.knnMatch(desA, desB, k=2)
        knnBA = bf.knnMatch(desB, desA, k=2)
        filtAB = [m for m, n in knnAB if m.distance < r * n.distance]
        filtBA = [m for m, n in knnBA if m.distance < r * n.distance]
        tableBA = {m.queryIdx: m.trainIdx for m in filtBA}
        checked = [m for m in filtAB if tableBA.get(m.trainIdx, -1) == m.queryIdx]
        return checked

    good_blob = cross_check_ratio(blob_des1, blob_des2, r=ratio)
    good_corner = cross_check_ratio(corner_des1, corner_des2, r=ratio)

    print(f"Feature matching completed in {time() - start_time:.2f} seconds")
    print(f"Blob matches: {len(good_blob)}, Corner matches: {len(good_corner)}")

    # Prepare matched points (OpenCV indices are 0-based)
    blob_pts1 = np.array([blob_desc1['locs'][m.queryIdx][:2] for m in good_blob]) if len(good_blob) > 0 else np.empty((0, 2))
    blob_pts2 = np.array([blob_desc2['locs'][m.trainIdx][:2] for m in good_blob]) if len(good_blob) > 0 else np.empty((0, 2))

    corner_pts1 = np.array([corner_desc1['locs'][m.queryIdx][:2] for m in good_corner]) if len(good_corner) > 0 else np.empty((0, 2))
    corner_pts2 = np.array([corner_desc2['locs'][m.trainIdx][:2] for m in good_corner]) if len(good_corner) > 0 else np.empty((0, 2))

    # Combine blob and corner matches
    all_pts1 = np.vstack([blob_pts1, corner_pts1]) if len(blob_pts1) > 0 and len(corner_pts1) > 0 else \
               blob_pts1 if len(blob_pts1) > 0 else corner_pts1
    all_pts2 = np.vstack([blob_pts2, corner_pts2]) if len(blob_pts2) > 0 and len(corner_pts2) > 0 else \
               blob_pts2 if len(blob_pts2) > 0 else corner_pts2

    # Prefilter matches with geometric prior (near 90° rotation around centroids and similar scale)
    if len(all_pts1) >= 3 and len(all_pts2) >= 3:
        c1 = np.mean(all_pts1, axis=0)
        c2 = np.mean(all_pts2, axis=0)
        v1 = all_pts1 - c1
        v2 = all_pts2 - c2
        ang1 = np.degrees(np.arctan2(v1[:, 1], v1[:, 0]))
        ang2 = np.degrees(np.arctan2(v2[:, 1], v2[:, 0]))
        dtheta = (ang2 - ang1 + 360.0) % 360.0
        ang_err = np.minimum(np.abs(dtheta - 90.0), np.abs(dtheta - 450.0))
        scale_ratio = (np.linalg.norm(v2, axis=1) + 1e-6) / (np.linalg.norm(v1, axis=1) + 1e-6)
        angle_th = 30.0
        scale_low, scale_high = 0.6, 1.6
        keep = (ang_err <= angle_th) & (scale_ratio >= scale_low) & (scale_ratio <= scale_high)
        if int(np.sum(keep)) >= 8:
            all_pts1 = all_pts1[keep]
            all_pts2 = all_pts2[keep]

    # 使用FSC相似变换匹配（匹配MATLAB方法），并对角度靠近90°施加软约束
    print("Estimating similarity transform with FSC (matching MATLAB)...")
    start_time = time()

    if len(all_pts1) >= 2 and len(all_pts2) >= 2:
        # Use FSC function from matching.py with similarity transformation
        solution, rmse, inlier_pts1, inlier_pts2 = fsc(all_pts1, all_pts2, change_form='similarity', error_t=3.0)
        
        print(f"FSC similarity estimation completed in {time() - start_time:.2f} seconds")
        
        # Extract rotation angle from the transformation matrix
        A_final = solution[:2, :2]
        U_final, S_final, Vt_final = np.linalg.svd(A_final)
        R_final = U_final @ Vt_final
        angle_final = np.degrees(np.arctan2(R_final[1, 0], R_final[0, 0]))
        ang_norm_final = (angle_final + 360.0) % 360.0
        ang_diff_final = min(abs(ang_norm_final - 90.0), abs(ang_norm_final - 450.0))
        
        print(f"Found {len(inlier_pts1)} inliers out of {len(all_pts1)} matches; RMSE={rmse:.3f}")
        print(f"Final rotation ≈ {angle_final:.1f}°, |Δ90°|={ang_diff_final:.2f}°")
        
        # Print transformation matrix for debugging
        print("Transformation matrix:")
        print(solution)

        np.savetxt('/Users/leolan/Downloads/WorkSpace/wssf_improve/wssf_python/results/solution.txt', solution)
        print("Transformation matrix saved to results/solution.txt")

        # Guided re-matching: project many keypoints from img1 to img2, find nearest neighbors
        def transform_points(X, sol):
            Xh = np.hstack([X, np.ones((len(X), 1))])
            Yh = (Xh @ sol.T)
            return Yh[:, :2] / np.maximum(Yh[:, 2:3], 1e-6)
    
        locs1_all = np.vstack([blob_desc1['locs'][:, :2], corner_desc1['locs'][:, :2]])
        locs2_all = np.vstack([blob_desc2['locs'][:, :2], corner_desc2['locs'][:, :2]])
    
        proj1 = transform_points(locs1_all, solution)
        radius = 6.0
        # 采样以避免O(N^2)过大
        sample_idx = np.linspace(0, len(locs1_all)-1, num=min(1500, len(locs1_all)), dtype=int)
        guided_1 = []
        guided_2 = []
        for idx in sample_idx:
            p = proj1[idx]
            d = np.linalg.norm(locs2_all - p, axis=1)
            j = int(np.argmin(d))
            if d[j] < radius:
                guided_1.append(locs1_all[idx])
                guided_2.append(locs2_all[j])
        if len(guided_1) >= 2:
            guided_1 = np.asarray(guided_1)
            guided_2 = np.asarray(guided_2)
            # 合并原始内点与引导匹配，去重
            combined_1 = np.vstack([inlier_pts1, guided_1])
            combined_2 = np.vstack([inlier_pts2, guided_2])
            # 二次FSC精炼
            solution_refined, rmse_refined, inlier_pts1_refined, inlier_pts2_refined = fsc(combined_1, combined_2, change_form='similarity', error_t=3.0)
            if len(inlier_pts1_refined) >= len(inlier_pts1):
                solution = solution_refined
                rmse = rmse_refined
                inlier_pts1 = inlier_pts1_refined
                inlier_pts2 = inlier_pts2_refined
                print("Refined transformation matrix:")
                print(solution)
                A_final = solution[:2, :2]
                U_final, S_final, Vt_final = np.linalg.svd(A_final)
                R_final = U_final @ Vt_final
                angle_final = np.degrees(np.arctan2(R_final[1, 0], R_final[0, 0]))
                ang_norm_final = (angle_final + 360.0) % 360.0
                ang_diff_final = min(abs(ang_norm_final - 90.0), abs(ang_norm_final - 450.0))
                print(f"Refined: inliers={len(inlier_pts1)}, RMSE={rmse:.3f}, angle≈{angle_final:.2f}°, |Δ90°|={ang_diff_final:.2f}°")
    
        save_dir = os.path.join(current_dir, "results")
        os.makedirs(save_dir, exist_ok=True)
        cp_show_match(img1_rgb, img2_rgb, inlier_pts1, inlier_pts2, None, os.path.join(save_dir, "matches.png"))
        fused_img = image_fusion(img1_rgb, img2_rgb, solution, os.path.join(save_dir, "fusion.png"))
        print(f"Results saved to {save_dir}")
    else:
        print("Not enough matches found to estimate similarity transform")
    
    print("WSSF demo completed")

if __name__ == "__main__":
    main()