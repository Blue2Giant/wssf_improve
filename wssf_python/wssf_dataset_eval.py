#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python /Users/leolan/Downloads/WorkSpace/wssf_improve/wssf_python/wssf_dataset_eval.py --data_dir "/Users/leolan/Downloads/WorkSpace/SRIF/dataset/Optical-SAR"  --out_csv   "./wssf_eval_results.csv"   --gt_dir   1to2 --grid      25
"""
import os
import re
import csv
import cv2
import numpy as np
from pathlib import Path
from time import time
import argparse

# ========== 你的模块 ==========
from core.image_space import create_image_space
from core.feature_extraction import wssf_features
from core.descriptors import gloh_descriptors
from core.matching import fsc

# ========== 辅助 ==========
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

PAIR_RE = re.compile(r"^pair[_-]?(\d+)[_-]?([12])\.(jpg|jpeg|png|bmp|webp|tif|tiff)$", re.IGNORECASE)
GT_RE   = re.compile(r"^gt[_-]?(\d+)\.txt$", re.IGNORECASE)

def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"fail to read image: {path}")
    # 自适应直方图均衡（与示例一致）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def read_gt_3x3(path: Path) -> np.ndarray:
    """
    读取 gt 矩阵：
    - 支持 3x3（9 个数）
    - 支持 2x3（6 个数）将嵌入为 [[a,b,c],[d,e,f],[0,0,1]]
    - 忽略分隔符（空格/逗号/制表）
    返回 np.float64 的 (3,3)
    """
    txt = Path(path).read_text().strip().replace(",", " ")
    nums = [float(x) for x in txt.split()]
    if len(nums) == 9:
        M = np.array(nums, dtype=np.float64).reshape(3,3)
    elif len(nums) == 6:
        M = np.array([[nums[0], nums[1], nums[2]],
                      [nums[3], nums[4], nums[5]],
                      [0.0,     0.0,     1.0   ]], dtype=np.float64)
    else:
        raise ValueError(f"unexpected GT format (need 9 or 6 numbers): {path}")
    return M

def invert_h(M: np.ndarray) -> np.ndarray:
    return np.linalg.inv(M)

def transform_points(pts_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    pts_xy: (N,2) in image-1 coords
    H: 3x3 mapping 1->2
    returns: (N,2) in image-2 coords
    """
    N = len(pts_xy)
    homo = np.hstack([pts_xy.astype(np.float64), np.ones((N,1), dtype=np.float64)])
    proj = (homo @ H.T)
    proj_xy = proj[:, :2] / np.maximum(proj[:, [2]], 1e-8)
    return proj_xy

def rmse_solution_vs_gt(H_est: np.ndarray, H_gt: np.ndarray, w: int, h: int, grid: int = 20) -> float:
    """
    在图1平面上采样 grid×grid 网格（含四角与中心），比较两变换投影到图2的差异。
    """
    xs = np.linspace(0, w-1, grid)
    ys = np.linspace(0, h-1, grid)
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)  # (N,2)

    P_est = transform_points(P, H_est)
    P_gt  = transform_points(P, H_gt)

    diff = P_est - P_gt
    se = np.sum(diff**2, axis=1)
    rmse = float(np.sqrt(np.mean(se)))
    return rmse

def bf_match_ratio_cross(desA: np.ndarray, desB: np.ndarray, ratio: float = 0.85):
    """
    OpenCV BFMatcher 双向 + ratio test。
    输入需 float32。
    """
    if desA is None or desB is None or len(desA)==0 or len(desB)==0:
        return []
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knnAB = bf.knnMatch(desA, desB, k=2)
    knnBA = bf.knnMatch(desB, desA, k=2)
    filtAB = [m for m, n in knnAB if len([m,n])==2 and m.distance < ratio*n.distance]
    filtBA = [m for m, n in knnBA if len([m,n])==2 and m.distance < ratio*n.distance]
    tableBA = {m.queryIdx: m.trainIdx for m in filtBA}
    checked = [m for m in filtAB if tableBA.get(m.trainIdx, -1) == m.queryIdx]
    return checked

def estimate_solution(img1_gray: np.ndarray, img2_gray: np.ndarray,
                      scale_num=3, scale_invariance='YES', scale_val=1.6,
                      ratio=0.5, sigma_1=1.0, filter_size=5,
                      ratio_test=0.85):
    """
    基于你现有流程：image space -> WSSF features -> GLOH -> BF+ratio+cross -> FSC(similarity)
    返回：H_est(3x3), rmse_inlier, n_inliers
    """
    # 1) image space
    spaces1 = create_image_space(img1_gray, scale_num, scale_invariance, scale_val, ratio, sigma_1, filter_size)
    spaces2 = create_image_space(img2_gray, scale_num, scale_invariance, scale_val, ratio, sigma_1, filter_size)
    nonlinear1, e1, max1, min1, phase1 = spaces1
    nonlinear2, e2, max2, min2, phase2 = spaces2

    # 2) WSSF features
    (blob_kpts1, corner_kpts1,
     blob_grad_cell1, corner_grad_cell1,
     blob_angle_cell1, corner_angle_cell1) = wssf_features(nonlinear1, e1, max1, min1, phase1, sigma_1, ratio, scale_invariance, scale_num)

    (blob_kpts2, corner_kpts2,
     blob_grad_cell2, corner_grad_cell2,
     blob_angle_cell2, corner_angle_cell2) = wssf_features(nonlinear2, e2, max2, min2, phase2, sigma_1, ratio, scale_invariance, scale_num)

    # 3) GLOH descriptors
    path_block = 10
    blob_desc1   = gloh_descriptors(blob_grad_cell1,   blob_angle_cell1,   blob_kpts1,   path_block, ratio=ratio, sigma_1=sigma_1)
    blob_desc2   = gloh_descriptors(blob_grad_cell2,   blob_angle_cell2,   blob_kpts2,   path_block, ratio=ratio, sigma_1=sigma_1)
    corner_desc1 = gloh_descriptors(corner_grad_cell1, corner_angle_cell1, corner_kpts1, path_block, ratio=ratio, sigma_1=sigma_1)
    corner_desc2 = gloh_descriptors(corner_grad_cell2, corner_angle_cell2, corner_kpts2, path_block, ratio=ratio, sigma_1=sigma_1)

    # 4) BF + ratio + cross
    blob_des1   = blob_desc1['des'].astype(np.float32)
    blob_des2   = blob_desc2['des'].astype(np.float32)
    corner_des1 = corner_desc1['des'].astype(np.float32)
    corner_des2 = corner_desc2['des'].astype(np.float32)

    good_blob   = bf_match_ratio_cross(blob_des1,   blob_des2,   ratio_test)
    good_corner = bf_match_ratio_cross(corner_des1, corner_des2, ratio_test)

    blob_pts1   = np.array([blob_desc1['locs'][m.queryIdx][:2]   for m in good_blob])   if good_blob   else np.empty((0,2))
    blob_pts2   = np.array([blob_desc2['locs'][m.trainIdx][:2]   for m in good_blob])   if good_blob   else np.empty((0,2))
    corner_pts1 = np.array([corner_desc1['locs'][m.queryIdx][:2] for m in good_corner]) if good_corner else np.empty((0,2))
    corner_pts2 = np.array([corner_desc2['locs'][m.trainIdx][:2] for m in good_corner]) if good_corner else np.empty((0,2))

    if len(blob_pts1) and len(corner_pts1):
        pts1 = np.vstack([blob_pts1, corner_pts1])
        pts2 = np.vstack([blob_pts2, corner_pts2])
    elif len(blob_pts1):
        pts1, pts2 = blob_pts1, blob_pts2
    else:
        pts1, pts2 = corner_pts1, corner_pts2

    if len(pts1) < 2 or len(pts2) < 2:
        return None, None, 0

    # 5) FSC 相似变换 + RANSAC（接口来自你的 matching.fsc）
    H_est, rmse_inlier, inlier_pts1, inlier_pts2 = fsc(pts1, pts2, change_form='similarity', error_t=3.0)
    n_in = len(inlier_pts1) if inlier_pts1 is not None else 0
    return H_est, float(rmse_inlier), n_in

def collect_pairs(dir_path: Path):
    """
    扫描目录，返回 {k: {"img1": Path, "img2": Path, "gt": Path or None}}
    匹配：
      pair{k}_1.jpg / pair{k}_2.jpg
      gt_{k}.txt
    """
    items = {}
    for p in dir_path.iterdir():
        if p.is_file():
            m = PAIR_RE.match(p.name)
            if m:
                k = int(m.group(1))
                which = m.group(2)
                d = items.setdefault(k, {})
                if which == "1":
                    d["img1"] = p
                else:
                    d["img2"] = p
                continue
            g = GT_RE.match(p.name)
            if g:
                k = int(g.group(1))
                d = items.setdefault(k, {})
                d["gt"] = p
    # 只保留成对图片的项
    pairs = {k:v for k,v in items.items() if "img1" in v and "img2" in v}
    return dict(sorted(pairs.items(), key=lambda x: x[0]))

def main():
    ap = argparse.ArgumentParser("Batch evaluate WSSF solution vs GT RMSE")
    ap.add_argument("--data_dir", required=True, help="Folder containing pair{k}_1.jpg, pair{k}_2.jpg and gt_{k}.txt")
    ap.add_argument("--out_csv", default="wssf_eval_results.csv", help="Output CSV path")
    ap.add_argument("--gt_dir", default="1to2", choices=["1to2","2to1"], help="Direction of GT homography")
    ap.add_argument("--grid", type=int, default=20, help="Grid sampling for RMSE (per side)")
    args = ap.parse_args()

    root = Path(args.data_dir).expanduser().resolve()
    pairs = collect_pairs(root)
    if not pairs:
        print(f"[!] no pairs found in {root}")
        return

    rows = []
    t0 = time()
    for k, rec in pairs.items():
        img1_path, img2_path = rec["img1"], rec["img2"]
        gt_path = rec.get("gt", None)

        print(f"\n=== Pair {k}: {img1_path.name}  /  {img2_path.name} ===")
        if gt_path is None:
            print(f"[warn] GT missing for pair {k} (expect gt_{k}.txt). Skip RMSE_vs_GT.")
        # 读图
        img1 = read_gray(img1_path)
        img2 = read_gray(img2_path)
        H, W = img1.shape[:2]
        # 估计
        est_t = time()
        H_est, rmse_inlier, n_in = estimate_solution(img1, img2)
        if H_est is None:
            print("[x] not enough matches. skip.")
            rows.append([k, img1_path.name, img2_path.name, str(gt_path.name if gt_path else ""), n_in, "", "", ""])
            continue

        # GT RMSE
        rmse_vs_gt = ""
        if gt_path is not None:
            H_gt = read_gt_3x3(gt_path)
            if args.gt_dir == "2to1":
                # gt 给的是 2->1，则取逆得到 1->2
                H_gt = invert_h(H_gt)
            rmse_vs_gt = rmse_solution_vs_gt(H_est, H_gt, W, H, grid=args.grid)
            print(f"[OK] inlier_RMSE={rmse_inlier:.3f} px,  RMSE(solution vs GT)={rmse_vs_gt:.3f} px  (grid={args.grid})")
        else:
            print(f"[OK] inlier_RMSE={rmse_inlier:.3f} px,  RMSE(solution vs GT)=N/A (no gt)")

        rows.append([k, img1_path.name, img2_path.name, str(gt_path.name if gt_path else ""),
                     n_in, f"{rmse_inlier:.6f}", f"{rmse_vs_gt:.6f}" if rmse_vs_gt!="" else "", args.gt_dir])

    # 写 CSV
    out_csv = Path(args.out_csv).resolve()
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_id","img1","img2","gt_txt","n_inliers","inlier_rmse_px","rmse_vs_gt_px","gt_dir"])
        w.writerows(rows)

    print(f"\nDone. {len(rows)} pairs. Time: {time()-t0:.1f}s")
    print(f"Results -> {out_csv}")

if __name__ == "__main__":
    main()
