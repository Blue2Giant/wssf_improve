#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批处理脚本：遍历 Optical-SAR 数据集，批量执行 WSSF 配准、精度评估与可视化，并生成汇总报告。

功能概述：
1) 递归扫描数据目录，自动识别配对图片与 GT 矩阵；校验每对完整性
2) 复用当前 python WSSF demo 的实现执行配准，得到估计投影矩阵
3) 精度评估：矩阵 MSE；在 SAR 上均匀采样 50 点进行位置 MSE
4) 可视化输出：
   - 特征点匹配图（黑底高对比彩色）
   - 融合棋盘图（保持原有风格）
   - 完全重叠融合图（alpha 混合）
   - 在三类图上叠加 RMSE/MSE 文本标注
5) 结果汇总：CSV、summary.txt、错误日志 errors.txt、批处理日志 batch_log.txt

技术栈：OpenCV、NumPy、Matplotlib（可选 seaborn）
异常处理：逐对 try/except，日志记录错误并继续处理后续数据
python3 wssf_batch_eval.py --data_dir ./Optical-SAR --out_dir ./wssf_python/results/batch --out_csv ./wssf_python/results/wssf_batch_results.csv --gt_dir 1to2
"""

import os
import re
import csv
import cv2
import numpy as np
from pathlib import Path
from time import time
import argparse
import logging

# 可选 seaborn 用于误差分布图；缺失时退化到 matplotlib
try:
    import seaborn as sns
    USE_SEABORN = True
except Exception:
    USE_SEABORN = False
import matplotlib.pyplot as plt

# 复用当前 WSSF python demo 的各模块
from core.image_space import create_image_space
from core.feature_extraction import wssf_features
from core.descriptors import gloh_descriptors
from core.matching import match_features, back_projection, fsc
from visualization.visualization import cp_show_match, image_fusion, image_fusion_overlapped

# 文件名匹配规则：pair{k}_{1|2}.jpg 与 gt_{k}.txt
PAIR_RE = re.compile(r"^pair[_-]?(\d+)[_-]?(1|2)\.(jpg|jpeg|png|bmp|webp|tif|tiff)$", re.IGNORECASE)
GT_RE   = re.compile(r"^gt[_-]?(\d+)\.txt$", re.IGNORECASE)

def read_gray(path: Path) -> np.ndarray:
    """读取灰度图并进行 CLAHE 预处理，以增强对比度。"""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"fail to read image: {path}")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def read_gt_3x3(path: Path) -> np.ndarray:
    """读取 GT 投影矩阵；支持 3x3（9 数）或 2x3（6 数，补齐为 3x3）。"""
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
    """矩阵求逆（用于 2→1 方向转换）。"""
    return np.linalg.inv(M)

def transform_points(pts_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
    """以齐次坐标进行 2D 投影变换：pts(1x) —H→ pts(2x)。"""
    N = len(pts_xy)
    homo = np.hstack([pts_xy.astype(np.float64), np.ones((N,1), dtype=np.float64)])
    proj = (homo @ H.T)
    proj_xy = proj[:, :2] / np.maximum(proj[:, [2]], 1e-8)
    return proj_xy

def collect_pairs_recursive(root: Path) -> dict:
    """递归扫描目录，收集 pair 图像与对应 GT；只保留完整项。"""
    items = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            m = PAIR_RE.match(fname)
            if m:
                k = int(m.group(1))
                which = m.group(2)
                d = items.setdefault(k, {})
                p = Path(dirpath) / fname
                if which == "1":
                    d["img1"] = p
                else:
                    d["img2"] = p
                continue
            g = GT_RE.match(fname)
            if g:
                k = int(g.group(1))
                d = items.setdefault(k, {})
                d["gt"] = Path(dirpath) / fname
    pairs = {k:v for k,v in items.items() if all(key in v for key in ("img1","img2","gt"))}
    return dict(sorted(pairs.items(), key=lambda x: x[0]))

def estimate_solution(img1_gray: np.ndarray, img2_gray: np.ndarray,
                      scale_num=3, scale_invariance='YES', scale_val=1.6,
                      ratio=2**(1/3), sigma_1=1.6, filter_size=5):
    """执行与 demo 同步的 WSSF 配准流程，返回仿射解与内点。"""
    # 1) 图像空间
    spaces1 = create_image_space(img1_gray, scale_num, scale_invariance, scale_val, ratio, sigma_1, filter_size)
    spaces2 = create_image_space(img2_gray, scale_num, scale_invariance, scale_val, ratio, sigma_1, filter_size)
    nonlinear1, e1, max1, min1, phase1 = spaces1
    nonlinear2, e2, max2, min2, phase2 = spaces2

    # 2) 特征与主方向
    blob_kpts1, corner_kpts1, blob_grad_cell1, corner_grad_cell1, blob_angle_cell1, corner_angle_cell1 = \
        wssf_features(nonlinear1, e1, max1, min1, phase1, sigma_1, ratio, scale_invariance, scale_num)
    blob_kpts2, corner_kpts2, blob_grad_cell2, corner_grad_cell2, blob_angle_cell2, corner_angle_cell2 = \
        wssf_features(nonlinear2, e2, max2, min2, phase2, sigma_1, ratio, scale_invariance, scale_num)

    # 3) GLOH 描述子
    path_block = 10
    blob_desc1 = gloh_descriptors(blob_grad_cell1, blob_angle_cell1, blob_kpts1, path_block, ratio=ratio, sigma_1=sigma_1)
    blob_desc2 = gloh_descriptors(blob_grad_cell2, blob_angle_cell2, blob_kpts2, path_block, ratio=ratio, sigma_1=sigma_1)
    corner_desc1 = gloh_descriptors(corner_grad_cell1, corner_angle_cell1, corner_kpts1, path_block, ratio=ratio, sigma_1=sigma_1)
    corner_desc2 = gloh_descriptors(corner_grad_cell2, corner_angle_cell2, corner_kpts2, path_block, ratio=ratio, sigma_1=sigma_1)

    # 4) 阈值+唯一性匹配并回投到图像坐标
    indexPairs_blob, _ = match_features(blob_desc1['des'], blob_desc2['des'], max_ratio=1, match_threshold=50, unique=True)
    indexPairs_corner, _ = match_features(corner_desc1['des'], corner_desc2['des'], max_ratio=1, match_threshold=50, unique=True)

    mp_1_1, mp_1_2 = back_projection(blob_desc1['locs'][indexPairs_blob[:, 0]-1, :3],
                                     blob_desc2['locs'][indexPairs_blob[:, 1]-1, :3], scale_val) if len(indexPairs_blob)>0 else (np.empty((0,2)), np.empty((0,2)))
    mp_2_1, mp_2_2 = back_projection(corner_desc1['locs'][indexPairs_corner[:, 0]-1, :3],
                                     corner_desc2['locs'][indexPairs_corner[:, 1]-1, :3], scale_val) if len(indexPairs_corner)>0 else (np.empty((0,2)), np.empty((0,2)))

    all_pts1 = np.vstack([mp_1_1, mp_2_1]) if len(mp_1_1)>0 and len(mp_2_1)>0 else (mp_1_1 if len(mp_1_1)>0 else mp_2_1)
    all_pts2 = np.vstack([mp_1_2, mp_2_2]) if len(mp_1_2)>0 and len(mp_2_2)>0 else (mp_1_2 if len(mp_1_2)>0 else mp_2_2)
    if len(all_pts1) < 3 or len(all_pts2) < 3:
        return None, None, np.empty((0,2)), np.empty((0,2))

    # 5) FSC 仿射估计
    H_est, rmse_inlier, inlier_pts1, inlier_pts2 = fsc(all_pts1, all_pts2, change_form='affine', error_t=3.0)
    return H_est, rmse_inlier, inlier_pts1, inlier_pts2

def mse_matrix(H_est: np.ndarray, H_gt: np.ndarray) -> float:
    """矩阵元素均方误差。"""
    diff = H_est.astype(np.float64) - H_gt.astype(np.float64)
    return float(np.mean(diff**2))

def uniform_sample_points(h: int, w: int, n_samples: int = 50) -> np.ndarray:
    """在图像上均匀采样 n_samples 个点（网格划分近似均匀）。"""
    cols = int(np.ceil(np.sqrt(n_samples * w / max(h,1))))
    rows = int(np.ceil(n_samples / max(cols,1)))
    xs = np.linspace(0, w-1, max(cols,1))
    ys = np.linspace(0, h-1, max(rows,1))
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
    return P[:n_samples]

def mse_points_on_sar(img2_gray: np.ndarray, H_est: np.ndarray, H_gt: np.ndarray, n_samples: int = 50) -> tuple:
    """在 SAR 图上均匀采样并计算位置 MSE；返回 (MSE, 误差向量, 采样点)。"""
    h, w = img2_gray.shape[:2]
    P = uniform_sample_points(h, w, n_samples)
    H_est_inv = invert_h(H_est)
    H_gt_inv  = invert_h(H_gt)
    P_est = transform_points(P, H_est_inv)
    P_gt  = transform_points(P, H_gt_inv)
    diffs = P_est - P_gt
    mse = float(np.mean(np.sum(diffs**2, axis=1)))
    return mse, diffs, P

def plot_error_distribution(pair_out: Path, P: np.ndarray, diffs: np.ndarray, mse_val: float):
    """绘制采样点投影误差分布图并标注 MSE。"""
    errs = np.sqrt(np.sum(diffs**2, axis=1))
    fig = plt.figure(figsize=(7,5))
    if USE_SEABORN:
        sns.histplot(errs, bins=20, kde=True, color='steelblue')
    else:
        plt.hist(errs, bins=20, color='steelblue', alpha=0.85)
    plt.title(f"Projection Error Distribution (MSE={mse_val:.4f})")
    plt.xlabel("Point-wise error (pixels)")
    plt.ylabel("Count")
    plt.tight_layout()
    out_path = pair_out / "sample_projection_error.png"
    fig.savefig(str(out_path))
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser("Batch run WSSF on Optical-SAR and evaluate vs GT")
    # 默认数据目录指向当前仓库下的 Optical-SAR
    parser.add_argument("--data_dir", default=str(Path(__file__).resolve().parents[1] / "Optical-SAR"))
    parser.add_argument("--out_dir", default=str(Path(__file__).resolve().parent / "results" / "batch"))
    parser.add_argument("--out_csv", default=str(Path(__file__).resolve().parent / "results" / "wssf_batch_results.csv"))
    parser.add_argument("--gt_dir", default="1to2", choices=["1to2","2to1"], help="GT 矩阵方向：1->2 或 2->1")
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 对 (0=全部)")
    args = parser.parse_args()

    # 输出目录与日志
    root = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "batch_log.txt"
    logging.basicConfig(filename=str(log_path), level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    logging.info(f"Start batch on {root}")

    # 递归收集配对
    pairs = collect_pairs_recursive(root)
    if not pairs:
        # 尝试常见的相对路径修正：若从 wssf_python 目录运行，真实数据在上级目录 ../Optical-SAR
        candidate_1 = (Path(__file__).resolve().parents[1] / "Optical-SAR").expanduser().resolve()
        candidate_2 = (root.parent / "Optical-SAR").expanduser().resolve()
        tried = []
        for cand in [candidate_1, candidate_2]:
            if cand != root:
                tried.append(str(cand))
                pairs = collect_pairs_recursive(cand)
                if pairs:
                    print(f"[info] fallback to {cand}")
                    logging.info(f"Fallback data_dir: {cand}")
                    root = cand
                    break
        if not pairs:
            print(f"[!] no pairs with GT found in {root}\n    Hint: if you run in wssf_python/, use --data_dir ../Optical-SAR")
            logging.error(f"No pairs with GT found. Tried: {tried}")
            return

    rows = []
    errors = []
    t0 = time()

    for idx, (k, rec) in enumerate(pairs.items()):
        if args.limit and idx >= args.limit:
            break
        img1_path, img2_path, gt_path = rec["img1"], rec["img2"], rec["gt"]
        print(f"\n=== Pair {k}: {img1_path.name} / {img2_path.name} ===")
        logging.info(f"Process pair {k}")

        try:
            # 读图与 GT
            img1 = read_gray(img1_path)
            img2 = read_gray(img2_path)
            H_gt = read_gt_3x3(gt_path)
            if args.gt_dir == "2to1":
                H_gt = invert_h(H_gt)

            # 配准估计
            H_est, rmse_inlier, inlier_pts1, inlier_pts2 = estimate_solution(img1, img2)
            if H_est is None:
                print("[x] not enough matches. skip.")
                logging.warning(f"Pair {k} skipped: not enough matches")
                rows.append([k, img1_path.name, img2_path.name, gt_path.name, 0, "", "", "", str(out_dir / f"pair_{k}")])
                continue

            # 误差度量（矩阵 & 均匀采样点）
            msem = mse_matrix(H_est, H_gt)
            mse_sar, diffs, P = mse_points_on_sar(img2, H_est, H_gt, n_samples=50)
            print(f"[OK] inlier_RMSE={rmse_inlier:.3f} px,  MSE(matrix)={msem:.6f},  MSE(SAR-50pts)={mse_sar:.6f}")
            logging.info(f"Pair {k}: inlier_RMSE={rmse_inlier:.3f}, MSE_matrix={msem:.6f}, MSE_points={mse_sar:.6f}")

            # 输出目录与可视化
            pair_out = out_dir / f"pair_{k}"
            pair_out.mkdir(parents=True, exist_ok=True)
            img1_rgb = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
            img2_rgb = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

            # 特征匹配图（黑底）
            cp_show_match(img1_rgb, img2_rgb, inlier_pts1, inlier_pts2, None, str(pair_out / "matches.png"))

            # 重叠融合（alpha）与棋盘格融合（原逻辑）
            image_fusion_overlapped(img1_rgb, img2_rgb, H_est, str(pair_out / "fusion.png"), alpha=0.5)
            image_fusion(img1_rgb, img2_rgb, H_est, str(pair_out / "fusion.png"))

            # 采样点误差分布图
            plot_error_distribution(pair_out, P, diffs, mse_sar)

            # 在三类可视化上叠加指标文本
            mf_path = str(pair_out / "matches_matched_features.png")
            fu_path = str(pair_out / "fusion_overlapped.png")
            cb_path = str(pair_out / "fusion_chessboard.png")
            for pth in (mf_path, fu_path, cb_path):
                img = cv2.imread(pth)
                if img is not None:
                    txt = f"RMSE(inlier)={rmse_inlier:.3f}  MSE(mat)={msem:.2e}  MSE(pts50)={mse_sar:.2e}"
                    cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
                    cv2.imwrite(pth, img)

            # 写入该对的指标汇总
            (pair_out / "metrics.txt").write_text(
                f"pair_id={k}\n"
                f"inliers={len(inlier_pts1)}\n"
                f"inlier_RMSE_px={rmse_inlier:.6f}\n"
                f"MSE_matrix={msem:.6f}\n"
                f"MSE_points_50={mse_sar:.6f}\n"
            )

            rows.append([k, img1_path.name, img2_path.name, gt_path.name, len(inlier_pts1), f"{rmse_inlier:.6f}", f"{msem:.6f}", f"{mse_sar:.6f}", str(pair_out)])
        except Exception as e:
            msg = f"Pair {k} failed: {e}"
            print(f"[!] {msg}")
            logging.exception(msg)
            errors.append((k, str(e)))
            rows.append([k, img1_path.name, img2_path.name, gt_path.name, 0, "", "", "", str(out_dir / f"pair_{k}")])

    # 写 CSV 汇总
    out_csv = Path(args.out_csv).resolve()
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_id","img1","img2","gt_txt","n_inliers","inlier_rmse_px","mse_matrix","mse_sar50","out_dir"])
        w.writerows(rows)

    # 统计与摘要
    try:
        mat_mses = [float(r[6]) for r in rows if r[6] != ""]
        pt_mses  = [float(r[7]) for r in rows if r[7] != ""]
        summary = (
            f"Total pairs processed: {len(rows)}\n"
            f"Matrix MSE: avg={np.mean(mat_mses):.6f}, max={np.max(mat_mses):.6f}, min={np.min(mat_mses):.6f}\n"
            f"Points MSE (50): avg={np.mean(pt_mses):.6f}, max={np.max(pt_mses):.6f}, min={np.min(pt_mses):.6f}\n"
            f"Errors: {len(errors)}\n"
        )
    except Exception:
        summary = (
            f"Total pairs processed: {len(rows)}\n"
            f"Errors: {len(errors)}\n"
        )
    (out_dir / "summary.txt").write_text(summary)
    if errors:
        with open(out_dir / "errors.txt", "w", encoding="utf-8") as ef:
            for k, emsg in errors:
                ef.write(f"pair_id={k}\t{emsg}\n")

    print(f"\nDone. {len(rows)} pairs. Time: {time()-t0:.1f}s")
    print(f"Results -> {out_csv}")

if __name__ == "__main__":
    main()
