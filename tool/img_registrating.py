import os
import cv2
import numpy as np
import SimpleITK as sitk
import tkinter as tk
from tkinter import filedialog


def remove_scale_from_affine(mat):
    R = mat[:, :2]
    t = mat[:, 2]
    U, _, Vt = np.linalg.svd(R)
    R_no_scale = U @ Vt
    mat_no_scale = np.hstack([R_no_scale, t.reshape(2, 1)])
    return mat_no_scale

def _create_feature_detector():
    """尝试创建特征检测器，按 SIFT → AKAZE → ORB 优先级回退。"""
    detectors = []
    try:
        detectors.append(("SIFT", cv2.SIFT_create()))
    except Exception:
        pass
    try:
        detectors.append(("AKAZE", cv2.AKAZE_create()))
    except Exception:
        pass
    try:
        detectors.append(("ORB", cv2.ORB_create(nfeatures=2000)))
    except Exception:
        pass
    if not detectors:
        raise RuntimeError("无可用的特征检测器（SIFT/AKAZE/ORB 均不可用）")
    return detectors


def rigid_registration(
        fixed_img,
        moving_img,
        n_matches=200,
        scale_factor=0.2,
        only_rigid=True
):
    if fixed_img.ndim == 2:
        fixed_gray = cv2.resize(fixed_img, (int(fixed_img.shape[1] * scale_factor), int(fixed_img.shape[0] * scale_factor)))
        moving_gray = cv2.resize(moving_img, (int(moving_img.shape[1] * scale_factor), int(moving_img.shape[0] * scale_factor)))
    elif fixed_img.ndim == 3:
        fixed_gray = cv2.cvtColor(cv2.resize(fixed_img, (int(fixed_img.shape[1] * scale_factor), int(fixed_img.shape[0] * scale_factor))), cv2.COLOR_BGR2GRAY)
        moving_gray = cv2.cvtColor(cv2.resize(moving_img, (int(moving_img.shape[1] * scale_factor), int(moving_img.shape[0] * scale_factor))), cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("不支持的图像维度")

    detectors = _create_feature_detector()

    kp1, des1, kp2, des2 = None, None, None, None
    used_detector = None
    norm_type = cv2.NORM_L2

    for name, detector in detectors:
        try:
            kp1, des1 = detector.detectAndCompute(fixed_gray, None)
            kp2, des2 = detector.detectAndCompute(moving_gray, None)
            if des1 is not None and des2 is not None and len(kp1) >= 5 and len(kp2) >= 5:
                used_detector = name
                if name == "ORB":
                    norm_type = cv2.NORM_HAMMING
                else:
                    norm_type = cv2.NORM_L2
                break
        except Exception:
            continue

    if used_detector is None:
        print(f"  警告: 所有特征检测器均未能提取足够特征点")
        return None

    print(f"  使用特征检测器: {used_detector} (kp1={len(kp1)}, kp2={len(kp2)})")

    # 特征匹配
    if used_detector == "ORB":
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:n_matches]
    if len(matches) < 5:
        print(f"  警告: 匹配点不足 ({len(matches)} < 5)")
        return None
    print(f"  匹配点数量: {len(matches)}")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    mat_low, mask = cv2.estimateAffinePartial2D(pts2, pts1)
    if only_rigid:
        mat_low = remove_scale_from_affine(mat_low)
    if mat_low is None:
        return None
    # 修正仿射矩阵到原图尺度
    s = np.array([[1/scale_factor, 0, 0], [0, 1/scale_factor, 0], [0, 0, 1]])
    s_ = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])
    mat_full = s @ np.vstack([mat_low, [0,0,1]]) @ s_
    mat_full = mat_full[:2, :]

    if fixed_img.ndim == 2:
        registered = cv2.warpAffine(moving_img, mat_full, (fixed_img.shape[1], fixed_img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    elif fixed_img.ndim == 3:
        registered = cv2.warpAffine(moving_img, mat_full, (fixed_img.shape[1], fixed_img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=[255, 255, 255])
    else:
        raise ValueError("不支持的图像维度")
    return registered

def elastic_registration(
    fixed,
    moving,
    mesh_size=4,
    shrink_factors=None,
    smoothing_sigmas=None,
    optimizer_iterations=50,
    optimizer_tol=1e-5,
    optimizer_bounds=(-100, 100)
):
    if shrink_factors is None:
        shrink_factors = [5]
    if smoothing_sigmas is None:
        smoothing_sigmas = [1]
    if isinstance(mesh_size, int):
        mesh_size = [mesh_size] * 2
    if fixed.ndim == 2:
        fixed_sitk = sitk.GetImageFromArray(fixed)
        moving_sitk = sitk.GetImageFromArray(moving)
        fixed_sitk = sitk.Cast(fixed_sitk, sitk.sitkFloat32)
        moving_sitk = sitk.Cast(moving_sitk, sitk.sitkFloat32)
        tx = sitk.BSplineTransformInitializer(fixed_sitk, mesh_size)
        R = sitk.ImageRegistrationMethod()
        lower_bound, upper_bound = optimizer_bounds
        R.SetMetricAsMeanSquares()
        R.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=optimizer_tol, numberOfIterations=optimizer_iterations, upperBound=upper_bound, lowerBound=lower_bound)
        R.SetInitialTransform(tx, True)
        R.SetInterpolator(sitk.sitkLinear)
        R.SetShrinkFactorsPerLevel(shrink_factors)
        R.SetSmoothingSigmasPerLevel(smoothing_sigmas)
        outTx = R.Execute(fixed_sitk, moving_sitk)
        moved = sitk.Resample(
            moving_sitk,
            fixed_sitk,
            outTx,
            sitk.sitkLinear,
            0,
            sitk.sitkFloat32
        )
        elastic_registered = sitk.GetArrayFromImage(moved).astype(np.uint8)
    elif fixed.ndim == 3:
        fixed_gray = cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY)
        moving_gray = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY)
        fixed_sitk = sitk.GetImageFromArray(fixed_gray)
        moving_sitk = sitk.GetImageFromArray(moving_gray)
        fixed_sitk = sitk.Cast(fixed_sitk, sitk.sitkFloat32)
        moving_sitk = sitk.Cast(moving_sitk, sitk.sitkFloat32)
        tx = sitk.BSplineTransformInitializer(fixed_sitk, mesh_size)
        R = sitk.ImageRegistrationMethod()
        lower_bound, upper_bound = optimizer_bounds
        R.SetMetricAsMeanSquares()
        R.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=optimizer_tol, numberOfIterations=optimizer_iterations, upperBound=upper_bound, lowerBound=lower_bound)
        R.SetInitialTransform(tx, True)
        R.SetInterpolator(sitk.sitkLinear)
        R.SetShrinkFactorsPerLevel(shrink_factors)
        R.SetSmoothingSigmasPerLevel(smoothing_sigmas)
        outTx = R.Execute(fixed_sitk, moving_sitk)
        moved_channels = []
        for c in range(3):
            moved = sitk.Resample(
                sitk.GetImageFromArray(moving[:,:,c].astype(np.float32)),
                fixed_sitk,
                outTx,
                sitk.sitkLinear,
                255,
                sitk.sitkFloat32
            )
            moved_channels.append(sitk.GetArrayFromImage(moved).astype(np.uint8))
        elastic_registered = cv2.merge(moved_channels)
    else:
        raise ValueError("不支持的图像维度")
    return elastic_registered

def batch_register_images_anchor_subseq(
        image_paths,
        output_folder,
        anchor_interval=10,
        mode='刚性',
        only_rigid=False,
        scale_factor=0.2,
        shrink_factors=None,
        smoothing_sigmas=None,
        n_matches=200,
        mesh_size=4,
        optimizer_iterations=50,
        optimizer_tol=1e-5,
        optimizer_bounds=(-100, 100),
        progress_callback=None
):
    if shrink_factors is None:
        shrink_factors = [5]
    if smoothing_sigmas is None:
        smoothing_sigmas = [1]
    os.makedirs(output_folder, exist_ok=True)
    if len(image_paths) == 0:
        return
    N = len(image_paths)
    anchor_indices = list(range(0, N, anchor_interval))
    if anchor_indices[-1] != N-1:
        anchor_indices.append(N-1)

    total_steps = len(anchor_indices)
    for i in range(len(anchor_indices)-1):
        total_steps += (anchor_indices[i+1] - anchor_indices[i] - 1)

    finished_steps = 0
    anchor_results = {}
    for idx, anchor_idx in enumerate(anchor_indices):
        img_path = image_paths[anchor_idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            finished_steps += 1
            if progress_callback:
                percent = int(finished_steps / total_steps * 100)
                progress_callback(percent)
            continue
        if idx == 0:
            anchor_results[anchor_idx] = img
            cv2.imwrite(os.path.join(output_folder, os.path.basename(img_path)), img)  # type: ignore
        else:
            prev_idx = anchor_indices[idx-1]
            prev_img = anchor_results[prev_idx]
            rigid_result = rigid_registration(
                prev_img, img,
                n_matches=n_matches,
                scale_factor=scale_factor,
                only_rigid=only_rigid
            )
            if rigid_result is None:
                finished_steps += 1
                if progress_callback:
                    percent = int(finished_steps / total_steps * 100)
                    progress_callback(percent)
                continue
            if mode == '刚性':
                result = rigid_result
            elif mode == '弹性':
                result = elastic_registration(
                    prev_img, img,
                    mesh_size=mesh_size,
                    shrink_factors=shrink_factors,
                    smoothing_sigmas=smoothing_sigmas,
                    optimizer_iterations=optimizer_iterations,
                    optimizer_tol=optimizer_tol,
                    optimizer_bounds=optimizer_bounds
                )
            elif mode == '刚性+弹性':
                result = elastic_registration(
                    prev_img, rigid_result,
                    mesh_size=mesh_size,
                    shrink_factors=shrink_factors,
                    smoothing_sigmas=smoothing_sigmas,
                    optimizer_iterations=optimizer_iterations,
                    optimizer_tol=optimizer_tol,
                    optimizer_bounds=optimizer_bounds
                )
            else:
                raise ValueError('mode参数应为"rigid"或"rigid+elastic"')
            anchor_results[anchor_idx] = result
            cv2.imwrite(os.path.join(output_folder, os.path.basename(img_path)), result)  # type: ignore
        finished_steps += 1
        if progress_callback:
            percent = int(finished_steps / total_steps * 100)
            progress_callback(percent)

    for i in range(len(anchor_indices)-1):
        anchor_idx = anchor_indices[i]
        next_anchor_idx = anchor_indices[i+1]
        anchor_img = anchor_results[anchor_idx]
        for j in range(anchor_idx+1, next_anchor_idx):
            img_path = image_paths[j]
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                finished_steps += 1
                if progress_callback:
                    percent = int(finished_steps / total_steps * 100)
                    progress_callback(percent)
                continue
            rigid_result = rigid_registration(
                anchor_img, img,
                n_matches=n_matches,
                scale_factor=scale_factor,
                only_rigid=only_rigid
            )
            if rigid_result is None:
                finished_steps += 1
                if progress_callback:
                    percent = int(finished_steps / total_steps * 100)
                    progress_callback(percent)
                continue
            if mode == '刚性':
                result = rigid_result
            elif mode == '弹性':
                result = elastic_registration(
                    prev_img, img,
                    mesh_size=mesh_size,
                    shrink_factors=shrink_factors,
                    smoothing_sigmas=smoothing_sigmas,
                    optimizer_iterations=optimizer_iterations,
                    optimizer_tol=optimizer_tol,
                    optimizer_bounds=optimizer_bounds
                )
            elif mode == '刚性+弹性':
                result = elastic_registration(
                    prev_img, rigid_result,
                    mesh_size=mesh_size,
                    shrink_factors=shrink_factors,
                    smoothing_sigmas=smoothing_sigmas,
                    optimizer_iterations=optimizer_iterations,
                    optimizer_tol=optimizer_tol,
                    optimizer_bounds=optimizer_bounds
                )
            else:
                raise ValueError('mode参数应为"rigid"或"rigid+elastic"')
            cv2.imwrite(os.path.join(output_folder, os.path.basename(img_path)), result)  # type: ignore
            finished_steps += 1
            if progress_callback:
                percent = int(finished_steps / total_steps * 100)
                progress_callback(percent)

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    # 0. 选择配准方式
    style_var = tk.StringVar(value="一对一")
    style_win = tk.Toplevel(root)
    style_win.title("选择配准方式")
    style_win.resizable(False, False)
    tk.Label(style_win, text="请选择配准方式：", font=("", 12)).pack(padx=20, pady=(15, 5))
    for text, val in [("一对一：所有图像配准到同一参考图像", "一对一"),
                       ("序列配准：按顺序依次配准到前一张", "序列")]:
        tk.Radiobutton(style_win, text=text, variable=style_var, value=val, font=("", 11)).pack(anchor="w", padx=20, pady=3)
    tk.Button(style_win, text="下一步", command=style_win.destroy, width=12).pack(pady=(10, 15))
    style_win.grab_set()
    root.wait_window(style_win)

    reg_style = style_var.get()

    if reg_style == "一对一":
        # ===== 一对一配准 =====
        # 1. 选择参考图像
        ref_path = filedialog.askopenfilename(
            title="选择参考图像（固定图像，配准目标）",
            filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                       ("所有文件", "*.*")],
        )
        if not ref_path:
            print("未选择参考图像，退出。")
            root.destroy()
            exit()

        # 2. 选择待配准图像（可多选）
        tgt_paths = filedialog.askopenfilenames(
            title="选择待配准图像（可多选，将逐一配准到参考图像）",
            filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                       ("所有文件", "*.*")],
        )
        if not tgt_paths:
            print("未选择待配准图像，退出。")
            root.destroy()
            exit()
    else:
        # ===== 序列配准 =====
        # 选择序列图像（可多选，按文件名排序作为序列顺序）
        seq_paths = filedialog.askopenfilenames(
            title="选择序列图像（可多选，按文件名自然排序确定顺序）",
            filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                       ("所有文件", "*.*")],
        )
        if not seq_paths:
            print("未选择图像，退出。")
            root.destroy()
            exit()
        from natsort import natsorted
        seq_paths = natsorted(list(seq_paths))
        ref_path = None  # 序列模式下不使用参考图像

    # 3. 选择输出文件夹
    output_dir = filedialog.askdirectory(title="选择输出文件夹")
    if not output_dir:
        print("未选择输出文件夹，退出。")
        root.destroy()
        exit()

    # 4. 选择配准模式（算法）
    mode_var = tk.StringVar(value="刚性")
    mode_win = tk.Toplevel(root)
    mode_win.title("选择配准算法")
    mode_win.resizable(False, False)
    tk.Label(mode_win, text="请选择配准算法：", font=("", 12)).pack(padx=20, pady=(15, 5))
    for text, val in [("刚性 (Rigid)", "刚性"), ("弹性 (Elastic)", "弹性"), ("刚性+弹性 (Rigid+Elastic)", "刚性+弹性")]:
        tk.Radiobutton(mode_win, text=text, variable=mode_var, value=val, font=("", 11)).pack(anchor="w", padx=30, pady=2)
    tk.Button(mode_win, text="开始配准", command=mode_win.destroy, width=12).pack(pady=(10, 15))
    mode_win.grab_set()
    root.wait_window(mode_win)

    mode = mode_var.get()
    root.destroy()

    os.makedirs(output_dir, exist_ok=True)
    print(f"配准方式: {reg_style}")
    print(f"配准算法: {mode}")
    print(f"输出目录: {output_dir}")

    count = 0

    if reg_style == "一对一":
        ref_img = cv2.imread(ref_path, cv2.IMREAD_COLOR)
        if ref_img is None:
            raise FileNotFoundError(f"读取参考图像失败: {ref_path}")
        print(f"参考图像: {ref_path}")

        for tgt_path in tgt_paths:
            try:
                tgt_img = cv2.imread(tgt_path, cv2.IMREAD_COLOR)
                if tgt_img is None:
                    print(f"读取失败，跳过: {tgt_path}")
                    continue

                if mode == "刚性":
                    result = rigid_registration(ref_img, tgt_img, n_matches=200, scale_factor=0.1, only_rigid=False)
                    if result is None:
                        print(f"刚性配准失败（特征点不足），跳过: {tgt_path}")
                        continue
                elif mode == "弹性":
                    result = elastic_registration(ref_img, tgt_img, mesh_size=8, shrink_factors=[10], smoothing_sigmas=[2],
                                                  optimizer_iterations=50, optimizer_tol=1e-5, optimizer_bounds=(-200, 200))
                elif mode == "刚性+弹性":
                    rigid_result = rigid_registration(ref_img, tgt_img, n_matches=200, scale_factor=0.1, only_rigid=True)
                    if rigid_result is None:
                        print(f"刚性配准失败（特征点不足），跳过: {tgt_path}")
                        continue
                    result = elastic_registration(ref_img, rigid_result, mesh_size=8, shrink_factors=[10], smoothing_sigmas=[2],
                                                  optimizer_iterations=50, optimizer_tol=1e-5, optimizer_bounds=(-200, 200))

                out_path = os.path.join(output_dir, os.path.basename(tgt_path))
                ok = cv2.imwrite(out_path, result)
                if not ok:
                    print(f"保存失败: {out_path}")
                    continue
                print(f"[{count + 1}/{len(tgt_paths)}] Saved: {out_path}")
                count += 1
            except Exception as e:
                print(f"处理失败 [{tgt_path}]: {e}")

        print(f"完成！共配准 {count}/{len(tgt_paths)} 张图像。")

    else:
        # ===== 序列配准 =====
        if len(seq_paths) < 2:
            print("至少需要 2 张图像进行序列配准，退出。")
            exit()

        # 读取第一张（锚点）
        anchor_img = cv2.imread(seq_paths[0], cv2.IMREAD_COLOR)
        if anchor_img is None:
            raise FileNotFoundError(f"读取失败: {seq_paths[0]}")

        # 保存第一张（不配准）
        first_out = os.path.join(output_dir, os.path.basename(seq_paths[0]))
        cv2.imwrite(first_out, anchor_img)
        print(f"[1/{len(seq_paths)}] Anchor: {first_out}")

        prev_img = anchor_img
        count = 1

        for i, tgt_path in enumerate(seq_paths[1:], 2):
            try:
                tgt_img = cv2.imread(tgt_path, cv2.IMREAD_COLOR)
                if tgt_img is None:
                    print(f"读取失败，跳过: {tgt_path}")
                    continue

                if mode == "刚性":
                    result = rigid_registration(prev_img, tgt_img, n_matches=200, scale_factor=0.1, only_rigid=False)
                    if result is None:
                        print(f"刚性配准失败（特征点不足），跳过: {tgt_path}")
                        continue
                elif mode == "弹性":
                    result = elastic_registration(prev_img, tgt_img, mesh_size=8, shrink_factors=[10], smoothing_sigmas=[2],
                                                  optimizer_iterations=50, optimizer_tol=1e-5, optimizer_bounds=(-200, 200))
                elif mode == "刚性+弹性":
                    rigid_result = rigid_registration(prev_img, tgt_img, n_matches=200, scale_factor=0.1, only_rigid=True)
                    if rigid_result is None:
                        print(f"刚性配准失败（特征点不足），跳过: {tgt_path}")
                        continue
                    result = elastic_registration(prev_img, rigid_result, mesh_size=8, shrink_factors=[10], smoothing_sigmas=[2],
                                                  optimizer_iterations=50, optimizer_tol=1e-5, optimizer_bounds=(-200, 200))

                # 将配准结果作为下一张的参考
                prev_img = result

                out_path = os.path.join(output_dir, os.path.basename(tgt_path))
                ok = cv2.imwrite(out_path, result)
                if not ok:
                    print(f"保存失败: {out_path}")
                    continue
                print(f"[{i}/{len(seq_paths)}] Saved: {out_path}")
                count += 1
            except Exception as e:
                print(f"处理失败 [{tgt_path}]: {e}")

        print(f"完成！共配准 {count}/{len(seq_paths)} 张图像。")