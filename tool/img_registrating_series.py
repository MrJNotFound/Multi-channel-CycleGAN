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


def rigid_registration_matrix(
        fixed_img,
        moving_img,
        n_matches=200,
        scale_factor=0.2,
        only_rigid=True
):
    """仅计算刚性配准的仿射矩阵，不执行 warp。

    Returns:
        mat_full: 2×3 仿射矩阵，或 None（特征点不足）
        fixed_shape: (w, h) 固定图像尺寸
    """
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
        return None, None

    print(f"  使用特征检测器: {used_detector} (kp1={len(kp1)}, kp2={len(kp2)})")

    if used_detector == "ORB":
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:n_matches]
    if len(matches) < 5:
        print(f"  警告: 匹配点不足 ({len(matches)} < 5)")
        return None, None
    print(f"  匹配点数量: {len(matches)}")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    mat_low, mask = cv2.estimateAffinePartial2D(pts2, pts1)
    if only_rigid:
        mat_low = remove_scale_from_affine(mat_low)
    if mat_low is None:
        return None, None
    s = np.array([[1/scale_factor, 0, 0], [0, 1/scale_factor, 0], [0, 0, 1]])
    s_ = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])
    mat_full = s @ np.vstack([mat_low, [0, 0, 1]]) @ s_
    mat_full = mat_full[:2, :]
    fixed_shape = (fixed_img.shape[1], fixed_img.shape[0])
    return mat_full, fixed_shape


def rigid_registration_apply(moving_img, mat_full, output_size):
    """使用预计算的仿射矩阵对图像进行 warp。

    Args:
        moving_img: 待配准图像
        mat_full: 2×3 仿射矩阵
        output_size: (w, h) 输出尺寸
    """
    h, w = output_size[1], output_size[0]
    if moving_img.ndim == 2:
        return cv2.warpAffine(moving_img, mat_full, (w, h),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    elif moving_img.ndim == 3:
        return cv2.warpAffine(moving_img, mat_full, (w, h),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=[255, 255, 255])
    else:
        raise ValueError("不支持的图像维度")


def elastic_registration_transform(
        fixed,
        moving,
        mesh_size=4,
        shrink_factors=None,
        smoothing_sigmas=None,
        optimizer_iterations=50,
        optimizer_tol=1e-5,
        optimizer_bounds=(-100, 100),
        scale_factor=1.0,
):
    """仅计算弹性配准的 BSplineTransform，不执行重采样。

    Args:
        scale_factor: < 1.0 时先将图像缩小再配准以节省内存，
                      返回的 transform 在缩小后的坐标空间。

    Returns:
        outTx: SimpleITK BSplineTransform
        fixed_sitk: 固定图像的 SimpleITK 对象（用于后续 resample）
        orig_shape: 原始固定图像尺寸 (h, w)，供 apply 时上采样回原尺寸
    """
    if shrink_factors is None:
        shrink_factors = [5]
    if smoothing_sigmas is None:
        smoothing_sigmas = [1]
    if isinstance(mesh_size, int):
        mesh_size = [mesh_size] * 2

    orig_h, orig_w = fixed.shape[:2]
    if scale_factor < 1.0:
        new_w = max(1, int(orig_w * scale_factor))
        new_h = max(1, int(orig_h * scale_factor))
        if fixed.ndim == 3:
            fixed_small = cv2.resize(fixed, (new_w, new_h), interpolation=cv2.INTER_AREA)
            moving_small = cv2.resize(moving, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            fixed_small = cv2.resize(fixed, (new_w, new_h), interpolation=cv2.INTER_AREA)
            moving_small = cv2.resize(moving, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        fixed_small, moving_small = fixed, moving

    if fixed_small.ndim == 2:
        fixed_sitk = sitk.GetImageFromArray(fixed_small)
        moving_sitk = sitk.GetImageFromArray(moving_small)
    elif fixed_small.ndim == 3:
        fixed_gray = cv2.cvtColor(fixed_small, cv2.COLOR_BGR2GRAY)
        moving_gray = cv2.cvtColor(moving_small, cv2.COLOR_BGR2GRAY)
        fixed_sitk = sitk.GetImageFromArray(fixed_gray)
        moving_sitk = sitk.GetImageFromArray(moving_gray)
    else:
        raise ValueError("不支持的图像维度")

    fixed_sitk = sitk.Cast(fixed_sitk, sitk.sitkFloat32)
    moving_sitk = sitk.Cast(moving_sitk, sitk.sitkFloat32)

    tx = sitk.BSplineTransformInitializer(fixed_sitk, mesh_size)
    R = sitk.ImageRegistrationMethod()
    lower_bound, upper_bound = optimizer_bounds
    R.SetMetricAsMeanSquares()
    R.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=optimizer_tol,
                           numberOfIterations=optimizer_iterations,
                           upperBound=upper_bound, lowerBound=lower_bound)
    R.SetInitialTransform(tx, True)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetShrinkFactorsPerLevel(shrink_factors)
    R.SetSmoothingSigmasPerLevel(smoothing_sigmas)
    outTx = R.Execute(fixed_sitk, moving_sitk)
    return outTx, fixed_sitk, (orig_h, orig_w)


def elastic_registration_apply(moving_img, fixed_sitk, outTx, output_size=None):
    """使用预计算的 BSplineTransform 对图像进行重采样。

    Args:
        moving_img: 待配准图像（原始尺寸）
        fixed_sitk: 固定图像的 SimpleITK 对象（用于确定输出空间）
        outTx: 预计算的 BSplineTransform
        output_size: (w, h) 若提供，输出上采样到该尺寸；
                     若 transform 是在缩小图上算的，传入原图尺寸即可还原
    """
    fw, fh = fixed_sitk.GetSize()
    mh, mw = moving_img.shape[:2]
    if (mw, mh) != (fw, fh):
        moving_small = cv2.resize(moving_img, (fw, fh), interpolation=cv2.INTER_AREA)
    else:
        moving_small = moving_img

    if moving_small.ndim == 2:
        moving_sitk = sitk.GetImageFromArray(moving_small)
        moving_sitk = sitk.Cast(moving_sitk, sitk.sitkFloat32)
        moved = sitk.Resample(moving_sitk, fixed_sitk, outTx,
                              sitk.sitkLinear, 0, sitk.sitkFloat32)
        result = sitk.GetArrayFromImage(moved).astype(np.uint8)
    elif moving_small.ndim == 3:
        moved_channels = []
        for c in range(3):
            moving_sitk = sitk.GetImageFromArray(moving_small[:, :, c].astype(np.float32))
            moved = sitk.Resample(moving_sitk, fixed_sitk, outTx,
                                  sitk.sitkLinear, 255, sitk.sitkFloat32)
            moved_channels.append(sitk.GetArrayFromImage(moved).astype(np.uint8))
        result = cv2.merge(moved_channels)
    else:
        raise ValueError("不支持的图像维度")

    if output_size is not None:
        ow, oh = output_size
        if (result.shape[1], result.shape[0]) != (ow, oh):
            result = cv2.resize(result, (ow, oh), interpolation=cv2.INTER_LANCZOS4)
    return result


def elastic_registration(
    fixed,
    moving,
    mesh_size=4,
    shrink_factors=None,
    smoothing_sigmas=None,
    optimizer_iterations=50,
    optimizer_tol=1e-5,
    optimizer_bounds=(-100, 100),
    scale_factor=1.0,
):
    """弹性配准（一步完成：缩放→计算变换→应用→还原尺寸）。

    Args:
        scale_factor: < 1.0 时先缩小再配准以节省内存，结果自动上采样回原尺寸。
    """
    if shrink_factors is None:
        shrink_factors = [5]
    if smoothing_sigmas is None:
        smoothing_sigmas = [1]
    if isinstance(mesh_size, int):
        mesh_size = [mesh_size] * 2

    orig_h, orig_w = fixed.shape[:2]
    if scale_factor < 1.0:
        new_w = max(1, int(orig_w * scale_factor))
        new_h = max(1, int(orig_h * scale_factor))
        fixed_small = cv2.resize(fixed, (new_w, new_h), interpolation=cv2.INTER_AREA)
        moving_small = cv2.resize(moving, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        fixed_small, moving_small = fixed, moving

    if fixed_small.ndim == 2:
        fixed_sitk = sitk.GetImageFromArray(fixed_small)
        moving_sitk = sitk.GetImageFromArray(moving_small)
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
    elif fixed_small.ndim == 3:
        fixed_gray = cv2.cvtColor(fixed_small, cv2.COLOR_BGR2GRAY)
        moving_gray = cv2.cvtColor(moving_small, cv2.COLOR_BGR2GRAY)
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
                sitk.GetImageFromArray(moving_small[:,:,c].astype(np.float32)),
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

    if scale_factor < 1.0:
        elastic_registered = cv2.resize(
            elastic_registered, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
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
    from natsort import natsorted

    root = tk.Tk()
    root.withdraw()

    # 0. 选择配准方式
    style_var = tk.StringVar(value="一对一")
    style_win = tk.Toplevel(root)
    style_win.title("选择配准方式")
    style_win.resizable(False, False)
    tk.Label(style_win, text="请选择配准方式：", font=("", 12)).pack(padx=20, pady=(15, 5))
    for text, val in [
        ("一对一：所有图像配准到同一参考图像", "一对一"),
        ("序列配准：按顺序依次配准到前一张", "序列"),
        ("锚点配准：按间隔选定锚点，非锚点配准到最近锚点", "锚点"),
    ]:
        tk.Radiobutton(style_win, text=text, variable=style_var, value=val, font=("", 11)).pack(anchor="w", padx=20, pady=3)
    tk.Button(style_win, text="下一步", command=style_win.destroy, width=12).pack(pady=(10, 15))
    style_win.grab_set()
    root.wait_window(style_win)

    reg_style = style_var.get()

    # ===== 选择图像 =====
    ref_path = None
    tgt_paths = None
    seq_paths = None

    if reg_style == "一对一":
        ref_path = filedialog.askopenfilename(
            title="选择参考图像（固定图像，配准目标）",
            filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("所有文件", "*.*")],
        )
        if not ref_path:
            print("未选择参考图像，退出。")
            root.destroy()
            exit()

        tgt_paths = filedialog.askopenfilenames(
            title="选择待配准图像（可多选，将逐一配准到参考图像）",
            filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("所有文件", "*.*")],
        )
        if not tgt_paths:
            print("未选择待配准图像，退出。")
            root.destroy()
            exit()
    else:
        # 序列 / 锚点：选择多张图像，按文件名排序
        title = "选择序列图像（可多选，按文件名自然排序确定顺序）" if reg_style == "序列" else "选择序列图像（可多选，按文件名自然排序）"
        seq_paths = filedialog.askopenfilenames(
            title=title,
            filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("所有文件", "*.*")],
        )
        if not seq_paths:
            print("未选择图像，退出。")
            root.destroy()
            exit()
        seq_paths = natsorted(list(seq_paths))

    # 输出目录
    output_dir = filedialog.askdirectory(title="选择输出文件夹")
    if not output_dir:
        print("未选择输出文件夹，退出。")
        root.destroy()
        exit()

    # ===== 参数设置窗口 =====
    mode_var = tk.StringVar(value="刚性")
    anchor_var = tk.IntVar(value=10)
    sf_var = tk.DoubleVar(value=0.1)
    nm_var = tk.IntVar(value=200)
    mesh_var = tk.IntVar(value=8)
    sh_var = tk.StringVar(value="10")
    sm_var = tk.StringVar(value="2")
    oi_var = tk.IntVar(value=50)
    ot_var = tk.StringVar(value="1e-5")
    ob_lo_var = tk.IntVar(value=-200)
    ob_hi_var = tk.IntVar(value=200)
    only_rigid_var = tk.BooleanVar(value=False)

    param_win = tk.Toplevel(root)
    param_win.title("配准参数设置")
    param_win.resizable(False, False)
    r = 0

    tk.Label(param_win, text="配准算法：", font=("", 11)).grid(row=r, column=0, sticky="e", padx=(15, 2), pady=(15, 3))
    mode_frame = tk.Frame(param_win)
    mode_frame.grid(row=r, column=1, sticky="w")
    for text, val in [("刚性", "刚性"), ("弹性", "弹性"), ("刚性+弹性", "刚性+弹性")]:
        tk.Radiobutton(mode_frame, text=text, variable=mode_var, value=val, font=("", 10)).pack(side="left", padx=2)
    r += 1

    if reg_style == "锚点":
        tk.Label(param_win, text="锚点间隔：", font=("", 11)).grid(row=r, column=0, sticky="e", padx=(15, 2), pady=3)
        tk.Entry(param_win, textvariable=anchor_var, width=6).grid(row=r, column=1, sticky="w")
        tk.Label(param_win, text="(每隔 N 张选一个锚点)", font=("", 9)).grid(row=r, column=2, sticky="w")
        r += 1

    tk.Label(param_win, text="缩放因子：", font=("", 11)).grid(row=r, column=0, sticky="e", padx=(15, 2), pady=3)
    tk.Entry(param_win, textvariable=sf_var, width=6).grid(row=r, column=1, sticky="w")
    tk.Label(param_win, text="(SIFT 匹配用的降采样比例)", font=("", 9)).grid(row=r, column=2, sticky="w")
    r += 1

    tk.Label(param_win, text="SIFT 匹配数：", font=("", 11)).grid(row=r, column=0, sticky="e", padx=(15, 2), pady=3)
    tk.Entry(param_win, textvariable=nm_var, width=6).grid(row=r, column=1, sticky="w")
    r += 1

    tk.Checkbutton(param_win, text="强制纯刚性（去除缩放分量）", variable=only_rigid_var, font=("", 10)).grid(row=r, column=0, columnspan=3, sticky="w", padx=15, pady=3)
    r += 1

    # 弹性参数（mode=弹性 / 刚性+弹性 时生效）
    tk.Label(param_win, text="--- 弹性配准参数 ---", font=("", 10, "bold")).grid(row=r, column=0, columnspan=3, sticky="w", padx=15, pady=(8, 2))
    r += 1

    tk.Label(param_win, text="B-spline 网格：", font=("", 11)).grid(row=r, column=0, sticky="e", padx=(15, 2), pady=3)
    tk.Entry(param_win, textvariable=mesh_var, width=6).grid(row=r, column=1, sticky="w")
    r += 1

    tk.Label(param_win, text="Shrink Factors：", font=("", 11)).grid(row=r, column=0, sticky="e", padx=(15, 2), pady=3)
    tk.Entry(param_win, textvariable=sh_var, width=6).grid(row=r, column=1, sticky="w")
    tk.Label(param_win, text="(逗号分隔，如 10,5)", font=("", 9)).grid(row=r, column=2, sticky="w")
    r += 1

    tk.Label(param_win, text="Smoothing Sigmas：", font=("", 11)).grid(row=r, column=0, sticky="e", padx=(15, 2), pady=3)
    tk.Entry(param_win, textvariable=sm_var, width=6).grid(row=r, column=1, sticky="w")
    tk.Label(param_win, text="(逗号分隔，如 2,1)", font=("", 9)).grid(row=r, column=2, sticky="w")
    r += 1

    tk.Label(param_win, text="优化迭代次数：", font=("", 11)).grid(row=r, column=0, sticky="e", padx=(15, 2), pady=3)
    tk.Entry(param_win, textvariable=oi_var, width=6).grid(row=r, column=1, sticky="w")
    r += 1

    tk.Label(param_win, text="优化容差：", font=("", 11)).grid(row=r, column=0, sticky="e", padx=(15, 2), pady=3)
    tk.Entry(param_win, textvariable=ot_var, width=6).grid(row=r, column=1, sticky="w")
    r += 1

    tk.Label(param_win, text="边界下/上限：", font=("", 11)).grid(row=r, column=0, sticky="e", padx=(15, 2), pady=3)
    frm = tk.Frame(param_win)
    frm.grid(row=r, column=1, sticky="w")
    tk.Entry(frm, textvariable=ob_lo_var, width=5).pack(side="left")
    tk.Label(frm, text=" ~ ", font=("", 10)).pack(side="left")
    tk.Entry(frm, textvariable=ob_hi_var, width=5).pack(side="left")
    r += 1

    tk.Button(param_win, text="开始配准", command=param_win.destroy, width=12).grid(row=r, column=0, columnspan=3, pady=(15, 15))

    # 参数收集（带输入验证，循环直到正确或取消）
    mode = anchor_interval = scale_factor = n_matches = None
    only_rigid = mesh_size = optimizer_iterations = optimizer_tol = None
    shrink_factors = smoothing_sigmas = optimizer_bounds = None

    while True:
        param_win.grab_set()
        root.wait_window(param_win)
        try:
            mode = mode_var.get()
            anchor_interval = anchor_var.get()
            scale_factor = sf_var.get()
            n_matches = nm_var.get()
            only_rigid = only_rigid_var.get()
            mesh_size = mesh_var.get()
            # 容错：替换中文逗号
            raw_sh = sh_var.get().replace("，", ",")
            raw_sm = sm_var.get().replace("，", ",")
            shrink_factors = [int(x.strip()) for x in raw_sh.split(",") if x.strip()]
            smoothing_sigmas = [float(x.strip()) for x in raw_sm.split(",") if x.strip()]
            if not shrink_factors:
                raise ValueError("Shrink Factors 不能为空")
            if not smoothing_sigmas:
                raise ValueError("Smoothing Sigmas 不能为空")
            optimizer_iterations = oi_var.get()
            raw_ot = ot_var.get().strip()
            if not raw_ot:
                raise ValueError("优化容差不能为空")
            optimizer_tol = float(raw_ot)
            optimizer_bounds = (ob_lo_var.get(), ob_hi_var.get())
            break  # 验证通过
        except ValueError as e:
            from tkinter import messagebox
            messagebox.showerror("输入格式错误", f"{e}\n请修正后重试。", parent=root)
            # 重建参数窗口（原窗口已被 destroy）
            param_win = tk.Toplevel(root)
            param_win.title("配准参数设置")
            param_win.resizable(False, False)
            r = 0
            tk.Label(param_win, text="配准算法：", font=("", 11)).grid(row=r, column=0, sticky="e", padx=(15, 2), pady=(15, 3))
            mode_frame2 = tk.Frame(param_win)
            mode_frame2.grid(row=r, column=1, sticky="w")
            for text, val in [("刚性", "刚性"), ("弹性", "弹性"), ("刚性+弹性", "刚性+弹性")]:
                tk.Radiobutton(mode_frame2, text=text, variable=mode_var, value=val, font=("", 10)).pack(side="left", padx=2)
            r += 1
            if reg_style == "锚点":
                tk.Label(param_win, text="锚点间隔：", font=("", 11)).grid(row=r, column=0, sticky="e", padx=(15, 2), pady=3)
                tk.Entry(param_win, textvariable=anchor_var, width=6).grid(row=r, column=1, sticky="w")
                tk.Label(param_win, text="(每隔 N 张选一个锚点)", font=("", 9)).grid(row=r, column=2, sticky="w")
                r += 1
            tk.Label(param_win, text="缩放因子：", font=("", 11)).grid(row=r, column=0, sticky="e", padx=(15, 2), pady=3)
            tk.Entry(param_win, textvariable=sf_var, width=6).grid(row=r, column=1, sticky="w")
            r += 1
            tk.Label(param_win, text="SIFT 匹配数：", font=("", 11)).grid(row=r, column=0, sticky="e", padx=(15, 2), pady=3)
            tk.Entry(param_win, textvariable=nm_var, width=6).grid(row=r, column=1, sticky="w")
            r += 1
            tk.Checkbutton(param_win, text="强制纯刚性（去除缩放分量）", variable=only_rigid_var, font=("", 10)).grid(row=r, column=0, columnspan=3, sticky="w", padx=15, pady=3)
            r += 1
            tk.Label(param_win, text="--- 弹性配准参数 ---", font=("", 10, "bold")).grid(row=r, column=0, columnspan=3, sticky="w", padx=15, pady=(8, 2))
            r += 1
            tk.Label(param_win, text="B-spline 网格：", font=("", 11)).grid(row=r, column=0, sticky="e", padx=(15, 2), pady=3)
            tk.Entry(param_win, textvariable=mesh_var, width=6).grid(row=r, column=1, sticky="w")
            r += 1
            tk.Label(param_win, text="Shrink Factors：", font=("", 11)).grid(row=r, column=0, sticky="e", padx=(15, 2), pady=3)
            tk.Entry(param_win, textvariable=sh_var, width=6).grid(row=r, column=1, sticky="w")
            tk.Label(param_win, text="(逗号分隔，如 10,5)", font=("", 9)).grid(row=r, column=2, sticky="w")
            r += 1
            tk.Label(param_win, text="Smoothing Sigmas：", font=("", 11)).grid(row=r, column=0, sticky="e", padx=(15, 2), pady=3)
            tk.Entry(param_win, textvariable=sm_var, width=6).grid(row=r, column=1, sticky="w")
            tk.Label(param_win, text="(逗号分隔，如 2,1)", font=("", 9)).grid(row=r, column=2, sticky="w")
            r += 1
            tk.Label(param_win, text="优化迭代次数：", font=("", 11)).grid(row=r, column=0, sticky="e", padx=(15, 2), pady=3)
            tk.Entry(param_win, textvariable=oi_var, width=6).grid(row=r, column=1, sticky="w")
            r += 1
            tk.Label(param_win, text="优化容差：", font=("", 11)).grid(row=r, column=0, sticky="e", padx=(15, 2), pady=3)
            tk.Entry(param_win, textvariable=ot_var, width=6).grid(row=r, column=1, sticky="w")
            r += 1
            tk.Label(param_win, text="边界下/上限：", font=("", 11)).grid(row=r, column=0, sticky="e", padx=(15, 2), pady=3)
            frm2 = tk.Frame(param_win)
            frm2.grid(row=r, column=1, sticky="w")
            tk.Entry(frm2, textvariable=ob_lo_var, width=5).pack(side="left")
            tk.Label(frm2, text=" ~ ", font=("", 10)).pack(side="left")
            tk.Entry(frm2, textvariable=ob_hi_var, width=5).pack(side="left")
            r += 1
            tk.Button(param_win, text="开始配准", command=param_win.destroy, width=12).grid(row=r, column=0, columnspan=3, pady=(15, 15))

    root.destroy()

    os.makedirs(output_dir, exist_ok=True)
    print(f"配准方式: {reg_style}")
    print(f"配准算法: {mode}")
    print(f"输出目录: {output_dir}")
    if reg_style == "锚点":
        print(f"锚点间隔: {anchor_interval}")
    print(f"参数: scale={scale_factor}, matches={n_matches}, only_rigid={only_rigid}")
    if mode in ("弹性", "刚性+弹性"):
        print(f"弹性: mesh={mesh_size}, shrink={shrink_factors}, smooth={smoothing_sigmas}, iter={optimizer_iterations}, tol={optimizer_tol}, bounds={optimizer_bounds}")
    print()

    count = 0

    if reg_style == "锚点":
        # ===== 锚点配准 =====
        if len(seq_paths) < 2:
            print("至少需要 2 张图像，退出。")
            exit()

        batch_register_images_anchor_subseq(
            image_paths=seq_paths,
            output_folder=output_dir,
            anchor_interval=anchor_interval,
            mode=mode,
            only_rigid=only_rigid,
            scale_factor=scale_factor,
            shrink_factors=shrink_factors,
            smoothing_sigmas=smoothing_sigmas,
            n_matches=n_matches,
            mesh_size=mesh_size,
            optimizer_iterations=optimizer_iterations,
            optimizer_tol=optimizer_tol,
            optimizer_bounds=optimizer_bounds,
        )
        print(f"完成！共处理 {len(seq_paths)} 张图像。")

    elif reg_style == "一对一":
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
                    result = rigid_registration(ref_img, tgt_img, n_matches=n_matches, scale_factor=scale_factor, only_rigid=only_rigid)
                    if result is None:
                        print(f"刚性配准失败，跳过: {tgt_path}")
                        continue
                elif mode == "弹性":
                    result = elastic_registration(ref_img, tgt_img, mesh_size=mesh_size, shrink_factors=shrink_factors, smoothing_sigmas=smoothing_sigmas,
                                                  optimizer_iterations=optimizer_iterations, optimizer_tol=optimizer_tol, optimizer_bounds=optimizer_bounds)
                elif mode == "刚性+弹性":
                    rigid_result = rigid_registration(ref_img, tgt_img, n_matches=n_matches, scale_factor=scale_factor, only_rigid=True)
                    if rigid_result is None:
                        print(f"刚性配准失败，跳过: {tgt_path}")
                        continue
                    result = elastic_registration(ref_img, rigid_result, mesh_size=mesh_size, shrink_factors=shrink_factors, smoothing_sigmas=smoothing_sigmas,
                                                  optimizer_iterations=optimizer_iterations, optimizer_tol=optimizer_tol, optimizer_bounds=optimizer_bounds)

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
            print("至少需要 2 张图像，退出。")
            exit()

        anchor_img = cv2.imread(seq_paths[0], cv2.IMREAD_COLOR)
        if anchor_img is None:
            raise FileNotFoundError(f"读取失败: {seq_paths[0]}")

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
                    result = rigid_registration(prev_img, tgt_img, n_matches=n_matches, scale_factor=scale_factor, only_rigid=only_rigid)
                    if result is None:
                        print(f"刚性配准失败，跳过: {tgt_path}")
                        continue
                elif mode == "弹性":
                    result = elastic_registration(prev_img, tgt_img, mesh_size=mesh_size, shrink_factors=shrink_factors, smoothing_sigmas=smoothing_sigmas,
                                                  optimizer_iterations=optimizer_iterations, optimizer_tol=optimizer_tol, optimizer_bounds=optimizer_bounds)
                elif mode == "刚性+弹性":
                    rigid_result = rigid_registration(prev_img, tgt_img, n_matches=n_matches, scale_factor=scale_factor, only_rigid=True)
                    if rigid_result is None:
                        print(f"刚性配准失败，跳过: {tgt_path}")
                        continue
                    result = elastic_registration(prev_img, rigid_result, mesh_size=mesh_size, shrink_factors=shrink_factors, smoothing_sigmas=smoothing_sigmas,
                                                  optimizer_iterations=optimizer_iterations, optimizer_tol=optimizer_tol, optimizer_bounds=optimizer_bounds)

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