import os
import cv2
import numpy as np
import SimpleITK as sitk
from natsort import natsorted


def remove_scale_from_affine(mat):
    R = mat[:, :2]
    t = mat[:, 2]
    U, _, Vt = np.linalg.svd(R)
    R_no_scale = U @ Vt
    mat_no_scale = np.hstack([R_no_scale, t.reshape(2, 1)])
    return mat_no_scale

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

    # 使用 SIFT 特征点检测和描述
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(fixed_gray, None)
    kp2, des2 = sift.detectAndCompute(moving_gray, None)

    # 用 NORM_L2 距离的 BFMatcher 匹配
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:n_matches]
    if len(matches) < 5:
        return None
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

input_dir = r"C:\Users\30927\Desktop\78"
ext=('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff')
images = []
for f in os.listdir(input_dir):
    if f.lower().endswith(ext):
        images.append(os.path.join(input_dir, f))
images = natsorted(images)
output_dir = r"C:\Users\30927\Desktop\kidney_paper_regist"
batch_register_images_anchor_subseq(image_paths=images,
                                    output_folder=output_dir,
                                    anchor_interval = 1,
                                    mode = '刚性',
                                    only_rigid = True,
                                    scale_factor = 0.1,
                                    shrink_factors = [20],
                                    smoothing_sigmas = [2],
                                    n_matches = 200,
                                    mesh_size = 8,
                                    optimizer_iterations = 50,
                                    optimizer_tol = 1e-5,
                                    optimizer_bounds = (-200, 200),
                                    progress_callback = None
                                    )