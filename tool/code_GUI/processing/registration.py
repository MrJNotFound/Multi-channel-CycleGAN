from importlib import import_module
from pathlib import Path
import cv2
import numpy as np
from .images import read_image, write_image

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

    orb = cv2.ORB_create(5000) # type: ignore
    kp1, des1 = orb.detectAndCompute(fixed_gray, None)
    kp2, des2 = orb.detectAndCompute(moving_gray, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
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
        # 边界采用纯白填充
        registered = cv2.warpAffine(moving_img, mat_full, (fixed_img.shape[1], fixed_img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    elif fixed_img.ndim == 3:
        registered = cv2.warpAffine(moving_img, mat_full, (fixed_img.shape[1], fixed_img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
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
    import SimpleITK as sitk

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
    mode="rigid",
    only_rigid=False,
    scale_factor=0.2,
    shrink_factors=None,
    smoothing_sigmas=None,
    n_matches=200,
    mesh_size=4,
    optimizer_iterations=50,
    optimizer_tol=1e-5,
    optimizer_bounds=(-100, 100),
    progress_callback=None,
    *,
    check_interrupted=None,
):
    modes = {"rigid": False, "刚性": False, "rigid+elastic": True, "刚性+弹性": True}
    if anchor_interval < 1 or mode not in modes:
        raise ValueError("配准模式或锚点间隔无效。")
    if modes[mode]:
        import_module("SimpleITK")
    paths = tuple(map(Path, image_paths))
    destination = Path(output_folder)
    targets = [destination / path.name for path in paths]
    resolved_targets = {path.resolve() for path in targets}
    if len(resolved_targets) != len(paths):
        raise ValueError("所选图像含有同名文件，请分开保存。")
    if resolved_targets.intersection(path.resolve() for path in paths):
        raise ValueError("输出路径会覆盖输入图像，请选择其他目录。")
    destination.mkdir(parents=True, exist_ok=True)

    anchor = None
    saved, errors = [], []
    for index, (path, target) in enumerate(zip(paths, targets)):
        if check_interrupted is not None:
            check_interrupted()
        try:
            image = read_image(path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("无法读取图像")
            result = image
            if anchor is not None:
                result = rigid_registration(anchor, image, n_matches=n_matches,
                                            scale_factor=scale_factor, only_rigid=only_rigid)
                if result is None:
                    raise ValueError("匹配点不足，无法进行刚性配准")
                if modes[mode]:
                    result = elastic_registration(
                        anchor, result, mesh_size=mesh_size, shrink_factors=shrink_factors,
                        smoothing_sigmas=smoothing_sigmas, optimizer_iterations=optimizer_iterations,
                        optimizer_tol=optimizer_tol, optimizer_bounds=optimizer_bounds,
                    )
            write_image(target, result)
            if anchor is None or index % anchor_interval == 0 or index == len(paths) - 1:
                anchor = result
            saved.append(target)
        except (OSError, ValueError, RuntimeError, cv2.error) as error:
            errors.append(f"{path.name}：{error}")
        if progress_callback:
            progress_callback(round(100 * (index + 1) / len(paths)))
    return saved, errors
