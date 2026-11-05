import cv2
import numpy as np
import os
import SimpleITK as sitk
from matplotlib import pyplot as plt


def expand_with_white(img, border=100):
    """用白色背景扩张图像四周"""
    return cv2.copyMakeBorder(
        img, border, border, border, border,
        cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )


# 绘制变形网格
def draw_deformation_grid(img, transform, mesh_size=(4, 4), color=(0, 0, 255), thickness=1):
    """
    img: 目标图像（BGR）
    transform: sitk.Transform 对象
    mesh_size: 网格行列数
    """
    h, w = img.shape[:2]
    # 生成网格点（物理空间坐标）
    x_lin = np.linspace(0, w - 1, mesh_size[0] + 1)
    y_lin = np.linspace(0, h - 1, mesh_size[1] + 1)
    grid_points = np.array([[x, y] for y in y_lin for x in x_lin], dtype=np.float32)

    # 对每个网格点做弹性变换
    warped_points = []
    for pt in grid_points:
        # 注意SimpleITK的坐标顺序为(x, y)
        warped_pt = transform.TransformPoint([float(pt[0]), float(pt[1])])
        warped_points.append(warped_pt)
    warped_points = np.array(warped_points).reshape((mesh_size[1] + 1, mesh_size[0] + 1, 2))

    # 在图像上画网格
    img_grid = img.copy()
    # 横线
    for i in range(mesh_size[1] + 1):
        for j in range(mesh_size[0]):
            pt1 = tuple(np.round(warped_points[i, j]).astype(int))
            pt2 = tuple(np.round(warped_points[i, j + 1]).astype(int))
            cv2.line(img_grid, pt1, pt2, color, thickness, cv2.LINE_AA)
    # 竖线
    for j in range(mesh_size[0] + 1):
        for i in range(mesh_size[1]):
            pt1 = tuple(np.round(warped_points[i, j]).astype(int))
            pt2 = tuple(np.round(warped_points[i + 1, j]).astype(int))
            cv2.line(img_grid, pt1, pt2, color, thickness, cv2.LINE_AA)
    return img_grid


# 弹性配准并返回变换对象
def elastic_registration(fixed, moving):
    # SimpleITK弹性配准（B样条）
    fixed_sitk = sitk.GetImageFromArray(cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY))
    moving_sitk = sitk.GetImageFromArray(cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY))
    fixed_sitk = sitk.Cast(fixed_sitk, sitk.sitkFloat32)
    moving_sitk = sitk.Cast(moving_sitk, sitk.sitkFloat32)

    transformDomainMeshSize = [4] * fixed_sitk.GetDimension()
    tx = sitk.BSplineTransformInitializer(fixed_sitk, transformDomainMeshSize)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=50, upperBound=100, lowerBound=-100)
    R.SetInitialTransform(tx, True)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    outTx = R.Execute(fixed_sitk, moving_sitk)

    # 对彩色图像应用弹性变换
    moved_channels = []
    for c in range(3):
        moved = sitk.Resample(
            sitk.GetImageFromArray(moving[:, :, c].astype(np.float32)),
            fixed_sitk,
            outTx,
            sitk.sitkLinear,
            255,
            sitk.sitkFloat32
        )
        moved_channels.append(sitk.GetArrayFromImage(moved).astype(np.uint8))
    elastic_registered = cv2.merge(moved_channels)
    return elastic_registered, outTx


# 处理流程
fixed_img = cv2.imread(r"C:\Users\30927\Desktop\111\1.png", cv2.IMREAD_COLOR)
moving_img = cv2.imread(r"C:\Users\30927\Desktop\111\2.png", cv2.IMREAD_COLOR)
fixed_img = expand_with_white(fixed_img)
moving_img = expand_with_white(moving_img)

# 2. 弹性配准
elastic_result, outTx = elastic_registration(fixed_img, moving_img)

# 3. 绘制变形网格（在弹性配准后的moving图像上展示形变网格）
mesh_size = (8, 8)  # 与transformDomainMeshSize一致
grid_img = draw_deformation_grid(elastic_result, outTx, mesh_size=mesh_size, color=(0, 0, 255), thickness=1)

cv2.imwrite("elastic_original1.png", fixed_img)
cv2.imwrite("elastic_result1.png", elastic_result)
cv2.imwrite("elastic_result_with_grid.png", grid_img)


