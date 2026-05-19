import numpy as np
import cv2
import os
import glob
from natsort import natsorted
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs): return iterable

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
        only_rigid=False
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

def batch_register(ref_dir, moving_dir, out_dir, scale_factor=0.2, only_rigid=True):
    os.makedirs(out_dir, exist_ok=True)

    valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
    ref_imgs = [f for f in natsorted(glob.glob(os.path.join(ref_dir, '*'))) if f.lower().endswith(valid_exts)]
    moving_imgs = [f for f in natsorted(glob.glob(os.path.join(moving_dir, '*'))) if f.lower().endswith(valid_exts)]

    paired = zip(ref_imgs, moving_imgs)
    total_imgs = min(len(ref_imgs), len(moving_imgs))

    print(f"Found {total_imgs} image pairs to register.")

    for ref_path, moving_path in tqdm(paired, total=total_imgs, desc="Registering"):
        ref_img = cv2.imread(ref_path)
        moving_img = cv2.imread(moving_path)

        if ref_img is None or moving_img is None:
            print(f"Failed to read image pair: {os.path.basename(ref_path)} & {os.path.basename(moving_path)}")
            continue

        registered = rigid_registration(ref_img, moving_img, scale_factor=scale_factor, only_rigid=only_rigid)

        if registered is not None:
            filename = os.path.basename(moving_path)
            out_path = os.path.join(out_dir, filename)
            cv2.imwrite(out_path, registered)
        else:
            print(f"Registration failed for: {os.path.basename(moving_path)}")

if __name__ == "__main__":
    # 请自行修改以下路径
    REFERENCE_DIR = r"C:\Users\30927\Desktop\img_histology\stain_kidney\kidney_Trans_regist"
    MOVING_DIR = r"C:\Users\30927\Desktop\img_histology\stain_kidney\kidney_DAPI_regist"
    OUTPUT_DIR = r"C:\Users\30927\Desktop\img_histology\stain_kidney\kidney_DAPI_registed_1"

    # 运行批量配准
    batch_register(
        REFERENCE_DIR,
        MOVING_DIR,
        OUTPUT_DIR,
        scale_factor=0.2,  # 可调整以加速配准，过小可能导致配准失败
        only_rigid=False    # 只保留旋转和平移，去除缩
        )
