import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from skimage.metrics import structural_similarity as ssim

import torch
import torch.nn as nn
from torchvision import models, transforms
from scipy import linalg


# ---------------------------------------------------------------------------
# FID feature extractor (Inception v3 → pool3, 2048-d)
# ---------------------------------------------------------------------------
class InceptionFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Inception v3 pretrained on ImageNet, output the final avg-pool
        inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        inception.fc = nn.Identity()          # strip classification head
        inception.aux_logits = False          # disable auxiliary classifier
        self.model = inception.eval()

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract_features(self, images):
        """images: list of np.ndarray (H, W, 3) uint8 BGR"""
        # infer device from model parameters (handles both cpu & cuda)
        dev = next(self.model.parameters()).device
        feats = []
        for img in images:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t = self.preprocess(rgb).unsqueeze(0).to(dev)
            feat = self.model(t).squeeze(0)           # (2048,)
            feats.append(feat.cpu().numpy())
        return np.stack(feats, axis=0)                 # (N, 2048)


def compute_fid(feat1, feat2, eps=1e-6):
    """Fréchet Inception Distance from two activation matrices (N1, D) / (N2, D).

    需要 N1, N2 >= 2，否则协方差无定义（FID 是分布层面的指标）。
    """
    mu1 = np.mean(feat1, axis=0)
    mu2 = np.mean(feat2, axis=0)
    sigma1 = np.cov(feat1, rowvar=False)
    sigma2 = np.cov(feat2, rowvar=False)

    diff = mu1 - mu2

    # sqrtm 不传 disp（已弃用）；旧版可能返回 (matrix, errest) 元组，做兼容
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if isinstance(covmean, tuple):
        covmean = covmean[0]

    # 数值不稳定（含 nan/inf）时，给协方差加对角偏移后重算
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
        if isinstance(covmean, tuple):
            covmean = covmean[0]

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))
    return fid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_images(paths):
    """Read images from a list of absolute paths, skip failures."""
    imgs = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  读取失败，跳过: {p}")
            continue
        imgs.append(img)
    return imgs


def _random_crop_patches(images, crop_size, crops_per_image, seed=42):
    """从大图列表中随机裁取图块，用于 FID 计算。

    每张大图随机裁 crops_per_image 个 crop_size × crop_size 的图块。
    若图像任一维度小于 crop_size，则 resize 后直接作为一个图块。
    """
    rng = np.random.RandomState(seed)
    patches = []
    for img in images:
        h, w = img.shape[:2]
        if h < crop_size or w < crop_size:
            resized = cv2.resize(img, (crop_size, crop_size),
                                 interpolation=cv2.INTER_LANCZOS4)
            patches.append(resized)
            continue
        for _ in range(crops_per_image):
            y = rng.randint(0, h - crop_size)
            x = rng.randint(0, w - crop_size)
            patches.append(img[y:y + crop_size, x:x + crop_size])
    return patches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    # 0. 选择要计算的指标
    metric_win = tk.Toplevel(root)
    metric_win.title("选择评估指标")
    metric_win.resizable(False, False)

    var_ssim = tk.BooleanVar(value=True)
    var_psnr = tk.BooleanVar(value=True)
    var_fid  = tk.BooleanVar(value=True)

    tk.Label(metric_win, text="请勾选要计算的指标：", font=("", 11)).pack(pady=(12, 8), padx=20)

    cb_ssim = tk.Checkbutton(metric_win, text="SSIM   (结构相似性，逐对计算取均值)", variable=var_ssim)
    cb_ssim.pack(anchor="w", padx=20, pady=2)

    cb_psnr = tk.Checkbutton(metric_win, text="PSNR   (峰值信噪比，逐对计算取均值)", variable=var_psnr)
    cb_psnr.pack(anchor="w", padx=20, pady=2)

    cb_fid = tk.Checkbutton(metric_win, text="FID    (Fréchet Inception Distance，分布层面)", variable=var_fid)
    cb_fid.pack(anchor="w", padx=20, pady=2)

    var_crop = tk.BooleanVar(value=False)
    cb_crop = tk.Checkbutton(metric_win, text="       └ 大图随机裁块（图少时用，裁出多个子图以稳定FID）", variable=var_crop)
    cb_crop.pack(anchor="w", padx=20, pady=2)

    tk.Button(metric_win, text="确认", command=metric_win.destroy,
              width=12, height=1).pack(pady=(10, 12))
    metric_win.grab_set()
    root.wait_window(metric_win)

    do_ssim = var_ssim.get()
    do_psnr = var_psnr.get()
    do_fid  = var_fid.get()
    do_crop = var_crop.get() if do_fid else False

    if not any([do_ssim, do_psnr, do_fid]):
        print("未选择任何指标，退出。")
        root.destroy()
        exit()

    # 0b. FID 裁块参数（仅当勾选 FID + 大图随机裁块时）
    crop_size = 299
    crops_per_image = 50
    if do_crop:
        crop_win = tk.Toplevel(root)
        crop_win.title("FID 随机裁块参数")
        crop_win.resizable(False, False)

        var_crop_size = tk.IntVar(value=299)
        var_crops_per = tk.IntVar(value=50)

        tk.Label(crop_win, text="FID 大图随机裁块参数", font=("", 11, "bold")).pack(pady=(12, 8), padx=20)

        row1 = tk.Frame(crop_win)
        row1.pack(anchor="w", padx=20, pady=2)
        tk.Label(row1, text="裁块尺寸：", font=("", 10)).pack(side="left")
        tk.Entry(row1, textvariable=var_crop_size, width=6).pack(side="left", padx=(4, 0))
        tk.Label(row1, text=" px  (建议 256~512，Inception 输入 299)", font=("", 9)).pack(side="left", padx=4)

        row2 = tk.Frame(crop_win)
        row2.pack(anchor="w", padx=20, pady=2)
        tk.Label(row2, text="每图裁块数：", font=("", 10)).pack(side="left")
        tk.Entry(row2, textvariable=var_crops_per, width=6).pack(side="left", padx=(4, 0))
        tk.Label(row2, text=" 块  (总数 = 图像数 × 每图裁块数 × 2 组)", font=("", 9)).pack(side="left", padx=4)

        tk.Button(crop_win, text="确认", command=crop_win.destroy,
                  width=12).pack(pady=(10, 12))
        crop_win.grab_set()
        root.wait_window(crop_win)

        crop_size = var_crop_size.get()
        crops_per_image = var_crops_per.get()
        print(f"FID 裁块: {crop_size}×{crop_size}, 每图 {crops_per_image} 块")

    # 1. 多选参考图像
    ref_paths = filedialog.askopenfilenames(
        title="多选参考图像（Ground Truth）—— Ctrl+A 全选或 Ctrl+Click 逐个选",
        filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.gif *.webp"),
                   ("所有文件", "*.*")],
    )
    if not ref_paths:
        print("未选择参考图像，退出。")
        root.destroy()
        exit()

    # 2. 多选生成图像
    gen_paths = filedialog.askopenfilenames(
        title="多选生成图像（Generated）—— Ctrl+A 全选或 Ctrl+Click 逐个选",
        filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.gif *.webp"),
                   ("所有文件", "*.*")],
    )
    if not gen_paths:
        print("未选择生成图像，退出。")
        root.destroy()
        exit()

    root.destroy()

    ref_list = list(ref_paths)
    gen_list = list(gen_paths)

    print(f"参考图像: {len(ref_list)} 张")
    print(f"生成图像: {len(gen_list)} 张")

    # SSIM / PSNR 配对策略：
    #   数量相等 → 按文件名排序后逐一配对（适配手选、参考/生成命名不一致的情况）
    #   数量不等 → 退回按同名文件配对
    if len(ref_list) == len(gen_list) and ref_list:
        ref_sorted = sorted(ref_list, key=os.path.basename)
        gen_sorted = sorted(gen_list, key=os.path.basename)
        pairs = list(zip(ref_sorted, gen_sorted))
        mismatched = [(a, b) for a, b in pairs
                      if os.path.basename(a) != os.path.basename(b)]
        if mismatched:
            a0, b0 = pairs[0]
            print("配对方式: 按排序顺序（文件名不一致，请核对首对映射）")
            print(f"           {os.path.basename(a0)}  ↔  {os.path.basename(b0)}")
        else:
            print("配对方式: 按同名文件")
    else:
        ref_map = {os.path.basename(p): p for p in ref_list}
        gen_map = {os.path.basename(p): p for p in gen_list}
        common = sorted(set(ref_map) & set(gen_map))
        pairs = [(ref_map[n], gen_map[n]) for n in common]
        print("配对方式: 数量不等，按同名文件")

    if not pairs:
        print("⚠ 无可配对图像，SSIM/PSNR 将无法计算。")
    else:
        print(f"SSIM/PSNR 配对数量: {len(pairs)} 对")

    # ---------- 逐对 SSIM & PSNR ----------
    ssim_vals: list[float] = []
    psnr_vals: list[float] = []

    if do_ssim or do_psnr:
        for ref_path, gen_path in pairs:
            img1 = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
            img2 = cv2.imread(str(gen_path), cv2.IMREAD_COLOR)

            if img1 is None:
                print(f"读取失败，跳过: {ref_path}")
                continue
            if img2 is None:
                print(f"读取失败，跳过: {gen_path}")
                continue

            # 尺寸对齐
            if img1.shape[:2] != img2.shape[:2]:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]),
                                  interpolation=cv2.INTER_LANCZOS4)

            # SSIM（彩色：channel_axis=2；灰度：data_range=255）
            if do_ssim:
                if img1.ndim == 3 and img2.ndim == 3:
                    ssim_val = ssim(img1, img2, channel_axis=2, data_range=255)
                else:
                    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
                    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2
                    ssim_val = ssim(gray1, gray2, data_range=255)
                ssim_vals.append(ssim_val)

            # PSNR
            if do_psnr:
                psnr_val = cv2.PSNR(img1, img2)
                psnr_vals.append(psnr_val)

    # ---------- FID（使用全部选中图像 / 随机裁块） ----------
    # FID 是分布层面的指标：每组至少需要 2 张图像才能估计协方差，
    # 图像越多越可靠（通常建议每组 ≥ 50 张）。
    # 若勾选"大图随机裁块"，则从每张大图中随机裁取多个子图块用于 FID。
    fid_n_ref = 0
    fid_n_gen = 0
    if not do_fid:
        fid_val = None
    else:
        n_src = min(len(ref_list), len(gen_list))
        if n_src < 1:
            print(f"\n⚠ FID 需要每组至少 1 张图像，跳过 FID 计算。")
            fid_val = None
        else:
            if do_crop:
                print(f"\n正在加载图像并随机裁块以计算 FID "
                      f"({crop_size}×{crop_size}, 每图 {crops_per_image} 块) ...")
            else:
                print("\n正在加载图像并提取 Inception 特征以计算 FID ...")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            extractor = InceptionFeatureExtractor().to(device)

            ref_imgs = _load_images(ref_list)
            gen_imgs = _load_images(gen_list)

            if do_crop:
                ref_patches = _random_crop_patches(ref_imgs, crop_size, crops_per_image)
                gen_patches = _random_crop_patches(gen_imgs, crop_size, crops_per_image)
                fid_n_ref = len(ref_patches)
                fid_n_gen = len(gen_patches)
                n_patch = min(fid_n_ref, fid_n_gen)
                print(f"  裁块结果: 参考 {fid_n_ref} 块, 生成 {fid_n_gen} 块")
                if n_patch < 2:
                    print(f"⚠ FID 需要每组至少 2 个图块（当前 {n_patch} 个），跳过 FID 计算。")
                    fid_val = None
                else:
                    act1 = extractor.extract_features(ref_patches)
                    act2 = extractor.extract_features(gen_patches)
                    print(f"  特征维度: 参考 {act1.shape}, 生成 {act2.shape}")
                    fid_val = compute_fid(act1, act2)
            else:
                fid_n_ref = len(ref_imgs)
                fid_n_gen = len(gen_imgs)
                if n_src < 2:
                    print(f"\n⚠ FID 需要每组至少 2 张图像（当前最少 {n_src} 张），跳过 FID 计算。")
                    fid_val = None
                else:
                    act1 = extractor.extract_features(ref_imgs)
                    act2 = extractor.extract_features(gen_imgs)
                    print(f"  特征维度: 参考 {act1.shape}, 生成 {act2.shape}")
                    fid_val = compute_fid(act1, act2)
                    if n_src < 50:
                        print(f"  注意：图像数量较少（最少 {n_src} 张），FID 估计偏差较大，仅供参考。")

    # ---------- 输出结果 ----------
    print("\n" + "=" * 50)
    print("评估结果")
    print("=" * 50)
    print(f"参考图像: {len(ref_list)} 张")
    print(f"生成图像: {len(gen_list)} 张")
    print(f"配对数量: {len(pairs)} 对")

    if do_ssim:
        if ssim_vals:
            print(f"\nSSIM:  {np.mean(ssim_vals):.6f}  (±{np.std(ssim_vals):.6f})   [均值 ± 标准差]")
        else:
            print("\nSSIM:  N/A  (无可配对图像)")
    else:
        print("\nSSIM:  (未勾选)")

    if do_psnr:
        if psnr_vals:
            print(f"PSNR:  {np.mean(psnr_vals):.4f} dB  (±{np.std(psnr_vals):.4f})   [均值 ± 标准差]")
        else:
            print("PSNR:  N/A  (无可配对图像)")
    else:
        print("PSNR:  (未勾选)")

    if do_fid:
        if fid_val is None:
            print("FID:   N/A  (图像/图块数量不足)")
        else:
            if do_crop:
                print(f"FID:   {fid_val:.4f}  (随机裁块, {crop_size}×{crop_size}, "
                      f"参考 {fid_n_ref} 块 / 生成 {fid_n_gen} 块)")
            else:
                print(f"FID:   {fid_val:.4f}")
    else:
        print("FID:   (未勾选)")
    print("=" * 50)
