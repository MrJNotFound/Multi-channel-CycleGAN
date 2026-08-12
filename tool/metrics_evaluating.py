import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

import torch
import torch.nn as nn
from torchvision import models, transforms
from scipy import linalg

# LPIPS — perceptual similarity (optional dependency)
try:
    import lpips as _lpips_lib
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False
    print("lpips 未安装，LPIPS 指标不可用。安装: pip install lpips")


# =========================================================================
#  WSI multi-scale stratified sampling
# =========================================================================

def _load_mask(path, h, w):
    """Load a tissue mask, resize to (h, w), return bool array."""
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.ones((h, w), dtype=bool)
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return m > 0


def _grid_sample_locations(mask, patch_size, n_per_region, rng):
    """Stratified grid sampling over tissue regions.

    1. Divide the mask into a grid of patch_size × patch_size cells.
    2. For cells with tissue coverage ≥ 50%, randomly sample up to
       n_per_region non-overlapping patch locations.
    3. Cells are shuffled so consecutive samples cover different regions.

    Returns list of (y, x) coordinates.
    """
    h, w = mask.shape
    n_rows = max(1, h // patch_size)
    n_cols = max(1, w // patch_size)

    cell_h = h // n_rows
    cell_w = w // n_cols

    locations = []
    for row in range(n_rows):
        for col in range(n_cols):
            r0 = row * cell_h
            c0 = col * cell_w
            r1 = min(r0 + cell_h, h)
            c1 = min(c0 + cell_w, w)
            cell_mask = mask[r0:r1, c0:c1]
            if cell_mask.mean() < 0.3:          # skip background cells
                continue
            for _ in range(n_per_region):
                max_y = r1 - patch_size
                max_x = c1 - patch_size
                if max_y <= r0 or max_x <= c0:
                    y, x = r0, c0
                else:
                    y = rng.randint(r0, max_y)
                    x = rng.randint(c0, max_x)
                locations.append((y, x))

    rng.shuffle(locations)
    return locations


def _crop_patches(img, locations, patch_size):
    """Crop patch_size × patch_size from img at each location."""
    patches = []
    for y, x in locations:
        patch = img[y:y + patch_size, x:x + patch_size]
        patches.append(patch)
    return patches


def wsi_multiscale_sample(ref_dir, gen_dir, mask_dir=None,
                          macro_size=512, macro_per=30,
                          micro_size=256, micro_per=50,
                          seed=42):
    """Multi-scale stratified sampling over paired WSI folders.

    Args:
        ref_dir, gen_dir: paths to reference / generated image folders
        mask_dir: optional tissue mask folder (same filenames)
        macro_size, macro_per: patch size & max per WSI for FID
        micro_size, micro_per: patch size & max per WSI for SSIM/LPIPS
        seed: random seed

    Returns:
        ref_macro, gen_macro: lists of np.ndarray (macro patches)
        ref_micro, gen_micro: lists of np.ndarray (paired micro patches)
    """
    rng = np.random.RandomState(seed)
    exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

    ref_files = sorted([f for f in os.listdir(ref_dir)
                        if Path(f).suffix.lower() in exts])
    gen_files = sorted([f for f in os.listdir(gen_dir)
                        if Path(f).suffix.lower() in exts])

    # Match by filename
    gen_set = set(gen_files)
    matched = [f for f in ref_files if f in gen_set]
    if not matched:
        # Try case-insensitive
        gen_lower = {f.lower(): f for f in gen_files}
        matched = [f for f in ref_files if f.lower() in gen_lower]
    print(f"WSI 匹配: {len(matched)} 对 ({len(ref_files)} ref, {len(gen_files)} gen)")

    ref_macro, gen_macro = [], []
    ref_micro, gen_micro = [], []

    for fname in matched:
        ref_path = os.path.join(ref_dir, fname)
        gen_path = os.path.join(gen_dir,
                                gen_set.get(fname) or gen_lower.get(fname.lower(), fname))

        ref_img = cv2.imread(ref_path, cv2.IMREAD_COLOR)
        gen_img = cv2.imread(gen_path, cv2.IMREAD_COLOR)
        if ref_img is None or gen_img is None:
            print(f"  读取失败: {fname}")
            continue

        # Align sizes
        if ref_img.shape[:2] != gen_img.shape[:2]:
            gen_img = cv2.resize(gen_img,
                                 (ref_img.shape[1], ref_img.shape[0]),
                                 interpolation=cv2.INTER_LANCZOS4)

        h, w = ref_img.shape[:2]

        # Load mask
        if mask_dir:
            mask_path = os.path.join(mask_dir, fname)
            # try common mask suffix conventions
            if not os.path.isfile(mask_path):
                stem = Path(fname).stem
                for ext in ['.png', '.jpg', '.tif', '_mask.png']:
                    alt = os.path.join(mask_dir, stem + ext)
                    if os.path.isfile(alt):
                        mask_path = alt
                        break
            mask = _load_mask(mask_path, h, w)
        else:
            mask = np.ones((h, w), dtype=bool)

        # --- Macro layer (FID: independent sampling for ref & gen) ---
        macro_locs = _grid_sample_locations(mask, macro_size, macro_per, rng)
        ref_macro += _crop_patches(ref_img, macro_locs, macro_size)
        gen_locs = _grid_sample_locations(mask, macro_size, macro_per, rng)
        gen_macro += _crop_patches(gen_img, gen_locs, macro_size)

        # --- Micro layer (SSIM/LPIPS: same locations for ref & gen) ---
        micro_locs = _grid_sample_locations(mask, micro_size, micro_per, rng)
        ref_micro += _crop_patches(ref_img, micro_locs, micro_size)
        gen_micro += _crop_patches(gen_img, micro_locs, micro_size)

    print(f"  宏观层: {len(ref_macro)} ref + {len(gen_macro)} gen  ({macro_size}×{macro_size})")
    print(f"  微观层: {len(ref_micro)} ref + {len(gen_micro)} gen  ({micro_size}×{micro_size})")
    return ref_macro, gen_macro, ref_micro, gen_micro, len(matched)


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


# ---------------------------------------------------------------------------
# LPIPS evaluator (AlexNet backbone, paired metric)
# ---------------------------------------------------------------------------
class LPIPSEvaluator:
    def __init__(self, device="cpu"):
        self.device = device
        self.loss_fn = _lpips_lib.LPIPS(net="alex", verbose=False).to(device)
        self.loss_fn.eval()

    @torch.no_grad()
    def compute(self, img1, img2):
        """img1, img2: np.ndarray (H, W, 3) uint8 BGR → LPIPS distance.
        Lower is better (more perceptually similar). Range ≈ [0, 1].
        """
        rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        # lpips expects [-1, 1] float tensors
        t1 = torch.from_numpy(rgb1).permute(2, 0, 1).float().div(127.5).sub(1.0)
        t2 = torch.from_numpy(rgb2).permute(2, 0, 1).float().div(127.5).sub(1.0)
        return self.loss_fn(t1.unsqueeze(0).to(self.device),
                            t2.unsqueeze(0).to(self.device)).item()


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
    root.attributes("-topmost", True)

    # ---- 1. 选择参考文件夹 ----
    ref_dir = filedialog.askdirectory(title="选择参考图像（Ground Truth H&E）文件夹")
    if not ref_dir:
        print("未选择参考文件夹，退出。"); root.destroy(); sys.exit()
    print(f"参考: {ref_dir}\n")

    # ---- 2. 选择生成父文件夹（自动以子文件夹为各方法） ----
    gen_parent = filedialog.askdirectory(
        title="选择生成图像父文件夹（其下每个子文件夹 = 一种方法）"
    )
    if not gen_parent:
        print("未选择文件夹，退出。"); root.destroy(); sys.exit()

    gen_parent = os.path.abspath(gen_parent)
    gen_folders = []  # list of (method_name, folder_path)
    for name in sorted(os.listdir(gen_parent)):
        full = os.path.join(gen_parent, name)
        if os.path.isdir(full):
            gen_folders.append((name, full))
    if not gen_folders:
        print("父文件夹下没有子文件夹，退出。"); root.destroy(); sys.exit()

    print(f"对比方法: {len(gen_folders)} 种")
    for name, d in gen_folders:
        print(f"  {name}: {d}")
    print()

    # ---- 3. 四指标参数配置 ----
    param_win = tk.Toplevel(root)
    param_win.title("指标采样参数")
    param_win.resizable(False, False)
    param_win.attributes("-topmost", True)

    var_do_ssim  = tk.BooleanVar(value=True)
    var_do_psnr  = tk.BooleanVar(value=True)
    var_do_lpips = tk.BooleanVar(value=True)
    var_do_fid   = tk.BooleanVar(value=True)

    var_lpips_min = tk.IntVar(value=128); var_lpips_max = tk.IntVar(value=256); var_lpips_n = tk.IntVar(value=50)
    var_fid_min   = tk.IntVar(value=256); var_fid_max   = tk.IntVar(value=512); var_fid_n   = tk.IntVar(value=30)
    var_seed      = tk.IntVar(value=42)

    def _make_range_row(parent, label, do_var, min_var, max_var, n_var):
        row = tk.Frame(parent)
        row.pack(anchor="w", padx=20, pady=3)
        tk.Checkbutton(row, text=label, variable=do_var, width=12, anchor="w").pack(side="left")
        tk.Label(row, text="尺寸:").pack(side="left", padx=(8, 2))
        tk.Entry(row, textvariable=min_var, width=5).pack(side="left")
        tk.Label(row, text="~").pack(side="left")
        tk.Entry(row, textvariable=max_var, width=5).pack(side="left")
        tk.Label(row, text="px　　每张:").pack(side="left")
        tk.Entry(row, textvariable=n_var, width=5).pack(side="left")
        tk.Label(row, text="块").pack(side="left")

    tk.Label(param_win, text="采样参数", font=("", 11, "bold")).pack(pady=(14, 4), padx=24)

    tk.Label(param_win, text="全图指标 (每张 WSI 直接计算，不裁块)",
             font=("", 9), fg="gray").pack(pady=(8, 2))
    row_full = tk.Frame(param_win)
    row_full.pack(anchor="w", padx=20, pady=3)
    tk.Checkbutton(row_full, text="SSIM", variable=var_do_ssim).pack(side="left", padx=(0, 10))
    tk.Checkbutton(row_full, text="PSNR", variable=var_do_psnr).pack(side="left")

    tk.Label(param_win, text="裁块指标 (随机尺寸，配对位置)")\
        .pack(anchor="w", padx=20, pady=(12, 2))
    _make_range_row(param_win, "LPIPS", var_do_lpips, var_lpips_min, var_lpips_max, var_lpips_n)
    _make_range_row(param_win, "FID",   var_do_fid,   var_fid_min,   var_fid_max,   var_fid_n)
    tk.Label(param_win, text="       FID 参考与生成独立采位，LPIPS 同位置采",
             font=("", 8), fg="gray").pack(anchor="w", padx=32)

    seed_row = tk.Frame(param_win)
    seed_row.pack(anchor="w", padx=20, pady=(8, 4))
    tk.Label(seed_row, text="随机种子:").pack(side="left")
    tk.Entry(seed_row, textvariable=var_seed, width=8).pack(side="left", padx=4)

    tk.Button(param_win, text="开始评估", command=param_win.destroy,
              width=14, height=1).pack(pady=(14, 14))
    param_win.grab_set()
    root.wait_window(param_win)
    root.destroy()

    do_ssim  = var_do_ssim.get()
    do_psnr  = var_do_psnr.get()
    do_lpips = var_do_lpips.get() and _LPIPS_AVAILABLE
    do_fid   = var_do_fid.get()
    seed     = var_seed.get()

    # Full-image metrics (no patching)
    full_metrics = [m for m, ok in [("SSIM", do_ssim), ("PSNR", do_psnr)] if ok]
    # Patch metrics: (name, min, max, n, paired)
    patch_metrics = []
    if do_lpips: patch_metrics.append(("LPIPS", var_lpips_min.get(), var_lpips_max.get(), var_lpips_n.get(), True))
    if do_fid:   patch_metrics.append(("FID",   var_fid_min.get(),   var_fid_max.get(),   var_fid_n.get(),   False))

    if not full_metrics and not patch_metrics:
        print("未选择任何指标，退出。"); sys.exit()

    print("采样配置:")
    if full_metrics:
        print(f"  全图: {', '.join(full_metrics)}")
    for name, lo, hi, n, paired in patch_metrics:
        tag = "配对采位" if paired else "独立采位"
        print(f"  {name:6s}: {lo}~{hi} px 随机, {n} 块/张 ({tag})")
    print(f"  随机种子: {seed}\n")

    # ---- 4. 预生成裁块位置 + 预加载参考图像 ----
    # 所有方法共用同一套裁块位置，确保指标公平对比
    rng = np.random.RandomState(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

    # Reference WSI list
    ref_files = sorted([f for f in os.listdir(ref_dir) if Path(f).suffix.lower() in exts])

    ref_imgs = {}           # {idx: np.ndarray or None}
    patch_plan = {}          # {idx: [(name, sz, y, x, y2, x2), ...] or None}

    print("预生成裁块位置...")
    import time as _time
    for i in range(len(ref_files)):
        ref_path = os.path.join(ref_dir, ref_files[i])
        ref_img = cv2.imread(ref_path, cv2.IMREAD_COLOR)
        if ref_img is None:
            ref_imgs[i] = None
            patch_plan[i] = None
            continue
        ref_imgs[i] = ref_img
        h, w = ref_img.shape[:2]
        positions = []
        for name, lo, hi, n, is_paired in patch_metrics:
            for _ in range(n):
                sz = rng.randint(lo, hi + 1)
                sz = min(sz, h, w)
                max_y = max(0, h - sz)
                max_x = max(0, w - sz)
                y = rng.randint(0, max_y + 1) if max_y > 0 else 0
                x = rng.randint(0, max_x + 1) if max_x > 0 else 0
                if is_paired:
                    y2, x2 = y, x
                else:
                    y2 = rng.randint(0, max_y + 1) if max_y > 0 else 0
                    x2 = rng.randint(0, max_x + 1) if max_x > 0 else 0
                positions.append((name, sz, y, x, y2, x2))
        patch_plan[i] = positions
    print(f"  完成: {sum(1 for v in patch_plan.values() if v is not None)} 张参考 WSI 就绪\n")

    # ---- 5. 对每种方法采样 + 评估 ----

    # LPIPS (shared across methods)
    lpips_eval = LPIPSEvaluator(device=device) if do_lpips else None

    # Results: method_results[method_name] = {metric_name: [values]}
    method_results = {}
    method_n_wsi = {}

    for mi, (method_name, gen_dir) in enumerate(gen_folders):
        print(f"\n{'='*50}")
        print(f"[{mi+1}/{len(gen_folders)}] 评估: {method_name}")
        print(f"{'='*50}")

        gen_files = sorted([f for f in os.listdir(gen_dir) if Path(f).suffix.lower() in exts])

        # Sorted-order matching: file names may differ, assume sorted lists correspond 1:1
        n_pair = min(len(ref_files), len(gen_files))
        ref_paired = ref_files[:n_pair]
        gen_paired = gen_files[:n_pair]
        if len(ref_files) != len(gen_files):
            print(f"  注意: 参考 {len(ref_files)} 张, 生成 {len(gen_files)} 张, "
                  f"按排序配对前 {n_pair} 张")
        else:
            print(f"  WSI 匹配: {n_pair} 对 (按排序顺序)")
        method_n_wsi[method_name] = n_pair

        if n_pair == 0:
            method_results[method_name] = {}
            continue

        all_metric_names = list(full_metrics) + [n for n, _, _, _, _ in patch_metrics]
        results = {name: [] for name in all_metric_names}
        fid_feats_ref, fid_feats_gen = [], []
        fid_extractor = InceptionFeatureExtractor().to(device) if do_fid else None

        n_patch_total = sum(n for _, _, _, n, _ in patch_metrics)
        print(f"  全图指标: {full_metrics if full_metrics else '无'}")
        if n_patch_total:
            print(f"  每张 WSI 裁块: {n_patch_total} 块 ({n_pair} 张, 共 {n_pair * n_patch_total} 块)")

        n_size_mismatch, n_processed = 0, 0
        t_start = _time.time()
        for i in range(n_pair):
            gen_path = os.path.join(gen_dir, gen_paired[i])

            # ref image from cache (pre-loaded)
            ref_img = ref_imgs.get(i)
            if ref_img is None:
                print(f"  [{i+1}/{n_pair}] 跳过(参考图读取失败): {Path(ref_paired[i]).name}")
                continue

            rh, rw = ref_img.shape[:2]

            gen_img = cv2.imread(gen_path, cv2.IMREAD_COLOR)
            if gen_img is None:
                print(f"  [{i+1}/{n_pair}] 跳过(生成图读取失败): {Path(gen_paired[i]).name}")
                continue

            gh, gw = gen_img.shape[:2]
            if (rh, rw) != (gh, gw):
                n_size_mismatch += 1
                if n_size_mismatch <= 3:
                    print(f"  [{i+1}/{n_pair}] 跳过(尺寸不匹配 {rw}×{rh} vs {gw}×{gh}): "
                          f"{Path(ref_paired[i]).name}")
                continue

            n_processed += 1
            print(f"  [{i+1}/{n_pair}] {Path(ref_paired[i]).name} ({rw}×{rh})", end="", flush=True)

            t_img = _time.time()

            # --- Full-image metrics ---
            if do_ssim:
                results["SSIM"].append(float(ssim(ref_img, gen_img, channel_axis=2, data_range=255)))
            if do_psnr:
                results["PSNR"].append(float(cv2.PSNR(ref_img, gen_img)))

            # --- Patch metrics (use pre-generated positions) ---
            positions = patch_plan.get(i)
            if positions:
                for name, sz, y, x, y2, x2 in positions:
                    r_patch = ref_img[y:y + sz, x:x + sz]
                    g_patch = gen_img[y2:y2 + sz, x2:x2 + sz]

                    if name == "LPIPS":
                        results[name].append(lpips_eval.compute(r_patch, g_patch))
                    elif name == "FID":
                        fid_feats_ref.append(r_patch)
                        fid_feats_gen.append(g_patch)

            elapsed = _time.time() - t_img
            eta = elapsed * (n_pair - i - 1)
            print(f"  ({elapsed:.1f}s, 预计剩余 {eta:.0f}s)", flush=True)

        # FID
        if do_fid and fid_feats_ref and fid_feats_gen:
            act1 = fid_extractor.extract_features(fid_feats_ref)
            act2 = fid_extractor.extract_features(fid_feats_gen)
            results["FID"] = [compute_fid(act1, act2)]
            print(f"  FID 图块: {len(fid_feats_ref)} ref + {len(fid_feats_gen)} gen")

        print(f"  有效配对: {n_processed}/{n_pair} (跳过 {n_pair - n_processed}: "
              f"尺寸不匹配 {n_size_mismatch}, 读取失败 {n_pair - n_processed - n_size_mismatch})")
        if n_size_mismatch > 0:
            print(f"  ⚠ 有 {n_size_mismatch} 对尺寸不匹配，请检查排序配对")
        method_results[method_name] = results

    # ---- 5. 构建输出表格 ----
    table_metrics = [m for m in ["SSIM", "PSNR"] if m in full_metrics]
    table_metrics += [n for n, _, _, _, _ in patch_metrics]

    col_width = max(18, max(len(m[0]) for m in gen_folders) + 4)

    # Best per metric
    best = {}
    for m in table_metrics:
        vals = []
        for mn, res in method_results.items():
            if m in res and res[m]:
                arr = np.array(res[m])
                vals.append((mn, arr.mean(), arr.std()))
        if vals:
            best[m] = (min if m in ("FID", "LPIPS") else max)(vals, key=lambda x: x[1])[0]

    # Build table lines (both print and save)
    lines = []
    lines.append("=" * 90)
    lines.append("多方法对比评估结果")
    lines.append("=" * 90)
    lines.append(f"参考: {ref_dir}  ({len(ref_files)} 张 WSI)")
    lines.append(f"方法: {len(gen_folders)} 种")
    if full_metrics:
        lines.append(f"  全图: {', '.join(full_metrics)}")
    for name, lo, hi, n, paired in patch_metrics:
        lines.append(f"  {name}: {lo}~{hi} px, {n} 块/张")
    lines.append("")

    # Header
    header = f"{'方法':>{col_width}s}"
    csv_header = "method"
    for m in table_metrics:
        if m == "FID":
            header += "  │  FID"
            csv_header += ",FID"
        else:
            header += f"  │  {m} (mean ± std)"
            csv_header += f",{m}_mean,{m}_std"
    lines.append(header)
    lines.append("─" * 90)

    csv_lines = [csv_header]

    # Rows
    for method_name, _ in gen_folders:
        res = method_results.get(method_name, {})
        row = f"{method_name:>{col_width}s}"
        csv_row = method_name
        for m in table_metrics:
            if m not in res or not res[m]:
                row += "  │  —"
                csv_row += "," if m == "FID" else ",,"
                continue
            vals = np.array(res[m])
            tag = " ★" if best.get(m) == method_name else ""
            if m == "FID":
                row += f"  │  {vals[0]:8.2f}{tag}"
                csv_row += f",{vals[0]:.4f}"
            elif m in ("SSIM", "LPIPS"):
                row += f"  │  {vals.mean():8.4f} ± {vals.std():6.4f}{tag}"
                csv_row += f",{vals.mean():.6f},{vals.std():.6f}"
            else:
                row += f"  │  {vals.mean():7.2f} ± {vals.std():5.2f} dB{tag}"
                csv_row += f",{vals.mean():.4f},{vals.std():.4f}"
        lines.append(row)
        csv_lines.append(csv_row)

    lines.append("─" * 90)
    if not _LPIPS_AVAILABLE:
        lines.append("  (LPIPS 未安装，跳过。安装: pip install lpips)")
    lines.append("  ★ = 该指标最优")
    n_wsi_list = [method_n_wsi.get(mn, 0) for mn, _ in gen_folders]
    if len(set(n_wsi_list)) > 1:
        lines.append("  注意: 各方法的 WSI 匹配数不一致 →")
        for mn, _ in gen_folders:
            lines.append(f"    {mn}: {method_n_wsi.get(mn, 0)} 对")
    lines.append("=" * 90)

    # Print to console
    print("\n")
    for line in lines:
        print(line)

    # Save to files
    ts = _time.strftime("%Y%m%d_%H%M%S")
    out_txt = Path(ref_dir) / f"eval_results_{ts}.txt"
    out_csv = Path(ref_dir) / f"eval_results_{ts}.csv"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines))
    print(f"\n结果已保存:")
    print(f"  {out_txt}")
    print(f"  {out_csv}")

    sys.exit()
