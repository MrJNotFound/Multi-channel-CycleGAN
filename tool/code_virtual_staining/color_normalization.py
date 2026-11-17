import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog


def main():
    # 隐藏主窗口，只显示对话框
    root = tk.Tk()
    root.withdraw()

    # 1. 选择参考图像
    ref_path = filedialog.askopenfilename(
        title="选择参考图像（颜色归一化目标）",
        filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                   ("所有文件", "*.*")],
    )
    if not ref_path:
        print("未选择参考图像，退出。")
        return

    # 2. 选择待归一化图像（可多选）
    tgt_paths = filedialog.askopenfilenames(
        title="选择待归一化图像（可多选）",
        filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                   ("所有文件", "*.*")],
    )
    if not tgt_paths:
        print("未选择待归一化图像，退出。")
        return

    # 3. 选择输出文件夹
    out_dir = filedialog.askdirectory(title="选择输出文件夹")
    if not out_dir:
        print("未选择输出文件夹，退出。")
        return

    root.destroy()

    # 读取参考图像
    ref = _read_bgr(ref_path)

    # 批量归一化并保存
    count = 0
    for tgt_path in tgt_paths:
        try:
            tgt = _read_bgr(tgt_path)
            norm = reinhard_normalize_bgr(ref, tgt)

            base_name = os.path.splitext(os.path.basename(tgt_path))[0]
            out_path = os.path.join(out_dir, f"{base_name}_normalized.png")

            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            ok = cv2.imwrite(out_path, norm)
            if not ok:
                print(f"保存失败: {out_path}")
                continue

            print(f"[{count + 1}/{len(tgt_paths)}] Saved: {out_path}")
            count += 1
        except Exception as e:
            print(f"处理失败 [{tgt_path}]: {e}")

    print(f"完成！共处理 {count}/{len(tgt_paths)} 张图像。")

def _mean_std(img_lab: np.ndarray, mask: np.ndarray | None = None):
    """计算每个通道的均值与标准差（可选 mask）。"""
    means = []
    stds = []
    for c in range(3):
        ch = img_lab[:, :, c]
        if mask is not None:
            vals = ch[mask]
        else:
            vals = ch.reshape(-1)
        vals = vals.astype(np.float32)
        mu = float(vals.mean()) if vals.size else 0.0
        sigma = float(vals.std(ddof=0)) if vals.size else 1.0
        if sigma < 1e-6:
            sigma = 1.0
        means.append(mu)
        stds.append(sigma)
    return np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32)


def reinhard_normalize_bgr(
    source_bgr: np.ndarray,
    target_bgr: np.ndarray,
    mask_source: np.ndarray | None = None,
    mask_target: np.ndarray | None = None,
) -> np.ndarray:
    """
    将 target 的颜色分布归一化到 source（参考）上（Reinhard，Lab 空间）。
    source_bgr: 参考图（BGR）
    target_bgr: 待归一化图（BGR）
    mask_source/mask_target: 可选，bool mask（True 表示参与统计的像素）
    """
    if source_bgr is None or target_bgr is None:
        raise ValueError("source_bgr/target_bgr 不能为空")

    src_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    src_mean, src_std = _mean_std(src_lab, mask_source)
    tgt_mean, tgt_std = _mean_std(tgt_lab, mask_target)

    out = tgt_lab.copy()
    for c in range(3):
        out[:, :, c] = (out[:, :, c] - tgt_mean[c]) * (src_std[c] / tgt_std[c]) + src_mean[c]

    # Lab in OpenCV: L,a,b in [0,255]
    out = np.clip(out, 0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
    return out_bgr


def _read_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"读取失败: {path}")
    return img

if __name__ == "__main__":
    main()
