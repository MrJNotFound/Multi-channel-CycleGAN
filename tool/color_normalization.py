import os
import cv2
import numpy as np

def main():
    # 修改为你的两张染色图片路径
    ref_path = r"C:\Users\30927\Desktop\img_histology\stain_kidney\kidney_HE\WSI\Slide 23-Region 008.jpg"  # 第 1 张：参考
    tgt_path = r"C:\Users\30927\Desktop\111.png"     # 第 2 张：待归一化

    out_path = r"C:\Users\30927\Desktop\1111.png"

    ref = _read_bgr(ref_path)
    tgt = _read_bgr(tgt_path)

    norm = reinhard_normalize_bgr(ref, tgt)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ok = cv2.imwrite(out_path, norm)
    if not ok:
        raise RuntimeError(f"保存失败: {out_path}")

    print(f"Saved: {out_path}")

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
