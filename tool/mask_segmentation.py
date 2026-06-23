"""
从翻转后的 AF 图像中分割组织主体，生成二值掩膜。

用法：
    python tool/mask_segmentation.py

交互流程：
    1. 多选翻转后的 AF 图像
    2. 选择输出文件夹
    3. OpenCV 图窗实时预览参数效果 → Enter 确认 → 批量处理
"""

import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from natsort import natsorted

PROC_DIM = 700  # 预览和最终处理都用这个分辨率

# trackbar 名称常量
TB_OTSU = "Otsu (0=off / 1=on)"
TB_THRESH = "Manual Thresh"
TB_INVERT = "Invert (0/1)"
TB_OPEN_K = "Open KSize"
TB_OPEN_ITERS = "Open Iters"
TB_MIN_AREA = "Min Area"
TB_CLOSE_K = "Close KSize"
TB_CLOSE_ITERS = "Close Iters"
TB_CONVEX = "Convex (0/1)"
TB_DILATE_K = "Dilate KSize"
TB_DILATE_ITERS = "Dilate Iters"

# 轨道条排列顺序（名称, 默认值, 最大值）
TRACKBARS = [
    (TB_OTSU,         1,   1),
    (TB_THRESH,      128, 255),
    (TB_INVERT,       0,   1),
    (TB_OPEN_K,       5,  51),
    (TB_OPEN_ITERS,   1,  10),
    (TB_MIN_AREA,   500, 5000),
    (TB_CLOSE_K,     40, 101),
    (TB_CLOSE_ITERS,  1,  10),
    (TB_CONVEX,       1,   1),
    (TB_DILATE_K,    15,  81),
    (TB_DILATE_ITERS, 1,  10),
]

WIN_NAME = "SPIF Mask Segmentation — Enter=确认  Esc=退出  ←→=切换预览图"


def _odd(v: int) -> int:
    return max(1, v if v % 2 == 1 else v + 1)


def process_mask(
    src: np.ndarray,
    use_otsu: bool,
    manual_thresh: int,
    invert: bool,
    open_ksize: int,
    open_iters: int,
    min_area: int,
    close_ksize: int,
    close_iters: int,
    convex: bool,
    dilate_ksize: int,
    dilate_iters: int,
):
    """对灰度 numpy 数组做组织分割，返回二值掩膜 (0/255)。

    所有尺寸参数基于 src 的实际尺寸（调用前应做好缩放）。
    """
    img = src.astype(np.uint8)

    # ---- 阈值 ----
    otsu_val = 0
    if use_otsu:
        otsu_val, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(img, manual_thresh, 255, cv2.THRESH_BINARY)

    if invert:
        binary = cv2.bitwise_not(binary)

    # ---- 开运算 ----
    if open_ksize > 0 and open_iters > 0:
        k = _odd(open_ksize)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=open_iters)

    # ---- 去小连通域 ----
    if min_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        cleaned = np.zeros_like(binary)
        for idx in range(1, num_labels):
            if stats[idx, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == idx] = 255
        binary = cleaned

    # ---- 闭运算 ----
    if close_ksize > 0 and close_iters > 0 and np.any(binary):
        k = _odd(close_ksize)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=close_iters)

    # ---- 凸包 / 轮廓填充 ----
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        all_pts = np.vstack(contours)
        if convex:
            hull = cv2.convexHull(all_pts)
            solid = np.zeros_like(binary)
            cv2.drawContours(solid, [hull], -1, 255, thickness=cv2.FILLED)
        else:
            solid = np.zeros_like(binary)
            cv2.drawContours(solid, contours, -1, 255, thickness=cv2.FILLED)
    else:
        solid = np.zeros_like(binary)

    # ---- 膨胀 ----
    if dilate_ksize > 0 and dilate_iters > 0 and np.any(solid):
        k = _odd(dilate_ksize)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        solid = cv2.dilate(solid, kernel, iterations=dilate_iters)

    return solid, otsu_val


def _load_gray(path: str):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"读取失败: {path}")
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, 3]
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _build_overlay(gray: np.ndarray, mask: np.ndarray):
    """灰度图 + 红色半透明掩膜叠加。"""
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    red = np.zeros_like(overlay)
    red[:, :, 2] = 255
    alpha = 0.35
    fg = (mask == 255)
    overlay[fg] = (overlay[fg] * (1 - alpha) + red[fg] * alpha).astype(np.uint8)
    # 画轮廓线
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
    return overlay


def preview_loop(first_gray):
    """OpenCV 实时预览循环（仅第一张图）。

    first_gray: 已下采样的灰度 numpy 数组
    返回用户确认的参数 dict，或 None (Esc)。
    """
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 1100, 680)

    for name, default, max_val in TRACKBARS:
        cv2.createTrackbar(name, WIN_NAME, default, max_val, lambda _: None)

    WAIT_MS = 30

    def _read():
        return {
            "use_otsu":      bool(cv2.getTrackbarPos(TB_OTSU,         WIN_NAME)),
            "manual_thresh":      cv2.getTrackbarPos(TB_THRESH,       WIN_NAME),
            "invert":         bool(cv2.getTrackbarPos(TB_INVERT,       WIN_NAME)),
            "open_ksize":          cv2.getTrackbarPos(TB_OPEN_K,       WIN_NAME),
            "open_iters":          cv2.getTrackbarPos(TB_OPEN_ITERS,   WIN_NAME),
            "min_area":            cv2.getTrackbarPos(TB_MIN_AREA,     WIN_NAME),
            "close_ksize":         cv2.getTrackbarPos(TB_CLOSE_K,      WIN_NAME),
            "close_iters":         cv2.getTrackbarPos(TB_CLOSE_ITERS,  WIN_NAME),
            "convex":          bool(cv2.getTrackbarPos(TB_CONVEX,       WIN_NAME)),
            "dilate_ksize":        cv2.getTrackbarPos(TB_DILATE_K,     WIN_NAME),
            "dilate_iters":        cv2.getTrackbarPos(TB_DILATE_ITERS, WIN_NAME),
        }

    while True:
        p = _read()
        mask, _ = process_mask(first_gray, **p)

        # 三栏拼接
        gray_bgr = cv2.cvtColor(first_gray, cv2.COLOR_GRAY2BGR)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        overlay = _build_overlay(first_gray, mask)
        panels = [gray_bgr, mask_bgr, overlay]
        labels = ["Original", "Mask", "Overlay"]
        labeled = []
        for panel, label in zip(panels, labels):
            canvas = cv2.copyMakeBorder(panel, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
            cv2.putText(canvas, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            labeled.append(canvas)
        row = np.hstack(labeled)

        # 状态栏
        fg_pct = 100 * (mask == 255).sum() / mask.size
        status = (f"Otsu={p['use_otsu']} Thresh={p['manual_thresh']} Invert={p['invert']}  |  "
                  f"Open={p['open_ksize']}x{p['open_iters']}  Area>{p['min_area']}  |  "
                  f"Close={p['close_ksize']}x{p['close_iters']}  Convex={p['convex']}  |  "
                  f"Dilate={p['dilate_ksize']}x{p['dilate_iters']}  |  "
                  f"FG={fg_pct:.1f}%  |  Enter=确认  Esc=退出")
        footer = np.full((30, row.shape[1], 3), 40, dtype=np.uint8)
        cv2.putText(footer, status, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

        cv2.imshow(WIN_NAME, np.vstack([row, footer]))
        key = cv2.waitKey(WAIT_MS) & 0xFF

        if key == 27:  # Esc
            cv2.destroyWindow(WIN_NAME)
            return None
        elif key == 13:  # Enter
            result = _read()
            cv2.destroyWindow(WIN_NAME)
            return result


# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    # 1. 多选图像
    raw_paths = filedialog.askopenfilenames(
        title="选择翻转后的 AF 图像（可多选）",
        filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("所有文件", "*.*")],
    )
    if not raw_paths:
        print("未选择图像，退出。"); root.destroy(); exit()
    img_paths = natsorted(list(raw_paths))
    print(f"已选择 {len(img_paths)} 张图像")

    # 2. 输出目录
    out_dir = filedialog.askdirectory(title="选择输出文件夹")
    if not out_dir:
        print("未选择输出文件夹，退出。"); root.destroy(); exit()

    # 3. 仅加载第一张图做预览
    first_gray = _load_gray(img_paths[0])
    h0_first, w0_first = first_gray.shape
    if max(h0_first, w0_first) > PROC_DIM:
        s = PROC_DIM / max(h0_first, w0_first)
        first_gray = cv2.resize(first_gray,
                                (int(round(w0_first * s)), int(round(h0_first * s))),
                                interpolation=cv2.INTER_AREA)

    # 销毁 tk，避免干扰 OpenCV 窗口
    root.destroy()

    # 4. 预览循环
    params = preview_loop(first_gray)
    if params is None:
        print("用户取消。"); exit()

    print("\n确认参数:")
    for k, v in params.items():
        print(f"  {k} = {v}")
    print(f"\n输出目录: {out_dir}\n")

    # 5. 批量处理：全部缩放到预览分辨率处理，掩膜还原到原始尺寸
    count = 0
    for i, img_path in enumerate(img_paths, 1):
        try:
            base = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(out_dir, f"{base}_mask.png")

            gray = _load_gray(img_path)
            h0, w0 = gray.shape[:2]

            # 缩放到与预览完全相同的分辨率
            if max(h0, w0) > PROC_DIM:
                s = PROC_DIM / max(h0, w0)
                gray = cv2.resize(gray, (int(round(w0 * s)), int(round(h0 * s))),
                                  interpolation=cv2.INTER_AREA)

            # 参数不做任何缩放，与预览完全一致
            mask, otsu_val = process_mask(gray, **params)

            # 还原到原始尺寸
            if mask.shape[0] != h0 or mask.shape[1] != w0:
                mask = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_NEAREST)

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, mask)

            fg_px = int((mask == 255).sum())
            total_px = mask.size
            print(
                f"[{i}/{len(img_paths)}] {os.path.basename(img_path)} -> {os.path.basename(out_path)}  "
                f"(FG: {fg_px}/{total_px}, {100*fg_px/total_px:.1f}%"
                + (f", Otsu={otsu_val}" if params["use_otsu"] else "")
                + ")"
            )
            count += 1
        except Exception as e:
            print(f"[{i}/{len(img_paths)}] 失败 [{img_path}]: {e}")

    print(f"\n完成！共处理 {count}/{len(img_paths)} 张。")
