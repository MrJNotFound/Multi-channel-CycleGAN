import os
import cv2


def clamp_roi(x: int, y: int, w: int, h: int, W: int, H: int):
    """把 (x,y,w,h) 限制在图像范围内，并保证 w/h 为正。"""
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h


# 只读取一张图片
img_path = r"C:\Users\30927\Desktop\img_histology\stain_kidney\kidney_Trans\WSI\Slide 40-Region 008.jpg" # 改为你的单张图片路径
save_dir = r"C:\Users\30927\Desktop\img_histology\stain_kidney\kidney_Trans\ROI"
save_name = "roi_downsampled_2.png"

# ROI 保存时的下采样比例：0.5 表示缩小到原来 50%
downsample = 1  # (0, 1] 建议；>1 会放大
if downsample <= 0:
    raise ValueError("downsample must be > 0")

os.makedirs(save_dir, exist_ok=True)

img = cv2.imread(img_path, cv2.IMREAD_COLOR)
if img is None:
    raise ValueError(f"读取失败: {img_path}")

h0, w0 = img.shape[:2]

# 显示窗口最大宽度（只影响显示与选框体验，不影响最终裁剪精度）
max_width = 700
scale = 1.0
if w0 > max_width:
    scale = max_width / w0
    disp_img = cv2.resize(
        img,
        (int(round(w0 * scale)), int(round(h0 * scale))),
        interpolation=cv2.INTER_AREA,
    )
else:
    disp_img = img.copy()

# 在缩放显示图上选择 ROI
roi_disp = cv2.selectROI(
    "Select ROI (on scaled image) - ENTER/SPACE confirm, C cancel",
    disp_img,
    showCrosshair=True,
    fromCenter=False,
)
cv2.destroyAllWindows()

x_disp, y_disp, w_disp, h_disp = map(int, roi_disp)
if w_disp <= 0 or h_disp <= 0:
    raise RuntimeError("ROI 为空或已取消（按了 C / Esc）。")

# 映射回原图坐标并裁剪
x = int(round(x_disp / scale))
y = int(round(y_disp / scale))
w = int(round(w_disp / scale))
h = int(round(h_disp / scale))
x, y, w, h = clamp_roi(x, y, w, h, w0, h0)

roi = img[y : y + h, x : x + w].copy()

# 保存前下采样
if downsample != 1.0:
    new_w = max(1, int(round(roi.shape[1] * downsample)))
    new_h = max(1, int(round(roi.shape[0] * downsample)))
    roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

save_path = os.path.join(save_dir, save_name)
ok = cv2.imwrite(save_path, roi)
if not ok:
    raise RuntimeError(f"保存失败: {save_path}")

print(f"Saved: {save_path}")
print(f"ROI original coords: x={x}, y={y}, w={w}, h={h}")
print(f"Downsample: {downsample}")
print(f"Saved size: W={roi.shape[1]}, H={roi.shape[0]}")
