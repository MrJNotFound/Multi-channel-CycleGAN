import os
from pathlib import Path
import cv2

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ==== 调试用：在这里写死路径（文件夹路径）====
INPUT_DIR = r"C:\Users\30927\Desktop\img_histology\stain_kidney\kidney_BF\WSI"  # 输入文件夹（会递归扫描）
OUTPUT_DIR = r"C:\Users\30927\Desktop\img_histology\stain_kidney\kidney_BF\WSI_re"


def invert_img(bgr_img):
    # 彩色图像:转换到HSV，反转V通道，再转换回BGR
    if len(bgr_img.shape) == 3 and bgr_img.shape[2] == 3:
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.subtract(255, v)  # V = 255 - V
        hsv_inv = cv2.merge((h, s, v))
        return cv2.cvtColor(hsv_inv, cv2.COLOR_HSV2BGR)
    # 灰度图像:直接反转像素值
    else:
        return cv2.subtract(255, bgr_img)


def main() -> None:
    in_dir = Path(INPUT_DIR).expanduser().resolve()
    out_dir = Path(OUTPUT_DIR).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for src in in_dir.rglob("*"):
        if not src.is_file() or src.suffix.lower() not in SUPPORTED_EXTS:
            continue

        rel = src.relative_to(in_dir)
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        out = invert_img(img)
        cv2.imwrite(str(dst), out)


if __name__ == "__main__":
    main()
