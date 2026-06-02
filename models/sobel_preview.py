"""Sobel gradient preview — compare AF, BF, HE edge maps.

Inputs three images (AF, BF, HE) and outputs four Sobel gradient magnitude
maps: AF, BF, HE, and the average of AF+BF.

Usage:
    python models/sobel_preview.py
"""

import os
import cv2
import numpy as np

# ======================
# Config
# ======================
img_af = r"C:\Users\30927\Desktop\epoch194_real_A_af.png"
img_bf = r"C:\Users\30927\Desktop\epoch194_real_A_bf.png"
img_he = r"C:\Users\30927\Desktop\epoch194_fake_B.png"

out_dir = r"C:\Users\30927\Desktop"

# ======================
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)


def gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude on channel-mean of input. (H, W, C) uint8 -> (H, W) uint8."""
    gray = np.mean(img.astype(np.float32), axis=2)
    gx = cv2.filter2D(gray, -1, sobel_x)
    gy = cv2.filter2D(gray, -1, sobel_y)
    mag = np.sqrt(gx ** 2 + gy ** 2 + 1e-6)
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8) * 255
    return mag.astype(np.uint8)


def main():
    af = cv2.imread(img_af, cv2.IMREAD_COLOR)
    bf = cv2.imread(img_bf, cv2.IMREAD_COLOR)
    he = cv2.imread(img_he, cv2.IMREAD_COLOR)

    for name, img in [("AF", af), ("BF", bf), ("HE", he)]:
        if img is None:
            raise FileNotFoundError(f"Failed to read: {globals()[f'img_{name.lower()}']}")

    sobel_af = gradient_magnitude(af)
    sobel_bf = gradient_magnitude(bf)
    sobel_he = gradient_magnitude(he)
    sobel_afbf_avg = ((sobel_af.astype(np.float32) + sobel_bf.astype(np.float32)) / 2).astype(np.uint8)

    for name, img in [("AF", sobel_af), ("BF", sobel_bf), ("HE", sobel_he), ("AF_BF_avg", sobel_afbf_avg)]:
        path = os.path.join(out_dir, f"sobel_{name}.png")
        cv2.imwrite(path, img)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
