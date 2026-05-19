import cv2
import numpy as np

# ======================
# 直接写死配置：你自己改这里
# ======================
img_path = r"C:\Users\30927\Desktop\img_histology\stain_kidney\kidney_fake\WSI_fake\BF_HE_UTOM_256.png"
mask_path = r"C:\Users\30927\Desktop\img_histology\stain_kidney\kidney_fake\WSI_fake\mask_2.png" # 二值掩膜：白=保留，黑=背景
out_path = r"C:\Users\30927\Desktop\img_histology\stain_kidney\kidney_fake\WSI_fake\BF_HE_UTOM_256_extracted.png"

# 掩膜阈值：>= thresh 视为前景(白)
thresh = 128


img = cv2.imread(img_path, cv2.IMREAD_COLOR)
if img is None:
    raise ValueError(f"读取失败: {img_path}")

mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
if mask is None:
    raise ValueError(f"读取失败: {mask_path}")

# 如果mask尺寸与img不一致，先缩放mask到img大小（最近邻，避免引入灰度）
if mask.shape[0] != img.shape[0] or mask.shape[1] != img.shape[1]:
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

# mask可能是：
# - 灰度 HxW
# - BGR HxWx3
# - 带alpha HxWx4
if mask.ndim == 3:
    # 若有alpha通道，优先用alpha；否则转灰度
    if mask.shape[2] == 4:
        mask_gray = mask[:, :, 3]
    else:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
else:
    mask_gray = mask

# 二值化：白色区域为True
fg = (mask_gray.astype(np.uint8) >= thresh)

# 输出：默认白底
out = np.full_like(img, 255)
out[fg] = img[fg]

ok = cv2.imwrite(out_path, out)
if not ok:
    raise RuntimeError(f"保存失败: {out_path}")

print("Done.")
print(f"Image: {img_path}")
print(f"Mask : {mask_path}")
print(f"Out  : {out_path}")
print(f"Size : W={img.shape[1]}, H={img.shape[0]}")
print(f"FG pixels: {int(fg.sum())} / {fg.size}")

