import cv2
import glob
import os
from natsort import natsorted

# 读取所有图片路径
img_dir = r"C:\Users\30927\Desktop\img_histology\regist_brain_HE\ROI_elastic_2"  # 修改为你的图片文件夹路径
img_paths = natsorted(glob.glob(os.path.join(img_dir, '*.png')))  # 支持jpg/png等
save_dir = r"C:\Users\30927\Desktop\img_histology\regist_brain_HE\ROI_elastic_2_2"

# 显示第一张图片，手动选择ROI
first_img = cv2.imread(img_paths[0])
h0, w0 = first_img.shape[:2]

# 设定显示窗口最大宽度
max_width = 1000
scale = 1.0
if w0 > max_width:
    scale = max_width / w0
    disp_img = cv2.resize(first_img, (int(w0 * scale), int(h0 * scale)))
else:
    disp_img = first_img.copy()

# 选择ROI
roi_disp = cv2.selectROI("Select ROI (on scaled image)", disp_img, showCrosshair=True, fromCenter=False)
cv2.destroyAllWindows()

# 将ROI坐标映射回原图
x_disp, y_disp, w_disp, h_disp = map(int, roi_disp)
x = int(x_disp / scale)
y = int(y_disp / scale)
w = int(w_disp / scale)
h = int(h_disp / scale)
print(f"ROI on original image: x={x}, y={y}, w={w}, h={h}")

# 批量裁剪并保存
os.makedirs(save_dir, exist_ok=True)

for idx, img_path in enumerate(img_paths):
    img = cv2.imread(img_path)
    crop = img[y:y+h, x:x+w]
    save_path = os.path.join(save_dir, f'ROI_{idx:04d}.png')
    cv2.imwrite(save_path, crop)
    print(f"Saved: {save_path}")

print("All images cropped and saved.")