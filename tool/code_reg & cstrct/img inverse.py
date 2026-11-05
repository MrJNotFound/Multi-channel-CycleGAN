import os
from PIL import Image, ImageOps

# 设置输入输出路径
input_dir = r'C:/Users/30927/Desktop/all/rigid_registered/registered_elastic'
output_dir = r'C:/Users/30927/Desktop/all/rigid_registered/registered_elastic_inverse'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 支持的图片格式
img_exts = ('.bmp', '.jpg', '.jpeg', '.png')

for fname in os.listdir(input_dir):
    if fname.lower().endswith(img_exts):
        img_path = os.path.join(input_dir, fname)
        img = Image.open(img_path)
        # 转为灰度
        gray_img = img.convert('L')
        # 反转灰度
        inverted_img = ImageOps.invert(gray_img)
        # 保存
        out_path = os.path.join(output_dir, fname)
        inverted_img.save(out_path)

print("批量灰度反转完成！")
