import os
import cv2
from natsort import natsorted


def convert_and_resize_images(input_dir, output_dir, scale=1.0, save_grayscale=False, rename=False):
    """
    将 input_dir 下所有图片（递归）转换为 PNG 并可缩放，输出到 output_dir。
    支持 jpg, jpeg, bmp, tiff, gif, webp, png 格式。
    输出文件名按自然排序，命名为 0001、0002、0003……

    参数:
      - input_dir: 输入目录
      - output_dir: 输出目录
      - scale: 缩放比例（浮点数），例如 0.5、1.0、2.0
      - save_grayscale: 是否保存为灰度图（布尔值，默认False，即保存为彩色）
    """
    img_exts = ('.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp', '.png')

    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    # 收集所有图片文件的完整路径
    img_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(img_exts):
                img_files.append(os.path.join(root, file))

    # 使用natsort进行自然排序
    img_files = natsorted(img_files)

    os.makedirs(output_dir, exist_ok=True)

    # 处理并保存
    for idx, img_path in enumerate(img_files, 1):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"无法读取文件: {img_path}")
            continue
        # 缩放
        if scale != 1.0:
            width = max(1, int(img.shape[1] * scale))
            height = max(1, int(img.shape[0] * scale))
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        # 转为灰度（如果需要）
        if save_grayscale:
            if len(img.shape) == 3:
                if img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 如果rename=False，使用原始文件名（但改扩展名为png）；如果rename=True，使用0001.png、0002.png……
        if rename:
            file_name = f"{idx:04d}.png"
        else:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            file_name = f"{base_name}.jpg"
        output_path = os.path.join(output_dir, file_name)
        cv2.imwrite(output_path, img)
        print(f"保存: {output_path}")


# 使用示例
if __name__ == "__main__":
    input_dir = r"C:\Users\30927\Desktop\img_histology\stain_kidney\kidney_fake\WSI_fake\new"
    output_dir = r"C:\Users\30927\Desktop\img_histology\stain_kidney\kidney_fake\WSI_fake\new_lowres"
    scale = 0.1  # 例如0.5表示缩小一半，1不缩放，2放大一倍
    save_grayscale = True # True=保存为灰度, False=彩色
    rename = False
    convert_and_resize_images(input_dir, output_dir, scale, save_grayscale, rename)