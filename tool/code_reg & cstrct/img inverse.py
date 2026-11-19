from pathlib import Path
import tkinter as tk
from tkinter import filedialog

from PIL import Image, ImageOps

# 本脚本处理手动选择的本地全幅切片，允许读取超过 Pillow 默认上限的大图。
Image.MAX_IMAGE_PIXELS = None


def invert_images(image_paths, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = {Path(path).resolve() for path in image_paths}
    succeeded = 0
    for index, image_path in enumerate(image_paths, 1):
        source = Path(image_path)
        output = output_dir / source.name
        try:
            if output.resolve() in sources:
                raise ValueError("输出路径与所选原图相同，请选择其他输出文件夹")
            if output.exists():
                raise FileExistsError("目标文件已存在，请选择其他输出文件夹")
            with Image.open(source) as image:
                options = {"dpi": image.info["dpi"]} if "dpi" in image.info else {}
                with image.convert("L") as gray:
                    with ImageOps.invert(gray) as inverted:
                        inverted.save(output, **options)
            succeeded += 1
            print(f"[{index}/{len(image_paths)}] 已保存：{output}", flush=True)
        except (OSError, ValueError) as error:
            print(f"[{index}/{len(image_paths)}] 失败：{source.name}：{error}", flush=True)
    print(f"批量灰度反转完成：成功 {succeeded}/{len(image_paths)} 张。", flush=True)
    return succeeded


def main():
    root = tk.Tk()
    root.withdraw()
    try:
        image_paths = filedialog.askopenfilenames(
            parent=root,
            title="选择需要灰度反转的图像（可多选）",
            filetypes=[("图像文件", "*.bmp *.jpg *.jpeg *.png *.tif *.tiff"),
                       ("所有文件", "*.*")],
        )
        if not image_paths:
            print("未选择图像，已取消。")
            return
        output_dir = filedialog.askdirectory(
            parent=root, title="选择反转图像的输出文件夹",
            initialdir=str(Path(image_paths[0]).parent),
        )
        if not output_dir:
            print("未选择输出文件夹，已取消。")
            return
    finally:
        root.destroy()
    invert_images(image_paths, output_dir)


if __name__ == "__main__":
    main()
