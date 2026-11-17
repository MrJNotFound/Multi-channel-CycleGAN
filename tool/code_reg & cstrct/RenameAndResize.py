import math
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import cv2
import numpy as np
from natsort import natsorted


INPUT_FORMATS = {
    "全部支持的格式": ('.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.gif', '.webp', '.png'),
    "PNG (.png)": ('.png',),
    "JPEG (.jpg / .jpeg)": ('.jpg', '.jpeg'),
    "TIFF (.tif / .tiff)": ('.tif', '.tiff'),
    "BMP (.bmp)": ('.bmp',),
    "GIF (.gif)": ('.gif',),
    "WebP (.webp)": ('.webp',),
}
OUTPUT_FORMATS = {
    "PNG (.png)": '.png',
    "JPEG (.jpg)": '.jpg',
    "TIFF (.tif)": '.tif',
    "BMP (.bmp)": '.bmp',
    "WebP (.webp)": '.webp',
}
FORMAT_NOTES = {
    '.png': "PNG：无损，支持 8/16 位和透明通道。",
    '.jpg': "JPEG：有损，仅支持 8 位、不支持透明通道；不兼容的图片会跳过并记录。",
    '.tif': "TIFF：支持 16 位和透明通道，适合保留图像精度。",
    '.bmp': "BMP：仅支持 8 位，文件通常较大；其他位深会跳过并记录。",
    '.webp': "WebP：仅支持 8 位和透明通道；灰度内容会保存为三通道。",
}


def validate_formats(input_extensions, output_format):
    if input_extensions is None:
        input_extensions = INPUT_FORMATS["全部支持的格式"]
    elif isinstance(input_extensions, str):
        input_extensions = (input_extensions,)
    extensions = tuple('.' + ext.strip().lower().lstrip('.') for ext in input_extensions)
    if not extensions or any(ext not in INPUT_FORMATS["全部支持的格式"] for ext in extensions):
        raise ValueError("请选择支持的输入图像格式。")
    output_format = '.' + output_format.strip().lower().lstrip('.')
    output_format = {'.jpeg': '.jpg', '.tiff': '.tif'}.get(output_format, output_format)
    if output_format not in OUTPUT_FORMATS.values():
        raise ValueError("请选择支持的输出图像格式。")
    if not cv2.haveImageWriter(output_format):
        raise ValueError(f"当前 OpenCV 环境不支持保存 {output_format} 格式。")
    return extensions, output_format


def check_output_compatibility(img, output_format):
    if output_format in ('.jpg', '.bmp', '.webp') and img.dtype != np.uint8:
        raise ValueError(f"{output_format} 仅支持 8 位图像，当前为 {img.dtype}；请改选 PNG 或 TIFF")
    if output_format == '.png' and img.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"PNG 仅支持 8/16 位图像，当前为 {img.dtype}；请改选 TIFF")
    if output_format == '.jpg' and img.ndim == 3 and img.shape[2] == 4:
        raise ValueError("JPEG 不支持透明通道；请改选 PNG、TIFF 或 WebP")


def validate_parameters(input_dir, output_dir, scale):
    if not input_dir or not os.path.isdir(input_dir):
        raise ValueError("请选择有效的输入文件夹。")
    if not output_dir:
        raise ValueError("请选择输出文件夹。")
    input_dir = os.path.realpath(os.path.abspath(input_dir))
    output_dir = os.path.realpath(os.path.abspath(output_dir))
    if os.path.normcase(input_dir) == os.path.normcase(output_dir):
        raise ValueError("输出文件夹需要与输入文件夹不同。")
    try:
        scale = float(scale)
    except (TypeError, ValueError):
        raise ValueError("缩放比例请输入大于 0 的数字。") from None
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("缩放比例请输入大于 0 的有限数字。")
    return input_dir, output_dir, scale


def convert_and_resize_images(
    input_dir,
    output_dir,
    scale=1.0,
    save_grayscale=False,
    *,
    include_subfolders=True,
    input_extensions=None,
    output_format='.png',
    log_callback=None,
    progress_callback=None,
    stop_event=None,
):
    """
    将 input_dir 下指定格式的图片转换为所选格式并可缩放，输出到 output_dir。
    支持读取 jpg, jpeg, bmp, tif, tiff, gif, webp, png 格式。
    输出文件名按自然排序，命名为 0001、0002、0003……

    参数:
      - input_dir: 输入目录
      - output_dir: 输出目录
      - scale: 缩放比例（浮点数），例如 0.5、1.0、2.0
      - save_grayscale: 是否转换为灰度图，默认保留原有颜色和通道
      - include_subfolders: 是否递归处理子文件夹
      - input_extensions: 筛选输入扩展名，例如 ('.jpg', '.jpeg')；None 表示全部支持格式
      - output_format: 输出格式扩展名，默认为 '.png'
      - log_callback: 接收处理日志的可选回调
      - progress_callback: 接收（已处理数量，总数量）的可选回调
      - stop_event: 可选 threading.Event，在两张图片之间响应停止请求

    返回包含 total、processed、saved、failed、cancelled 的统计字典。
    """
    input_dir, output_dir, scale = validate_parameters(input_dir, output_dir, scale)
    img_exts, output_format = validate_formats(input_extensions, output_format)
    log = log_callback if log_callback is not None else print
    output_key = os.path.normcase(output_dir)

    # 收集所有图片文件的完整路径
    log("正在扫描图片文件……")
    img_files = []
    for root, directories, files in os.walk(input_dir):
        if stop_event is not None and stop_event.is_set():
            break
        if include_subfolders:
            # 输出在输入目录内时，排除它，避免再次处理之前的结果。
            directories[:] = [
                directory for directory in directories
                if os.path.normcase(os.path.realpath(os.path.join(root, directory))) != output_key
            ]
        else:
            directories.clear()
        for file in files:
            if file.lower().endswith(img_exts):
                img_files.append(os.path.join(root, file))

    # 使用natsort进行自然排序
    img_files = natsorted(img_files)
    total = len(img_files)
    summary = dict(total=total, processed=0, saved=0, failed=0, cancelled=False)
    if progress_callback is not None:
        progress_callback(0, total)
    if stop_event is not None and stop_event.is_set():
        summary["cancelled"] = True
        return summary
    if not img_files:
        log("未找到符合所选输入格式的图片文件。")
        return summary
    log(f"共找到 {total} 张图片，开始处理。")
    os.makedirs(output_dir, exist_ok=True)

    # 处理并保存
    for idx, img_path in enumerate(img_files, 1):
        if stop_event is not None and stop_event.is_set():
            summary["cancelled"] = True
            break
        try:
            # 先读取文件字节，再解码，兼容 Windows 中文路径。
            image_data = np.fromfile(img_path, dtype=np.uint8)
            img = cv2.imdecode(image_data, cv2.IMREAD_UNCHANGED) if image_data.size else None
            if img is None:
                raise ValueError("文件为空、图片损坏或格式不支持")
            if scale != 1.0:
                width = max(1, int(img.shape[1] * scale))
                height = max(1, int(img.shape[0] * scale))
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            if save_grayscale and img.ndim == 3:
                conversion = cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY
                img = cv2.cvtColor(img, conversion)
            check_output_compatibility(img, output_format)
            output_path = os.path.join(output_dir, f"{idx:04d}{output_format}")
            # 编码后由 NumPy 写入文件，输出路径同样支持中文。
            success, encoded = cv2.imencode(output_format, img)
            if not success:
                raise ValueError(f"{output_format} 编码失败")
            encoded.tofile(output_path)
            summary["saved"] += 1
            log(f"保存: {output_path}")
        except (OSError, ValueError, cv2.error) as exc:
            summary["failed"] += 1
            log(f"处理失败: {img_path}（{exc}）")
        finally:
            summary["processed"] = idx
            if progress_callback is not None:
                progress_callback(idx, total)
    return summary


class RenameResizeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图片批量重命名与缩放")
        self.root.geometry("820x680")
        self.root.minsize(780, 620)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.events = queue.Queue()
        self.stop_event = threading.Event()
        self.running = False
        self.closing = False
        self.worker = None

        self.input_var = tk.StringVar(root)
        self.output_var = tk.StringVar(root)
        self.scale_var = tk.StringVar(root, value="0.535")
        self.grayscale_var = tk.BooleanVar(root, value=False)
        self.recursive_var = tk.BooleanVar(root, value=True)
        self.input_format_var = tk.StringVar(root, value="全部支持的格式")
        self.output_format_var = tk.StringVar(root, value="PNG (.png)")
        self.format_hint_var = tk.StringVar(root)
        self.status_var = tk.StringVar(root, value="请选择文件夹并设置处理参数。")
        self.controls = []
        self.build_widgets()
        self.output_format_var.trace_add("write", self.update_format_hint)
        self.update_format_hint()
        self.root.after(100, self.poll_events)

    def build_widgets(self):
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(5, weight=1)

        folders = ttk.LabelFrame(container, text="输入与输出", padding=12)
        folders.grid(row=0, column=0, sticky="ew")
        folders.columnconfigure(1, weight=1)
        for row, label, variable, command in (
            (0, "输入文件夹", self.input_var, self.browse_input),
            (1, "输出文件夹", self.output_var, self.browse_output),
        ):
            ttk.Label(folders, text=label).grid(row=row, column=0, padx=(0, 10), pady=5)
            entry = ttk.Entry(folders, textvariable=variable)
            entry.grid(row=row, column=1, sticky="ew", pady=5)
            button = ttk.Button(folders, text="浏览……", command=command)
            button.grid(row=row, column=2, padx=(10, 0), pady=5)
            self.controls.extend((entry, button))

        parameters = ttk.LabelFrame(container, text="处理参数", padding=12)
        parameters.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(parameters, text="缩放比例").grid(row=0, column=0, sticky="w")
        scale_entry = ttk.Entry(parameters, textvariable=self.scale_var, width=12)
        scale_entry.grid(row=0, column=1, padx=10, sticky="w")
        ttk.Label(parameters, text="0.5 缩小一半；1 保持原尺寸；2 放大一倍").grid(
            row=0, column=2, columnspan=2, sticky="w"
        )
        grayscale_check = ttk.Checkbutton(
            parameters, text="转换为灰度图", variable=self.grayscale_var
        )
        grayscale_check.grid(row=1, column=0, columnspan=2, sticky="w", pady=(12, 0))
        recursive_check = ttk.Checkbutton(
            parameters, text="同时处理子文件夹", variable=self.recursive_var
        )
        recursive_check.grid(row=1, column=2, columnspan=2, sticky="w", pady=(12, 0))
        ttk.Label(parameters, text="输入格式").grid(row=2, column=0, sticky="w", pady=(12, 0))
        self.input_format_combo = ttk.Combobox(
            parameters, textvariable=self.input_format_var,
            values=tuple(INPUT_FORMATS), state="readonly", width=23,
        )
        self.input_format_combo.grid(row=2, column=1, sticky="w", padx=10, pady=(12, 0))
        ttk.Label(parameters, text="输出格式").grid(row=2, column=2, sticky="w", pady=(12, 0))
        self.output_format_combo = ttk.Combobox(
            parameters, textvariable=self.output_format_var,
            values=tuple(OUTPUT_FORMATS), state="readonly", width=19,
        )
        self.output_format_combo.grid(row=2, column=3, sticky="w", padx=(10, 0), pady=(12, 0))
        ttk.Label(parameters, textvariable=self.format_hint_var, wraplength=690).grid(
            row=3, column=0, columnspan=4, sticky="w", pady=(12, 0)
        )
        self.controls.extend((
            scale_entry, grayscale_check, recursive_check,
            self.input_format_combo, self.output_format_combo,
        ))

        actions = ttk.Frame(container)
        actions.grid(row=2, column=0, sticky="ew", pady=12)
        self.start_button = ttk.Button(actions, text="开始处理", command=self.start)
        self.start_button.pack(side="left")
        self.controls.append(self.start_button)
        self.stop_button = ttk.Button(actions, text="停止", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=10)
        ttk.Label(container, textvariable=self.status_var).grid(row=3, column=0, sticky="w")
        self.progress = ttk.Progressbar(container, mode="determinate", maximum=1)
        self.progress.grid(row=4, column=0, sticky="ew", pady=(8, 10))
        self.log_box = scrolledtext.ScrolledText(
            container, height=10, wrap="word", state="disabled"
        )
        self.log_box.grid(row=5, column=0, sticky="nsew")

    def browse_input(self):
        input_dir = filedialog.askdirectory(
            parent=self.root,
            title="选择输入图片文件夹",
            initialdir=self.input_var.get() or os.getcwd(),
            mustexist=True,
        )
        if input_dir:
            self.input_var.set(input_dir)
            if not self.output_var.get():
                self.output_var.set(os.path.normpath(input_dir) + "_resized")

    def browse_output(self):
        selected = self.output_var.get()
        initialdir = selected if os.path.isdir(selected) else os.path.dirname(selected)
        output_dir = filedialog.askdirectory(
            parent=self.root,
            title="选择输出文件夹（可新建文件夹）",
            initialdir=initialdir or self.input_var.get() or os.getcwd(),
            mustexist=False,
        )
        if output_dir:
            self.output_var.set(output_dir)

    def append_log(self, message):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def update_format_hint(self, *_):
        extension = OUTPUT_FORMATS.get(self.output_format_var.get(), '.png')
        self.format_hint_var.set(
            f"自然顺序编号：0001{extension}、0002{extension}……\n{FORMAT_NOTES[extension]}"
        )

    def set_running(self, running):
        self.running = running
        for control in self.controls:
            normal_state = "readonly" if isinstance(control, ttk.Combobox) else "normal"
            control.configure(state="disabled" if running else normal_state)
        self.stop_button.configure(state="normal" if running else "disabled")

    def start(self):
        if self.running:
            return
        try:
            input_dir, output_dir, scale = validate_parameters(
                self.input_var.get().strip().strip('"'),
                self.output_var.get().strip().strip('"'),
                self.scale_var.get(),
            )
            input_extensions, output_format = validate_formats(
                INPUT_FORMATS[self.input_format_var.get()],
                OUTPUT_FORMATS[self.output_format_var.get()],
            )
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc), parent=self.root)
            return
        options = (
            input_dir, output_dir, scale, self.grayscale_var.get(), self.recursive_var.get(),
            input_extensions, output_format,
        )
        self.stop_event.clear()
        self.progress.configure(value=0, maximum=1)
        self.status_var.set("正在扫描图片文件……")
        self.append_log(f"\n输入：{input_dir}\n输出：{output_dir}\n缩放比例：{scale}")
        self.append_log(f"输入格式：{self.input_format_var.get()}；输出格式：{self.output_format_var.get()}")
        self.set_running(True)
        # 工作线程仅发送消息；所有窗口更新都在主线程中进行。
        self.worker = threading.Thread(target=self.process, args=options, daemon=True)
        self.worker.start()

    def process(self, input_dir, output_dir, scale, grayscale, recursive, input_extensions, output_format):
        try:
            summary = convert_and_resize_images(
                input_dir, output_dir, scale, grayscale,
                include_subfolders=recursive,
                input_extensions=input_extensions,
                output_format=output_format,
                log_callback=lambda message: self.events.put(("log", message)),
                progress_callback=lambda done, total: self.events.put(("progress", (done, total))),
                stop_event=self.stop_event,
            )
            self.events.put(("done", summary))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def stop(self):
        if self.running:
            self.stop_event.set()
            self.stop_button.configure(state="disabled")
            self.status_var.set("已请求停止，将在当前图片处理完成后结束。")

    def poll_events(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "log":
                    self.append_log(value)
                elif kind == "progress":
                    done, total = value
                    self.progress.configure(maximum=max(1, total), value=done)
                    if not self.stop_event.is_set():
                        self.status_var.set(f"已处理 {done} / {total} 张")
                elif kind == "done":
                    self.set_running(False)
                    state = "已停止" if value["cancelled"] else "处理完成"
                    text = f"{state}：已处理 {value['processed']} / {value['total']} 张，保存 {value['saved']} 张，失败 {value['failed']} 张。"
                    self.status_var.set(text)
                    self.append_log(text)
                elif kind == "error":
                    self.set_running(False)
                    self.status_var.set("处理失败，请查看日志。")
                    self.append_log(f"错误：{value}")
                    if not self.closing:
                        messagebox.showerror("处理失败", value, parent=self.root)
        except queue.Empty:
            pass
        if self.closing and not self.running:
            self.root.destroy()
            return
        self.root.after(100, self.poll_events)

    def close(self):
        if self.running:
            self.closing = True
            self.stop()
        else:
            self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    RenameResizeApp(root)
    root.mainloop()
