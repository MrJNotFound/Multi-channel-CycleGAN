"""图形界面/命令行批量旋转图像，支持中文路径、格式筛选和有界并行。"""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
import os
from pathlib import Path
import queue
import sys
import tempfile
import threading
from time import perf_counter
from typing import Callable

import cv2
import numpy as np


DEFAULT_WORKERS = min(4, os.cpu_count() or 1)
IMAGE_SUFFIXES = frozenset({'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp', '.jp2', '.gif'})
INPUT_FORMATS = {
    "全部支持的格式": None,
    "PNG (.png)": ('.png',),
    "JPEG (.jpg / .jpeg)": ('.jpg', '.jpeg'),
    "TIFF (.tif / .tiff)": ('.tif', '.tiff'),
    "BMP (.bmp)": ('.bmp',),
    "WebP (.webp)": ('.webp',),
    "JPEG 2000 (.jp2)": ('.jp2',),
    "GIF (.gif)": ('.gif',),
}
OUTPUT_FORMATS = {
    "保持原格式": None, "PNG (.png)": '.png', "JPEG (.jpg)": '.jpg',
    "TIFF (.tif)": '.tif', "BMP (.bmp)": '.bmp', "WebP (.webp)": '.webp',
}
ROTATIONS = {"顺时针 90°": 90, "逆时针 90°": 270, "旋转 180°": 180, "不旋转，仅转换格式": 0}
ROTATION_CODES = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
_OPENCV_THREAD_LOCK = threading.Lock()


@dataclass(frozen=True)
class RotationOptions:
    rotation: int = 90
    save_grayscale: bool = False
    recursive: bool = False
    input_extensions: tuple[str, ...] | None = None
    output_format: str | None = None
    prefix: str = "flipped_"
    png_compression: int = 1
    jpeg_quality: int = 95

    def __post_init__(self):
        if self.rotation not in (0, 90, 180, 270):
            raise ValueError("旋转角度只能是 0、90、180 或 270（顺时针）。")
        if not isinstance(self.png_compression, int) or not 0 <= self.png_compression <= 9:
            raise ValueError("PNG 压缩级别必须为 0–9 的整数。")
        if not isinstance(self.jpeg_quality, int) or not 1 <= self.jpeg_quality <= 100:
            raise ValueError("JPEG 质量必须为 1–100 的整数。")
        if any(character in self.prefix for character in '<>:"/\\|?*\0'):
            raise ValueError("文件名前缀不能包含路径分隔符或文件名禁用字符。")
        if self.input_extensions is not None:
            extensions = self.input_extensions
            if isinstance(extensions, str):
                extensions = (extensions,)
            extensions = tuple('.' + value.strip().lower().lstrip('.') for value in extensions)
            if not extensions or not set(extensions).issubset(IMAGE_SUFFIXES):
                raise ValueError("输入格式包含不支持的扩展名。")
            object.__setattr__(self, 'input_extensions', extensions)
        if self.output_format is not None:
            extension = '.' + self.output_format.strip().lower().lstrip('.')
            extension = {'.jpeg': '.jpg', '.tiff': '.tif'}.get(extension, extension)
            if extension not in OUTPUT_FORMATS.values():
                raise ValueError("不支持所选输出格式。")
            object.__setattr__(self, 'output_format', extension)


@dataclass
class RotationResult:
    total: int
    succeeded: int
    errors: list[tuple[str, str]]
    elapsed_seconds: float
    processed: int = 0
    cancelled: bool = False
    workers: int = 1


@dataclass(frozen=True)
class RotationProgress:
    completed: int
    total: int
    succeeded: int
    failed: int
    elapsed_seconds: float

    @property
    def images_per_second(self):
        return self.completed / max(self.elapsed_seconds, 1e-9)

    @property
    def remaining_seconds(self):
        return (self.total - self.completed) / self.images_per_second if self.completed else 0


def validate_directories(source, destination, workers):
    if not source or not Path(source).is_dir():
        raise ValueError(f"请选择有效的输入目录：{source}")
    if not destination:
        raise ValueError("请选择输出目录。")
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("并行数量必须至少为 1。")
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if os.path.normcase(str(source)) == os.path.normcase(str(destination)):
        raise ValueError("输入目录和输出目录不能相同。")
    return source, destination


def collect_images(source, destination, options, stop_event):
    extensions = frozenset(options.input_extensions or IMAGE_SUFFIXES)
    destination_key = os.path.normcase(str(destination))
    files = []
    for root, directories, names in os.walk(source):
        if stop_event.is_set():
            break
        if options.recursive:
            directories[:] = [
                name for name in directories
                if os.path.normcase(os.path.join(root, name)) != destination_key
            ]
        else:
            directories.clear()
        files.extend(Path(root) / name for name in names if Path(name).suffix.lower() in extensions)
    return sorted(files)


def plan_outputs(files, source, destination, options):
    jobs, seen, extensions = [], {}, set()
    source_keys = {os.path.normcase(str(path)) for path in files}
    for path in files:
        suffix = options.output_format or path.suffix
        target = destination / path.relative_to(source).parent / f"{options.prefix}{path.stem}{suffix}"
        key = os.path.normcase(str(target))
        if key in source_keys:
            raise ValueError(f"输出会覆盖输入图像：{target}；请更换输出目录或文件名前缀。")
        if key in seen:
            raise ValueError(
                f"转换后文件名重复：{seen[key].name} 和 {path.name} → {target.name}；"
                "请选择单一输入格式，或保持原格式。"
            )
        seen[key] = path
        extensions.add(suffix.lower())
        jobs.append((path, target))
    for extension in extensions:
        if not cv2.haveImageWriter(extension):
            raise ValueError(f"当前 OpenCV 环境不支持保存 {extension}，请改选其他输出格式。")
    return jobs


def read_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    if not data.size:
        raise ValueError("文件为空")
    # 保留灰度、16 位和透明通道，减少灰度大图的解码及内存开销。
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("无法解码图像")
    return image


def encode_image(image, extension, options):
    extension = extension.lower()
    if extension in ('.jpg', '.jpeg', '.bmp', '.webp', '.gif') and image.dtype != np.uint8:
        raise ValueError(f"{extension} 仅支持 8 位图像，当前为 {image.dtype}；请选择 PNG 或 TIFF")
    if extension in ('.png', '.jp2') and image.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"{extension} 仅支持 8/16 位图像，当前为 {image.dtype}；请选择 TIFF")
    if extension in ('.jpg', '.jpeg') and image.ndim == 3 and image.shape[2] == 4:
        raise ValueError("JPEG 不支持透明通道；请选择 PNG、TIFF 或 WebP")
    parameters = []
    if extension == '.png':
        parameters = [cv2.IMWRITE_PNG_COMPRESSION, options.png_compression]
    elif extension in ('.jpg', '.jpeg'):
        parameters = [cv2.IMWRITE_JPEG_QUALITY, options.jpeg_quality]
    ok, encoded = cv2.imencode(extension, image, parameters)
    if not ok:
        raise OSError("图像编码失败")
    return encoded


def atomic_write(destination, encoded):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=".img_rotate_", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(memoryview(encoded))
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def rotate_one(source: Path, destination: Path, options: RotationOptions | None = None) -> None:
    options = options or RotationOptions()
    image = read_image(source)
    if options.save_grayscale and image.ndim == 3:
        mode = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        image = cv2.cvtColor(image, mode)
    if options.rotation:
        # 直角旋转直接重排像素，不做仿射插值。
        image = cv2.rotate(image, ROTATION_CODES[options.rotation])
    encoded = encode_image(image, destination.suffix, options)
    del image
    atomic_write(destination, encoded)


def rotate_images(
    source: str | Path,
    destination: str | Path,
    workers: int = DEFAULT_WORKERS,
    *,
    progress: bool = True,
    options: RotationOptions | None = None,
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[RotationProgress], None] | None = None,
    stop_event: threading.Event | None = None,
    progress_interval: float = 0.2,
) -> RotationResult:
    source, destination = validate_directories(source, destination, workers)
    options = options or RotationOptions()
    stop_event = stop_event if stop_event is not None else threading.Event()
    log = log_callback if log_callback is not None else (print if progress else lambda _: None)
    started = perf_counter()
    log("正在扫描图片文件……")
    files = collect_images(source, destination, options, stop_event)
    if stop_event.is_set():
        return RotationResult(len(files), 0, [], perf_counter() - started, cancelled=True)
    if not files:
        raise ValueError("输入目录中没有符合所选格式的图像。")
    jobs = plan_outputs(files, source, destination, options)
    for directory in {target.parent for _, target in jobs}:
        directory.mkdir(parents=True, exist_ok=True)
    workers = min(workers, len(jobs))
    result = RotationResult(len(jobs), 0, [], 0, workers=workers)
    log(f"共 {len(jobs)} 张图像，并行任务数 {workers}，顺时针角度 {options.rotation}°。")
    last_report = started

    def report(force=False):
        nonlocal last_report
        now = perf_counter()
        if not force and now - last_report < progress_interval:
            return
        update = RotationProgress(result.processed, result.total, result.succeeded, len(result.errors), now - started)
        if progress_callback is not None:
            progress_callback(update)
        if progress:
            print(f"{update.completed}/{update.total} | {update.images_per_second:.2f} 张/秒 | "
                  f"已用 {update.elapsed_seconds:.1f}s | 剩余约 {update.remaining_seconds:.0f}s | "
                  f"失败 {update.failed}", flush=True)
        last_report = now

    def execute(job):
        if stop_event.is_set():
            return False
        rotate_one(*job, options)
        return True

    def record(job, action):
        try:
            if not action():
                return
            result.succeeded += 1
        except Exception as error:
            name = str(job[0].relative_to(source))
            result.errors.append((name, str(error)))
            log(f"失败：{name}：{error}")
        result.processed += 1
        report()

    report(force=True)
    # 控制 OpenCV 内部线程，避免与图片级并行叠加；批次结束恢复设置。
    with _OPENCV_THREAD_LOCK:
        previous_threads = cv2.getNumThreads()
        if workers > 1:
            cv2.setNumThreads(1)
        try:
            if workers == 1:
                for job in jobs:
                    if stop_event.is_set():
                        break
                    record(job, lambda job=job: execute(job))
            else:
                iterator = iter(jobs)
                pending = {}
                pool = ThreadPoolExecutor(max_workers=workers)
                try:
                    while True:
                        # 待执行任务最多为并发数的两倍，限制调度开销并及时响应停止。
                        while not stop_event.is_set() and len(pending) < workers * 2:
                            job = next(iterator, None)
                            if job is None:
                                break
                            pending[pool.submit(execute, job)] = job
                        if not pending:
                            break
                        done, _ = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                        for future in done:
                            job = pending.pop(future)
                            if not future.cancelled():
                                record(job, future.result)
                        if stop_event.is_set():
                            for future in list(pending):
                                if future.cancel():
                                    pending.pop(future)
                except BaseException:
                    stop_event.set()
                    raise
                finally:
                    pool.shutdown(wait=True, cancel_futures=True)
        finally:
            if workers > 1:
                cv2.setNumThreads(previous_threads)
    result.cancelled = stop_event.is_set() and result.processed < result.total
    result.elapsed_seconds = perf_counter() - started
    report(force=True)
    return result


class RotationApp:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import scrolledtext, ttk

        self.root, self.ttk = root, ttk
        root.title("图像批量旋转与格式转换")
        root.geometry("900x740")
        root.minsize(840, 680)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.events = queue.Queue()
        self.stop_event = threading.Event()
        self.running = self.closing = False
        self.worker = None
        self.input_var = tk.StringVar(root)
        self.output_var = tk.StringVar(root)
        self.rotation_var = tk.StringVar(root, value="顺时针 90°")
        self.workers_var = tk.StringVar(root, value=str(DEFAULT_WORKERS))
        self.input_format_var = tk.StringVar(root, value="全部支持的格式")
        self.output_format_var = tk.StringVar(root, value="保持原格式")
        self.prefix_var = tk.StringVar(root, value="flipped_")
        self.png_compression_var = tk.StringVar(root, value="1")
        self.jpeg_quality_var = tk.StringVar(root, value="95")
        self.grayscale_var = tk.BooleanVar(root, value=False)
        self.recursive_var = tk.BooleanVar(root, value=False)
        self.status_var = tk.StringVar(root, value="请选择文件夹并设置旋转参数。")
        self.format_hint_var = tk.StringVar(root)
        self.controls = []

        container = ttk.Frame(root, padding=16)
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
        for column in (1, 3):
            parameters.columnconfigure(column, weight=1)

        def add_control(row, column, label, variable, values=None, bounds=None):
            ttk.Label(parameters, text=label).grid(row=row, column=column, sticky="w", pady=5)
            if values is not None:
                control = ttk.Combobox(parameters, textvariable=variable, values=tuple(values), state="readonly", width=24)
            elif bounds is not None:
                control = ttk.Spinbox(parameters, textvariable=variable, from_=bounds[0], to=bounds[1], width=10)
            else:
                control = ttk.Entry(parameters, textvariable=variable, width=26)
            control.grid(row=row, column=column + 1, sticky="ew", padx=(10, 20 if column == 0 else 0), pady=5)
            self.controls.append(control)
            return control

        self.rotation_combo = add_control(0, 0, "旋转方式", self.rotation_var, values=ROTATIONS)
        add_control(0, 2, "并行任务数", self.workers_var, bounds=(1, max(32, os.cpu_count() or 1)))
        self.input_format_combo = add_control(1, 0, "输入格式", self.input_format_var, values=INPUT_FORMATS)
        self.output_format_combo = add_control(1, 2, "输出格式", self.output_format_var, values=OUTPUT_FORMATS)
        add_control(2, 0, "文件名前缀", self.prefix_var)
        self.png_spin = add_control(2, 2, "PNG 压缩级别", self.png_compression_var, bounds=(0, 9))
        self.jpeg_spin = add_control(3, 0, "JPEG 质量", self.jpeg_quality_var, bounds=(1, 100))
        ttk.Label(parameters, text="PNG：0 最快，9 压缩最强；大图建议并发 1–2。", wraplength=350).grid(
            row=3, column=2, columnspan=2, sticky="w", pady=5
        )
        for column, text, variable in (
            (0, "转换为灰度图", self.grayscale_var),
            (2, "同时处理子文件夹（保留目录结构）", self.recursive_var),
        ):
            control = ttk.Checkbutton(parameters, text=text, variable=variable)
            control.grid(row=4, column=column, columnspan=2, sticky="w", pady=5)
            self.controls.append(control)
        ttk.Label(parameters, textvariable=self.format_hint_var, wraplength=790).grid(
            row=5, column=0, columnspan=4, sticky="w", pady=(8, 0)
        )
        actions = ttk.Frame(container)
        actions.grid(row=2, column=0, sticky="ew", pady=12)
        self.start_button = ttk.Button(actions, text="开始处理", command=self.start)
        self.start_button.pack(side="left")
        self.controls.append(self.start_button)
        self.stop_button = ttk.Button(actions, text="停止", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=10)
        ttk.Label(container, textvariable=self.status_var, wraplength=790).grid(row=3, column=0, sticky="w")
        self.progress = ttk.Progressbar(container, maximum=1, mode="determinate")
        self.progress.grid(row=4, column=0, sticky="ew", pady=(8, 10))
        self.log_box = scrolledtext.ScrolledText(container, height=10, wrap="word", state="disabled")
        self.log_box.grid(row=5, column=0, sticky="nsew")
        self.output_format_var.trace_add("write", self.refresh_format_controls)
        self.refresh_format_controls()
        root.after(100, self.poll_events)

    def browse_input(self):
        from tkinter import filedialog
        selected = filedialog.askdirectory(parent=self.root, title="选择输入图片文件夹", mustexist=True)
        if selected:
            self.input_var.set(selected)
            if not self.output_var.get():
                self.output_var.set(os.path.normpath(selected) + "_rotated")

    def browse_output(self):
        from tkinter import filedialog
        selected = filedialog.askdirectory(
            parent=self.root, title="选择输出文件夹（可新建）", mustexist=False,
            initialdir=os.path.dirname(self.output_var.get()) or self.input_var.get() or os.getcwd(),
        )
        if selected:
            self.output_var.set(selected)

    def refresh_format_controls(self, *_):
        extension = OUTPUT_FORMATS[self.output_format_var.get()]
        note = {
            None: "保留原图位深和通道；格式不兼容时会记录失败。直角旋转不做插值。",
            '.png': "PNG：无损，支持 8/16 位和透明通道。压缩级别只影响速度和文件大小。",
            '.jpg': "JPEG：有损，仅支持 8 位、无透明通道；不兼容图像会记录失败。",
            '.tif': "TIFF：支持 16 位和透明通道。",
            '.bmp': "BMP：仅支持 8 位图像，文件通常较大。",
            '.webp': "WebP：仅支持 8 位，支持透明通道；灰度内容保存为三通道。",
        }[extension]
        self.format_hint_var.set(note)
        if not self.running:
            self.png_spin.configure(state="normal" if extension in (None, '.png') else "disabled")
            self.jpeg_spin.configure(state="normal" if extension in (None, '.jpg') else "disabled")

    def append_log(self, message):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def set_running(self, running):
        self.running = running
        for control in self.controls:
            state = "readonly" if isinstance(control, self.ttk.Combobox) else "normal"
            control.configure(state="disabled" if running else state)
        self.stop_button.configure(state="normal" if running else "disabled")
        self.refresh_format_controls()

    def start(self):
        from tkinter import messagebox
        if self.running:
            return
        try:
            workers = int(self.workers_var.get())
            source, destination = validate_directories(
                self.input_var.get().strip().strip('"'), self.output_var.get().strip().strip('"'), workers
            )
            output_format = OUTPUT_FORMATS[self.output_format_var.get()]
            options = RotationOptions(
                rotation=ROTATIONS[self.rotation_var.get()],
                save_grayscale=self.grayscale_var.get(), recursive=self.recursive_var.get(),
                input_extensions=INPUT_FORMATS[self.input_format_var.get()],
                output_format=output_format, prefix=self.prefix_var.get(),
                png_compression=int(self.png_compression_var.get()) if output_format in (None, '.png') else 1,
                jpeg_quality=int(self.jpeg_quality_var.get()) if output_format in (None, '.jpg') else 95,
            )
        except (ValueError, KeyError) as error:
            messagebox.showerror("参数错误", str(error), parent=self.root)
            return
        self.stop_event.clear()
        self.progress.configure(value=0, maximum=1)
        self.status_var.set("正在扫描图片文件……")
        self.append_log(f"\n输入：{source}\n输出：{destination}\n{self.rotation_var.get()}；"
                        f"{workers} 个并行任务；输出：{self.output_format_var.get()}")
        self.set_running(True)
        self.worker = threading.Thread(target=self.process, args=(source, destination, workers, options), daemon=True)
        self.worker.start()

    def process(self, source, destination, workers, options):
        try:
            result = rotate_images(
                source, destination, workers, progress=False, options=options, stop_event=self.stop_event,
                log_callback=lambda text: self.events.put(("log", text)),
                progress_callback=lambda update: self.events.put(("progress", update)),
            )
            self.events.put(("done", result))
        except Exception as error:
            self.events.put(("error", str(error)))

    def stop(self):
        if self.running:
            self.stop_event.set()
            self.stop_button.configure(state="disabled")
            self.status_var.set("已请求停止，正在完成当前运行的图片。")

    def poll_events(self):
        from tkinter import messagebox
        try:
            for _ in range(200):
                kind, value = self.events.get_nowait()
                if kind == "log":
                    self.append_log(value)
                elif kind == "progress":
                    self.progress.configure(maximum=max(1, value.total), value=value.completed)
                    if not self.stop_event.is_set():
                        text = (
                            f"已处理 {value.completed}/{value.total} | {value.images_per_second:.2f} 张/秒 | "
                            f"剩余约 {value.remaining_seconds:.0f} 秒 | 失败 {value.failed}"
                        )
                        if not value.completed:
                            text = f"共 {value.total} 张图像，正在处理首批图片……"
                        self.status_var.set(text)
                elif kind == "done":
                    self.set_running(False)
                    state = "已停止" if value.cancelled else "处理完成"
                    text = (f"{state}：已处理 {value.processed}/{value.total}，成功 {value.succeeded}，"
                            f"失败 {len(value.errors)}，用时 {value.elapsed_seconds:.1f} 秒。")
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


def launch_gui():
    import tkinter as tk
    root = tk.Tk()
    RotationApp(root)
    root.mainloop()
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        return launch_gui()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true", help="打开参数设置窗口")
    parser.add_argument("--input", help="输入图像目录")
    parser.add_argument("--output", help="输出目录")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="同时处理的图像数；大图建议 1–2")
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=90, help="顺时针角度")
    parser.add_argument("--input-formats", nargs='+', help="筛选输入扩展名，例如 png jpg jpeg")
    parser.add_argument("--format", choices=('same', 'png', 'jpg', 'tif', 'bmp', 'webp'), default='same', help="输出格式")
    parser.add_argument("--grayscale", action="store_true", help="转换为灰度图")
    parser.add_argument("--recursive", action="store_true", help="处理子文件夹，保留目录结构")
    parser.add_argument("--prefix", default="flipped_", help="输出文件名前缀")
    parser.add_argument("--png-compression", type=int, default=1, help="PNG 压缩级别 0–9；0 最快")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG 质量 1–100")
    args = parser.parse_args(argv)
    if args.gui:
        return launch_gui()
    if not args.input or not args.output:
        parser.error("命令行模式需要同时指定 --input 和 --output；直接运行脚本可打开窗口。")
    try:
        options = RotationOptions(
            rotation=args.rotation, save_grayscale=args.grayscale, recursive=args.recursive,
            input_extensions=args.input_formats, output_format=None if args.format == 'same' else args.format,
            prefix=args.prefix, png_compression=args.png_compression, jpeg_quality=args.jpeg_quality,
        )
        result = rotate_images(args.input, args.output, args.workers, options=options)
    except (OSError, ValueError) as error:
        parser.exit(1, f"错误：{error}\n")
    except KeyboardInterrupt:
        print("\n已中止处理。")
        return 130
    print(f"完成：成功 {result.succeeded}，失败 {len(result.errors)}，用时 {result.elapsed_seconds:.1f}s。")
    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

