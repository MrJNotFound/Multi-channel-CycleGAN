"""并行读取、顺时针旋转 90° 并保存图像，保持原文件名的 flipped_ 前缀。"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from time import perf_counter

import cv2
import numpy as np


# 直接运行时使用下面的目录，也可以通过 --input / --output 指定。
input_path = r"C:\Users\30927\Desktop\img_histology\stitch\bone\merged_54x50"
output_path = r"C:\Users\30927\Desktop\img_histology\stitch\bone\human_bone"
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".jp2"}


@dataclass
class RotationResult:
    total: int
    succeeded: int
    errors: list[tuple[str, str]]
    elapsed_seconds: float


def rotate_one(source: Path, destination: Path) -> None:
    # OpenCV 的文件路径接口在部分 Windows 版本中不支持中文路径。
    # 保持原脚本的彩色读取和默认编码参数，PNG 的像素不会因提速而改变。
    data = np.fromfile(source, dtype=np.uint8)
    if not data.size:
        raise ValueError("文件为空")
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法解码图像")
    del data
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    del img
    ok, encoded = cv2.imencode(source.suffix, rotated)
    if not ok:
        raise OSError("图像编码失败")
    del rotated

    # 完成写入后再替换目标，写入失败时保留已有的完整结果。
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".img_rotate_",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(memoryview(encoded))
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def rotate_images(source: str | Path, destination: str | Path,
                  workers: int = DEFAULT_WORKERS, *, progress: bool = True) -> RotationResult:
    source, destination = Path(source), Path(destination)
    if workers < 1:
        raise ValueError("并行数量必须至少为 1")
    if not source.is_dir():
        raise ValueError(f"输入目录不存在：{source}")
    if source.resolve() == destination.resolve():
        raise ValueError("输入目录和输出目录不能相同")
    files = sorted(path for path in source.iterdir()
                   if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise ValueError(f"输入目录中没有支持的图像：{source}")
    destination.mkdir(parents=True, exist_ok=True)
    workers = min(workers, len(files))
    if progress:
        print(f"共 {len(files)} 张图像，{workers} 个并行任务，顺时针旋转 90°。", flush=True)
    started = last_report = perf_counter()
    errors = []
    # 解码/编码在 OpenCV 内运行，可用线程并行而无需跨进程传输大图。
    # 每个任务处理完一张才读取下一张，内存占用由 workers 控制。
    previous_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            futures = {pool.submit(rotate_one, path, destination / f"flipped_{path.name}"): path
                       for path in files}
            for completed, future in enumerate(as_completed(futures), 1):
                try:
                    future.result()
                except Exception as error:
                    errors.append((futures[future].name, str(error)))
                    if progress:
                        print(f"\n失败：{futures[future].name}：{error}", flush=True)
                now = perf_counter()
                if progress and (now - last_report >= 0.5 or completed == len(files)):
                    elapsed = now - started
                    rate = completed / max(elapsed, 1e-9)
                    eta = (len(files) - completed) / rate
                    print(f"\r{completed}/{len(files)} | {rate:.2f} 张/秒 | "
                          f"已用 {elapsed:.1f}s | 剩余约 {eta:.0f}s | 失败 {len(errors)}    ",
                          end="\n" if completed == len(files) else "", flush=True)
                    last_report = now
        finally:
            # Ctrl+C 时取消尚未开始的任务，只等当前正在处理的图像收尾。
            pool.shutdown(wait=True, cancel_futures=True)
    finally:
        cv2.setNumThreads(previous_threads)
    return RotationResult(len(files), len(files) - len(errors), errors, perf_counter() - started)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=input_path, help="输入图像目录")
    parser.add_argument("--output", default=output_path, help="输出目录")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"同时处理的图像数，默认 {DEFAULT_WORKERS}；内存不足时调小")
    args = parser.parse_args(argv)
    try:
        result = rotate_images(args.input, args.output, args.workers)
    except (OSError, ValueError) as error:
        parser.exit(1, f"错误：{error}\n")
    print(f"完成：成功 {result.succeeded}，失败 {len(result.errors)}，"
          f"用时 {result.elapsed_seconds:.1f}s。")
    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

