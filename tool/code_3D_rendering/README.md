# 3D Rendering (Standalone)

Interactive 3D visualization for aligned histology slices.

## Features
- Remove white background before stacking.
- 3D volume rendering with MIP/MINIP modes.
- Orthogonal slices (XY/XZ/YZ) with sliders.
- 3D clipping to expose the selected planes.

## Run
1. Ensure dependencies are installed (see `requirements.txt`).
2. Launch the GUI:

```powershell
python -u main.py
```

## Notes
- Use the left panel to set spacing, downsample, and background threshold.
- Use the right sliders to move the orthogonal planes.

## Batch image rotation

`img_rotate.py` rotates images 90° clockwise and saves them with the `flipped_` prefix.
Run it directly to use the input/output paths configured at the top of the script,
or pass directories explicitly:

```powershell
python img_rotate.py --input "D:\images" --output "D:\rotated" --workers 8
```

Reading, rotation, and encoding run concurrently (up to 8 workers by default).
Reduce `--workers` for large images or limited memory. PNG encoding keeps the
original OpenCV defaults and remains lossless. The script supports Unicode paths,
reports progress and remaining time, and ignores non-image files and subdirectories.
Existing outputs are replaced only after a complete file has been written.
Unreadable images or write failures are reported and produce a nonzero exit code.

