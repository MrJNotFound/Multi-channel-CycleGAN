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

