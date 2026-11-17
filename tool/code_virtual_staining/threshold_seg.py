"""
Threshold-based tissue segmentation tool for UTOM.

Two modes:
  A) Adaptive (Otsu) — auto-compute threshold per image, use mean for all
  B) Manual — user-specified threshold applied to all images

Workflow:
  1. Select images (multi-file picker)
  2. Select output folder (folder picker)
  3. Choose mode: adaptive or manual
  4. (Manual mode) Enter threshold value
  5. Generate mask overlays + thresholds.txt

Output:
  <output_folder>/
      <image_name>_mask.png    -- overlay using selected threshold
      thresholds.txt            -- per-image thresholds + summary
"""

import numpy as np
from pathlib import Path
from PIL import Image
import tkinter as tk
from tkinter import filedialog

Image.MAX_IMAGE_PIXELS = None  # allow large microscopy images


# ---------------------------------------------------------------------------
# Threshold computation
# ---------------------------------------------------------------------------

def otsu_threshold(img_gray: np.ndarray) -> float:
    """Otsu's method — returns threshold in [0, 255]."""
    hist, _ = np.histogram(img_gray.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 128.0

    sum_all = np.dot(np.arange(256), hist)
    weight_bg = 0.0
    sum_bg = 0.0
    max_variance = 0.0
    best_threshold = 0

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        variance_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance_between > max_variance:
            max_variance = variance_between
            best_threshold = t

    return float(best_threshold)


# ---------------------------------------------------------------------------
# Mask & overlay
# ---------------------------------------------------------------------------

def foreground_mask(img_gray: np.ndarray, threshold: float,
                    tissue: str = "auto") -> np.ndarray:
    """Create foreground (tissue) mask.

    Args:
        img_gray: (H, W) uint8 grayscale
        threshold: intensity threshold (0-255)
        tissue: "bright" = bright=tissue, "dark" = dark=tissue, "auto" = auto-detect

    Returns:
        (H, W) bool mask (True = foreground/tissue)
    """
    if tissue == "auto":
        # Auto-detect: if >50% above threshold, tissue is likely dark-on-bright
        mask_above = img_gray > threshold
        tissue = "dark" if mask_above.mean() > 0.5 else "bright"

    if tissue == "bright":
        return img_gray > threshold
    else:
        return img_gray < threshold


def create_overlay(img_gray: np.ndarray, mask: np.ndarray,
                   mask_color=(0, 255, 0), alpha=0.4) -> np.ndarray:
    """Overlay colored mask on grayscale, return RGB uint8."""
    overlay = np.stack([img_gray] * 3, axis=-1).astype(np.float64)
    color = np.array(mask_color, dtype=np.float64)
    overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Mode dialogs
# ---------------------------------------------------------------------------

def choose_mode(root: tk.Tk) -> tuple[str, float | None]:
    """Ask user to select mode and (if manual) threshold value.

    Returns:
        (mode, manual_threshold_or_None)
    """
    dialog = tk.Toplevel(root)
    dialog.title("Threshold Mode")
    dialog.attributes("-topmost", True)
    dialog.geometry("380x280")
    dialog.resizable(False, False)

    result = {"mode": None, "value": None}

    def on_adaptive():
        result["mode"] = "adaptive"
        dialog.destroy()

    def on_manual():
        dialog.destroy()
        # Second dialog for threshold value
        val_dlg = tk.Toplevel(root)
        val_dlg.title("Manual Threshold")
        val_dlg.attributes("-topmost", True)
        val_dlg.geometry("300x140")
        val_dlg.resizable(False, False)

        tk.Label(val_dlg, text="Enter threshold value (0-255):",
                 font=("", 11)).pack(pady=(15, 5))
        entry = tk.Entry(val_dlg, font=("", 14), width=10, justify="center")
        entry.insert(0, "128")
        entry.pack(pady=5)
        entry.select_range(0, tk.END)
        entry.focus_set()

        def ok():
            try:
                v = float(entry.get())
                if 0 <= v <= 255:
                    result["mode"] = "manual"
                    result["value"] = v
                    val_dlg.destroy()
                else:
                    entry.configure(fg="red")
            except ValueError:
                entry.configure(fg="red")

        def cancel():
            val_dlg.destroy()

        btn_frame = tk.Frame(val_dlg)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="OK", width=8, command=ok).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Cancel", width=8, command=cancel).pack(side="left", padx=5)
        val_dlg.bind("<Return>", lambda _e: ok())

        root.wait_window(val_dlg)

    # Main choice dialog
    tk.Label(dialog, text="Choose segmentation method:",
             font=("", 12, "bold")).pack(pady=(25, 20))

    tk.Button(dialog, text="Adaptive (Otsu)", font=("", 11),
              width=30, height=2, command=on_adaptive).pack(pady=5)
    tk.Label(dialog, text="Auto-compute threshold per image, use mean for all",
             fg="gray").pack()

    tk.Button(dialog, text="Manual", font=("", 11),
              width=30, height=2, command=on_manual).pack(pady=(15, 5))
    tk.Label(dialog, text="Apply a fixed threshold to all images",
             fg="gray").pack()

    root.wait_window(dialog)
    return result["mode"], result["value"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    print("=" * 60)
    print("UTOM Threshold Segmentation Tool")
    print("=" * 60)
    print()

    # 1. Select images
    image_paths = filedialog.askopenfilenames(
        title="Select images for segmentation",
        filetypes=[
            ("All images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
            ("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"),
            ("TIFF", "*.tif *.tiff"), ("All files", "*.*"),
        ],
    )
    if not image_paths:
        print("No images selected. Exiting.")
        root.destroy()
        return

    print(f"Selected {len(image_paths)} image(s):")
    for p in image_paths:
        print(f"  - {Path(p).name}")
    print()

    # 2. Select output folder
    output_dir = filedialog.askdirectory(
        title="Select output folder for mask overlays and threshold file",
    )
    if not output_dir:
        print("No output folder selected. Exiting.")
        root.destroy()
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {output_dir}")
    print()

    # 3. Choose mode
    mode, manual_value = choose_mode(root)
    root.destroy()

    if mode is None:
        print("Cancelled. Exiting.")
        return

    # 4. Preload all images
    preloaded = []  # (name, img_array)
    for path in image_paths:
        try:
            img = Image.open(path).convert("L")
            preloaded.append((Path(path).name, np.array(img)))
        except Exception as e:
            print(f"  ERROR loading {Path(path).name}: {e}")

    if not preloaded:
        print("No images loaded successfully.")
        return

    # 5. Determine threshold
    if mode == "adaptive":
        print("Mode: Adaptive (Otsu)")
        print("-" * 40)
        thresholds = []
        for name, img_array in preloaded:
            t = otsu_threshold(img_array)
            thresholds.append(t)
            print(f"  {name}: Otsu = {t:.1f}")
        threshold = float(np.mean(thresholds))
        print(f"  => Mean threshold = {threshold:.1f}")
        mode_label = f"adaptive (Otsu mean = {threshold:.1f})"
    else:
        threshold = manual_value
        print(f"Mode: Manual")
        print(f"  Threshold = {threshold:.1f}")
        # Compute Otsu for reference
        thresholds = [otsu_threshold(arr) for _, arr in preloaded]
        mode_label = f"manual (value = {threshold:.1f})"

    print()

    # 6. Apply threshold to all images
    print("Generating masks and overlays...")
    print("-" * 40)

    for name, img_array in preloaded:
        mask = foreground_mask(img_array, threshold, tissue="auto")
        overlay = create_overlay(img_array, mask)
        out_path = output_dir / (Path(name).stem + "_mask.png")
        Image.fromarray(overlay).save(out_path)
        polarity = "dark-on-bright" if img_array[mask].mean() < img_array[~mask].mean() else "bright-on-dark"
        print(f"  {name}: fg={mask.mean() * 100:.1f}% ({polarity})")

    # 7. Save thresholds.txt
    thresh_file = output_dir / "thresholds.txt"
    with open(thresh_file, "w", encoding="utf-8") as f:
        f.write("# UTOM Threshold Segmentation Results\n")
        f.write(f"# Mode: {mode_label}\n")
        f.write(f"# {len(preloaded)} image(s) processed\n")
        f.write("#\n")
        f.write("# Per-image Otsu thresholds (0-255, for reference):\n")
        for path, t in zip(image_paths, thresholds):
            f.write(f"  {Path(path).name:45s} {t:6.1f}\n")
        f.write("#\n")
        f.write(f"# Applied threshold: {threshold:.1f} ({mode})\n")
        f.write("#\n")
        f.write("# --- Statistics (Otsu, for reference) ---\n")
        t_arr = np.array(thresholds)
        f.write(f"  {'mean':45s} {t_arr.mean():6.1f}\n")
        f.write(f"  {'median':45s} {np.median(t_arr):6.1f}\n")
        f.write(f"  {'std':45s} {t_arr.std():6.1f}\n")
        f.write(f"  {'min':45s} {t_arr.min():6.1f}\n")
        f.write(f"  {'max':45s} {t_arr.max():6.1f}\n")
        f.write("#\n")
        f.write(f"# Recommended --threshold_A (UTOM): {threshold:.0f}\n")

    # 8. Summary
    print("-" * 40)
    print()
    print(f"Mode:      {mode_label}")
    print(f"Threshold: {threshold:.1f}")
    print(f"Images:    {len(preloaded)}")
    print()
    print(f"Output ({output_dir}):")
    print(f"  {len(preloaded)} overlay image(s)")
    print(f"  thresholds.txt")
    print()
    print(f"UTOM training:  --threshold_A  {threshold:.0f}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
