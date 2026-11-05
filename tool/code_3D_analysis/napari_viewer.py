"""
Open the synthetic glomerulus dataset in napari with two label layers:

  Layer 1 — "原始数据集"   原始 label（1–7 肾小球，100+ 假阳性）
  Layer 2 — "追踪结果"      追踪后的 label（FP 滤除，缺失插值补全）
                             同一 track ID 跨层一致

Usage:
    python napari_viewer.py

Scale (μm/pixel):  Z=5.0,  Y=0.5,  X=0.5
"""

import sys, os
import numpy as np
import napari

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_cell_tracking_v2 import track_glomeruli, build_detection_to_track

np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════
# 1.  Generate dataset
# ══════════════════════════════════════════════════════════════════════
IMG_SIZE = 800
SLICE_SPACING = 5.0
PIXEL_SIZE = 0.5
N_SLICES = 60
N_GLOM = 7

_YY, _XX = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]


def make_circle_mask(cx, cy, r, _):
    return (_XX - cx)**2 + (_YY - cy)**2 <= r**2


def make_blob_mask(cx, cy, r, shp):
    dx = _XX - cx; dy = _YY - cy
    ca, sa = np.cos(shp['rot']), np.sin(shp['rot'])
    xr = dx * ca + dy * sa; yr = -dx * sa + dy * ca
    rr = np.sqrt(xr**2 + (yr / shp['axis_ratio'])**2)
    theta = np.arctan2(yr, xr)
    mod = np.ones_like(theta)
    for k, a, p in shp['harm']:
        mod += a * np.cos(k * theta + p)
    return rr <= r * mod


def make_glom_shape():
    harm = [(int(k), float(np.random.uniform(0.08, 0.22)),
             float(np.random.uniform(0, 2 * np.pi)))
            for k in np.random.choice([2, 3, 4, 5], size=3, replace=False)]
    return {'axis_ratio': float(np.random.uniform(0.72, 0.92)),
            'rot': float(np.random.uniform(0, 2 * np.pi)), 'harm': harm}


# Positions
glomeruli = []
for _ in range(5000):
    if len(glomeruli) >= N_GLOM: break
    r_um = np.random.uniform(50, 80); r_px = r_um / PIXEL_SIZE; rz_sl = r_um / SLICE_SPACING
    cx = np.random.uniform(r_px + 20, IMG_SIZE - r_px - 20)
    cy = np.random.uniform(r_px + 20, IMG_SIZE - r_px - 20)
    zc = np.random.uniform(rz_sl + 3, N_SLICES - rz_sl - 3)
    ok = True
    for (ex, ey, ez, er_px, erz) in glomeruli:
        dxy = np.hypot(cx - ex, cy - ey); dz = abs(zc - ez)
        sep = np.sqrt((dxy/(r_px + er_px))**2 + (dz/(rz_sl + erz))**2)
        xy_frac = dxy/(r_px + er_px); z_extent_gap = dz - (rz_sl + erz)
        if sep < 1.05 or (xy_frac < 0.45 and z_extent_gap < 5): ok = False; break
    if ok: glomeruli.append((cx, cy, zc, r_px, rz_sl))

glom_shapes = [make_glom_shape() for _ in range(N_GLOM)]
glom_jitter = {gid: np.random.uniform(-0.08, 0.08, N_SLICES) for gid in range(N_GLOM)}

# Defect assignments
best_pair = None; best_score = float('inf')
for i in range(N_GLOM):
    for j in range(i + 1, N_GLOM):
        xi, yi, zi, ri, rzi = glomeruli[i]; xj, yj, zj, rj, rzj = glomeruli[j]
        dxy = np.hypot(xi - xj, yi - yj)
        z_overlap = min(zi + rzi, zj + rzj) - max(zi - rzi, zj - rzj)
        if z_overlap > 3 and dxy < 250 and dxy < best_score: best_score = dxy; best_pair = (i, j)

MERGE_G1, MERGE_G2 = best_pair
zi1, rzi1 = glomeruli[MERGE_G1][2], glomeruli[MERGE_G1][4]
zi2, rzi2 = glomeruli[MERGE_G2][2], glomeruli[MERGE_G2][4]
z_common = int((max(zi1 - rzi1, zi2 - rzi2) + min(zi1 + rzi1, zi2 + rzi2)) / 2)
MERGE_SLICES = [max(0, z_common - 1), z_common]

used = {MERGE_G1, MERGE_G2}
avail = [i for i in range(N_GLOM) if i not in used]
MISSING_GLOM = avail[0]; SPLIT_GLOM = avail[1]
gz_m = glomeruli[MISSING_GLOM][2]
MISSING_START = int(round(gz_m)); MISSING_END = MISSING_START + 1
MISSING2_GLOM = avail[3] if len(avail) > 3 else avail[-1]
MISSING2_SLICE = int(round(glomeruli[MISSING2_GLOM][2]))
SPLIT_SLICE = int(glomeruli[SPLIT_GLOM][2])


def glom_radius_at(gid, z):
    cx, cy, zc, r_px, rz_sl = glomeruli[gid]
    dz = abs(z - zc); return r_px * np.sqrt(max(0.0, 1 - (dz/rz_sl)**2)) if dz < rz_sl else 0.0


_merge_mid_y = (glomeruli[MERGE_G1][1] + glomeruli[MERGE_G2][1]) / 2 / IMG_SIZE
_fp_candidates = [
    (int(N_SLICES * 0.18), 0.72, 0.74, 20, 1),
    (int(N_SLICES * 0.30), 0.16, 0.82, 15, 1),
    (int(N_SLICES * 0.45), 0.87, 0.28, 24, 2),
    (int(N_SLICES * 0.55), 0.30, 0.11, 14, 1),
    (int(N_SLICES * 0.68), 0.63, 0.90, 18, 1),
    (int(N_SLICES * 0.80), 0.90, 0.58, 16, 1),
    (MERGE_SLICES[0] + 3, 650.0 / IMG_SIZE, _merge_mid_y, 38, 2),
]

from scipy.ndimage import binary_fill_holes, binary_dilation
import cv2

# ══════════════════════════════════════════════════════════════════════
# 2.  Build raw label volume + per-slice PNGs for tracker
# ══════════════════════════════════════════════════════════════════════
ROOT = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.join(ROOT, 'output', '_track_slices')
os.makedirs(TMP_DIR, exist_ok=True)

raw_volume = np.zeros((N_SLICES, IMG_SIZE, IMG_SIZE), dtype=np.uint16)
# tracker_labels: per-slice label images with UNIQUE IDs for the tracker
tracker_labels = np.zeros((N_SLICES, IMG_SIZE, IMG_SIZE), dtype=np.uint16)
fp_counter = 100

for z in range(N_SLICES):
    lbl_tracker = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint16)
    skip_glom = set()

    if z in MERGE_SLICES:
        g1, g2 = glomeruli[MERGE_G1], glomeruli[MERGE_G2]
        dz1, dz2 = abs(z - g1[2]), abs(z - g2[2])
        t1 = dz1 / g1[4] if g1[4] > 0 else 99
        t2 = dz2 / g2[4] if g2[4] > 0 else 99
        if t1 < 1 and t2 < 1:
            r1 = g1[3] * np.sqrt(max(0, 1 - t1**2))
            r2 = g2[3] * np.sqrt(max(0, 1 - t2**2))
            m1 = make_blob_mask(g1[0], g1[1], r1, glom_shapes[MERGE_G1])
            m2 = make_blob_mask(g2[0], g2[1], r2, glom_shapes[MERGE_G2])
            merged = m1 | m2
            bridge = np.zeros((IMG_SIZE, IMG_SIZE), np.uint8)
            cv2.line(bridge, (int(g1[0]), int(g1[1])), (int(g2[0]), int(g2[1])),
                     1, thickness=max(10, int(min(r1, r2) * 0.75)))
            merged = merged | (bridge > 0)
            merged = binary_fill_holes(binary_dilation(merged, iterations=2))
            if merged.sum() > 30: lbl_tracker[merged] = 99
        skip_glom = {MERGE_G1, MERGE_G2}

    for gid, (cx, cy, zc, r_px, rz_sl) in enumerate(glomeruli):
        if gid in skip_glom: continue
        if gid == MISSING_GLOM and MISSING_START <= z <= MISSING_END: continue
        if gid == MISSING2_GLOM and z == MISSING2_SLICE: continue
        dz = abs(z - zc)
        if dz >= rz_sl: continue
        t = dz / rz_sl
        r_eff = r_px * np.sqrt(max(0, 1 - t**2))
        r_eff *= (1.0 + glom_jitter[gid][z])
        if r_eff < 4: continue
        mask = make_blob_mask(cx, cy, r_eff, glom_shapes[gid])
        if gid == SPLIT_GLOM and z == SPLIT_SLICE and mask.sum() > 40:
            y1 = cy - r_eff * 0.3; y2 = cy + r_eff * 0.3
            m1 = mask & (np.arange(IMG_SIZE)[:, None] < y1)
            m2 = mask & (np.arange(IMG_SIZE)[:, None] >= y1) & (np.arange(IMG_SIZE)[:, None] < y2)
            m3 = mask & (np.arange(IMG_SIZE)[:, None] >= y2)
            for m in [m1, m2, m3]:
                if m.sum() > 20: lbl_tracker[m] = gid + 1
        else:
            if mask.sum() > 20: lbl_tracker[mask] = gid + 1

    if 99 in lbl_tracker: lbl_tracker[lbl_tracker == 99] = MERGE_G1 + 1

    for (z0, fx, fy, fr, span) in _fp_candidates:
        px, py = fx * IMG_SIZE, fy * IMG_SIZE
        if z0 <= z < z0 + span:
            collide = any(np.hypot(px - glomeruli[g][0], py - glomeruli[g][1])
                          < glom_radius_at(g, z) + fr + 6 for g in range(N_GLOM))
            if not collide:
                fp_mask = make_circle_mask(px, py, fr, None)
                if fp_mask.sum() > 15: lbl_tracker[fp_mask] = fp_counter; fp_counter += 1

    # raw_volume: all detections → label 8 (avoid clash with track IDs 1-7)
    raw_volume[z] = (lbl_tracker > 0).astype(np.uint16) * 8
    tracker_labels[z] = lbl_tracker
    cv2.imwrite(os.path.join(TMP_DIR, f'slice_{z:03d}.png'), lbl_tracker.astype(np.uint16))

print(f"Dataset: {N_GLOM} glomeruli, {fp_counter - 100} FP instances")

# ══════════════════════════════════════════════════════════════════════
# 3.  Run tracker & build tracked label volume
# ══════════════════════════════════════════════════════════════════════
paths = [os.path.join(TMP_DIR, f'slice_{z:03d}.png') for z in range(N_SLICES)]
confirmed_tracks, fp_tracks, events = track_glomeruli(
    paths, max_missing=3, confirm_thresh=3, pos_gate=0.45)
print(f"Tracker: {len(confirmed_tracks)} confirmed tracks, {len(fp_tracks)} FP filtered")

# Build per-slice detection→track mapping
detection_to_track = build_detection_to_track(confirmed_tracks)

# ---- Build tracked volume: matched masks + shape-interpolated gaps ----
tracked_volume = np.zeros_like(tracker_labels)

for t in confirmed_tracks:
    # Collect matched z-slices and their masks for this track
    matched_z = []
    matched_masks = {}   # z -> binary mask
    for z in range(N_SLICES):
        dmap = detection_to_track.get(z, {})
        for lid, tt in dmap.items():
            if tt.id == t.id:
                mask = (tracker_labels[z] == lid)
                if mask.sum() > 10:
                    matched_z.append(z)
                    matched_masks[z] = mask
                break  # first match per slice

    if not matched_z:
        continue

    matched_z = sorted(matched_z)

    # Fill matched slices
    for z in matched_z:
        tracked_volume[z][matched_masks[z]] = t.id + 1

    # Fill interpolated gaps with shape morphing
    for idx_gap in range(len(t.history)):
        zz, hx, hy, ha, st = t.history[idx_gap]
        if st != 'interpolated':
            continue
        if tracked_volume[zz].max() > 0 and (tracked_volume[zz] == t.id + 1).sum() > 0:
            # Already filled by a nearby matched mask via tracker_labels remnant
            continue

        # Find nearest matched slices before and after
        before = [mz for mz in matched_z if mz < zz]
        after = [mz for mz in matched_z if mz > zz]

        if before and after:
            zb, za = before[-1], after[0]
            mask_b = matched_masks[zb]
            mask_a = matched_masks[za]
            w = (zz - zb) / (za - zb)

            # Align centroids: shift mask_a so its centroid matches mask_b
            cy_b, cx_b = np.argwhere(mask_b).mean(axis=0)
            cy_a, cx_a = np.argwhere(mask_a).mean(axis=0)
            dy = int(round(cy_b - cy_a))
            dx = int(round(cx_b - cx_a))
            mask_a_shifted = np.roll(mask_a, (dy, dx), axis=(0, 1))

            # Weighted average → threshold
            blend = (1 - w) * mask_b.astype(float) + w * mask_a_shifted.astype(float)
            interp_mask = blend > 0.45
            # Only fill where raw_volume is empty (true gap)
            gap = interp_mask & (tracked_volume[zz] == 0)
            tracked_volume[zz][gap] = t.id + 1

        elif before:
            zb = before[-1]
            tracked_volume[zz][matched_masks[zb]] = t.id + 1

        elif after:
            za = after[0]
            tracked_volume[zz][matched_masks[za]] = t.id + 1

print(f"Tracked volume labels: {sorted(np.unique(tracked_volume))}")
print(f"  1..{len(confirmed_tracks)} = confirmed tracks (FP excluded)")

# ══════════════════════════════════════════════════════════════════════
# 4.  Launch napari  (let napari auto-assign label colours)
# ══════════════════════════════════════════════════════════════════════
SCALE = (SLICE_SPACING, PIXEL_SIZE, PIXEL_SIZE)

viewer = napari.Viewer(title='肾小球合成数据集 — 原始 vs 追踪')

layer_raw = viewer.add_labels(
    raw_volume, name='原始数据集', scale=SCALE,
    opacity=0.50, blending='translucent',
)

layer_tracked = viewer.add_labels(
    tracked_volume, name='追踪结果', scale=SCALE,
    opacity=0.75, blending='translucent',
)

viewer.dims.ndisplay = 3
viewer.camera.angles = (0.0, 30.0, 120.0)
viewer.camera.zoom = 1.5

print("""
Ready.
  原始数据集  — 灰色 (label=1, 含缺失/合并/分裂缺陷)
  追踪结果    — 各 track 独立着色 (FP 已剔除, 缺失间隙形态插值补全)

  Toggle layers with the checkbox in the layer panel.
""")

napari.run()
