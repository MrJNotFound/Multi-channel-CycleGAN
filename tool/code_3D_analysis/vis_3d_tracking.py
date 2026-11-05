"""
3D + 2D cross-section combined visualization of glomerulus instance tracking.

Same 3D instance → same color across all views (3D surface, 2D contours,
tracking centroids).  This produces two figures:
  fig_d — 3D ellipsoid surfaces + tracking chains
  fig_e — 2D cross-sections at key z-slices with track-colored overlays
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from skimage.measure import regionprops, find_contours
from scipy.ndimage import binary_fill_holes, binary_dilation
import cv2

from func_cell_tracking_v2 import track_glomeruli, build_detection_to_track

# ── CJK font ──
import matplotlib.font_manager as fm
for fname in fm.fontManager.ttflist:
    if 'Microsoft YaHei' in fname.name and 'Light' not in fname.name and 'UI' not in fname.name:
        plt.rcParams['font.family'] = fname.name
        break
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════
# 1.  Re-generate the synthetic dataset (same seed → same geometry)
# ══════════════════════════════════════════════════════════════════════
IMG_SIZE = 800
SLICE_SPACING = 5.0
PIXEL_SIZE = 0.5
N_SLICES = 60
N_GLOM = 7

_YY, _XX = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]


def make_circle_mask(cx, cy, r, shp):
    return (_XX - cx)**2 + (_YY - cy)**2 <= r**2


def make_blob_mask(cx, cy, r, shp):
    dx = _XX - cx
    dy = _YY - cy
    ca, sa = np.cos(shp['rot']), np.sin(shp['rot'])
    xr = dx * ca + dy * sa
    yr = -dx * sa + dy * ca
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
            'rot': float(np.random.uniform(0, 2 * np.pi)),
            'harm': harm}


# -- Generate glomeruli --
glomeruli = []
glom_shapes = []
for _ in range(5000):
    if len(glomeruli) >= N_GLOM:
        break
    r_um = np.random.uniform(50, 80)
    r_px = r_um / PIXEL_SIZE
    rz_sl = r_um / SLICE_SPACING
    cx = np.random.uniform(r_px + 20, IMG_SIZE - r_px - 20)
    cy = np.random.uniform(r_px + 20, IMG_SIZE - r_px - 20)
    zc = np.random.uniform(rz_sl + 3, N_SLICES - rz_sl - 3)
    ok = True
    for (ex, ey, ez, er_px, erz) in glomeruli:
        dxy = np.hypot(cx - ex, cy - ey)
        dz = abs(zc - ez)
        sep = np.sqrt((dxy / (r_px + er_px))**2 + (dz / (rz_sl + erz))**2)
        xy_frac = dxy / (r_px + er_px)
        z_extent_gap = dz - (rz_sl + erz)
        if sep < 1.05 or (xy_frac < 0.45 and z_extent_gap < 5):
            ok = False
            break
    if ok:
        glomeruli.append((cx, cy, zc, r_px, rz_sl))

# Generate shapes AFTER all positions are fixed (decouples RNG)
glom_shapes = [make_glom_shape() for _ in range(N_GLOM)]

# -- Per-slice radius jitter (makes cross-sections irregular layer-to-layer) --
glom_jitter = {}
for gid in range(N_GLOM):
    glom_jitter[gid] = np.random.uniform(-0.08, 0.08, N_SLICES)

# -- Assign defects (same logic as run_tracking_v2) --
best_pair = None
best_score = float('inf')
for i in range(len(glomeruli)):
    for j in range(i + 1, len(glomeruli)):
        xi, yi, zi, ri, rzi = glomeruli[i]
        xj, yj, zj, rj, rzj = glomeruli[j]
        dxy = np.hypot(xi - xj, yi - yj)
        z_overlap = min(zi + rzi, zj + rzj) - max(zi - rzi, zj - rzj)
        if z_overlap > 3 and dxy < 250 and dxy < best_score:
            best_score = dxy
            best_pair = (i, j)

MERGE_G1, MERGE_G2 = best_pair
zi1, rzi1 = glomeruli[MERGE_G1][2], glomeruli[MERGE_G1][4]
zi2, rzi2 = glomeruli[MERGE_G2][2], glomeruli[MERGE_G2][4]
z_common = int((max(zi1 - rzi1, zi2 - rzi2) + min(zi1 + rzi1, zi2 + rzi2)) / 2)
MERGE_SLICES = [max(0, z_common - 1), z_common]

used = {MERGE_G1, MERGE_G2}
avail = [i for i in range(len(glomeruli)) if i not in used]
MISSING_GLOM = avail[0]
SPLIT_GLOM = avail[1]
gz_m = glomeruli[MISSING_GLOM][2]
MISSING_START = int(round(gz_m))
MISSING_END = MISSING_START + 1
MISSING2_GLOM = avail[3] if len(avail) > 3 else avail[-1]
MISSING2_SLICE = int(round(glomeruli[MISSING2_GLOM][2]))
SPLIT_SLICE = int(glomeruli[SPLIT_GLOM][2])


def glom_radius_at(gid, z):
    cx, cy, zc, r_px, rz_sl = glomeruli[gid]
    dz = abs(z - zc)
    if dz >= rz_sl:
        return 0.0
    return r_px * np.sqrt(max(0.0, 1 - (dz / rz_sl)**2))


_fp_candidates = [
    (int(N_SLICES * 0.18), 0.72, 0.74, 20, 1),
    (int(N_SLICES * 0.30), 0.16, 0.82, 15, 1),
    (int(N_SLICES * 0.45), 0.87, 0.28, 24, 2),
    (int(N_SLICES * 0.55), 0.30, 0.11, 14, 1),
    (int(N_SLICES * 0.68), 0.63, 0.90, 18, 1),
    (int(N_SLICES * 0.80), 0.90, 0.58, 16, 1),
]
# Add one prominent FP at the section Y but far from all glomeruli in X.
_merge_mid_y = (glomeruli[MERGE_G1][1] + glomeruli[MERGE_G2][1]) / 2 / IMG_SIZE
# Place FP at far-right X=650 (clear of glomeruli), at Y=Y_SECTION.
_fp_candidates.append((MERGE_SLICES[0] + 3, 650.0 / IMG_SIZE, _merge_mid_y, 38, 2))

fp_by_slice = {}
FP_MARKERS = []
for z0, fx, fy, fr, span in _fp_candidates:
    px, py = fx * IMG_SIZE, fy * IMG_SIZE
    for z in range(z0, z0 + span):
        if not (0 <= z < N_SLICES):
            continue
        collide = any(np.hypot(px - glomeruli[g][0], py - glomeruli[g][1])
                      < glom_radius_at(g, z) + fr + 6 for g in range(N_GLOM))
        if collide:
            continue
        fp_by_slice.setdefault(z, []).append((px, py, fr))
        FP_MARKERS.append((z, px, py))

# -- Generate slice labels --
slice_labels = []
for z in range(N_SLICES):
    lbl = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.int32)
    next_id = 1
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
            if merged.sum() > 30:
                lbl[merged] = next_id
                next_id += 1
        skip_glom = {MERGE_G1, MERGE_G2}

    for gid, (cx, cy, zc, r_px, rz_sl) in enumerate(glomeruli):
        if gid in skip_glom:
            continue
        if gid == MISSING_GLOM and MISSING_START <= z <= MISSING_END:
            continue
        if gid == MISSING2_GLOM and z == MISSING2_SLICE:
            continue
        dz = abs(z - zc)
        if dz >= rz_sl:
            continue
        t = dz / rz_sl
        r_eff = r_px * np.sqrt(max(0, 1 - t**2))
        r_eff *= (1.0 + glom_jitter[gid][z])  # per-slice irregularity
        if r_eff < 4:
            continue
        mask = make_blob_mask(cx, cy, r_eff, glom_shapes[gid])
        if gid == SPLIT_GLOM and z == SPLIT_SLICE and mask.sum() > 40:
            y1 = cy - r_eff * 0.3
            y2 = cy + r_eff * 0.3
            m1 = mask & (np.arange(IMG_SIZE)[:, None] < y1)
            m2 = mask & (np.arange(IMG_SIZE)[:, None] >= y1) & (np.arange(IMG_SIZE)[:, None] < y2)
            m3 = mask & (np.arange(IMG_SIZE)[:, None] >= y2)
            for m in [m1, m2, m3]:
                if m.sum() > 20:
                    lbl[m] = next_id
                    next_id += 1
        else:
            if mask.sum() > 20:
                lbl[mask] = next_id
                next_id += 1

    for (px, py, fr) in fp_by_slice.get(z, []):
        fp_mask = make_circle_mask(px, py, fr, None)
        if fp_mask.sum() > 15:
            lbl[fp_mask] = next_id
            next_id += 1

    slice_labels.append(lbl)

# Save slices
TMP_DIR = os.path.join(OUTPUT_DIR, 'slices')
os.makedirs(TMP_DIR, exist_ok=True)
for z, lbl in enumerate(slice_labels):
    cv2.imwrite(os.path.join(TMP_DIR, f'slice_{z:03d}.png'), lbl.astype(np.uint16))

paths = [os.path.join(TMP_DIR, f'slice_{z:03d}.png') for z in range(N_SLICES)]

# ══════════════════════════════════════════════════════════════════════
# 2.  Run Kalman+Hungarian tracking
# ══════════════════════════════════════════════════════════════════════
print("Running Kalman+Hungarian tracking...")
confirmed_tracks, fp_tracks, events = track_glomeruli(
    paths, max_missing=3, confirm_thresh=3, pos_gate=0.45)

# Map each glomerulus → its track
glom_to_track = {}
for gid in range(N_GLOM):
    cx, cy, zc, r_px, rz_sl = glomeruli[gid]
    best_overlap = 0
    best_track = None
    for t in confirmed_tracks:
        overlap = 0
        for (z, hx, hy, ha, status) in t.history:
            dz = abs(z - zc)
            if dz >= rz_sl:
                continue
            d = np.hypot(hx - cx, hy - cy)
            if d < r_px * 0.8:
                overlap += 1
        if overlap > best_overlap:
            best_overlap = overlap
            best_track = t
    if best_overlap >= 3:
        glom_to_track[gid] = best_track

print(f"  Confirmed tracks: {len(confirmed_tracks)}")
print(f"  Glomeruli mapped to tracks: {len(glom_to_track)}")

# Compute the Y position for the X-Z cross-section plane (used by both
# fig_d for the annotation plane and fig_e for the knife-cut slice).
Y_SECTION = int((glomeruli[MERGE_G1][1] + glomeruli[MERGE_G2][1]) / 2)
# Nudge toward nearest FP if none visible
best_d = 9999
for (z0, fx, fy, fr, span) in _fp_candidates:
    d = abs(fy * IMG_SIZE - Y_SECTION)
    if d < best_d:
        best_d = d
if best_d > 50:
    Y_SECTION = int(_fp_candidates[0][1] * IMG_SIZE)
print(f"  X-Z section plane: Y = {Y_SECTION} px")

# ══════════════════════════════════════════════════════════════════════
# 3.  Build per-slice detection→track mapping (for 2D contour colours)
# ══════════════════════════════════════════════════════════════════════
# detection_to_track[z][label_id] → track (or None)
detection_to_track = build_detection_to_track(confirmed_tracks)

# Build a forward track→detection lookup for convenience
# track_det_at_z[track.id][z] → (cx, cy, area, label) or None
track_det_at_z = {}
for t in confirmed_tracks:
    tmap = {}
    for z in range(N_SLICES):
        # look through our detection_to_track for this z
        found = False
        for lid, tt in detection_to_track.get(z, {}).items():
            if tt.id == t.id:
                # get detection info
                for rp in regionprops(slice_labels[z]):
                    if rp.label == lid:
                        tmap[z] = (rp.centroid[1], rp.centroid[0], rp.area, lid)
                        found = True
                        break
                if found:
                    break
        if not found:
            tmap[z] = None
    track_det_at_z[t.id] = tmap

# ══════════════════════════════════════════════════════════════════════
# 4.  Colour palette — one colour per track, consistent across views
# ══════════════════════════════════════════════════════════════════════
# Sort tracks by z-midpoint for colour ordering
track_order = sorted(confirmed_tracks, key=lambda t: np.mean([h[0] for h in t.history]))
track_colors = {}
cmap = plt.cm.tab10
for i, t in enumerate(track_order):
    track_colors[t.id] = cmap(i % 10)

# Also assign a colour to each glomerulus (inherited from its track)
glom_colors = {}
for gid, t in glom_to_track.items():
    glom_colors[gid] = track_colors[t.id]

# ══════════════════════════════════════════════════════════════════════
# 5.  fig_d — 3D schematic: instance shapes + cross-section plane only
# ══════════════════════════════════════════════════════════════════════
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

fig_d = plt.figure(figsize=(16, 11))
ax3d = fig_d.add_subplot(1, 1, 1, projection='3d')

# 5a. Draw glomerulus surfaces as thick, opaque wireframes
theta = np.linspace(0, 2 * np.pi, 40)
phi = np.linspace(0, np.pi, 20)
THETA, PHI = np.meshgrid(theta, phi)

for gid, (cx, cy, zc, r_px, rz_sl) in enumerate(glomeruli):
    shp = glom_shapes[gid]
    color = glom_colors.get(gid, (0.5, 0.5, 0.5))

    a = r_px
    b = r_px * shp['axis_ratio']
    c = rz_sl

    lx = a * np.sin(PHI) * np.cos(THETA)
    ly = b * np.sin(PHI) * np.sin(THETA)
    lz = c * np.cos(PHI)

    ca, sa = np.cos(shp['rot']), np.sin(shp['rot'])
    xr = lx * ca - ly * sa
    yr = lx * sa + ly * ca
    local_theta = np.arctan2(yr, xr)
    mod = np.ones_like(local_theta)
    for k, amp, phase in shp['harm']:
        mod += amp * np.cos(k * local_theta + phase)
    mod = np.clip(mod, 0.5, 1.5)
    xr *= mod
    yr *= mod

    gx = xr + cx
    gy = yr + cy
    gz = lz + zc

    ax3d.plot_wireframe(gx, gy, gz, color=color, alpha=0.55, linewidth=0.6,
                        rstride=1, cstride=2)

# 5b. Cross-section indicator plane at Y = Y_SECTION
_zl = ax3d.get_zlim()
_yl = ax3d.get_ylim()
_xl = ax3d.get_xlim()
plane_verts = [[(_xl[0], _yl[0], Y_SECTION),
                (_xl[1], _yl[0], Y_SECTION),
                (_xl[1], _yl[1], Y_SECTION),
                (_xl[0], _yl[1], Y_SECTION)]]
plane = Poly3DCollection(plane_verts, alpha=0.22, color='gold',
                          edgecolor='goldenrod', linewidth=1.0, linestyle='--')
ax3d.add_collection3d(plane)
ax3d.text(_xl[1], _yl[1], Y_SECTION, f' X-Z截面 Y={Y_SECTION} ', fontsize=9,
          color='darkgoldenrod', fontweight='bold', ha='right')

ax3d.set_xlabel('Z (slice)', fontsize=11)
ax3d.set_ylabel('X (pixels)', fontsize=11)
ax3d.set_zlabel('Y (pixels)', fontsize=11)
ax3d.set_title('合成数据集三维示意图  (金色平面 = X-Z 截面位置)',
               fontsize=13, fontweight='bold')
# Fix squeezed aspect: Z (slice) vs X/Y (pixels) — 1 slice ≈ 10 px visually
ax3d.set_box_aspect((N_SLICES, IMG_SIZE, IMG_SIZE))
ax3d.view_init(elev=22, azim=-55)

fig_d.tight_layout()
fig_d.savefig(os.path.join(OUTPUT_DIR, 'fig_d_3d_surface_tracking.png'), dpi=200, bbox_inches='tight')
plt.close(fig_d)
print("  fig_d saved.")

# ══════════════════════════════════════════════════════════════════════
# 6.  fig_e — X-Z cross-section at Y = Y_SECTION (merge pair + FP)
# ══════════════════════════════════════════════════════════════════════
y0 = Y_SECTION

fig_e, ax = plt.subplots(1, 1, figsize=(18, 9))
ax.invert_yaxis()

# -- Precompute per-glomerulus ground-truth masks --
glom_masks = {}
for gid, (cx, cy, zc, r_px, rz_sl) in enumerate(glomeruli):
    glom_masks[gid] = {}
    for z in range(N_SLICES):
        dz = abs(z - zc)
        if dz >= rz_sl:
            glom_masks[gid][z] = None
            continue
        t_s = dz / rz_sl
        r_eff = r_px * np.sqrt(max(0, 1 - t_s**2))
        r_eff *= (1.0 + glom_jitter[gid][z])
        if r_eff < 3:
            glom_masks[gid][z] = None
            continue
        if gid == MISSING_GLOM and MISSING_START <= z <= MISSING_END:
            glom_masks[gid][z] = None; continue
        if gid == MISSING2_GLOM and z == MISSING2_SLICE:
            glom_masks[gid][z] = None; continue
        if gid in {MERGE_G1, MERGE_G2} and z in MERGE_SLICES:
            glom_masks[gid][z] = None; continue
        glom_masks[gid][z] = make_blob_mask(cx, cy, r_eff, glom_shapes[gid])


def find_x_ranges(row_vals):
    if len(row_vals) == 0:
        return []
    runs, cur_label, cur_start = [], row_vals[0], 0
    for x in range(1, len(row_vals)):
        if row_vals[x] != cur_label:
            if cur_label > 0:
                runs.append((cur_start, x - 1, cur_label))
            cur_label, cur_start = row_vals[x], x
    if cur_label > 0:
        runs.append((cur_start, len(row_vals) - 1, cur_label))
    return runs


# -- 6a. Ground-truth outlines (dashed) --
for gid, (cx, cy, zc, r_px, rz_sl) in enumerate(glomeruli):
    color = glom_colors.get(gid, (0.5, 0.5, 0.5))
    outline_z, outline_xl, outline_xr = [], [], []
    for z in range(N_SLICES):
        mask = glom_masks[gid].get(z)
        if mask is None:
            if outline_z:
                ax.plot(outline_xl, outline_z, '--', color=color, linewidth=1.0, alpha=0.35)
                ax.plot(outline_xr, outline_z, '--', color=color, linewidth=1.0, alpha=0.35)
                outline_z, outline_xl, outline_xr = [], [], []
            continue
        row = mask[y0, :]
        if not row.any():
            if outline_z:
                ax.plot(outline_xl, outline_z, '--', color=color, linewidth=1.0, alpha=0.35)
                ax.plot(outline_xr, outline_z, '--', color=color, linewidth=1.0, alpha=0.35)
                outline_z, outline_xl, outline_xr = [], [], []
            continue
        xs = np.where(row)[0]
        outline_z.append(z); outline_xl.append(xs[0]); outline_xr.append(xs[-1])
    if outline_z:
        ax.plot(outline_xl, outline_z, '--', color=color, linewidth=1.0, alpha=0.35)
        ax.plot(outline_xr, outline_z, '--', color=color, linewidth=1.0, alpha=0.35)

# -- 6b. Detected instances — filled rectangles, NO inter-layer gaps --
det_map_z = detection_to_track
for z in range(N_SLICES):
    lbl = slice_labels[z]
    row = lbl[y0, :]
    dmap = det_map_z.get(z, {})
    for x0, x1, lid in find_x_ranges(row):
        t = dmap.get(lid)
        color = track_colors[t.id] if t else 'red'
        alpha = 0.60 if t else 0.55
        # exact ±0.5 → rectangles touch, zero gap
        rect = plt.Rectangle((x0, z - 0.5), x1 - x0 + 1, 1.0,
                             facecolor=color, alpha=alpha, edgecolor='none')
        ax.add_patch(rect)
        # Highlight FP with a red border
        if t is None:
            rect2 = plt.Rectangle((x0, z - 0.5), x1 - x0 + 1, 1.0,
                                  facecolor='none', edgecolor='red',
                                  linewidth=0.8, linestyle='-', alpha=0.9)
            ax.add_patch(rect2)

# -- 6c. Tracking path lines (follow cross-section centre, not centroid) --
# For each track at each z, compute the X-midpoint of its filled region
# at Y=y0.  This guarantees the path runs through the visible structure.
track_x_centers = {}   # track_id → {z: x_center}
track_status_at_z = {}  # track_id → {z: 'matched'|'interpolated'}
for z in range(N_SLICES):
    lbl = slice_labels[z]
    row = lbl[y0, :]
    dmap = det_map_z.get(z, {})
    track_runs = {}   # track_id → list of (x0, x1)
    for x0, x1, lid in find_x_ranges(row):
        t = dmap.get(lid)
        if t is not None:
            track_runs.setdefault(t.id, []).append((x0, x1))
    for tid, runs in track_runs.items():
        # Use the overall X-midpoint across all runs for this track
        all_x = []
        for x0, x1 in runs:
            all_x.extend(range(x0, x1 + 1))
        if all_x:
            track_x_centers.setdefault(tid, {})[z] = (min(all_x) + max(all_x)) / 2

# Also collect interpolated points from tracker history
for t in confirmed_tracks:
    for (zz, hx, hy, ha, st) in t.history:
        if st == 'interpolated' and abs(hy - y0) <= 30:
            # Only add if no matched detection at this z
            if zz not in track_x_centers.get(t.id, {}):
                track_x_centers.setdefault(t.id, {})[zz] = hx
                track_status_at_z.setdefault(t.id, {})[zz] = 'interpolated'

for t in confirmed_tracks:
    if t.id not in track_x_centers:
        continue
    color = track_colors[t.id]
    centers = track_x_centers[t.id]
    zs = sorted(centers.keys())
    xs = [centers[z] for z in zs]
    ax.plot(xs, zs, '-', color=color, linewidth=2.5, alpha=0.85, zorder=8)

# -- 6d. Tracking centroids --
for t in confirmed_tracks:
    color = track_colors[t.id]
    for (zz, hx, hy, ha, st) in t.history:
        if abs(hy - y0) > 30:
            continue
        if st == 'interpolated':
            ax.plot(hx, zz, 's', color=color, markersize=8,
                    markerfacecolor='none', markeredgewidth=2.2, zorder=10)
        else:
            ax.plot(hx, zz, 'o', color='white', markersize=8, zorder=9)
            ax.plot(hx, zz, 'o', color=color, markersize=5.5, zorder=10)

# -- 6e. FP centroid markers (red X) --
for t in fp_tracks:
    for (zz, hx, hy, ha, st) in t.history:
        if abs(hy - y0) > 15:
            continue
        ax.plot(hx, zz, 'x', color='red', markersize=7, markeredgewidth=1.5, zorder=10)

# -- 6f.  Annotate the largest FP (at far-right X ~650, away from instances) --
fp_labeled = False
for z in range(N_SLICES):
    lbl = slice_labels[z]
    row = lbl[y0, :]
    dmap = det_map_z.get(z, {})
    for x0, x1, lid in find_x_ranges(row):
        t = dmap.get(lid)
        # FP: not tracked, large enough, near X=650
        if t is None and (x1 - x0) > 25 and x0 > 500 and not fp_labeled:
            ax.annotate('假阳性\n(FP 过滤)', xy=((x0 + x1) / 2, z),
                        xytext=((x0 + x1) / 2 - 100, z - 8),
                        fontsize=11, fontweight='bold', color='darkred',
                        arrowprops=dict(arrowstyle='->', color='darkred', lw=2.0),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  edgecolor='darkred', alpha=0.85),
                        zorder=20)
            fp_labeled = True
            break
    if fp_labeled:
        break

ax.set_xlabel('X (pixels)', fontsize=13)
ax.set_ylabel('Z (slice)', fontsize=13)
ax.set_xlim(0, IMG_SIZE)
ax.set_title(f'X-Z 截面  (Y = {y0} px) — 合并对 · 缺失插值 · FP 过滤', fontsize=13, fontweight='bold')

# ---- Legend ----
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
legend_handles, legend_labels = [], []
for gid in sorted(glom_to_track.keys()):
    t = glom_to_track[gid]
    color = track_colors[t.id]
    legend_handles.append(Patch(facecolor=color, alpha=0.60))
    legend_labels.append(f'G{gid} → Track {t.id}')
legend_handles.append(Patch(facecolor='red', alpha=0.55, edgecolor='red', linewidth=0.8))
legend_labels.append('FP 检测 (假阳性)')
legend_handles.append(Line2D([0], [0], color='gray', linestyle='--', linewidth=1.0, alpha=0.4))
legend_labels.append('真值轮廓')
legend_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                              markersize=5, linewidth=0))
legend_labels.append('匹配质心')
legend_handles.append(Line2D([0], [0], marker='s', color='gray', markerfacecolor='none',
                              markersize=7, markeredgewidth=2, linewidth=0))
legend_labels.append('插值质心')
legend_handles.append(Line2D([0], [0], color='gray', linewidth=2.2, alpha=0.75))
legend_labels.append('追踪路径')

fig_e.legend(legend_handles, legend_labels, fontsize=9, loc='lower center',
             ncol=min(5, len(legend_handles)), frameon=True, fancybox=True)

fig_e.suptitle('X-Z 截面视图 — 实例不规则形状 · 缺失插值 · FP 过滤  (同色 = 同一三维实例)',
               fontsize=14, fontweight='bold')
fig_e.tight_layout(rect=[0, 0.07, 1, 0.97])
fig_e.savefig(os.path.join(OUTPUT_DIR, 'fig_e_xz_cross_sections.png'), dpi=200, bbox_inches='tight')
plt.close(fig_e)
print("  fig_e saved.")

# ══════════════════════════════════════════════════════════════════════
# 7.  Summary
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"3D + 2D combined visualization — Done")
print(f"{'='*60}")
print(f"  Glomeruli: {N_GLOM}, Tracks: {len(confirmed_tracks)}")
print(f"  Colour mapping: {len(glom_to_track)} glom → track")
print(f"  Views: 3D wireframe + X-Z side + Y-Z side")
print(f"\n  Figures: {OUTPUT_DIR}")
