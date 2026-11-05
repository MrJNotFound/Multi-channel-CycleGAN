"""
Run Kalman+Hungarian tracking on synthetic glomerulus dataset.
Generate report-ready figures.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Circle
from skimage.measure import regionprops, find_contours
from scipy.ndimage import binary_fill_holes, binary_dilation
import cv2

from func_cell_tracking_v2 import track_glomeruli, extract_features

# ── CJK font ──
for fname in fm.fontManager.ttflist:
    if 'Microsoft YaHei' in fname.name and 'Light' not in fname.name and 'UI' not in fname.name:
        plt.rcParams['font.family'] = fname.name
        break
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(42)

# ── 1. Generate the same synthetic glomerulus dataset ────────────────
IMG_SIZE = 800
SLICE_SPACING = 5.0
PIXEL_SIZE = 0.5
N_SLICES = 60
N_GLOM = 8


def make_circle_mask(cx, cy, r, shape):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return (xx - cx)**2 + (yy - cy)**2 <= r**2


# Precomputed coordinate grid (reused by the irregular-blob rasteriser).
_YY, _XX = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]


def make_blob_mask(cx, cy, r, shp):
    """Rasterise an irregular, glomerulus-like cross-section.

    A real glomerulus is not a perfect sphere: its outline is a lumpy,
    slightly elongated spheroid. ``shp`` holds per-glomerulus shape params
    (axis ratio, rotation, a few low-order angular harmonics) that are FIXED
    across all slices, so the 3D instance is a consistent irregular solid
    whose cross-sections scale smoothly with ``r``.
    """
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
    """Random but smooth shape descriptor for one glomerulus (3D-consistent)."""
    harm = [(int(k), float(np.random.uniform(0.04, 0.11)),
             float(np.random.uniform(0, 2 * np.pi)))
            for k in np.random.choice([2, 3, 4], size=2, replace=False)]
    return {'axis_ratio': float(np.random.uniform(0.80, 0.98)),
            'rot': float(np.random.uniform(0, 2 * np.pi)),
            'harm': harm}


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
        dxy = np.sqrt((cx - ex)**2 + (cy - ey)**2)
        dz = abs(zc - ez)
        # Treat each glomerulus as an ellipsoid; require a clear 3D gap so no
        # two instances share space. ``sep < 1`` means the ellipsoids overlap;
        # we demand a 5% margin. Additionally, two xy-coincident instances must
        # be separated in z by several empty slices, otherwise their poles touch
        # and the tracker cannot tell them apart from one continuous instance.
        sep = np.sqrt((dxy / (r_px + er_px))**2 + (dz / (rz_sl + erz))**2)
        xy_frac = dxy / (r_px + er_px)
        z_extent_gap = dz - (rz_sl + erz)
        if sep < 1.05 or (xy_frac < 0.45 and z_extent_gap < 5):
            ok = False
            break
    if ok:
        glomeruli.append((cx, cy, zc, r_px, rz_sl))
        glom_shapes.append(make_glom_shape())

# Assign defects
best_pair = None
best_score = float('inf')
for i in range(len(glomeruli)):
    for j in range(i + 1, len(glomeruli)):
        xi, yi, zi, ri, rzi = glomeruli[i]
        xj, yj, zj, rj, rzj = glomeruli[j]
        dxy = np.sqrt((xi - xj)**2 + (yi - yj)**2)
        z_overlap = min(zi + rzi, zj + rzj) - max(zi - rzi, zj - rzj)
        if z_overlap > 3 and dxy < 250:
            if dxy < best_score:
                best_score = dxy
                best_pair = (i, j)

MERGE_G1, MERGE_G2 = best_pair if best_pair else (0, 1)
zi1, rzi1 = glomeruli[MERGE_G1][2], glomeruli[MERGE_G1][4]
zi2, rzi2 = glomeruli[MERGE_G2][2], glomeruli[MERGE_G2][4]
z_common = int((max(zi1 - rzi1, zi2 - rzi2) + min(zi1 + rzi1, zi2 + rzi2)) / 2)
MERGE_SLICES = [max(0, z_common - 1), z_common]

used = {MERGE_G1, MERGE_G2}
avail = [i for i in range(len(glomeruli)) if i not in used]
MISSING_GLOM = avail[0]
SPLIT_GLOM = avail[1] if len(avail) > 1 else 0

# Missing defect: a short 2-layer gap (1-2 dropped slices is the common case;
# a 4-slice dropout is rare in practice).
gz_m = glomeruli[MISSING_GLOM][2]
MISSING_START = int(round(gz_m))
MISSING_END = MISSING_START + 1          # inclusive → 2 layers
# A second, 1-layer dropout on another glomerulus for variety.
MISSING2_GLOM = avail[3] if len(avail) > 3 else avail[-1]
MISSING2_SLICE = int(round(glomeruli[MISSING2_GLOM][2]))

SPLIT_SLICE = int(glomeruli[SPLIT_GLOM][2])


def glom_radius_at(gid, z):
    cx, cy, zc, r_px, rz_sl = glomeruli[gid]
    dz = abs(z - zc)
    if dz >= rz_sl:
        return 0.0
    return r_px * np.sqrt(max(0.0, 1 - (dz / rz_sl)**2))


# False positives: several isolated small blobs scattered across slices and the
# field of view (vessel / tubule cross-sections mistaken for glomeruli). Each
# lives on 1-2 slices only, so a robust tracker must reject them as unconfirmed.
_fp_candidates = [
    (int(N_SLICES * 0.18), 0.72, 0.74, 20, 1),
    (int(N_SLICES * 0.30), 0.16, 0.82, 15, 1),
    (int(N_SLICES * 0.45), 0.87, 0.28, 24, 2),   # spans 2 slices
    (int(N_SLICES * 0.55), 0.30, 0.11, 14, 1),
    (int(N_SLICES * 0.68), 0.63, 0.90, 18, 1),
    (int(N_SLICES * 0.80), 0.90, 0.58, 16, 1),
]
fp_by_slice = {}
FP_MARKERS = []          # (z, cx, cy) for figure annotation
for z0, fx, fy, fr, span in _fp_candidates:
    px, py = fx * IMG_SIZE, fy * IMG_SIZE
    for z in range(z0, z0 + span):
        if not (0 <= z < N_SLICES):
            continue
        # keep FPs clear of real glomeruli at this slice
        collide = any(np.hypot(px - glomeruli[g][0], py - glomeruli[g][1])
                      < glom_radius_at(g, z) + fr + 6 for g in range(len(glomeruli)))
        if collide:
            continue
        fp_by_slice.setdefault(z, []).append((px, py, fr))
        FP_MARKERS.append((z, px, py))
FP_SLICES = sorted(fp_by_slice.keys())

# Generate slices
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
            # Bridge the two into ONE connected instance (a realistic merge).
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
        scale = np.sqrt(max(0, 1 - t**2))
        r_eff = r_px * scale
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
        fp_mask = make_circle_mask(px, py, fr, (IMG_SIZE, IMG_SIZE))
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

# ── 2. Run Kalman+Hungarian tracking ───────────────────────────────
print("Running Kalman+Hungarian tracking...")
confirmed_tracks, fp_tracks, events = track_glomeruli(
    paths, max_missing=3, confirm_thresh=3, pos_gate=0.45
)

print(f"\n{'='*60}")
print(f"Tracking Results")
print(f"{'='*60}")
print(f"  Confirmed tracks: {len(confirmed_tracks)}")
print(f"  FP (candidate) tracks: {len(fp_tracks)}")
print(f"  Total events: {len(events)}")

# Classify events
event_counts = {}
for e in events:
    t = e['type']
    event_counts[t] = event_counts.get(t, 0) + 1
for k, v in event_counts.items():
    print(f"    {k}: {v}")

# ── 3. Build ground-truth mapping ──────────────────────────────────
# Map each glomerulus to its track
glom_to_track = {}
for gid in range(len(glomeruli)):
    cx, cy, zc, r_px, rz_sl = glomeruli[gid]
    best_overlap = 0
    best_track = None
    for t in confirmed_tracks:
        # Count how many history points overlap with this glomerulus
        overlap = 0
        for (z, hx, hy, ha, status) in t.history:
            dz = abs(z - zc)
            if dz >= rz_sl:
                continue
            t_scale = dz / rz_sl
            r_eff = r_px * np.sqrt(max(0, 1 - t_scale**2))
            d = np.sqrt((hx - cx)**2 + (hy - cy)**2)
            if d < r_eff * 0.8:
                overlap += 1
        if overlap > best_overlap:
            best_overlap = overlap
            best_track = t
    if best_overlap >= 3:
        glom_to_track[gid] = best_track

# ── 4. Visualizations ───────────────────────────────────────────────

# --- Fig A: 3D tracking chains overview ---
fig_a = plt.figure(figsize=(14, 10))
ax3d = fig_a.add_subplot(1, 1, 1, projection='3d')
colors = plt.cm.tab10(np.linspace(0, 1, len(confirmed_tracks)))

for ti, t in enumerate(confirmed_tracks):
    zs = [h[0] for h in t.history]
    cxs = [h[1] for h in t.history]
    cys = [h[2] for h in t.history]
    areas = [h[3] for h in t.history]
    statuses = [h[4] for h in t.history]

    # Split matched and interpolated
    mz, mx, my = [], [], []
    iz, ix, iy = [], [], []
    for z, cx, cy, a, s in t.history:
        if s == 'interpolated':
            iz.append(z); ix.append(cx); iy.append(cy)
        else:
            mz.append(z); mx.append(cx); my.append(cy)

    ax3d.plot(mz, mx, my, '-', color=colors[ti], linewidth=1.5, alpha=0.8, label=f'Track {t.id}')
    if mz:
        ax3d.scatter(mz, mx, my, c=[colors[ti]], s=15, alpha=0.8)
    if iz:
        ax3d.scatter(iz, ix, iy, c=[colors[ti]], s=30, marker='x', linewidths=1.5, alpha=0.9)

# Mark FP candidates
for t in fp_tracks:
    zs = [h[0] for h in t.history]
    cxs = [h[1] for h in t.history]
    cys = [h[2] for h in t.history]
    ax3d.scatter(zs, cxs, cys, c='red', s=80, marker='x', linewidths=2, zorder=10, label='FP candidate' if t == fp_tracks[0] else '')

ax3d.set_xlabel('Slice Z'); ax3d.set_ylabel('X (pixels)'); ax3d.set_zlabel('Y (pixels)')
ax3d.set_title('3D Tracking Chains: Confirmed Tracks & FP Filtering', fontsize=12, fontweight='bold')
ax3d.legend(fontsize=7, loc='upper left')
ax3d.view_init(elev=20, azim=-50)

fig_a.tight_layout()
fig_a.savefig(os.path.join(OUTPUT_DIR, 'fig_a_3d_tracks.png'), dpi=200, bbox_inches='tight')
plt.close(fig_a)

# --- Fig B: Tracking detail — Normal vs Missing glomerulus ---
fig_b, axes_b = plt.subplots(2, 6, figsize=(18, 6))

clean_g = avail[2] if len(avail) > 2 else 0
track_clean = glom_to_track.get(clean_g)
track_miss = glom_to_track.get(MISSING_GLOM)

# Normal
gz_n = int(glomeruli[clean_g][2])
for i in range(6):
    ax = axes_b[0, i]; zs = gz_n - 2 + i
    if zs < 0 or zs >= N_SLICES: ax.axis('off'); continue
    ax.imshow(slice_labels[zs] > 0, cmap='gray', interpolation='none')
    cx, cy, _, r_px, rz_sl = glomeruli[clean_g]
    dz_a = abs(zs - glomeruli[clean_g][2])
    if dz_a < rz_sl:
        t_s = dz_a / rz_sl; r = r_px * np.sqrt(max(0, 1 - t_s**2))
        ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor='lime', linewidth=1.5))
    # Show tracking point
    if track_clean:
        for h in track_clean.history:
            if h[0] == zs:
                ax.plot(h[1], h[2], 'go', markersize=4)
    ax.set_title(f'z={zs}', fontsize=9); ax.axis('off')
axes_b[0, 0].set_title(f'G{clean_g} (正常) — 追踪链', fontsize=10, fontweight='bold')

# Missing
gz_m = int(glomeruli[MISSING_GLOM][2])
for i in range(6):
    ax = axes_b[1, i]; zs = gz_m - 2 + i
    if zs < 0 or zs >= N_SLICES: ax.axis('off'); continue
    ax.imshow(slice_labels[zs] > 0, cmap='gray', interpolation='none')
    cx, cy, _, r_px, rz_sl = glomeruli[MISSING_GLOM]
    dz_a = abs(zs - glomeruli[MISSING_GLOM][2])
    in_gap = MISSING_START <= zs <= MISSING_END
    if dz_a < rz_sl and not in_gap:
        t_s = dz_a / rz_sl; r = r_px * np.sqrt(max(0, 1 - t_s**2))
        ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor='cyan', linewidth=1.5))
    if in_gap:
        ax.plot(cx, cy, 'rx', markersize=10, markeredgewidth=2)
    # Show tracking point (interpolated in gap)
    if track_miss:
        for h in track_miss.history:
            if h[0] == zs:
                color = 'co' if h[4] == 'interpolated' else 'go'
                ax.plot(h[1], h[2], 'c+' if h[4] == 'interpolated' else 'go', markersize=6)
    ax.set_title(f'z={zs}', fontsize=9); ax.axis('off')
axes_b[1, 0].set_title(f'G{MISSING_GLOM} (缺失) — 追踪链（+插值）', fontsize=10, fontweight='bold')

fig_b.suptitle('图8 Kalman+Hungarian多目标追踪：正常肾小球与缺失肾小球', fontsize=13, fontweight='bold')
fig_b.tight_layout()
fig_b.savefig(os.path.join(OUTPUT_DIR, 'fig_b_tracking_detail.png'), dpi=200, bbox_inches='tight')
plt.close(fig_b)

# --- Fig C: Area evolution curves (split/merge/normal comparison) ---
fig_c, axes_c = plt.subplots(1, 3, figsize=(18, 5))
z_axis = np.arange(N_SLICES)

# C1: Normal glomerulus area curve
if clean_g in glom_to_track:
    t = glom_to_track[clean_g]
    zs = [h[0] for h in t.history]
    areas = [h[3] for h in t.history]
    statuses = [h[4] for h in t.history]
    axes_c[0].plot(zs, areas, 'o-', color='steelblue', linewidth=1.5, markersize=4)
    # Mark interpolated
    for zi, ai, si in zip(zs, areas, statuses):
        if si == 'interpolated':
            axes_c[0].plot(zi, ai, 's', color='darkred', markersize=8,
                           markerfacecolor='none', markeredgewidth=2)
    axes_c[0].set_title('正常肾小球 — 面积平滑变化', fontsize=11, fontweight='bold')
    axes_c[0].set_xlabel('切片编号'); axes_c[0].set_ylabel('截面面积 (pixels²)')

# C2: Missing glomerulus area curve with interpolation
if MISSING_GLOM in glom_to_track:
    t = glom_to_track[MISSING_GLOM]
    zs = [h[0] for h in t.history]
    areas = [h[3] for h in t.history]
    statuses = [h[4] for h in t.history]
    axes_c[1].plot(zs, areas, 'o-', color='tomato', linewidth=1.5, markersize=4)
    for ms in range(MISSING_START, MISSING_END + 1):
        axes_c[1].axvspan(ms - 0.4, ms + 0.4, color='red', alpha=0.08)
    for zi, ai, si in zip(zs, areas, statuses):
        if si == 'interpolated':
            axes_c[1].plot(zi, ai, 's', color='darkred', markersize=10,
                           markerfacecolor='none', markeredgewidth=2)
    axes_c[1].set_title('缺失肾小球 — 间隙插值补全', fontsize=11, fontweight='bold')
    axes_c[1].set_xlabel('切片编号')
    leg = [plt.Line2D([0], [0], color='tomato', marker='o', linewidth=1.5, label='匹配点'),
           plt.Line2D([0], [0], color='darkred', marker='s', linewidth=0,
                      markerfacecolor='none', markeredgewidth=2, label='卡尔曼插值'),
           plt.Line2D([0], [0], color='red', alpha=0.3, linewidth=5, label='缺失区间')]
    axes_c[1].legend(handles=leg, fontsize=8)

# C3: Tracking quality summary
# Count: correct tracks, FP filtered, interpolations
n_glom_matched = len(glom_to_track)
n_total_tracks = len(confirmed_tracks)
fp_filtered = len(fp_tracks)
n_interpolated = sum(1 for t in confirmed_tracks
                     for h in t.history if h[4] == 'interpolated')
# Count affected track/slice associations for multi-track merge conflicts too.
n_merge_conflicts = sum(len(e['track_ids']) for e in events if e['type'] == 'merge_conflict')
n_rej_merge = sum(1 for e in events if e['type'] == 'match_rejected' and e['reason'] == 'merge') + n_merge_conflicts
n_rej_split = sum(1 for e in events if e['type'] == 'match_rejected' and e['reason'] == 'split')
n_rejected = sum(1 for e in events if e['type'] == 'match_rejected') + n_merge_conflicts

metrics = [
    f'肾小球总数: {N_GLOM}',
    f'成功追踪: {n_glom_matched}/{N_GLOM}',
    f'额外追踪链: {n_total_tracks - n_glom_matched}',
    f'FP候选(已滤除): {fp_filtered}',
    f'插值补全点: {n_interpolated}',
    f'合并异常剔除: {n_rej_merge}',
    f'分裂异常剔除: {n_rej_split}',
    f'异常匹配拒绝: {n_rejected}',
]
axes_c[2].axis('off')
for i, m in enumerate(metrics):
    color = 'green' if ('成功' in m or 'FP' in m) else ('black' if '额外' not in m else 'darkorange')
    axes_c[2].text(0.08, 0.9 - i * 0.11, m, fontsize=13,
                   color=color, transform=axes_c[2].transAxes)
axes_c[2].set_title('追踪质量汇总', fontsize=11, fontweight='bold')

fig_c.suptitle('图9 Kalman+Hungarian追踪：面积演化与追踪质量', fontsize=13, fontweight='bold')
fig_c.tight_layout()
fig_c.savefig(os.path.join(OUTPUT_DIR, 'fig_c_tracking_quality.png'), dpi=200, bbox_inches='tight')
plt.close(fig_c)

# ── 5. Summary ─────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Kalman+Hungarian Tracking — Final Summary")
print(f"{'='*60}")
print(f"  Glomeruli: {N_GLOM}")
print(f"  Successfully tracked: {n_glom_matched}/{N_GLOM}")
print(f"  Extra (non-glomerulus) tracks: {n_total_tracks - n_glom_matched}")
print(f"  FP candidates filtered: {fp_filtered}")
print(f"  Total interpolations: {n_interpolated}")
print(f"  Anomalous matches rejected: {n_rejected} (merge={n_rej_merge}, split={n_rej_split})")
print(f"\n  Each confirmed track outputs:")
print(f"    - Full slice range with (z, cx, cy, area)")
print(f"    - Status per point: matched | interpolated")
print(f"    - 3D centroid and volume ready for quantification")
print(f"\n  Figures: {OUTPUT_DIR}")
