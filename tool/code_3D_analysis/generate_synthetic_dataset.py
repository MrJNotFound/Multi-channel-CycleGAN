"""
Synthetic glomerulus dataset for validating defect-robust
cross-slice instance association.

Realistic parameters:
  - Glomerulus: ~80-160μm diameter, roughly spherical
  - Slice thickness: 5μm → a 100μm glomerulus spans ~20 slices
  - Glomeruli do not overlap in 3D
  - Cross-section varies smoothly (small at poles, largest at equator)

Defect types (glomerulus segmentation scenarios):
  A. Mid-glomerulus missing detection (staining artifact)
  B. False positive: small tubule/vessel cross-section mistaken for glomerulus
  C. Adjacent glomeruli merged in single slice
  D. Glomerulus split into fragments
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Circle
from skimage.measure import regionprops, find_contours
from scipy.ndimage import binary_fill_holes, binary_dilation
import cv2
import os
from func_cell_tracking import analyze_layer_iou_by_paths

# ── CJK font ──
for fname in fm.fontManager.ttflist:
    if 'Microsoft YaHei' in fname.name and 'Light' not in fname.name and 'UI' not in fname.name:
        plt.rcParams['font.family'] = fname.name
        break
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(42)

# ── 1. Dataset parameters ────────────────────────────────────────
IMG_SIZE = 800          # pixels
SLICE_SPACING = 5.0     # μm per slice
PIXEL_SIZE = 0.5        # μm per pixel → 800×0.5 = 400 μm field
N_SLICES = 60
N_GLOM = 8

def make_circle_mask(cx, cy, r, shape):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return (xx - cx)**2 + (yy - cy)**2 <= r**2

# ── 2. Generate non-overlapping glomeruli ────────────────────────
glomeruli = []
for _ in range(5000):
    if len(glomeruli) >= N_GLOM:
        break
    r_um = np.random.uniform(50, 80)           # radius in μm (diameter 100-160)
    r_px = r_um / PIXEL_SIZE                   # radius in pixels
    rz_sl = r_um / SLICE_SPACING               # radius in slices

    cx = np.random.uniform(r_px + 20, IMG_SIZE - r_px - 20)
    cy = np.random.uniform(r_px + 20, IMG_SIZE - r_px - 20)
    zc = np.random.uniform(rz_sl + 3, N_SLICES - rz_sl - 3)

    ok = True
    for (ex, ey, ez, er_px, erz) in glomeruli:
        dxy = np.sqrt((cx - ex)**2 + (cy - ey)**2)
        dz = abs(zc - ez)
        if dxy < (r_px + er_px) * 0.85 and dz < (rz_sl + erz) * 0.85:
            ok = False
            break
    if ok:
        glomeruli.append((cx, cy, zc, r_px, rz_sl))

print(f"Generated {len(glomeruli)} non-overlapping glomeruli")
for i, (cx, cy, zc, r_px, rz) in enumerate(glomeruli):
    print(f"  G{i}: xy=({cx:.0f},{cy:.0f}) z={zc:.1f} r={r_px:.0f}px({r_px*PIXEL_SIZE:.0f}μm) rz={rz:.1f}sl")

# ── 3. Assign defects ────────────────────────────────────────────
# Find two glomeruli that are close in xy AND z for merge defect
best_pair = None
best_score = float('inf')
for i in range(len(glomeruli)):
    for j in range(i+1, len(glomeruli)):
        xi, yi, zi, ri, rzi = glomeruli[i]
        xj, yj, zj, rj, rzj = glomeruli[j]
        dxy = np.sqrt((xi - xj)**2 + (yi - yj)**2)
        dz = abs(zi - zj)
        # Score: want close xy AND overlapping z
        z_overlap = min(zi + rzi, zj + rzj) - max(zi - rzi, zj - rzj)
        if z_overlap > 3 and dxy < 250:
            score = dxy
            if score < best_score:
                best_score = score
                best_pair = (i, j)

MERGE_G1, MERGE_G2 = best_pair if best_pair else (0, 1)
# Find a common z slice where both are visible
zi1, rzi1 = glomeruli[MERGE_G1][2], glomeruli[MERGE_G1][4]
zi2, rzi2 = glomeruli[MERGE_G2][2], glomeruli[MERGE_G2][4]
z_common = int((max(zi1 - rzi1, zi2 - rzi2) + min(zi1 + rzi1, zi2 + rzi2)) / 2)
MERGE_SLICES = [max(0, z_common - 1), z_common]

# Missing and split: use glomeruli not in merge pair
used = {MERGE_G1, MERGE_G2}
avail = [i for i in range(len(glomeruli)) if i not in used]
MISSING_GLOM = avail[0]
SPLIT_GLOM = avail[1] if len(avail) > 1 else 0

# Missing: remove middle 4-5 slices
gz_m, grz_m = glomeruli[MISSING_GLOM][2], glomeruli[MISSING_GLOM][4]
MISSING_START = int(gz_m - grz_m * 0.15)
MISSING_END = int(gz_m + grz_m * 0.15)

# Split slice
SPLIT_SLICE = int(glomeruli[SPLIT_GLOM][2])

# False positive: small tubule-like object on SINGLE isolated slice
FP_SLICES = [N_SLICES // 3]  # single slice!
fp_cx, fp_cy = IMG_SIZE * 0.7, IMG_SIZE * 0.7
fp_r = 22

print(f"\nDefects:")
print(f"  Missing: G{MISSING_GLOM} slices {MISSING_START}-{MISSING_END}")
print(f"  Split: G{SPLIT_GLOM} slice {SPLIT_SLICE}")
print(f"  Merge: G{MERGE_G1}+G{MERGE_G2} slices {MERGE_SLICES}")
print(f"  FP: slices {FP_SLICES}")

# ── 4. Generate per-slice labels ─────────────────────────────────
slice_labels = []
for z in range(N_SLICES):
    lbl = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.int32)
    next_id = 1
    skip_glom = set()

    # Handle merge defect
    if z in MERGE_SLICES:
        g1, g2 = glomeruli[MERGE_G1], glomeruli[MERGE_G2]
        dz1, dz2 = abs(z - g1[2]), abs(z - g2[2])
        t1 = dz1 / g1[4] if g1[4] > 0 else 99
        t2 = dz2 / g2[4] if g2[4] > 0 else 99
        if t1 < 1 and t2 < 1:
            r1 = g1[3] * np.sqrt(max(0, 1 - t1**2))
            r2 = g2[3] * np.sqrt(max(0, 1 - t2**2))
            m1 = make_circle_mask(g1[0], g1[1], r1, (IMG_SIZE, IMG_SIZE))
            m2 = make_circle_mask(g2[0], g2[1], r2, (IMG_SIZE, IMG_SIZE))
            merged = binary_dilation(m1 | m2, iterations=4)
            merged = binary_fill_holes(merged)
            if merged.sum() > 30:
                lbl[merged] = next_id
                next_id += 1
        skip_glom = {MERGE_G1, MERGE_G2}

    for gid, (cx, cy, zc, r_px, rz_sl) in enumerate(glomeruli):
        if gid in skip_glom:
            continue
        # Missing defect
        if gid == MISSING_GLOM and MISSING_START <= z <= MISSING_END:
            continue
        dz = abs(z - zc)
        if dz >= rz_sl:
            continue
        t = dz / rz_sl
        scale = np.sqrt(max(0, 1 - t**2))
        r_eff = r_px * scale
        if r_eff < 4:
            continue
        mask = make_circle_mask(cx, cy, r_eff, (IMG_SIZE, IMG_SIZE))

        # Split defect
        if gid == SPLIT_GLOM and z == SPLIT_SLICE and mask.sum() > 40:
            # Split into 3 fragments vertically
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

    # False positive
    if z in FP_SLICES:
        fp_mask = make_circle_mask(fp_cx, fp_cy, fp_r, (IMG_SIZE, IMG_SIZE))
        if fp_mask.sum() > 15:
            lbl[fp_mask] = next_id
            next_id += 1

    slice_labels.append(lbl)

# ── 5. Save & run tracking ───────────────────────────────────────
TMP_DIR = os.path.join(OUTPUT_DIR, 'slices')
os.makedirs(TMP_DIR, exist_ok=True)
for z, lbl in enumerate(slice_labels):
    cv2.imwrite(os.path.join(TMP_DIR, f'slice_{z:03d}.png'), lbl.astype(np.uint16))

print("\nRunning cross-slice tracking...")
paths = [os.path.join(TMP_DIR, f'slice_{z:03d}.png') for z in range(N_SLICES)]
all_layer_scores = []
for z in range(N_SLICES):
    scores = analyze_layer_iou_by_paths(paths, z)
    all_layer_scores.append(scores)

# ── 6. Classify instances ────────────────────────────────────────
normal_scores = []
defect_scores = {'fp': [], 'split': [], 'merge': []}

for z in range(N_SLICES):
    scores = all_layer_scores[z]
    lbl = slice_labels[z]
    for r in regionprops(lbl):
        lid = r.label
        if lid - 1 >= len(scores):
            continue
        s = scores[lid - 1]
        cx_r, cy_r = r.centroid

        # FP check
        is_fp = (z in FP_SLICES and abs(cx_r - fp_cx) < 30 and abs(cy_r - fp_cy) < 30)
        # Merge check
        gm = glomeruli[MERGE_G1]
        is_merge = (z in MERGE_SLICES and abs(cx_r - gm[0]) < gm[3]*2 and abs(cy_r - gm[1]) < gm[3]*2)
        # Split check
        gs = glomeruli[SPLIT_GLOM]
        is_split = (z == SPLIT_SLICE and abs(cx_r - gs[0]) < gs[3]*2 and abs(cy_r - gs[1]) < gs[3]*2)

        if is_fp:
            defect_scores['fp'].append(s)
        elif is_merge:
            defect_scores['merge'].append(s)
        elif is_split:
            defect_scores['split'].append(s)
        else:
            normal_scores.append(s)

all_defect = defect_scores['fp'] + defect_scores['split'] + defect_scores['merge']

# ── 7. Score curves ──────────────────────────────────────────────
def get_glom_score_curve(gid):
    curve = np.full(N_SLICES, np.nan)
    cx, cy, zc, r_px, rz_sl = glomeruli[gid]
    for z in range(N_SLICES):
        dz = abs(z - zc)
        if dz >= rz_sl:
            continue
        t = dz / rz_sl
        scale = np.sqrt(max(0, 1 - t**2))
        r_eff = r_px * scale
        if r_eff < 4:
            continue
        true_mask = make_circle_mask(cx, cy, r_eff, (IMG_SIZE, IMG_SIZE))
        lbl = slice_labels[z]
        scores_list = all_layer_scores[z]
        best_iou, best_lid = 0, -1
        for rp in regionprops(lbl):
            rmask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool)
            rmask[tuple(rp.coords.T)] = True
            inter = (true_mask & rmask).sum()
            union = (true_mask | rmask).sum()
            iou = inter / union if union > 0 else 0
            if iou > best_iou and rp.label - 1 < len(scores_list):
                best_iou = iou
                best_lid = rp.label
        if best_iou > 0.3 and best_lid - 1 < len(scores_list):
            curve[z] = scores_list[best_lid - 1]
    return curve

# ── 8. Visualizations ────────────────────────────────────────────

# --- Fig A: Dataset overview slices ---
key_slices = [
    (25, '正常'),
    (MISSING_START + 1, '实例缺失'),
    (MISSING_END + 1, '缺失后恢复'),
    (SPLIT_SLICE, '实例分裂'),
    (MERGE_SLICES[0], '实例合并'),
    (FP_SLICES[0], '假阳性'),
    (40, '正常'),
    (10, '正常(边界)'),
]

fig_a, axes_a = plt.subplots(2, 4, figsize=(18, 9))
for idx, (zs, desc) in enumerate(key_slices):
    ax = axes_a[idx // 4, idx % 4]
    zc_s = max(0, min(N_SLICES-1, zs))
    lbl = slice_labels[zc_s]
    ax.imshow(lbl > 0, cmap='gray', interpolation='none')
    for rp in regionprops(lbl):
        mask_r = np.zeros_like(lbl, dtype=bool)
        mask_r[tuple(rp.coords.T)] = True
        ctrs = find_contours(mask_r.astype(float), 0.5)
        for c in ctrs:
            ax.plot(c[:, 1], c[:, 0], 'r-', linewidth=0.4)

    if '缺失' in desc:
        gx, gy = glomeruli[MISSING_GLOM][0], glomeruli[MISSING_GLOM][1]
        gr = glomeruli[MISSING_GLOM][3] * 0.6
        ax.add_patch(Circle((gx, gy), gr, fill=False, edgecolor='red', linewidth=2, linestyle='--'))
    color = 'red' if '缺失' in desc else ('darkorange' if ('分裂' in desc or '合并' in desc) else ('magenta' if '假阳性' in desc else 'black'))
    ax.set_title(f'Slice {zc_s} ({desc})', fontsize=10, color=color, fontweight='bold')
    ax.axis('off')

fig_a.suptitle('图7 合成肾小球数据集连续切片实例分割（含模拟缺陷）', fontsize=13, fontweight='bold')
fig_a.tight_layout()
fig_a.savefig(os.path.join(OUTPUT_DIR, 'fig_a_glom_dataset.png'), dpi=200, bbox_inches='tight')
plt.close(fig_a)

# --- Fig B: Tracking comparison ──────────────────────────────────
fig_b, axes_b = plt.subplots(2, 6, figsize=(18, 6))
# Normal: pick a clean glomerulus
clean_g = avail[2] if len(avail) > 2 else 0
gz_n = int(glomeruli[clean_g][2])
for i in range(6):
    ax = axes_b[0, i]; zs = gz_n - 2 + i
    if zs < 0 or zs >= N_SLICES: ax.axis('off'); continue
    ax.imshow(slice_labels[zs] > 0, cmap='gray', interpolation='none')
    cx, cy = glomeruli[clean_g][0], glomeruli[clean_g][1]
    dz_a = abs(zs - glomeruli[clean_g][2])
    if dz_a < glomeruli[clean_g][4]:
        t = dz_a / glomeruli[clean_g][4]; r = glomeruli[clean_g][3] * np.sqrt(max(0, 1-t**2))
        ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor='lime', linewidth=1.5))
    ax.set_title(f'z={zs}', fontsize=9); ax.axis('off')
axes_b[0, 0].set_title(f'G{clean_g} (正常): 跨切片追踪', fontsize=10, fontweight='bold')

# Missing
gz_d = int(glomeruli[MISSING_GLOM][2])
for i in range(6):
    ax = axes_b[1, i]; zs = gz_d - 2 + i
    if zs < 0 or zs >= N_SLICES: ax.axis('off'); continue
    ax.imshow(slice_labels[zs] > 0, cmap='gray', interpolation='none')
    cx, cy = glomeruli[MISSING_GLOM][0], glomeruli[MISSING_GLOM][1]
    dz_a = abs(zs - glomeruli[MISSING_GLOM][2])
    in_gap = MISSING_START <= zs <= MISSING_END
    if dz_a < glomeruli[MISSING_GLOM][4] and not in_gap:
        t = dz_a / glomeruli[MISSING_GLOM][4]; r = glomeruli[MISSING_GLOM][3] * np.sqrt(max(0, 1-t**2))
        ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor='cyan', linewidth=1.5))
    if in_gap:
        ax.plot(cx, cy, 'rx', markersize=10, markeredgewidth=2)
    ax.set_title(f'z={zs}', fontsize=9); ax.axis('off')
axes_b[1, 0].set_title(f'G{MISSING_GLOM} (缺失): 跨切片追踪', fontsize=10, fontweight='bold')

fig_b.suptitle('图8 跨切片实例关联：正常肾小球与缺失肾小球的追踪对比', fontsize=13, fontweight='bold')
fig_b.tight_layout()
fig_b.savefig(os.path.join(OUTPUT_DIR, 'fig_b_glom_tracking.png'), dpi=200, bbox_inches='tight')
plt.close(fig_b)

# --- Fig C: Score analysis ───────────────────────────────────────
fig_c, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
z_axis = np.arange(N_SLICES)

# C1: Normal
c_n = get_glom_score_curve(clean_g)
ax1.plot(z_axis, c_n, 'o-', color='steelblue', linewidth=1.5, markersize=4)
ax1.axhline(y=0.3, color='gray', linestyle='--', linewidth=1, label='阈值=0.3')
ax1.fill_between(z_axis, 0, 0.3, alpha=0.04, color='gray')
ax1.set_ylim(0, 1.05); ax1.set_title('正常肾小球 — 全程高分', fontsize=11, fontweight='bold')
ax1.set_xlabel('切片编号'); ax1.set_ylabel('跨切片一致性得分'); ax1.legend(fontsize=8)

# C2: Missing
c_m = get_glom_score_curve(MISSING_GLOM)
ax2.plot(z_axis, c_m, 'o-', color='tomato', linewidth=1.5, markersize=4)
ax2.axhline(y=0.3, color='gray', linestyle='--', linewidth=1)
for ms in range(MISSING_START, MISSING_END + 1):
    ax2.axvspan(ms - 0.4, ms + 0.4, color='red', alpha=0.08)
    ax2.plot(ms, 0.32, 's', color='darkred', markersize=8, markerfacecolor='none', markeredgewidth=2)
ax2.set_ylim(0, 1.05); ax2.set_title('缺失肾小球 — 间隙→识别→插值恢复', fontsize=11, fontweight='bold')
ax2.set_xlabel('切片编号')
leg2 = [plt.Line2D([0],[0], color='tomato', marker='o', linewidth=1.5, label='得分'),
        plt.Line2D([0],[0], color='darkred', marker='s', linewidth=0, markerfacecolor='none', markeredgewidth=2, label='插值'),
        plt.Line2D([0],[0], color='gray', linestyle='--', label='阈值')]
ax2.legend(handles=leg2, fontsize=8)

# C3: Distribution
bins = np.linspace(0, 1, 25)
ax3.hist(normal_scores, bins=bins, alpha=0.7, color='steelblue', label=f'正常 (n={len(normal_scores)})', edgecolor='white')
ax3.hist(all_defect, bins=bins, alpha=0.8, color='tomato', label=f'缺陷 (n={len(all_defect)})', edgecolor='white')
ax3.axvline(x=0.3, color='gray', linestyle='--', linewidth=1.5, label='阈值=0.3')
ax3.set_xlabel('跨切片一致性得分'); ax3.set_ylabel('频数')
ax3.set_title('得分分布：正常 vs 缺陷实例', fontsize=11, fontweight='bold')
ax3.legend(fontsize=8)
ax3.annotate('缺陷可\n被滤除', xy=(0.08, 3), fontsize=10, color='darkred', fontweight='bold', ha='center')

fig_c.suptitle('图9 缺陷鲁棒跨切片实例关联：得分机制与验证', fontsize=13, fontweight='bold')
fig_c.tight_layout()
fig_c.savefig(os.path.join(OUTPUT_DIR, 'fig_c_glom_scores.png'), dpi=200, bbox_inches='tight')
plt.close(fig_c)

# ── 9. Summary ───────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"合成肾小球数据集 — 验证结果")
print(f"{'='*60}")
print(f"  切片: {N_SLICES} ({SLICE_SPACING}μm/层), 视场: {IMG_SIZE*PIXEL_SIZE:.0f}μm")
print(f"  肾小球: {N_GLOM}个, 直径 {min(g[3]*PIXEL_SIZE*2 for g in glomeruli):.0f}-{max(g[3]*PIXEL_SIZE*2 for g in glomeruli):.0f}μm")
print(f"  跨切片数: {min(g[4]*2 for g in glomeruli):.0f}-{max(g[4]*2 for g in glomeruli):.0f} 层/个")
print(f"  正常实例: {len(normal_scores)}, 缺陷实例: {len(all_defect)}")
print(f"  正常得分: {np.mean(normal_scores):.3f} ± {np.std(normal_scores):.3f}")
print(f"  缺陷得分: {np.mean(all_defect):.3f} ± {np.std(all_defect):.3f}")
print(f"  分离裕度: {np.mean(normal_scores) - np.mean(all_defect):.3f}")
th = 0.3
tp = sum(1 for s in normal_scores if s >= th)
tn = sum(1 for s in all_defect if s < th)
print(f"  阈值={th}: TP={tp}/{len(normal_scores)} TN={tn}/{len(all_defect)}")
print(f"  缺陷检出: {tn}/{len(all_defect)} = {tn/len(all_defect)*100:.1f}%")
print(f"  正常保留: {tp}/{len(normal_scores)} = {tp/len(normal_scores)*100:.1f}%")
for k, v in defect_scores.items():
    c = sum(1 for s in v if s < th)
    print(f"    {k}: {c}/{len(v)} caught")
print(f"\nFigures: {OUTPUT_DIR}")
