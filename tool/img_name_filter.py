from pathlib import Path
import shutil

# ======================
# 直接写死配置：你自己改这里
# ======================
img_dir = r"C:\Users\30927\Desktop\img_histology\stain_kidney\kidney_fake\kidney\mouse_kidney_dual_stack_256\test_latest\images"# 输入文件夹（会递归扫描）
out_dir = r"C:\Users\30927\Desktop\img_histology\stain_kidney\kidney_fake\kidney\mouse_kidney_dual_stack_256\test_latest\images_fake"  # 输出文件夹

# 支持的图像扩展名
exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

src_root = Path(img_dir)
dst_root = Path(out_dir)
dst_root.mkdir(parents=True, exist_ok=True)

if not src_root.exists():
    raise FileNotFoundError(f"Input folder not found: {src_root}")

picked = []

# 递归扫描
for p in src_root.rglob("*"):
    if not p.is_file():
        continue

    if p.suffix.lower() not in exts:
        continue

    # 规则：扩展名前以 _fake 结尾
    # 例：xxx_fake.png  -> picked
    #     xxx_fake_001.png -> NOT picked（如果你也想要这种，把 endswith('_fake') 改成 `'_fake' in name`）
    if not p.stem.endswith("_fake"):
        continue

    # 为避免不同子目录同名冲突：保留相对目录结构
    rel = p.relative_to(src_root)
    dst_path = dst_root / rel
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(str(p), str(dst_path))
    picked.append(str(rel))

print("Done.")
print(f"Scanned: {src_root}")
print(f"Saved to: {dst_root}")
print(f"Picked: {len(picked)} files")

# 打印前100个，避免太长
for s in picked[:100]:
    print(s)

if len(picked) > 100:
    print(f"... ({len(picked) - 100} more)")
