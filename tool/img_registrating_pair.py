"""
批量配对配准工具
---------------
选取一组参考图像，再选多组待配准图像（每组数量与参考一致）。
图像按文件名排序后一一配对，每组独立输出到子文件夹。

依赖: img_registrating_series.py（同目录下的刚性/弹性配准函数）
"""

import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

# 允许从同目录导入配准函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from img_registrating_series import rigid_registration, elastic_registration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sorted_paths(paths):
    """按文件名排序，返回 list。"""
    return sorted(list(paths), key=os.path.basename)


def _pick_images(root, title, expect_n=None):
    """弹出多选文件对话框。

    若 expect_n 不为 None，选中数量不符时弹出警告并返回 None。
    返回排序后的路径列表，或 None（取消 / 数量不符）。
    """
    paths = filedialog.askopenfilenames(
        title=title,
        filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.gif *.webp"),
                   ("所有文件", "*.*")],
    )
    if not paths:
        return None
    selected = _sorted_paths(paths)
    if expect_n is not None and len(selected) != expect_n:
        messagebox.showwarning(
            "数量不匹配",
            f"需要选择 {expect_n} 张图像，当前选择了 {len(selected)} 张。\n请重新选择。",
            parent=root,
        )
        return None
    return selected


def _continue_dialog(root, group_idx):
    """询问是否继续添加下一组。返回 True/False。"""
    return messagebox.askyesno(
        "继续添加？",
        f"已添加第 {group_idx} 组。\n是否继续添加下一组待配准图像？",
        parent=root,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    # ===== Step 1: 选择参考图像 =====
    ref_paths = _pick_images(
        root,
        title="选择参考图像（可多选，Ctrl+A 全选或 Ctrl+Click 逐个选）",
    )
    if not ref_paths:
        print("未选择参考图像，退出。")
        root.destroy()
        exit()
    N = len(ref_paths)
    print(f"参考图像: {N} 张")
    for i, p in enumerate(ref_paths):
        print(f"  [{i + 1}] {os.path.basename(p)}")

    # ===== Step 2: 选择输出根目录 =====
    output_root = filedialog.askdirectory(title="选择输出根目录（各组将创建子文件夹）")
    if not output_root:
        print("未选择输出目录，退出。")
        root.destroy()
        exit()
    print(f"输出根目录: {output_root}")

    # ===== Step 3: 选择多组待配准图像 =====
    groups: list[list[str]] = []
    group_idx = 1
    while True:
        mov_paths = _pick_images(
            root,
            title=f"选择第 {group_idx} 组待配准图像（需选 {N} 张）—— 取消则结束选择",
            expect_n=N,
        )
        if mov_paths is None:
            if group_idx == 1:
                print("未选择任何待配准图像，退出。")
                root.destroy()
                exit()
            else:
                print("已结束组添加。")
                break

        groups.append(mov_paths)
        print(f"\n第 {group_idx} 组: {len(mov_paths)} 张")
        for j, p in enumerate(mov_paths):
            print(f"  [{j + 1}] {os.path.basename(p)}  ↔  {os.path.basename(ref_paths[j])}")

        if not _continue_dialog(root, group_idx):
            break
        group_idx += 1

    print(f"\n共 {len(groups)} 组待配准，每组 {N} 对。")

    # ===== Step 4: 参数设置 =====
    mode_var = tk.StringVar(value="刚性")
    sf_var = tk.DoubleVar(value=0.1)
    nm_var = tk.IntVar(value=200)
    only_rigid_var = tk.BooleanVar(value=False)
    mesh_var = tk.IntVar(value=8)
    sh_var = tk.StringVar(value="10")
    sm_var = tk.StringVar(value="2")
    oi_var = tk.IntVar(value=50)
    ot_var = tk.StringVar(value="1e-5")
    ob_lo_var = tk.IntVar(value=-200)
    ob_hi_var = tk.IntVar(value=200)

    param_win = tk.Toplevel(root)
    param_win.title("配准参数设置")
    param_win.resizable(False, False)
    r = 0

    tk.Label(param_win, text="配准算法：", font=("", 11)).grid(
        row=r, column=0, sticky="e", padx=(15, 2), pady=(15, 3))
    mf = tk.Frame(param_win)
    mf.grid(row=r, column=1, sticky="w")
    for text, val in [("刚性", "刚性"), ("弹性", "弹性"), ("刚性+弹性", "刚性+弹性")]:
        tk.Radiobutton(mf, text=text, variable=mode_var, value=val,
                       font=("", 10)).pack(side="left", padx=2)
    r += 1

    tk.Label(param_win, text="缩放因子：", font=("", 11)).grid(
        row=r, column=0, sticky="e", padx=(15, 2), pady=3)
    tk.Entry(param_win, textvariable=sf_var, width=6).grid(row=r, column=1, sticky="w")
    tk.Label(param_win, text="(特征匹配降采样比例)", font=("", 9)).grid(
        row=r, column=2, sticky="w")
    r += 1

    tk.Label(param_win, text="匹配点数：", font=("", 11)).grid(
        row=r, column=0, sticky="e", padx=(15, 2), pady=3)
    tk.Entry(param_win, textvariable=nm_var, width=6).grid(row=r, column=1, sticky="w")
    r += 1

    tk.Checkbutton(param_win, text="强制纯刚性（去除缩放分量）",
                   variable=only_rigid_var, font=("", 10)).grid(
        row=r, column=0, columnspan=3, sticky="w", padx=15, pady=3)
    r += 1

    # 弹性参数
    tk.Label(param_win, text="--- 弹性配准参数 ---", font=("", 10, "bold")).grid(
        row=r, column=0, columnspan=3, sticky="w", padx=15, pady=(8, 2))
    r += 1

    tk.Label(param_win, text="B-spline 网格：", font=("", 11)).grid(
        row=r, column=0, sticky="e", padx=(15, 2), pady=3)
    tk.Entry(param_win, textvariable=mesh_var, width=6).grid(row=r, column=1, sticky="w")
    r += 1

    tk.Label(param_win, text="Shrink Factors：", font=("", 11)).grid(
        row=r, column=0, sticky="e", padx=(15, 2), pady=3)
    tk.Entry(param_win, textvariable=sh_var, width=6).grid(row=r, column=1, sticky="w")
    tk.Label(param_win, text="(逗号分隔，如 10,5)", font=("", 9)).grid(
        row=r, column=2, sticky="w")
    r += 1

    tk.Label(param_win, text="Smoothing Sigmas：", font=("", 11)).grid(
        row=r, column=0, sticky="e", padx=(15, 2), pady=3)
    tk.Entry(param_win, textvariable=sm_var, width=6).grid(row=r, column=1, sticky="w")
    tk.Label(param_win, text="(逗号分隔，如 2,1)", font=("", 9)).grid(
        row=r, column=2, sticky="w")
    r += 1

    tk.Label(param_win, text="优化迭代次数：", font=("", 11)).grid(
        row=r, column=0, sticky="e", padx=(15, 2), pady=3)
    tk.Entry(param_win, textvariable=oi_var, width=6).grid(row=r, column=1, sticky="w")
    r += 1

    tk.Label(param_win, text="优化容差：", font=("", 11)).grid(
        row=r, column=0, sticky="e", padx=(15, 2), pady=3)
    tk.Entry(param_win, textvariable=ot_var, width=6).grid(row=r, column=1, sticky="w")
    r += 1

    tk.Label(param_win, text="边界下/上限：", font=("", 11)).grid(
        row=r, column=0, sticky="e", padx=(15, 2), pady=3)
    bf = tk.Frame(param_win)
    bf.grid(row=r, column=1, sticky="w")
    tk.Entry(bf, textvariable=ob_lo_var, width=5).pack(side="left")
    tk.Label(bf, text=" ~ ", font=("", 10)).pack(side="left")
    tk.Entry(bf, textvariable=ob_hi_var, width=5).pack(side="left")
    r += 1

    tk.Button(param_win, text="开始配准", command=param_win.destroy,
              width=12).grid(row=r, column=0, columnspan=3, pady=(15, 15))

    # 参数验证循环
    while True:
        param_win.grab_set()
        root.wait_window(param_win)
        try:
            mode = mode_var.get()
            scale_factor = sf_var.get()
            n_matches = nm_var.get()
            only_rigid = only_rigid_var.get()
            mesh_size = mesh_var.get()
            raw_sh = sh_var.get().replace("，", ",")
            raw_sm = sm_var.get().replace("，", ",")
            shrink_factors = [int(x.strip()) for x in raw_sh.split(",") if x.strip()]
            smoothing_sigmas = [float(x.strip()) for x in raw_sm.split(",") if x.strip()]
            if not shrink_factors:
                raise ValueError("Shrink Factors 不能为空")
            if not smoothing_sigmas:
                raise ValueError("Smoothing Sigmas 不能为空")
            optimizer_iterations = oi_var.get()
            raw_ot = ot_var.get().strip()
            if not raw_ot:
                raise ValueError("优化容差不能为空")
            optimizer_tol = float(raw_ot)
            optimizer_bounds = (ob_lo_var.get(), ob_hi_var.get())
            break
        except ValueError as e:
            messagebox.showerror("输入格式错误", f"{e}\n请修正后重试。", parent=root)
            # 重建参数窗口
            param_win = tk.Toplevel(root)
            param_win.title("配准参数设置")
            param_win.resizable(False, False)
            r = 0
            tk.Label(param_win, text="配准算法：", font=("", 11)).grid(
                row=r, column=0, sticky="e", padx=(15, 2), pady=(15, 3))
            mf2 = tk.Frame(param_win)
            mf2.grid(row=r, column=1, sticky="w")
            for text, val in [("刚性", "刚性"), ("弹性", "弹性"), ("刚性+弹性", "刚性+弹性")]:
                tk.Radiobutton(mf2, text=text, variable=mode_var, value=val,
                               font=("", 10)).pack(side="left", padx=2)
            r += 1
            tk.Label(param_win, text="缩放因子：", font=("", 11)).grid(
                row=r, column=0, sticky="e", padx=(15, 2), pady=3)
            tk.Entry(param_win, textvariable=sf_var, width=6).grid(row=r, column=1, sticky="w")
            r += 1
            tk.Label(param_win, text="匹配点数：", font=("", 11)).grid(
                row=r, column=0, sticky="e", padx=(15, 2), pady=3)
            tk.Entry(param_win, textvariable=nm_var, width=6).grid(row=r, column=1, sticky="w")
            r += 1
            tk.Checkbutton(param_win, text="强制纯刚性（去除缩放分量）",
                           variable=only_rigid_var, font=("", 10)).grid(
                row=r, column=0, columnspan=3, sticky="w", padx=15, pady=3)
            r += 1
            tk.Label(param_win, text="--- 弹性配准参数 ---", font=("", 10, "bold")).grid(
                row=r, column=0, columnspan=3, sticky="w", padx=15, pady=(8, 2))
            r += 1
            tk.Label(param_win, text="B-spline 网格：", font=("", 11)).grid(
                row=r, column=0, sticky="e", padx=(15, 2), pady=3)
            tk.Entry(param_win, textvariable=mesh_var, width=6).grid(row=r, column=1, sticky="w")
            r += 1
            tk.Label(param_win, text="Shrink Factors：", font=("", 11)).grid(
                row=r, column=0, sticky="e", padx=(15, 2), pady=3)
            tk.Entry(param_win, textvariable=sh_var, width=6).grid(row=r, column=1, sticky="w")
            tk.Label(param_win, text="(逗号分隔)", font=("", 9)).grid(row=r, column=2, sticky="w")
            r += 1
            tk.Label(param_win, text="Smoothing Sigmas：", font=("", 11)).grid(
                row=r, column=0, sticky="e", padx=(15, 2), pady=3)
            tk.Entry(param_win, textvariable=sm_var, width=6).grid(row=r, column=1, sticky="w")
            tk.Label(param_win, text="(逗号分隔)", font=("", 9)).grid(row=r, column=2, sticky="w")
            r += 1
            tk.Label(param_win, text="优化迭代次数：", font=("", 11)).grid(
                row=r, column=0, sticky="e", padx=(15, 2), pady=3)
            tk.Entry(param_win, textvariable=oi_var, width=6).grid(row=r, column=1, sticky="w")
            r += 1
            tk.Label(param_win, text="优化容差：", font=("", 11)).grid(
                row=r, column=0, sticky="e", padx=(15, 2), pady=3)
            tk.Entry(param_win, textvariable=ot_var, width=6).grid(row=r, column=1, sticky="w")
            r += 1
            tk.Label(param_win, text="边界下/上限：", font=("", 11)).grid(
                row=r, column=0, sticky="e", padx=(15, 2), pady=3)
            bf2 = tk.Frame(param_win)
            bf2.grid(row=r, column=1, sticky="w")
            tk.Entry(bf2, textvariable=ob_lo_var, width=5).pack(side="left")
            tk.Label(bf2, text=" ~ ", font=("", 10)).pack(side="left")
            tk.Entry(bf2, textvariable=ob_hi_var, width=5).pack(side="left")
            r += 1
            tk.Button(param_win, text="开始配准", command=param_win.destroy,
                      width=12).grid(row=r, column=0, columnspan=3, pady=(15, 15))

    root.destroy()

    # ===== 打印配置 =====
    print(f"\n配准算法: {mode}")
    print(f"参数: scale_factor={scale_factor}, n_matches={n_matches}, only_rigid={only_rigid}")
    if mode in ("弹性", "刚性+弹性"):
        print(f"弹性: mesh={mesh_size}, shrink={shrink_factors}, smooth={smoothing_sigmas}, "
              f"iter={optimizer_iterations}, tol={optimizer_tol}, bounds={optimizer_bounds}")
    print()

    # ===== Step 5: 逐组逐对配准 =====
    ref_imgs_cache: dict[str, np.ndarray] = {}

    for g_idx, mov_paths in enumerate(groups, 1):
        group_dir = os.path.join(output_root, f"group_{g_idx:02d}")
        os.makedirs(group_dir, exist_ok=True)
        print(f"\n{'=' * 50}")
        print(f"第 {g_idx}/{len(groups)} 组 → {group_dir}")
        print(f"{'=' * 50}")

        ok_count = 0
        for p_idx, (ref_path, mov_path) in enumerate(zip(ref_paths, mov_paths), 1):
            ref_name = os.path.basename(ref_path)
            mov_name = os.path.basename(mov_path)
            print(f"  [{p_idx}/{N}] {mov_name}  →  {ref_name}  ...", end=" ")

            try:
                # 加载图像（参考图缓存复用）
                if ref_path not in ref_imgs_cache:
                    ref_img = cv2.imread(ref_path, cv2.IMREAD_COLOR)
                    if ref_img is None:
                        print(f"✗ 参考图读取失败")
                        continue
                    ref_imgs_cache[ref_path] = ref_img
                else:
                    ref_img = ref_imgs_cache[ref_path]

                mov_img = cv2.imread(mov_path, cv2.IMREAD_COLOR)
                if mov_img is None:
                    print(f"✗ 待配准图读取失败")
                    continue

                # 配准
                if mode == "刚性":
                    result = rigid_registration(
                        ref_img, mov_img,
                        n_matches=n_matches,
                        scale_factor=scale_factor,
                        only_rigid=only_rigid,
                    )
                    if result is None:
                        print(f"✗ 刚性配准失败（特征点不足）")
                        continue
                elif mode == "弹性":
                    result = elastic_registration(
                        ref_img, mov_img,
                        mesh_size=mesh_size,
                        shrink_factors=shrink_factors,
                        smoothing_sigmas=smoothing_sigmas,
                        optimizer_iterations=optimizer_iterations,
                        optimizer_tol=optimizer_tol,
                        optimizer_bounds=optimizer_bounds,
                    )
                elif mode == "刚性+弹性":
                    rigid_result = rigid_registration(
                        ref_img, mov_img,
                        n_matches=n_matches,
                        scale_factor=scale_factor,
                        only_rigid=True,  # 刚性阶段强制去缩放
                    )
                    if rigid_result is None:
                        print(f"✗ 刚性预配准失败（特征点不足）")
                        continue
                    result = elastic_registration(
                        ref_img, rigid_result,
                        mesh_size=mesh_size,
                        shrink_factors=shrink_factors,
                        smoothing_sigmas=smoothing_sigmas,
                        optimizer_iterations=optimizer_iterations,
                        optimizer_tol=optimizer_tol,
                        optimizer_bounds=optimizer_bounds,
                    )
                else:
                    raise ValueError(f"未知配准模式: {mode}")

                # 保存（保留原文件名）
                out_path = os.path.join(group_dir, mov_name)
                ok = cv2.imwrite(out_path, result)
                if not ok:
                    print(f"✗ 保存失败")
                    continue
                print(f"✓")
                ok_count += 1

            except Exception as e:
                print(f"✗ {e}")

        print(f"  完成: {ok_count}/{N} 对")

    print(f"\n{'=' * 50}")
    print(f"全部完成！共 {len(groups)} 组 × {N} 对，输出至: {output_root}")
    print(f"{'=' * 50}")
