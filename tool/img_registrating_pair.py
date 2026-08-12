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
import SimpleITK as sitk
import tkinter as tk
from tkinter import filedialog, messagebox

# 允许从同目录导入配准函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from img_registrating_series import (
    rigid_registration, elastic_registration,
    rigid_registration_matrix, rigid_registration_apply,
    elastic_registration_transform, elastic_registration_apply,
)


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


def _save_params(params_root, mode, cached_rigid, cached_elastic):
    """将缓存的变换参数保存到磁盘，供以后直接导入复用。

    目录结构:
        params_root/rigid/pair_01.npz      (mat_full, fixed_shape)
        params_root/elastic/pair_01_transform.tfm
        params_root/elastic/pair_01_fixed.mha
    """
    if mode in ("刚性", "刚性+弹性") and cached_rigid:
        rigid_dir = os.path.join(params_root, "rigid")
        os.makedirs(rigid_dir, exist_ok=True)
        for p_idx, (mat_full, fixed_shape) in cached_rigid.items():
            np.savez(os.path.join(rigid_dir, f"pair_{p_idx:02d}.npz"),
                     mat=mat_full, shape=np.array(fixed_shape))
        print(f"  刚性参数已保存: {len(cached_rigid)} 对 → {rigid_dir}")
    if mode in ("弹性", "刚性+弹性") and cached_elastic:
        elastic_dir = os.path.join(params_root, "elastic")
        os.makedirs(elastic_dir, exist_ok=True)
        for p_idx, (outTx, fixed_sitk, orig_shape) in cached_elastic.items():
            sitk.WriteTransform(outTx, os.path.join(elastic_dir, f"pair_{p_idx:02d}_transform.tfm"))
            sitk.WriteImage(fixed_sitk, os.path.join(elastic_dir, f"pair_{p_idx:02d}_fixed.mha"))
            np.savez(os.path.join(elastic_dir, f"pair_{p_idx:02d}_shape.npz"),
                     shape=np.array(orig_shape))
        print(f"  弹性参数已保存: {len(cached_elastic)} 对 → {elastic_dir}")


def _load_params(params_root, mode, n_pairs):
    """从磁盘加载变换参数。返回 (cached_rigid, cached_elastic)。

    Raises:
        FileNotFoundError: 缺少某个参数文件时。
    """
    cached_rigid = {}
    cached_elastic = {}
    if mode in ("刚性", "刚性+弹性"):
        rigid_dir = os.path.join(params_root, "rigid")
        for p_idx in range(1, n_pairs + 1):
            path = os.path.join(rigid_dir, f"pair_{p_idx:02d}.npz")
            if not os.path.exists(path):
                raise FileNotFoundError(f"缺少刚性参数文件: {path}")
            data = np.load(path)
            cached_rigid[p_idx] = (data["mat"], tuple(int(x) for x in data["shape"]))
    if mode in ("弹性", "刚性+弹性"):
        elastic_dir = os.path.join(params_root, "elastic")
        for p_idx in range(1, n_pairs + 1):
            tfm_path = os.path.join(elastic_dir, f"pair_{p_idx:02d}_transform.tfm")
            fixed_path = os.path.join(elastic_dir, f"pair_{p_idx:02d}_fixed.mha")
            shape_path = os.path.join(elastic_dir, f"pair_{p_idx:02d}_shape.npz")
            if not os.path.exists(tfm_path) or not os.path.exists(fixed_path):
                raise FileNotFoundError(f"缺少弹性参数文件: {tfm_path}")
            orig_shape = None
            if os.path.exists(shape_path):
                orig_shape = tuple(int(x) for x in np.load(shape_path)["shape"])
            cached_elastic[p_idx] = (sitk.ReadTransform(tfm_path), sitk.ReadImage(fixed_path), orig_shape)
    return cached_rigid, cached_elastic


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
    elastic_sf_var = tk.DoubleVar(value=0.3)

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

    reuse_rigid_var = tk.BooleanVar(value=False)
    tk.Checkbutton(param_win, text="复用刚性参数（仅首对计算仿射矩阵，其余复用）",
                   variable=reuse_rigid_var, font=("", 10)).grid(
        row=r, column=0, columnspan=3, sticky="w", padx=15, pady=3)
    r += 1

    reuse_elastic_var = tk.BooleanVar(value=False)
    tk.Checkbutton(param_win, text="复用弹性参数（仅首对计算形变场，其余复用）",
                   variable=reuse_elastic_var, font=("", 10)).grid(
        row=r, column=0, columnspan=3, sticky="w", padx=15, pady=3)
    r += 1

    # ---- 参数保存 / 导入 ----
    save_params_var = tk.BooleanVar(value=False)
    load_params_var = tk.BooleanVar(value=False)
    save_params_dir_var = tk.StringVar()
    load_params_dir_var = tk.StringVar()

    def _pick_save_dir():
        d = filedialog.askdirectory(title="选择参数保存文件夹")
        if d:
            save_params_dir_var.set(d)

    def _pick_load_dir():
        d = filedialog.askdirectory(title="选择参数导入文件夹")
        if d:
            load_params_dir_var.set(d)

    tk.Checkbutton(param_win, text="保存首组配准参数（保存到文件夹，供以后直接套用）",
                   variable=save_params_var, font=("", 10)).grid(
        row=r, column=0, columnspan=3, sticky="w", padx=15, pady=(8, 0))
    r += 1
    sf_frame = tk.Frame(param_win)
    sf_frame.grid(row=r, column=0, columnspan=3, sticky="w", padx=15)
    tk.Entry(sf_frame, textvariable=save_params_dir_var, width=30).pack(side="left")
    tk.Button(sf_frame, text="选择…", command=_pick_save_dir).pack(side="left", padx=5)
    r += 1

    tk.Checkbutton(param_win, text="导入已有参数（跳过配准计算，直接套用）",
                   variable=load_params_var, font=("", 10)).grid(
        row=r, column=0, columnspan=3, sticky="w", padx=15, pady=(8, 0))
    r += 1
    lf_frame = tk.Frame(param_win)
    lf_frame.grid(row=r, column=0, columnspan=3, sticky="w", padx=15)
    tk.Entry(lf_frame, textvariable=load_params_dir_var, width=30).pack(side="left")
    tk.Button(lf_frame, text="选择…", command=_pick_load_dir).pack(side="left", padx=5)
    r += 1

    # 弹性参数
    tk.Label(param_win, text="--- 弹性配准参数 ---", font=("", 10, "bold")).grid(
        row=r, column=0, columnspan=3, sticky="w", padx=15, pady=(8, 2))
    r += 1

    tk.Label(param_win, text="B-spline 网格：", font=("", 11)).grid(
        row=r, column=0, sticky="e", padx=(15, 2), pady=3)
    tk.Entry(param_win, textvariable=mesh_var, width=6).grid(row=r, column=1, sticky="w")
    r += 1

    tk.Label(param_win, text="弹性缩放因子：", font=("", 11)).grid(
        row=r, column=0, sticky="e", padx=(15, 2), pady=3)
    tk.Entry(param_win, textvariable=elastic_sf_var, width=6).grid(row=r, column=1, sticky="w")
    tk.Label(param_win, text="(<1 降采样节省内存，>1 放大, 1=原始)",
             font=("", 9)).grid(row=r, column=2, sticky="w")
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
            save_params = save_params_var.get()
            load_params = load_params_var.get()
            save_params_dir = save_params_dir_var.get().strip()
            load_params_dir = load_params_dir_var.get().strip()
            if save_params and load_params:
                raise ValueError("不能同时勾选保存参数和导入参数")
            if save_params and not save_params_dir:
                raise ValueError("保存参数需要选择保存文件夹")
            if load_params and not load_params_dir:
                raise ValueError("导入参数需要选择参数文件夹")
            if load_params and not os.path.isdir(load_params_dir):
                raise ValueError(f"参数文件夹不存在: {load_params_dir}")
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
            tk.Checkbutton(param_win, text="复用刚性参数（仅首对计算仿射矩阵，其余复用）",
                           variable=reuse_rigid_var, font=("", 10)).grid(
                row=r, column=0, columnspan=3, sticky="w", padx=15, pady=3)
            r += 1
            tk.Checkbutton(param_win, text="复用弹性参数（仅首对计算形变场，其余复用）",
                           variable=reuse_elastic_var, font=("", 10)).grid(
                row=r, column=0, columnspan=3, sticky="w", padx=15, pady=3)
            r += 1
            tk.Checkbutton(param_win, text="保存首组配准参数（保存到文件夹，供以后直接套用）",
                           variable=save_params_var, font=("", 10)).grid(
                row=r, column=0, columnspan=3, sticky="w", padx=15, pady=(8, 0))
            r += 1
            sf_frame2 = tk.Frame(param_win)
            sf_frame2.grid(row=r, column=0, columnspan=3, sticky="w", padx=15)
            tk.Entry(sf_frame2, textvariable=save_params_dir_var, width=30).pack(side="left")
            tk.Button(sf_frame2, text="选择…", command=_pick_save_dir).pack(side="left", padx=5)
            r += 1
            tk.Checkbutton(param_win, text="导入已有参数（跳过配准计算，直接套用）",
                           variable=load_params_var, font=("", 10)).grid(
                row=r, column=0, columnspan=3, sticky="w", padx=15, pady=(8, 0))
            r += 1
            lf_frame2 = tk.Frame(param_win)
            lf_frame2.grid(row=r, column=0, columnspan=3, sticky="w", padx=15)
            tk.Entry(lf_frame2, textvariable=load_params_dir_var, width=30).pack(side="left")
            tk.Button(lf_frame2, text="选择…", command=_pick_load_dir).pack(side="left", padx=5)
            r += 1
            tk.Label(param_win, text="--- 弹性配准参数 ---", font=("", 10, "bold")).grid(
                row=r, column=0, columnspan=3, sticky="w", padx=15, pady=(8, 2))
            r += 1
            tk.Label(param_win, text="B-spline 网格：", font=("", 11)).grid(
                row=r, column=0, sticky="e", padx=(15, 2), pady=3)
            tk.Entry(param_win, textvariable=mesh_var, width=6).grid(row=r, column=1, sticky="w")
            r += 1
            tk.Label(param_win, text="弹性缩放因子：", font=("", 11)).grid(
                row=r, column=0, sticky="e", padx=(15, 2), pady=3)
            tk.Entry(param_win, textvariable=elastic_sf_var, width=6).grid(row=r, column=1, sticky="w")
            tk.Label(param_win, text="(<1 降采样节省内存)", font=("", 9)).grid(row=r, column=2, sticky="w")
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
    reuse_rigid = reuse_rigid_var.get()
    reuse_elastic = reuse_elastic_var.get()
    elastic_sf = elastic_sf_var.get()
    print(f"\n配准算法: {mode}")
    print(f"参数: scale_factor={scale_factor}, n_matches={n_matches}, only_rigid={only_rigid}, "
          f"复用刚性={reuse_rigid}, 复用弹性={reuse_elastic}")
    if save_params:
        print(f"保存参数到: {save_params_dir}")
    if load_params:
        print(f"导入参数自: {load_params_dir}")
    if mode in ("弹性", "刚性+弹性"):
        print(f"弹性: mesh={mesh_size}, shrink={shrink_factors}, smooth={smoothing_sigmas}, "
              f"iter={optimizer_iterations}, tol={optimizer_tol}, bounds={optimizer_bounds}, "
              f"elastic_sf={elastic_sf}")
    print()

    # ===== Step 5: 逐组逐对配准 =====
    ref_imgs_cache: dict[str, np.ndarray] = {}

    # 缓存：仅首对计算参数，后续复用
    #   rigid:  {p_idx: (mat_full, fixed_shape)}
    #   elastic: {p_idx: (outTx, fixed_sitk)}
    cached_rigid: dict[int, tuple] = {}
    cached_elastic: dict[int, tuple] = {}

    # 导入已有参数：所有组都直接套用，不再计算
    if load_params:
        try:
            cached_rigid, cached_elastic = _load_params(load_params_dir, mode, N)
        except FileNotFoundError as e:
            print(f"✗ 参数导入失败: {e}")
            exit(1)
        print(f"已导入参数: 刚性 {len(cached_rigid)} 对, 弹性 {len(cached_elastic)} 对")

    # 保存参数时也默认复用本次计算的首组参数
    use_rigid_cache = load_params or reuse_rigid or save_params
    use_elastic_cache = load_params or reuse_elastic or save_params

    for g_idx, mov_paths in enumerate(groups, 1):
        group_dir = os.path.join(output_root, f"group_{g_idx:02d}")
        os.makedirs(group_dir, exist_ok=True)
        print(f"\n{'=' * 50}")
        print(f"第 {g_idx}/{len(groups)} 组 → {group_dir}")
        if use_rigid_cache or use_elastic_cache:
            flags = []
            if use_rigid_cache:
                flags.append("刚性")
            if use_elastic_cache:
                flags.append("弹性")
            if load_params:
                print(f"  [导入参数: {', '.join(flags)}]")
            else:
                print(f"  [复用: {', '.join(flags)}]")
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

                # ---- 模式: 刚性 ----
                if mode == "刚性":
                    if use_rigid_cache and p_idx in cached_rigid:
                        mat_full, fixed_shape = cached_rigid[p_idx]
                        result = rigid_registration_apply(mov_img, mat_full, fixed_shape)
                        print(f"♻", end="")
                    elif use_rigid_cache:
                        mat_full, fixed_shape = rigid_registration_matrix(
                            ref_img, mov_img,
                            n_matches=n_matches,
                            scale_factor=scale_factor,
                            only_rigid=only_rigid,
                        )
                        if mat_full is None:
                            print(f"✗ 刚性配准失败（特征点不足）")
                            continue
                        cached_rigid[p_idx] = (mat_full, fixed_shape)
                        result = rigid_registration_apply(mov_img, mat_full, fixed_shape)
                    else:
                        result = rigid_registration(
                            ref_img, mov_img,
                            n_matches=n_matches,
                            scale_factor=scale_factor,
                            only_rigid=only_rigid,
                        )
                        if result is None:
                            print(f"✗ 刚性配准失败（特征点不足）")
                            continue

                # ---- 模式: 弹性 ----
                elif mode == "弹性":
                    if use_elastic_cache and p_idx in cached_elastic:
                        outTx, fixed_sitk, orig_shape = cached_elastic[p_idx]
                        result = elastic_registration_apply(mov_img, fixed_sitk, outTx,
                                                            output_size=orig_shape)
                        print(f"♻", end="")
                    elif use_elastic_cache:
                        outTx, fixed_sitk, orig_shape = elastic_registration_transform(
                            ref_img, mov_img,
                            mesh_size=mesh_size,
                            shrink_factors=shrink_factors,
                            smoothing_sigmas=smoothing_sigmas,
                            optimizer_iterations=optimizer_iterations,
                            optimizer_tol=optimizer_tol,
                            optimizer_bounds=optimizer_bounds,
                            scale_factor=elastic_sf,
                        )
                        cached_elastic[p_idx] = (outTx, fixed_sitk, orig_shape)
                        result = elastic_registration_apply(mov_img, fixed_sitk, outTx,
                                                            output_size=orig_shape)
                    else:
                        result = elastic_registration(
                            ref_img, mov_img,
                            mesh_size=mesh_size,
                            shrink_factors=shrink_factors,
                            smoothing_sigmas=smoothing_sigmas,
                            optimizer_iterations=optimizer_iterations,
                            optimizer_tol=optimizer_tol,
                            optimizer_bounds=optimizer_bounds,
                            scale_factor=elastic_sf,
                        )

                # ---- 模式: 刚性+弹性 ----
                elif mode == "刚性+弹性":
                    # -- 刚性阶段 --
                    if use_rigid_cache and p_idx in cached_rigid:
                        mat_full, fixed_shape = cached_rigid[p_idx]
                        rigid_result = rigid_registration_apply(mov_img, mat_full, fixed_shape)
                        print(f"♻", end="")
                    elif use_rigid_cache:
                        mat_full, fixed_shape = rigid_registration_matrix(
                            ref_img, mov_img,
                            n_matches=n_matches,
                            scale_factor=scale_factor,
                            only_rigid=True,
                        )
                        if mat_full is None:
                            print(f"✗ 刚性预配准失败（特征点不足）")
                            continue
                        cached_rigid[p_idx] = (mat_full, fixed_shape)
                        rigid_result = rigid_registration_apply(mov_img, mat_full, fixed_shape)
                    else:
                        rigid_result = rigid_registration(
                            ref_img, mov_img,
                            n_matches=n_matches,
                            scale_factor=scale_factor,
                            only_rigid=True,
                        )
                        if rigid_result is None:
                            print(f"✗ 刚性预配准失败（特征点不足）")
                            continue

                    # -- 弹性阶段 --
                    if use_elastic_cache and p_idx in cached_elastic:
                        outTx, fixed_sitk, orig_shape = cached_elastic[p_idx]
                        result = elastic_registration_apply(rigid_result, fixed_sitk, outTx,
                                                            output_size=orig_shape)
                        if use_rigid_cache:
                            print(f"+♻", end="")
                        else:
                            print(f"♻", end="")
                    elif use_elastic_cache:
                        outTx, fixed_sitk, orig_shape = elastic_registration_transform(
                            ref_img, rigid_result,
                            mesh_size=mesh_size,
                            shrink_factors=shrink_factors,
                            smoothing_sigmas=smoothing_sigmas,
                            optimizer_iterations=optimizer_iterations,
                            optimizer_tol=optimizer_tol,
                            optimizer_bounds=optimizer_bounds,
                            scale_factor=elastic_sf,
                        )
                        cached_elastic[p_idx] = (outTx, fixed_sitk, orig_shape)
                        result = elastic_registration_apply(rigid_result, fixed_sitk, outTx,
                                                            output_size=orig_shape)
                    else:
                        result = elastic_registration(
                            ref_img, rigid_result,
                            mesh_size=mesh_size,
                            shrink_factors=shrink_factors,
                            smoothing_sigmas=smoothing_sigmas,
                            optimizer_iterations=optimizer_iterations,
                            optimizer_tol=optimizer_tol,
                            optimizer_bounds=optimizer_bounds,
                            scale_factor=elastic_sf,
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

        # 首组处理完后，将缓存的参数保存到磁盘
        if save_params and g_idx == 1:
            _save_params(save_params_dir, mode, cached_rigid, cached_elastic)
            print(f"  参数已保存到: {save_params_dir}")

    print(f"\n{'=' * 50}")
    print(f"全部完成！共 {len(groups)} 组 × {N} 对，输出至: {output_root}")
    print(f"{'=' * 50}")
