"""Guided desktop UI for module-driven training and inference."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui.launcher import (  # noqa: E402
    ESSENTIAL_FIELDS,
    RECOMMENDED_COMMON,
    SECTION_FIELDS,
    ROOT,
    SECTION_ORDER,
    ProgressParser,
    build_command,
    coerce_bool,
    dataset_info,
    default_values,
    discover_datasets,
    discover_models,
    display_value,
    load_checkpoint_configuration,
    module_info,
    option_label,
    option_specs,
    output_directory,
    quote_command,
    recommended_dataset_for,
    validate_configuration,
)
from models.registry import (
    COMPONENT_LABELS, COMPONENT_TITLES, MODEL_SPECS, composition_candidates,
    composition_errors,
)


C = {
    "bg": "#F3F5F6",
    "surface": "#FFFFFF",
    "soft": "#F7F9F9",
    "line": "#DDE4E5",
    "ink": "#162226",
    "muted": "#68777D",
    "brand": "#087E8B",
    "brand_dark": "#055F68",
    "accent": "#E98A3A",
    "success": "#248A63",
    "warning": "#AD6819",
    "danger": "#B64141",
    "console": "#111A1E",
    "console_text": "#D7E2E5",
}


class ScrollableFrame(ttk.Frame):
    def __init__(self, master: tk.Misc, background: str = C["surface"]):
        super().__init__(master)
        self.canvas = tk.Canvas(self, bg=background, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.content = ttk.Frame(self.canvas, style="Surface.TFrame")
        self.window = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.content.bind(
            "<Configure>",
            lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.bind(
            "<Configure>",
            lambda event: self.canvas.itemconfigure(self.window, width=event.width),
        )
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self._bind_wheel(self.canvas)
        self._bind_wheel(self.content)

    def _bind_wheel(self, widget: tk.Widget) -> None:
        widget.bind(
            "<MouseWheel>",
            lambda event: self.canvas.yview_scroll(int(-event.delta / 120), "units"),
        )


class AdvancedDialog(tk.Toplevel):
    def __init__(self, owner: "ControlCenter") -> None:
        super().__init__(owner)
        self.owner = owner
        self.title("专家参数 · 完整模块配置")
        self.geometry("980x760")
        self.minsize(780, 580)
        self.configure(bg=C["bg"])
        self.transient(owner)
        self.grab_set()
        self.working = dict(owner.values)
        self.variables: dict[str, tk.Variable] = {}
        self.search_var = tk.StringVar()
        self.changed_only_var = tk.BooleanVar(value=False)

        header = ttk.Frame(self, padding=(20, 16, 20, 10))
        header.pack(fill="x")
        ttk.Label(header, text="专家参数", style="DialogTitle.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="这里包含当前模型与数据模块公开的全部参数。日常使用无需进入此页。",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(3, 0))

        filters = ttk.Frame(self, style="Surface.TFrame", padding=(14, 10))
        filters.pack(fill="x", padx=20)
        ttk.Label(filters, text="搜索参数", style="Field.TLabel").pack(side="left")
        search = ttk.Entry(filters, textvariable=self.search_var, width=28)
        search.pack(side="left", padx=(8, 18))
        ttk.Checkbutton(
            filters,
            text="只看已修改",
            variable=self.changed_only_var,
            command=self._rebuild,
        ).pack(side="left")
        ttk.Button(
            filters,
            text="恢复模块推荐值",
            style="Quiet.TButton",
            command=self._reset,
        ).pack(side="right")
        self.search_var.trace_add("write", lambda *_args: self.after(120, self._rebuild))

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=20, pady=12)
        self.pages: dict[str, ScrollableFrame] = {}
        for section in SECTION_ORDER:
            page = ScrollableFrame(self.notebook)
            self.notebook.add(page, text=section)
            self.pages[section] = page

        footer = ttk.Frame(self, padding=(20, 0, 20, 16))
        footer.pack(fill="x")
        ttk.Button(footer, text="取消", style="Quiet.TButton", command=self.destroy).pack(side="right")
        ttk.Button(footer, text="应用", style="Brand.TButton", command=self._apply).pack(side="right", padx=(0, 8))
        self._rebuild()

    def _capture(self) -> None:
        for dest, variable in self.variables.items():
            self.working[dest] = variable.get()

    def _rebuild(self) -> None:
        self._capture()
        for page in self.pages.values():
            for child in page.content.winfo_children():
                child.destroy()
        self.variables = {}
        search = self.search_var.get().strip().lower()
        counts = {section: 0 for section in SECTION_ORDER}
        for spec in self.owner.specs:
            text = f"{spec.dest} {option_label(spec.dest)} {spec.help}".lower()
            if search and search not in text:
                continue
            if self.changed_only_var.get() and self.owner.is_baseline_value(
                spec.dest, self.working.get(spec.dest, spec.default)
            ):
                continue
            variable = self.owner.create_editor(
                self.pages[spec.section].content,
                spec,
                self.working.get(spec.dest, spec.default),
                compact=False,
                on_change=lambda: None,
            )
            self.variables[spec.dest] = variable
            counts[spec.section] += 1
        for index, section in enumerate(SECTION_ORDER):
            self.notebook.tab(index, text=f"{section} · {counts[section]}")
            if counts[section] == 0:
                ttk.Label(
                    self.pages[section].content,
                    text="没有匹配的参数",
                    style="MutedSurface.TLabel",
                ).pack(pady=28)

    def _reset(self) -> None:
        self.working = dict(self.owner.baseline_values)
        self.variables = {}
        self._rebuild()

    def _apply(self) -> None:
        self._capture()
        self.owner.values.update(self.working)
        self.owner.rebuild_parameter_panels(capture=False)
        self.owner.schedule_refresh()
        self.destroy()


class ControlCenter(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Multi-layer CycleGAN · 模块化训练与推理")
        self.geometry("1500x920")
        self.minsize(1180, 740)
        self.configure(bg=C["bg"])
        self._styles()

        self.task_var = tk.StringVar(value="train")
        self.init_var = tk.StringVar(value="fresh")
        self.module_var = tk.StringVar(value="cycle_gan_dual_fusion_structure")
        self.dataset_var = tk.StringVar(value="dual_channel")
        self.version_var = tk.StringVar(value="v2")
        self.component_vars = {name: tk.StringVar() for name in COMPONENT_LABELS}
        self.component_combos = {}
        self.dataset_display_var = tk.StringVar()
        self.status_var = tk.StringVar(value="就绪")
        self.stage_var = tk.StringVar(value="等待配置")
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_text_var = tk.StringVar(value="0%")
        self.elapsed_var = tk.StringVar(value="00:00:00")
        self.validation_title_var = tk.StringVar(value="尚未完成配置")
        self.validation_detail_var = tk.StringVar(value="请选择数据路径并填写实验名称。")
        self.command_var = tk.StringVar()
        self.help_var = tk.StringVar(value="参数说明会显示在这里。")

        self.model_names = discover_models()
        self.values = default_values("train", self.module_var.get(), "dual_channel")
        self.baseline_values = dict(self.values)
        self.parser = None
        self.specs = []
        self.spec_by_dest = {}
        self.field_vars: dict[str, tk.Variable] = {}
        self.method_expanded = False
        self._refresh_timer: str | None = None

        self.process: subprocess.Popen[str] | None = None
        self.output_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.progress_parser: ProgressParser | None = None
        self.started_at: float | None = None
        self.stop_requested = False
        self.metric_history: dict[str, list[float]] = {}
        self.image_refs: list[Any] = []

        self._build()
        self.configure_schema(reset=True)
        self.after(100, self._drain_output)
        self.after(1000, self._tick)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=C["bg"])
        style.configure("Surface.TFrame", background=C["surface"])
        style.configure("Soft.TFrame", background=C["soft"])
        style.configure("Title.TLabel", background=C["bg"], foreground=C["ink"], font=("Microsoft YaHei UI", 20, "bold"))
        style.configure("DialogTitle.TLabel", background=C["bg"], foreground=C["ink"], font=("Microsoft YaHei UI", 16, "bold"))
        style.configure("Section.TLabel", background=C["surface"], foreground=C["ink"], font=("Microsoft YaHei UI", 11, "bold"))
        style.configure("Field.TLabel", background=C["surface"], foreground=C["ink"], font=("Microsoft YaHei UI", 9))
        style.configure("Muted.TLabel", background=C["bg"], foreground=C["muted"], font=("Microsoft YaHei UI", 9))
        style.configure("MutedSurface.TLabel", background=C["surface"], foreground=C["muted"], font=("Microsoft YaHei UI", 9))
        style.configure("Code.TLabel", background=C["surface"], foreground=C["muted"], font=("Cascadia Mono", 8))
        style.configure("CardValue.TLabel", background=C["soft"], foreground=C["ink"], font=("Microsoft YaHei UI", 14, "bold"))
        style.configure("CardCaption.TLabel", background=C["soft"], foreground=C["muted"], font=("Microsoft YaHei UI", 8))
        style.configure("Brand.TButton", background=C["brand"], foreground="white", padding=(16, 9), borderwidth=0, font=("Microsoft YaHei UI", 10, "bold"))
        style.map("Brand.TButton", background=[("active", C["brand_dark"]), ("disabled", "#AAB8BC")])
        style.configure("Quiet.TButton", padding=(10, 6), font=("Microsoft YaHei UI", 9))
        style.configure("Danger.TButton", foreground=C["danger"], padding=(10, 6))
        style.configure("Segment.TRadiobutton", background=C["soft"], foreground=C["ink"], padding=(12, 7), font=("Microsoft YaHei UI", 9, "bold"))
        style.map("Segment.TRadiobutton", background=[("selected", "#DDEFF1")], foreground=[("selected", C["brand_dark"])])
        style.configure("Horizontal.TProgressbar", background=C["brand"], troughcolor="#E5EAEC", borderwidth=0)
        style.configure("TNotebook", background=C["surface"], borderwidth=0)
        style.configure("TNotebook.Tab", padding=(12, 7), font=("Microsoft YaHei UI", 9))
        style.map("TNotebook.Tab", foreground=[("selected", C["brand_dark"])])
        style.configure("Treeview", rowheight=24, background=C["surface"], fieldbackground=C["surface"])
        style.configure("Treeview.Heading", font=("Microsoft YaHei UI", 9, "bold"))

    def _build(self) -> None:
        header = ttk.Frame(self, padding=(24, 17, 24, 12))
        header.pack(fill="x")
        title = ttk.Frame(header)
        title.pack(side="left", fill="x", expand=True)
        ttk.Label(title, text="MODEL LAB", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            title,
            text="模型模块 × 数据模块 × 参数配置 · 统一训练与推理入口",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(2, 0))
        ttk.Label(header, text=self.device_summary(), style="Muted.TLabel").pack(side="right")

        pane = ttk.Panedwindow(self, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=24, pady=(0, 16))
        left_host = ttk.Frame(pane, style="Surface.TFrame")
        right = ttk.Frame(pane, style="Surface.TFrame", padding=16)
        pane.add(left_host, weight=5)
        pane.add(right, weight=6)
        self.left = ScrollableFrame(left_host)
        self.left.pack(fill="both", expand=True)
        self._build_configuration(self.left.content)
        self._build_monitor(right)

    def device_summary(self) -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return f"● CUDA · {torch.cuda.device_count()} GPU"
            return "● CPU · CUDA 不可用"
        except Exception:
            return "● 设备状态未知"

    def card(self, parent: tk.Misc, title: str, subtitle: str = "") -> ttk.Frame:
        frame = ttk.Frame(parent, style="Surface.TFrame", padding=14)
        frame.pack(fill="x", padx=14, pady=(12, 0))
        ttk.Label(frame, text=title, style="Section.TLabel").pack(anchor="w")
        if subtitle:
            ttk.Label(frame, text=subtitle, style="MutedSurface.TLabel", wraplength=570).pack(anchor="w", pady=(2, 10))
        return frame

    def _build_configuration(self, parent: ttk.Frame) -> None:
        task_card = self.card(parent, "1 · 选择任务", "训练与推理共用同一套模块配置，不再映射到专用脚本。")
        task_row = ttk.Frame(task_card, style="Soft.TFrame", padding=5)
        task_row.pack(fill="x")
        for key, label in (("train", "训练模型"), ("inference", "运行推理")):
            ttk.Radiobutton(
                task_row,
                text=label,
                value=key,
                variable=self.task_var,
                style="Segment.TRadiobutton",
                command=self.on_task_changed,
            ).pack(side="left", fill="x", expand=True)

        self.init_frame = ttk.Frame(task_card, style="Surface.TFrame")
        self.init_frame.pack(fill="x", pady=(10, 0))
        ttk.Label(self.init_frame, text="权重初始化", style="Field.TLabel").pack(side="left", padx=(0, 10))
        for key, label in (("fresh", "从头训练"), ("resume", "继续实验"), ("finetune", "载入预训练后微调")):
            ttk.Radiobutton(
                self.init_frame,
                text=label,
                value=key,
                variable=self.init_var,
                command=self.on_initialization_changed,
            ).pack(side="left", padx=(0, 8))

        module_card = self.card(parent, "2 · 组合模型", "依次选择四个组成元素；下游仅显示已实现的组合。RGB 通道不是图像类别。")
        selectors = ttk.Frame(module_card, style="Surface.TFrame")
        selectors.pack(fill="x")
        selectors.columnconfigure((0, 1), weight=1)
        for index, name in enumerate(COMPONENT_LABELS):
            box = ttk.Frame(selectors, style="Surface.TFrame")
            box.grid(row=index // 2, column=index % 2, sticky="ew", padx=(0, 8), pady=(0, 10))
            ttk.Label(box, text=COMPONENT_TITLES[name], style="Field.TLabel").pack(anchor="w")
            combo = ttk.Combobox(box, textvariable=self.component_vars[name], state="readonly", width=24)
            combo.pack(fill="x", pady=(4, 0))
            combo.bind("<<ComboboxSelected>>", lambda event, part=name: self.on_component_changed(part))
            self.component_combos[name] = combo
            if name == "adapter":
                self.adapter_frame = box
        data_box = ttk.Frame(selectors, style="Surface.TFrame")
        data_box.grid(row=3, column=0, columnspan=2, sticky="ew")
        ttk.Label(data_box, text="数据载入模块（按组合推荐，可调整）", style="Field.TLabel").pack(anchor="w")
        self.dataset_combo = ttk.Combobox(data_box, textvariable=self.dataset_display_var, state="readonly")
        self.dataset_combo.pack(fill="x", pady=(4, 0))
        self.dataset_combo.bind("<<ComboboxSelected>>", self.on_dataset_changed)

        self.version_frame = ttk.Frame(selectors, style="Surface.TFrame")
        self.version_frame.grid(row=2, column=1, sticky="ew", pady=(0, 10))
        ttk.Label(self.version_frame, text="SPIF 版本：v1 通道 / v2 通道—空间", style="Field.TLabel").pack(anchor="w")
        self.version_combo = ttk.Combobox(self.version_frame, textvariable=self.version_var,
                                          values=("v1", "v2"), state="readonly")
        self.version_combo.pack(fill="x", pady=(4, 0))
        self.version_combo.bind("<<ComboboxSelected>>", self.on_version_changed)

        info = ttk.Frame(module_card, style="Soft.TFrame", padding=10)
        info.pack(fill="x", pady=(10, 0))
        self.module_title = tk.Label(info, bg=C["soft"], fg=C["brand_dark"], anchor="w", font=("Microsoft YaHei UI", 10, "bold"))
        self.module_title.pack(fill="x")
        self.module_description = tk.Label(info, bg=C["soft"], fg=C["muted"], anchor="w", justify="left", wraplength=560, font=("Microsoft YaHei UI", 9))
        self.module_description.pack(fill="x", pady=(3, 5))
        self.layout_label = tk.Label(info, bg=C["soft"], fg=C["ink"], anchor="w", justify="left", font=("Cascadia Mono", 8))
        self.layout_label.pack(fill="x")

        self.essential_card = self.card(parent, "3 · 必要配置", "只填写路径与实验身份即可形成可执行配置。")
        self.essential_body = ttk.Frame(self.essential_card, style="Surface.TFrame")
        self.essential_body.pack(fill="x")

        common_card = self.card(parent, "4 · 推荐设置", "常用尺寸、训练计划和运行参数；其余参数保持模块默认值。")
        self.common_body = ttk.Frame(common_card, style="Surface.TFrame")
        self.common_body.pack(fill="x")

        self.method_card = self.card(parent, "方法专用设置")
        self.method_toggle = ttk.Button(self.method_card, style="Quiet.TButton", command=self.toggle_method_parameters)
        self.method_toggle.pack(fill="x")
        self.method_body = ttk.Frame(self.method_card, style="Surface.TFrame")

        tools = ttk.Frame(parent, style="Surface.TFrame", padding=14)
        tools.pack(fill="x", padx=14, pady=(12, 14))
        self.advanced_button = ttk.Button(tools, text="专家参数", style="Quiet.TButton", command=self.open_advanced)
        self.advanced_button.pack(side="left")
        ttk.Button(tools, text="恢复模块默认", style="Quiet.TButton", command=self.reset_defaults).pack(side="left", padx=6)
        ttk.Button(
            tools,
            text="读取权重配置",
            style="Quiet.TButton",
            command=self.import_checkpoint_configuration,
        ).pack(side="left")
        ttk.Button(tools, text="载入配置", style="Quiet.TButton", command=self.load_config).pack(side="right")
        ttk.Button(tools, text="保存配置", style="Quiet.TButton", command=self.save_config).pack(side="right", padx=6)
        ttk.Label(parent, textvariable=self.help_var, style="MutedSurface.TLabel", wraplength=580).pack(fill="x", padx=22, pady=(0, 14))

    def _build_monitor(self, parent: ttk.Frame) -> None:
        header = ttk.Frame(parent, style="Surface.TFrame")
        header.pack(fill="x")
        ttk.Label(header, text="运行与反馈", style="Section.TLabel").pack(side="left")
        self.status_badge = tk.Label(header, textvariable=self.status_var, bg="#E4F3EF", fg=C["success"], padx=10, pady=4, font=("Microsoft YaHei UI", 9, "bold"))
        self.status_badge.pack(side="right")

        validation = ttk.Frame(parent, style="Soft.TFrame", padding=11)
        validation.pack(fill="x", pady=(12, 10))
        self.validation_icon = tk.Label(validation, text="○", bg=C["soft"], fg=C["muted"], font=("Microsoft YaHei UI", 16, "bold"))
        self.validation_icon.pack(side="left", padx=(0, 9))
        validation_text = ttk.Frame(validation, style="Soft.TFrame")
        validation_text.pack(side="left", fill="x", expand=True)
        self.validation_title = tk.Label(validation_text, textvariable=self.validation_title_var, bg=C["soft"], fg=C["ink"], anchor="w", font=("Microsoft YaHei UI", 10, "bold"))
        self.validation_title.pack(fill="x")
        self.validation_detail = tk.Label(validation_text, textvariable=self.validation_detail_var, bg=C["soft"], fg=C["muted"], anchor="w", justify="left", wraplength=610, font=("Microsoft YaHei UI", 8))
        self.validation_detail.pack(fill="x", pady=(2, 0))

        stats = ttk.Frame(parent, style="Surface.TFrame")
        stats.pack(fill="x")
        stats.columnconfigure((0, 1, 2), weight=1)
        self._stat(stats, "当前阶段", self.stage_var, 0)
        self._stat(stats, "总体进度", self.progress_text_var, 1)
        self._stat(stats, "运行时间", self.elapsed_var, 2)
        ttk.Progressbar(parent, variable=self.progress_var, maximum=100).pack(fill="x", pady=(8, 10))

        metrics = ttk.Frame(parent, style="Soft.TFrame", padding=9)
        metrics.pack(fill="x", pady=(0, 10))
        metrics.columnconfigure(0, weight=3)
        metrics.columnconfigure(1, weight=2)
        self.chart = tk.Canvas(metrics, height=145, bg=C["soft"], highlightthickness=0)
        self.chart.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.chart.bind("<Configure>", lambda _event: self.draw_chart())
        self.metrics_tree = ttk.Treeview(metrics, columns=("name", "value"), show="headings", height=5)
        self.metrics_tree.heading("name", text="最新指标")
        self.metrics_tree.heading("value", text="值")
        self.metrics_tree.column("name", width=120, anchor="w")
        self.metrics_tree.column("value", width=80, anchor="e")
        self.metrics_tree.grid(row=0, column=1, sticky="nsew")

        tabs = ttk.Notebook(parent)
        tabs.pack(fill="both", expand=True)
        log_page = ttk.Frame(tabs, style="Surface.TFrame")
        image_page = ttk.Frame(tabs, style="Surface.TFrame", padding=7)
        tabs.add(log_page, text="实时日志")
        tabs.add(image_page, text="最新图像")
        self._build_console(log_page)
        self._build_images(image_page)

        command_box = ttk.Frame(parent, style="Soft.TFrame", padding=8)
        command_box.pack(fill="x", pady=(10, 8))
        command_head = ttk.Frame(command_box, style="Soft.TFrame")
        command_head.pack(fill="x")
        ttk.Label(command_head, text="统一入口命令", style="CardCaption.TLabel").pack(side="left")
        ttk.Button(command_head, text="复制", style="Quiet.TButton", command=self.copy_command).pack(side="right")
        self.command_text = tk.Text(command_box, height=3, wrap="word", borderwidth=0, bg=C["soft"], fg=C["ink"], font=("Cascadia Mono", 8))
        self.command_text.pack(fill="x", pady=(3, 0))
        self.command_text.configure(state="disabled")

        buttons = ttk.Frame(parent, style="Surface.TFrame")
        # Reserve space for the primary action before the expanding log area.
        # Packing it last at the bottom can hide it on smaller/scaled screens.
        buttons.pack(fill="x", pady=(12, 0), before=validation)
        self.start_button = ttk.Button(buttons, text="▶ 开始运行", style="Brand.TButton", command=self.start)
        self.start_button.pack(side="left", fill="x", expand=True)
        self.stop_button = ttk.Button(buttons, text="■ 停止", style="Danger.TButton", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="打开输出", style="Quiet.TButton", command=self.open_output).pack(side="left", padx=(8, 0))

    def _stat(self, parent: ttk.Frame, caption: str, variable: tk.StringVar, column: int) -> None:
        frame = ttk.Frame(parent, style="Soft.TFrame", padding=(10, 8))
        frame.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 6, 0))
        ttk.Label(frame, textvariable=variable, style="CardValue.TLabel").pack(anchor="w")
        ttk.Label(frame, text=caption, style="CardCaption.TLabel").pack(anchor="w")

    def _build_console(self, parent: ttk.Frame) -> None:
        scrollbar = ttk.Scrollbar(parent)
        scrollbar.pack(side="right", fill="y")
        self.console = tk.Text(parent, wrap="word", state="disabled", yscrollcommand=scrollbar.set, bg=C["console"], fg=C["console_text"], insertbackground="white", borderwidth=0, padx=11, pady=9, font=("Cascadia Mono", 9))
        self.console.pack(fill="both", expand=True)
        scrollbar.configure(command=self.console.yview)
        self.console.tag_configure("warning", foreground="#F1BC72")
        self.console.tag_configure("error", foreground="#FF8585")
        self.console.tag_configure("success", foreground="#72D2AE")

    def _build_images(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent, style="Surface.TFrame")
        top.pack(fill="x", pady=(0, 5))
        ttk.Label(top, text="自动读取当前输出目录", style="Code.TLabel").pack(side="left")
        ttk.Button(top, text="刷新", style="Quiet.TButton", command=self.refresh_images).pack(side="right")
        grid = ttk.Frame(parent, style="Surface.TFrame")
        grid.pack(fill="both", expand=True)
        grid.columnconfigure((0, 1), weight=1)
        grid.rowconfigure((0, 1), weight=1)
        self.image_labels = []
        for index in range(4):
            label = tk.Label(grid, text="暂无图像", bg=C["soft"], fg=C["muted"], compound="top", font=("Microsoft YaHei UI", 8), padx=5, pady=5)
            label.grid(row=index // 2, column=index % 2, sticky="nsew", padx=3, pady=3)
            self.image_labels.append(label)

    def dataset_display(self, name: str) -> str:
        info = dataset_info(name)
        return f"{info.label}   [{name}]"

    def _refresh_selectors(self) -> None:
        spec = MODEL_SPECS[self.module_var.get()]
        display_spec = spec
        filters = {}
        for name, labels in COMPONENT_LABELS.items():
            candidates = composition_candidates(filters, self.model_names)
            choices = list(dict.fromkeys(getattr(candidate, name) for candidate in candidates))
            self.component_combos[name].configure(values=[labels[value] for value in choices])
            value = getattr(display_spec, name)
            self.component_vars[name].set(labels[value])
            filters[name] = value
            if name == "adapter":
                if len(choices) > 1:
                    self.adapter_frame.grid()
                else:
                    self.adapter_frame.grid_remove()
        self.version_var.set(str(self.values.get("spif_version", "v2")))
        if spec.fusion == "fusion":
            self.version_frame.grid()
        else:
            self.version_frame.grid_remove()
        datasets = discover_datasets(self.task_var.get())
        self.dataset_display_map = {self.dataset_display(name): name for name in datasets}
        self.dataset_combo.configure(values=list(self.dataset_display_map))
        self.dataset_display_var.set(self.dataset_display(self.dataset_var.get()))

    def capture_fields(self) -> None:
        for dest, variable in self.field_vars.items():
            self.values[dest] = variable.get()

    def configure_schema(self, reset: bool = False, preserved: dict[str, Any] | None = None) -> None:
        task = self.task_var.get()
        module = self.module_var.get()
        dataset = self.dataset_var.get()
        fresh = default_values(task, module, dataset)
        if reset:
            self.values = fresh
        elif preserved is not None:
            fresh.update(preserved)
            self.values = fresh
        else:
            fresh.update(self.values)
            self.values = fresh
        self.values["source_model"] = module
        self.values["dataset_mode"] = dataset
        self.sync_initialization_values()
        self.parser, self.specs = option_specs(task, module, dataset, self.values)
        self.spec_by_dest = {spec.dest: spec for spec in self.specs}
        self.baseline_values = default_values(task, module, dataset)
        self._refresh_selectors()
        self.update_module_guidance()
        self.rebuild_parameter_panels(capture=False)
        self.schedule_refresh()

    def sync_initialization_values(self) -> None:
        if self.task_var.get() != "train":
            self.values["continue_train"] = False
            self.values["pretrained_name"] = ""
            return
        mode = self.init_var.get()
        self.values["continue_train"] = mode == "resume"
        if mode != "finetune":
            self.values["pretrained_name"] = ""

    def on_task_changed(self) -> None:
        if self.running_guard():
            return
        self.capture_fields()
        keep = {
            key: self.values.get(key)
            for key in (
                "dataroot",
                "name",
                "checkpoints_dir",
                "results_dir",
                "gpu_ids",
            )
            if self.values.get(key) is not None
        }
        # Task switching must not change the generator used by a checkpoint.
        for key in SECTION_FIELDS["网络结构"] | {"input_nc", "output_nc"}:
            if key in self.values:
                keep[key] = self.values[key]
        module = self.module_var.get()
        if module not in self.model_names:
            module = "cycle_gan_single_none_base"
            self.module_var.set(module)
        dataset = recommended_dataset_for(self.task_var.get(), module)
        available = discover_datasets(self.task_var.get())
        if dataset not in available:
            dataset = available[0]
        self.dataset_var.set(dataset)
        if self.task_var.get() == "train":
            self.init_frame.pack(fill="x", pady=(10, 0))
        else:
            self.init_frame.pack_forget()
        self.configure_schema(reset=False, preserved=keep)

    def on_version_changed(self, _event=None) -> None:
        if self.running_guard():
            self.version_var.set(str(self.values.get("spif_version", "v2")))
            return
        self.values["spif_version"] = self.version_var.get()
        self.schedule_refresh()

    def on_component_changed(self, changed: str) -> None:
        if self.running_guard():
            self._refresh_selectors()
            return
        filters = {}
        for name, labels in COMPONENT_LABELS.items():
            available = composition_candidates(filters, self.model_names)
            current = next((key for key, label in labels.items() if label == self.component_vars[name].get()), None)
            choices = list(dict.fromkeys(getattr(spec, name) for spec in available))
            filters[name] = current if current in choices else choices[0]
        selected = composition_candidates(filters, self.model_names)[0].key
        if selected != self.module_var.get():
            self._select_module(selected)
        else:
            self._refresh_selectors()

    def _select_module(self, selected: str) -> None:
        self.capture_fields()
        keep_names = {
            "dataroot",
            "name",
            "checkpoints_dir",
            "results_dir",
            "gpu_ids",
            "num_threads",
            "n_epochs",
            "n_epochs_decay",
            "batch_size",
            "save_epoch_freq",
        }
        keep = {key: value for key, value in self.values.items() if key in keep_names}
        if (MODEL_SPECS[selected].fusion == "fusion"
                and MODEL_SPECS[self.module_var.get()].fusion == "fusion"):
            keep["spif_version"] = self.values.get("spif_version", "v2")
        self.module_var.set(selected)
        recommended_dataset = recommended_dataset_for(
            self.task_var.get(), selected
        )
        if recommended_dataset in discover_datasets(self.task_var.get()):
            self.dataset_var.set(recommended_dataset)
        self.method_expanded = False
        self.configure_schema(reset=False, preserved=keep)
        self.status_var.set("已按组合更新参数；路径与训练计划保留")

    def on_dataset_changed(self, _event=None) -> None:
        if self.running_guard():
            return
        selected = self.dataset_display_map.get(self.dataset_display_var.get())
        if not selected:
            return
        self.capture_fields()
        current = dict(self.values)
        self.dataset_var.set(selected)
        self.configure_schema(reset=False, preserved=current)

    def on_initialization_changed(self) -> None:
        if self.running_guard():
            return
        self.capture_fields()
        self.sync_initialization_values()
        self.rebuild_parameter_panels()
        self.schedule_refresh()

    def running_guard(self) -> bool:
        if self.process and self.process.poll() is None:
            messagebox.showwarning("任务运行中", "请先停止当前任务再修改任务或模块。")
            return True
        return False

    def update_module_guidance(self) -> None:
        info = module_info(self.module_var.get())
        data = dataset_info(self.dataset_var.get())
        self.module_title.configure(text=f"{info.family}  ·  {data.label}")
        spec = MODEL_SPECS[self.module_var.get()]
        base_losses = {"cycle_gan": "GAN + 循环一致性 + 可选恒等", "cut": "GAN + PatchNCE", "pix2pix": "条件 GAN + L1"}
        note = f"基础损失：{base_losses[spec.architecture]}。额外损失在此基础上叠加。"
        self.module_description.configure(text=f"{info.description}\n{note}\n{data.description}")
        layout = data.layout_train if self.task_var.get() == "train" else data.layout_inference
        layout_text = "需要的目录：" + ("   ".join(layout) if layout else "直接使用所选路径")
        self.layout_label.configure(text=layout_text)

    def rebuild_parameter_panels(self, capture: bool = True) -> None:
        if capture:
            self.capture_fields()
        self.field_vars = {}
        self.version_var.set(str(self.values.get("spif_version", "v2")))
        for body in (self.essential_body, self.common_body, self.method_body):
            for child in body.winfo_children():
                child.destroy()

        essential = list(ESSENTIAL_FIELDS[self.task_var.get()])
        if self.task_var.get() == "train":
            if self.init_var.get() == "resume":
                essential.extend(["epoch", "load_iter", "epoch_count"])
            elif self.init_var.get() == "finetune":
                essential.extend(["pretrained_name", "epoch", "load_iter"])
        self._populate(self.essential_body, essential)
        self._populate(self.common_body, RECOMMENDED_COMMON[self.task_var.get()])

        featured = [dest for dest in module_info(self.module_var.get()).featured if dest in self.spec_by_dest]
        self.method_toggle.configure(
            text=("▾ 隐藏" if self.method_expanded else "▸ 展开")
            + f" 当前组合的 {len(featured)} 个关键参数"
        )
        if self.method_expanded:
            self.method_body.pack(fill="x", pady=(8, 0))
            self._populate(self.method_body, featured)
        else:
            self.method_body.pack_forget()
        self.update_advanced_count()

    def _populate(self, parent: ttk.Frame, destinations) -> None:
        for dest in destinations:
            if dest == "spif_version":
                continue
            spec = self.spec_by_dest.get(dest)
            if not spec or dest in self.field_vars:
                continue
            variable = self.create_editor(
                parent,
                spec,
                self.values.get(dest, spec.default),
                compact=True,
                on_change=self.on_field_changed,
            )
            self.field_vars[dest] = variable

    def create_editor(
        self,
        parent: ttk.Frame,
        spec,
        value: Any,
        compact: bool,
        on_change: Callable[[], None],
    ) -> tk.Variable:
        row = ttk.Frame(parent, style="Surface.TFrame", padding=(3, 5))
        row.pack(fill="x")
        row.columnconfigure(1, weight=1)
        label_box = ttk.Frame(row, style="Surface.TFrame", width=170 if compact else 210)
        label_box.grid(row=0, column=0, sticky="nw", padx=(0, 10))
        label_box.grid_propagate(False)
        ttk.Label(label_box, text=option_label(spec.dest), style="Field.TLabel").pack(anchor="w")
        ttk.Label(label_box, text=spec.flag, style="Code.TLabel").pack(anchor="w")

        if spec.boolean:
            variable: tk.Variable = tk.BooleanVar(value=coerce_bool(value))
            widget = ttk.Checkbutton(row, text="启用", variable=variable)
            if (
                bool(spec.default)
                and not spec.store_false
                and not spec.boolean_value
            ):
                widget.configure(state="disabled", text="启用（模块固定默认）")
            widget.grid(row=0, column=1, sticky="w")
        else:
            variable = tk.StringVar(value=display_value(value))
            holder = ttk.Frame(row, style="Surface.TFrame")
            holder.grid(row=0, column=1, sticky="ew")
            holder.columnconfigure(0, weight=1)
            if spec.choices:
                widget = ttk.Combobox(holder, textvariable=variable, values=[str(item) for item in spec.choices], state="readonly")
            else:
                widget = ttk.Entry(holder, textvariable=variable)
            widget.grid(row=0, column=0, sticky="ew")
            if spec.dest in {"dataroot", "checkpoints_dir", "results_dir"}:
                ttk.Button(holder, text="浏览", style="Quiet.TButton", command=lambda var=variable: self.choose_directory(var)).grid(row=0, column=1, padx=(6, 0))

        help_text = spec.help or "由当前模块公开的参数。"
        widget.bind("<Enter>", lambda _event, text=help_text: self.help_var.set(text))
        widget.bind("<FocusIn>", lambda _event, text=help_text: self.help_var.set(text))
        variable.trace_add("write", lambda *_args: on_change())
        return variable

    def choose_directory(self, variable: tk.StringVar) -> None:
        current = Path(variable.get().strip() or ROOT)
        if not current.is_absolute():
            current = ROOT / current
        if not current.exists():
            current = ROOT
        selected = filedialog.askdirectory(initialdir=current)
        if not selected:
            return
        try:
            variable.set(str(Path(selected).relative_to(ROOT)))
        except ValueError:
            variable.set(selected)

    def on_field_changed(self) -> None:
        self.schedule_refresh()

    def toggle_method_parameters(self) -> None:
        self.capture_fields()
        self.method_expanded = not self.method_expanded
        self.rebuild_parameter_panels()

    def is_baseline_value(self, dest: str, value: Any) -> bool:
        baseline = self.baseline_values.get(dest, self.spec_by_dest.get(dest).default if dest in self.spec_by_dest else None)
        if isinstance(value, bool) or isinstance(baseline, bool):
            return coerce_bool(value) == coerce_bool(baseline)
        return display_value(value).strip() == display_value(baseline).strip()

    def update_advanced_count(self) -> None:
        self.capture_fields()
        presented = set(ESSENTIAL_FIELDS[self.task_var.get()])
        presented.update(RECOMMENDED_COMMON[self.task_var.get()])
        presented.update(module_info(self.module_var.get()).featured)
        if self.task_var.get() == "train":
            if self.init_var.get() == "resume":
                presented.update(("epoch", "load_iter", "epoch_count"))
            elif self.init_var.get() == "finetune":
                presented.update(("pretrained_name", "epoch", "load_iter"))
        changed = sum(
            1
            for spec in self.specs
            if spec.dest not in presented
            and not self.is_baseline_value(
                spec.dest, self.values.get(spec.dest, spec.default)
            )
        )
        self.advanced_button.configure(text=f"专家参数 · {changed} 项自定义")

    def open_advanced(self) -> None:
        self.capture_fields()
        AdvancedDialog(self)

    def reset_defaults(self) -> None:
        if self.running_guard():
            return
        self.capture_fields()
        paths = {key: self.values.get(key) for key in ("dataroot", "name", "checkpoints_dir", "results_dir", "gpu_ids")}
        self.values = default_values(self.task_var.get(), self.module_var.get(), self.dataset_var.get())
        self.values.update({key: value for key, value in paths.items() if value not in (None, "")})
        self.sync_initialization_values()
        self.baseline_values = default_values(self.task_var.get(), self.module_var.get(), self.dataset_var.get())
        self.rebuild_parameter_panels(capture=False)
        self.schedule_refresh()

    def import_checkpoint_configuration(self) -> None:
        """Bring architecture/options from a saved training experiment."""
        if self.running_guard():
            return
        self.capture_fields()
        source_name = str(self.values.get("name", "")).strip()
        if self.task_var.get() == "train" and self.init_var.get() == "finetune":
            source_name = str(self.values.get("pretrained_name", "")).strip()
        if not source_name:
            messagebox.showwarning(
                "缺少实验名称",
                "请先填写实验名称；微调模式下请填写预训练实验名称。",
            )
            return

        checkpoint_root = Path(
            str(self.values.get("checkpoints_dir", "./checkpoints"))
        ).expanduser()
        if not checkpoint_root.is_absolute():
            checkpoint_root = ROOT / checkpoint_root
        source_dir = checkpoint_root / source_name
        try:
            saved = load_checkpoint_configuration(source_dir)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            messagebox.showerror("无法读取权重配置", str(exc))
            return

        saved_module = str(saved.get("model", ""))
        if saved_module not in self.model_names or composition_errors(saved_module, saved):
            messagebox.showerror(
                "模型模块不可用",
                "权重配置不符合当前接口；请手动选择模型和 v1/v2 版本。权重文件无需重命名。",
            )
            return

        identity_fields = {
            key: self.values.get(key)
            for key in (
                "dataroot",
                "name",
                "pretrained_name",
                "checkpoints_dir",
                "results_dir",
                "gpu_ids",
                "phase",
            )
            if key in self.values
        }
        for key in (
            "model",
            "dataset_mode",
            "source_model",
            "isTrain",
            "continue_train",
            "counter",
            "data_size",
        ):
            saved.pop(key, None)
        saved.update(identity_fields)
        self.module_var.set(saved_module)
        self.dataset_var.set(recommended_dataset_for(self.task_var.get(), saved_module))
        self.configure_schema(reset=False, preserved=saved)
        self.set_status("已读取权重配置", "success")

    def schedule_refresh(self) -> None:
        if self._refresh_timer:
            self.after_cancel(self._refresh_timer)
        self._refresh_timer = self.after(180, self.refresh_configuration)

    def refresh_configuration(self) -> None:
        self._refresh_timer = None
        self.capture_fields()
        self.sync_initialization_values()
        try:
            command = build_command(
                self.task_var.get(),
                self.module_var.get(),
                self.dataset_var.get(),
                self.values,
            )
            command_text = quote_command(command)
        except Exception as exc:
            command_text = f"参数尚未形成有效命令：{exc}"
        self.command_text.configure(state="normal")
        self.command_text.delete("1.0", "end")
        self.command_text.insert("1.0", command_text)
        self.command_text.configure(state="disabled")
        self.refresh_validation()
        self.update_advanced_count()

    def refresh_validation(self) -> None:
        report = validate_configuration(
            self.task_var.get(),
            self.module_var.get(),
            self.dataset_var.get(),
            self.init_var.get(),
            self.values,
        )
        self.validation_report = report
        if report.errors:
            self.validation_title_var.set(f"需要处理 {len(report.errors)} 个问题")
            self.validation_detail_var.set(" · ".join(item.replace("\n", " ") for item in report.errors[:3]))
            self.validation_icon.configure(text="!", fg=C["danger"])
            self.validation_title.configure(fg=C["danger"])
        elif report.warnings:
            self.validation_title_var.set(f"配置可运行，存在 {len(report.warnings)} 条提示")
            self.validation_detail_var.set(" · ".join(report.warnings[:2]))
            self.validation_icon.configure(text="△", fg=C["warning"])
            self.validation_title.configure(fg=C["warning"])
        else:
            self.validation_title_var.set("配置检查通过")
            self.validation_detail_var.set("模块、数据目录、尺寸和检查点配置一致。")
            self.validation_icon.configure(text="✓", fg=C["success"])
            self.validation_title.configure(fg=C["success"])

    def save_config(self) -> None:
        self.capture_fields()
        path = filedialog.asksaveasfilename(title="保存模块配置", defaultextension=".json", filetypes=[("JSON", "*.json")], initialfile=f"{self.values.get('name') or self.module_var.get()}.json")
        if not path:
            return
        payload = {
            "schema": 4,
            "task": self.task_var.get(),
            "model_module": self.module_var.get(),
            "dataset_mode": self.dataset_var.get(),
            "initialization": self.init_var.get(),
            "values": self.values,
            "components": MODEL_SPECS[self.module_var.get()].components if self.module_var.get() in MODEL_SPECS else None,
        }
        try:
            Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("保存失败", str(exc))
            return
        self.set_status("配置已保存", "success")

    def load_config(self) -> None:
        if self.running_guard():
            return
        path = filedialog.askopenfilename(title="载入模块配置", filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if payload.get("schema") != 4:
                raise ValueError("请用新版界面重新保存配置（仅支持 schema 4），不再自动迁移旧配置。")
            task = payload["task"]
            module = payload["model_module"]
            dataset = payload["dataset_mode"]
            values = payload["values"]
            problems = composition_errors(module, {**values, "dataset_mode": dataset})
            if problems:
                raise ValueError("\n".join(problems))
            if task not in {"train", "inference"} or module not in self.model_names:
                raise ValueError("配置引用了当前仓库中不可用的任务或模型模块。")
            if payload.get("components") is not None and payload["components"] != MODEL_SPECS[module].components:
                raise ValueError("配置中的模型名称与组成元素不一致。")
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            messagebox.showerror("载入失败", str(exc))
            return
        self.task_var.set(task)
        self.module_var.set(module)
        self.dataset_var.set(dataset)
        self.init_var.set(payload.get("initialization", "fresh"))
        if task == "train":
            self.init_frame.pack(fill="x", pady=(10, 0))
        else:
            self.init_frame.pack_forget()
        self.configure_schema(reset=False, preserved=values)
        self.set_status("配置已载入", "success")

    def copy_command(self) -> None:
        value = self.command_text.get("1.0", "end-1c")
        self.clipboard_clear()
        self.clipboard_append(value)
        self.set_status("命令已复制", "success")

    def start(self) -> None:
        if self.process and self.process.poll() is None:
            return
        self.refresh_configuration()
        if not self.validation_report.valid:
            messagebox.showerror("配置检查未通过", "\n\n".join(self.validation_report.errors))
            return
        self.capture_fields()
        self.sync_initialization_values()
        try:
            command = build_command(self.task_var.get(), self.module_var.get(), self.dataset_var.get(), self.values)
        except ValueError as exc:
            messagebox.showerror("参数无效", str(exc))
            return
        self.clear_console()
        if self.validation_report.warnings:
            self.append_console("[配置提示]\n" + "\n".join(self.validation_report.warnings) + "\n\n", "warning")
        self.append_console(f"$ {quote_command(command)}\n\n", "success")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            self.process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
        except OSError as exc:
            messagebox.showerror("启动失败", str(exc))
            self.process = None
            return
        self.progress_parser = ProgressParser(self.task_var.get())
        self.metric_history = {}
        self.draw_chart()
        self.started_at = time.monotonic()
        self.stop_requested = False
        self.progress_var.set(0)
        self.progress_text_var.set("0%")
        self.stage_var.set("启动统一引擎")
        self.set_status("运行中", "running")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        threading.Thread(target=self.read_output, args=(self.process,), daemon=True).start()
        self.after(1200, self.refresh_images)

    def read_output(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        try:
            for line in iter(process.stdout.readline, ""):
                self.output_queue.put(("line", line))
        finally:
            process.stdout.close()
            self.output_queue.put(("done", process.wait()))

    def _drain_output(self) -> None:
        try:
            while True:
                kind, payload = self.output_queue.get_nowait()
                if kind == "line":
                    line = str(payload)
                    structured = line.startswith(ProgressParser.EVENT_PREFIX)
                    if not structured:
                        lowered = line.lower()
                        tag = "error" if "traceback" in lowered or "error" in lowered else ("warning" if "warning" in lowered else None)
                        self.append_console(line, tag)
                    if self.progress_parser:
                        state = self.progress_parser.feed(line)
                        self.update_progress(state)
                        if structured and state.last_event.get("type") == "metrics":
                            self.update_metrics(state.metrics)
                else:
                    self.process_finished(int(payload))
        except queue.Empty:
            pass
        self.after(100, self._drain_output)

    def update_progress(self, state) -> None:
        self.stage_var.set(state.phase)
        percent = state.fraction * 100
        self.progress_var.set(percent)
        self.progress_text_var.set(f"{percent:.1f}%")

    def update_metrics(self, metrics: dict[str, float]) -> None:
        for item in self.metrics_tree.get_children():
            self.metrics_tree.delete(item)
        for name, value in metrics.items():
            self.metrics_tree.insert("", "end", values=(name, f"{value:.6g}"))
            self.metric_history.setdefault(name, []).append(float(value))
            self.metric_history[name] = self.metric_history[name][-120:]
        self.draw_chart()

    def draw_chart(self) -> None:
        if not hasattr(self, "chart"):
            return
        self.chart.delete("all")
        width = max(self.chart.winfo_width(), 320)
        height = max(self.chart.winfo_height(), 145)
        margin = 24
        self.chart.create_text(8, 8, text="LOSS TREND", anchor="nw", fill=C["muted"], font=("Cascadia Mono", 8, "bold"))
        series = [(name, values) for name, values in self.metric_history.items() if len(values) >= 1][:4]
        if not series:
            self.chart.create_text(width / 2, height / 2, text="开始训练后显示损失趋势", fill=C["muted"], font=("Microsoft YaHei UI", 9))
            return
        all_values = [value for _, values in series for value in values]
        low, high = min(all_values), max(all_values)
        if high == low:
            high = low + 1
        colors = (C["brand"], C["accent"], "#6F62B5", "#4C9A61")
        self.chart.create_line(margin, height - margin, width - 8, height - margin, fill=C["line"])
        for series_index, (name, values) in enumerate(series):
            color = colors[series_index]
            points = []
            for index, value in enumerate(values):
                x = margin + (width - margin - 12) * (index / max(len(values) - 1, 1))
                y = 24 + (height - margin - 30) * (1 - (value - low) / (high - low))
                points.extend((x, y))
            if len(points) >= 4:
                self.chart.create_line(*points, fill=color, width=2, smooth=True)
            elif points:
                self.chart.create_oval(points[0] - 2, points[1] - 2, points[0] + 2, points[1] + 2, fill=color, outline="")
            self.chart.create_text(92 + series_index * 95, 9, text=name, fill=color, anchor="nw", font=("Cascadia Mono", 7))

    def process_finished(self, return_code: int) -> None:
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        if self.stop_requested:
            self.set_status("已停止", "warning")
            self.stage_var.set("任务已由用户停止")
        elif return_code == 0:
            self.set_status("已完成", "success")
            self.stage_var.set("运行完成")
            self.progress_var.set(100)
            self.progress_text_var.set("100%")
            self.append_console("\n[GUI] 任务成功完成。\n", "success")
        else:
            self.set_status(f"失败 · {return_code}", "error")
            self.stage_var.set("运行失败，请查看日志")
            self.append_console(f"\n[GUI] 子进程退出代码：{return_code}\n", "error")
        self.process = None
        self.refresh_images()

    def stop(self) -> None:
        if not self.process or self.process.poll() is not None:
            return
        self.stop_requested = True
        self.set_status("正在停止", "warning")
        self.stage_var.set("等待训练进程退出")
        self.append_console("\n[GUI] 正在停止任务……\n", "warning")
        process = self.process
        try:
            process.terminate()
        except OSError:
            return
        self.after(3000, lambda: self.force_kill(process))

    @staticmethod
    def force_kill(process: subprocess.Popen[str]) -> None:
        if process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass

    def append_console(self, text: str, tag: str | None = None) -> None:
        self.console.configure(state="normal")
        self.console.insert("end", text, tag or ())
        lines = int(self.console.index("end-1c").split(".")[0])
        if lines > 8000:
            self.console.delete("1.0", "1001.0")
        self.console.see("end")
        self.console.configure(state="disabled")

    def clear_console(self) -> None:
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")

    def set_status(self, text: str, state: str) -> None:
        palette = {
            "running": ("#DFF0F2", C["brand_dark"]),
            "success": ("#E3F2EC", C["success"]),
            "warning": ("#FFF0DC", C["warning"]),
            "error": ("#F9E3E3", C["danger"]),
        }
        bg, fg = palette.get(state, ("#E7ECEE", C["muted"]))
        self.status_var.set(text)
        self.status_badge.configure(bg=bg, fg=fg)

    def _tick(self) -> None:
        if self.started_at is not None and self.process and self.process.poll() is None:
            elapsed = int(time.monotonic() - self.started_at)
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.elapsed_var.set(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        self.after(1000, self._tick)

    def refresh_images(self) -> None:
        try:
            base = output_directory(self.task_var.get(), self.values)
            image_dir = base / "web" / "images" if self.task_var.get() == "train" else base / "images"
            paths = []
            if image_dir.is_dir():
                paths = sorted(
                    (path for path in image_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}),
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )[:4]
            self.show_images(paths)
        except (OSError, ValueError):
            pass
        if self.process and self.process.poll() is None:
            self.after(2500, self.refresh_images)

    def show_images(self, paths: list[Path]) -> None:
        self.image_refs = []
        try:
            from PIL import Image, ImageTk
        except ImportError:
            for label in self.image_labels:
                label.configure(text="需要 Pillow 才能显示预览", image="")
            return
        for index, label in enumerate(self.image_labels):
            if index >= len(paths):
                label.configure(text="暂无图像", image="")
                continue
            path = paths[index]
            try:
                with Image.open(path) as source:
                    image = source.convert("RGB")
                    image.thumbnail((280, 180))
                    photo = ImageTk.PhotoImage(image)
                self.image_refs.append(photo)
                label.configure(image=photo, text=path.name)
            except OSError:
                label.configure(text=f"无法读取\n{path.name}", image="")

    def open_output(self) -> None:
        path = output_directory(self.task_var.get(), self.values)
        path.mkdir(parents=True, exist_ok=True)
        try:
            if os.name == "nt":
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except OSError as exc:
            messagebox.showerror("无法打开目录", str(exc))

    def _on_close(self) -> None:
        if self.process and self.process.poll() is None:
            if not messagebox.askyesno("任务运行中", "关闭界面会停止任务，确定继续吗？"):
                return
            self.stop()
        self.destroy()


def main() -> None:
    ControlCenter().mainloop()


if __name__ == "__main__":
    main()
