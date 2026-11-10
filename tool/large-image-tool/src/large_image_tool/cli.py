from __future__ import annotations

import argparse
import json
from pathlib import Path

from .file_index import discover_numeric_tiles
from .models import FastAxis, GridGeometry, MarkFilterSettings, ScanPattern, StartCorner, ToolMarkMethod
from .registration import AutoStitcher
from .synthetic import generate_turning_dataset
from .toolmarks import process_toolmark_input


def _pattern(args: argparse.Namespace) -> ScanPattern:
    return ScanPattern(StartCorner(args.corner), FastAxis(args.axis), args.snake)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="显微图像工具的测试与诊断命令")
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser("synthetic", help="生成带圆环刀痕的测试扫描图")
    synthetic.add_argument("output", type=Path)
    synthetic.add_argument("--rows", type=int, default=5)
    synthetic.add_argument("--cols", type=int, default=6)
    synthetic.add_argument("--width", type=int, default=640)
    synthetic.add_argument("--height", type=int, default=360)
    synthetic.add_argument("--overlap-x", type=float, default=14.0)
    synthetic.add_argument("--overlap-y", type=float, default=12.0)
    synthetic.add_argument("--rotation", "--angle", dest="rotation", type=float, default=1.4)
    synthetic.add_argument("--corner", choices=[item.value for item in StartCorner], default=StartCorner.TOP_LEFT.value)
    synthetic.add_argument("--axis", choices=[item.value for item in FastAxis], default=FastAxis.HORIZONTAL.value)
    synthetic.add_argument("--snake", action=argparse.BooleanOptionalAction, default=True)
    synthetic.add_argument("--seed", type=int, default=7)

    analyze = subparsers.add_parser("analyze", help="对数字命名的扫描图执行自动配准诊断")
    analyze.add_argument("folder", type=Path)
    analyze.add_argument("--rows", type=int, required=True)
    analyze.add_argument("--cols", type=int, required=True)
    analyze.add_argument("--corner", choices=[item.value for item in StartCorner], default=StartCorner.TOP_LEFT.value)
    analyze.add_argument("--axis", choices=[item.value for item in FastAxis], default=FastAxis.HORIZONTAL.value)
    analyze.add_argument("--snake", action=argparse.BooleanOptionalAction, default=True)
    analyze.add_argument("--overlap-min", type=float, default=0.0)
    analyze.add_argument("--overlap-max", type=float, default=30.0)
    analyze.add_argument("--rotation-min", type=float, default=-8.0)
    analyze.add_argument("--rotation-max", type=float, default=8.0)
    analyze.add_argument("--sample-pairs", type=int, default=24)
    analyze.add_argument("--registration-size", type=int, default=1024)
    analyze.add_argument("--output", type=Path, help="可选的结果 JSON 文件")

    marks = subparsers.add_parser("remove-marks", help="去除单图、大图或文件夹中的刀纹")
    marks.add_argument("input", type=Path)
    marks.add_argument("output", type=Path)
    marks.add_argument("--method", choices=[item.value for item in ToolMarkMethod], default=ToolMarkMethod.FFT_GAUSSIAN_NOTCH.value)
    marks.add_argument("--strength", type=float, default=0.72)
    marks.add_argument("--orientation", type=float, help="手动法向角；省略时自动识别")
    marks.add_argument("--period", type=float, help="手动周期；省略时自动识别")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "synthetic":
        geometry = GridGeometry(args.overlap_x, args.overlap_y, args.rotation)
        geometry.validate()
        output = generate_turning_dataset(
            args.output,
            rows=args.rows,
            cols=args.cols,
            tile_size=(args.width, args.height),
            geometry=geometry,
            pattern=_pattern(args),
            seed=args.seed,
        )
        print(output)
        return 0

    if args.command == "remove-marks":
        settings = MarkFilterSettings(
            method=args.method,
            strength=args.strength,
            auto_orientation=args.orientation is None,
            orientation_deg=args.orientation or 0.0,
            auto_period=args.period is None,
            period_px=args.period or 18.0,
        )
        result = process_toolmark_input(args.input, args.output, settings)
        print(result.output_path)
        print(result.settings_path)
        return 0

    tile_set = discover_numeric_tiles(args.folder, args.rows, args.cols, _pattern(args))

    def progress(current: int, total: int, message: str) -> None:
        print(f"\r{message}: {current}/{total}", end="", flush=True)

    result = AutoStitcher(
        tile_set,
        overlap_range=(args.overlap_min, args.overlap_max),
        rotation_range=(args.rotation_min, args.rotation_max),
        max_registration_size=args.registration_size,
        sample_pairs_per_direction=args.sample_pairs,
    ).run(progress=progress)
    print()
    payload = {
        "geometry": {
            "overlap_x_percent": result.geometry.overlap_x_percent,
            "overlap_y_percent": result.geometry.overlap_y_percent,
            "tile_rotation_deg": result.geometry.tile_rotation_deg,
        },
        "edge_count": len(result.edges),
        "low_confidence_edge_count": len(result.low_confidence_edges),
        "warnings": tile_set.warnings,
        "messages": result.messages,
        "positions": {
            f"{row},{col}": [float(value[0]), float(value[1])]
            for (row, col), value in result.positions.items()
        },
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(args.output.resolve())
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
