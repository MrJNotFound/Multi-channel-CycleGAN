"""Compatibility entry point for the original GUI script and module name."""

from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "tool.code_GUI"

from .ui.main_window import MainWindow, main


if __name__ == "__main__":
    raise SystemExit(main())
