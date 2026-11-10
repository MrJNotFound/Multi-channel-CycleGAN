from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    try:
        from .app import MainWindow, create_application
    except ModuleNotFoundError as exc:
        if exc.name in {"numpy", "PIL", "scipy", "PySide6", "tifffile", "imagecodecs", "zarr"}:
            project_dir = Path(__file__).resolve().parents[2]
            install_target = (
                f'-e "{project_dir}"'
                if (project_dir / "pyproject.toml").is_file()
                else "large-image-tool"
            )
            command = f'"{sys.executable}" -m pip install {install_target}'
            if sys.platform == "win32":
                command = "& " + command  # PowerShell invokes quoted paths with &.
            print(
                f"Missing dependency: {exc.name}\n"
                f"Python: {sys.executable}\n"
                f"Install dependencies with:\n{command}",
                file=sys.stderr,
            )
            return 2
        raise
    app = create_application(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

