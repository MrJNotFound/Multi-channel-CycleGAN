"""Launch the source checkout, using its local virtual environment if present."""

from pathlib import Path
import subprocess
import sys


def main() -> int:
    project_dir = Path(__file__).resolve().parent
    environment_dir = project_dir / ".venv"
    environment_python = environment_dir / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    )
    if environment_python.is_file() and Path(sys.prefix).resolve() != environment_dir.resolve():
        return subprocess.run(
            [str(environment_python), str(project_dir / "run.py"), *sys.argv[1:]],
            check=False,
        ).returncode

    if sys.version_info < (3, 12):
        print("large-image-tool requires Python 3.12 or newer.", file=sys.stderr)
        return 2

    # Resolve src relative to this file so IDEs and other working directories work.
    sys.path.insert(0, str(project_dir / "src"))
    from large_image_tool.main import main as application_main

    return application_main()


if __name__ == "__main__":
    raise SystemExit(main())

