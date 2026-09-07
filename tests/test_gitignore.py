"""Check repository scope without staging, modifying or deleting any files."""

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def ignored_paths(paths):
    result = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-c", "core.quotepath=false",
         "check-ignore", "--no-index", "--stdin", "-z"],
        input=("\0".join(paths) + "\0").encode("utf-8"), cwd=ROOT,
        capture_output=True, check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    return set(filter(None, result.stdout.decode("utf-8").split("\0")))


class GitignoreTests(unittest.TestCase):
    def test_core_sources_are_trackable(self):
        paths = [".gitignore", "README.md", "LICENSE", "launch_gui.py",
                 "requirements.txt", "requirements-dev.txt", "pyproject.toml",
                 "environment.yml", ".github/workflows/tests.yml", ".env.example"]
        for directory in ("models", "data", "options", "util", "gui", "tests"):
            paths.extend(path.relative_to(ROOT).as_posix()
                         for path in (ROOT / directory).rglob("*.py")
                         if "__pycache__" not in path.parts)
        paths.extend(["models/README.md", "training/train.py", "training/finetune.py",
                      "training/train_single.py", "training/train_dual.py",
                      "inference/infer.py", "inference/infer_single.py", "inference/infer_dual.py"])
        self.assertEqual(ignored_paths(paths), set())

    def test_curated_docs_and_tools_are_trackable(self):
        paths = ["docs/GUI_CN.md", "docs/REPOSITORY_SCOPE.md",
                 "docs/spif_v2/architecture.md", "docs/spif_v2/spif_v2_generator.svg",
                 "docs/spif_v2/spif_v2_generator.png", "docs/spif_v2/sccf_principle_diagram.png"]
        tools = ["build_dual_channel_dataset", "patch_generate_multi_mode", "patch_stitching",
                 "img_invert", "img_registrating_pair", "img_registrating_series",
                 "color_normalization", "auto_freq_filter", "ROI_crop", "threshold_seg",
                 "metrics_evaluating", "loss_plot", "visualize_content_loss"]
        paths.extend(f"tool/{name}.py" for name in tools)
        self.assertEqual(ignored_paths(paths), set())

    def test_local_outputs_experiments_and_secrets_are_ignored(self):
        paths = ["datasets/trainA/image.png", "checkpoints/latest.pth", "results/image.png",
                 "output/docx/manuscript.docx", "tmp/pdfs/page.png", "out/render.svg",
                 "paper/manuscript.tex", "scripts/eval_cityscapes/evaluate.py",
                 "scripts/install_deps.sh", "run_ablation.py", "finetune.py", "train_spif_af.py",
                 "training/run_dual_5epoch_20runs.py", "training/train_cut_af.py",
                 "training/train_spif_bf.py", "inference/infer_utom.py",
                 "docs/spif_v2/MANUSCRIPT_REMAINING_REVISIONS.md",
                 "docs/spif_v2/CMPB_SUBMISSION_REQUIREMENTS_AND_CHECKLIST.md",
                 "tool/recycle_bin/patch_generate.py", "tool/3D-stack-with-slicing-master/main.py",
                 "tool/volume_rendering.py", "tool/draw_spif_v2_generator.py",
                 "models/temporary_weights.pt", "models/weights.safetensors",
                 "gui/run.log", "tests/__pycache__/test_model.cpython-312.pyc",
                 "tests/.pytest_cache/state", "gui/.env", ".env.local",
                 ".idea/workspace.xml", ".codex/config.toml", ".venv/pyvenv.cfg",
                 "unreviewed_local_folder/anything.py", "local_notes.md"]
        self.assertEqual(ignored_paths(paths), set(paths))


if __name__ == "__main__":
    unittest.main()
