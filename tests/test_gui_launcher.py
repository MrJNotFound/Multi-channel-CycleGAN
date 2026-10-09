import argparse
import json
import tempfile
import unittest
from pathlib import Path

from gui.launcher import (
    OptionSpec,
    ProgressParser,
    build_cli_args,
    build_command,
    default_values,
    load_checkpoint_configuration,
    recommended_dataset_for,
    validate_configuration,
)


def _spec(dest, flag, default, *, boolean=False, nargs=None, required=False):
    return OptionSpec(
        dest=dest,
        flag=flag,
        aliases=(flag,),
        default=default,
        help="",
        choices=(),
        required=required,
        boolean=boolean,
        store_false=False,
        nargs=nargs,
        value_type=str,
        section="其他",
    )


def _event(payload):
    return "@@MLCGUI " + json.dumps(payload)


class LauncherTests(unittest.TestCase):
    def test_progress_parser_tracks_structured_training_events(self):
        parser = ProgressParser("train")
        parser.feed(
            _event(
                {
                    "type": "run_started",
                    "task": "train",
                    "dataset_size": 200,
                    "start_epoch": 1,
                    "final_epoch": 20,
                }
            )
        )
        state = parser.feed(
            _event(
                {
                    "type": "progress",
                    "task": "train",
                    "epoch": 3,
                    "final_epoch": 20,
                    "epoch_iter": 50,
                    "dataset_size": 200,
                }
            )
        )
        parser.feed(
            _event(
                {
                    "type": "metrics",
                    "values": {"G_A": 1.25, "D_A": 0.375},
                }
            )
        )

        self.assertEqual((state.current, state.total), (450, 4000))
        self.assertEqual(state.metrics, {"G_A": 1.25, "D_A": 0.375})

    def test_progress_parser_tracks_structured_inference_events(self):
        parser = ProgressParser("inference")
        parser.feed(
            _event(
                {
                    "type": "run_started",
                    "task": "inference",
                    "total": 23,
                }
            )
        )
        state = parser.feed(
            _event(
                {
                    "type": "progress",
                    "task": "inference",
                    "current": 16,
                    "total": 23,
                }
            )
        )

        self.assertEqual(state.current, 16)
        self.assertEqual(state.total, 23)
        self.assertAlmostEqual(state.fraction, 16 / 23)

    def test_cli_builder_emits_only_changed_flags_and_multi_values(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--dataroot", required=True)
        parser.add_argument("--enabled", action="store_true")
        parser.add_argument("--weights", nargs="+", type=float)
        specs = [
            _spec("dataroot", "--dataroot", None, required=True),
            _spec("enabled", "--enabled", False, boolean=True),
            _spec("weights", "--weights", None, nargs="+"),
        ]

        args = build_cli_args(
            parser,
            specs,
            {
                "dataroot": "data with spaces",
                "enabled": True,
                "weights": "0.4, 0.6",
            },
        )

        self.assertEqual(
            args,
            [
                "--dataroot",
                "data with spaces",
                "--enabled",
                "--weights",
                "0.4",
                "0.6",
            ],
        )

    def test_cli_builder_rejects_missing_required_value(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--dataroot", required=True)
        spec = _spec("dataroot", "--dataroot", None, required=True)

        with self.assertRaisesRegex(ValueError, "不能为空"):
            build_cli_args(parser, [spec], {"dataroot": ""})

    def test_cli_builder_omits_false_default_flag(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--dataroot", required=True)
        parser.add_argument("--verbose", action="store_true")
        specs = [
            _spec("dataroot", "--dataroot", None, required=True),
            _spec("verbose", "--verbose", False, boolean=True),
        ]

        args = build_cli_args(parser, specs, {"dataroot": "images"})

        self.assertNotIn("--verbose", args)

    def test_typed_boolean_can_override_true_default(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--enabled",
            type=lambda value: value.lower() == "true",
            nargs="?",
            const=True,
            default=True,
        )
        spec = OptionSpec(
            dest="enabled",
            flag="--enabled",
            aliases=("--enabled",),
            default=True,
            help="",
            choices=(),
            required=False,
            boolean=True,
            store_false=False,
            nargs="?",
            value_type=bool,
            section="其他",
            boolean_value=True,
        )

        args = build_cli_args(parser, [spec], {"enabled": False})

        self.assertEqual(args, ["--enabled", "False"])

    def test_training_command_uses_unified_engine_and_plugins(self):
        values = default_values("train", "cycle_gan_dual_fusion_structure", "dual_channel")
        values.update({"dataroot": "datasets/sample", "name": "sample_run"})

        command = build_command(
            "train", "cycle_gan_dual_fusion_structure", "dual_channel", values, "python"
        )

        self.assertTrue(
            Path(command[1]).as_posix().endswith("training/train.py")
        )
        self.assertEqual(
            command[-5:],
            [
                "--model",
                "cycle_gan_dual_fusion_structure",
                "--dataset_mode",
                "dual_channel",
                "--gui_events",
            ],
        )
        self.assertNotIn("train_dual.py", " ".join(command))

    def test_inference_separates_source_module_from_runner(self):
        values = default_values("inference", "cut_dual_stack_base", "dual_channel")
        values.update({"dataroot": "datasets/sample", "name": "sample_run"})

        command = build_command(
            "inference", "cut_dual_stack_base", "dual_channel", values, "python"
        )

        self.assertTrue(
            Path(command[1]).as_posix().endswith("inference/infer.py")
        )
        self.assertEqual(command[command.index("--model") + 1], "test")
        self.assertEqual(
            command[command.index("--source_model") + 1], "cut_dual_stack_base"
        )
        self.assertEqual(
            command[command.index("--dataset_mode") + 1], "dual_channel"
        )

    def test_task_specific_dataset_guidance_is_not_a_lock(self):
        self.assertEqual(
            recommended_dataset_for("train", "cycle_gan_single_none_base"), "unaligned"
        )
        self.assertEqual(
            recommended_dataset_for("inference", "cycle_gan_single_none_base"), "single"
        )
        self.assertEqual(
            recommended_dataset_for("inference", "cycle_gan_dual_fusion_structure"), "dual_channel"
        )

    def test_validation_accepts_fresh_modular_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "trainA").mkdir()
            (root / "trainB").mkdir()
            values = default_values("train", "cycle_gan_single_none_base", "unaligned")
            values.update({"dataroot": str(root), "name": "experiment"})

            report = validate_configuration(
                "train", "cycle_gan_single_none_base", "unaligned", "fresh", values
            )

        self.assertEqual(report.errors, ())

    def test_checkpoint_text_options_are_importable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train_opt.txt").write_text(
                "----------------- Options ---------------\n"
                "                    model: cycle_gan_dual_fusion_structure\n"
                "                 input_nc: 2\n"
                "                no_dropout: True\t[default: False]\n"
                "                   gpu_ids: [0]\n"
                "----------------- End -------------------\n",
                encoding="utf-8",
            )

            values = load_checkpoint_configuration(root)

        self.assertEqual(values["model"], "cycle_gan_dual_fusion_structure")
        self.assertEqual(values["input_nc"], 2)
        self.assertIs(values["no_dropout"], True)
        self.assertEqual(values["gpu_ids"], [0])


if __name__ == "__main__":
    unittest.main()
