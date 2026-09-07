"""Unified generator inference driven by model and dataset modules."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import create_dataset  # noqa: E402
from models import create_model  # noqa: E402
from options.test_options import TestOptions  # noqa: E402
from util import util  # noqa: E402
from util.util import cleanup_ddp  # noqa: E402


GUI_EVENTS = False


def emit_event(event_type: str, **payload) -> None:
    if not GUI_EVENTS:
        return
    print(
        "@@MLCGUI "
        + json.dumps({"type": event_type, **payload}, ensure_ascii=False),
        flush=True,
    )


def _has_option(arguments: list[str], option: str) -> bool:
    return option in arguments or any(value.startswith(f"{option}=") for value in arguments)


def run_inference(opt) -> None:
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    total_images = min(dataset_size, opt.num_test)
    if total_images <= 0:
        raise RuntimeError(f"Inference dataset is empty: {opt.dataroot}")

    model = create_model(opt)
    model.setup(opt)
    if opt.eval:
        model.eval()

    web_dir = Path(opt.results_dir) / opt.name / f"{opt.phase}_{opt.epoch}"
    if opt.load_iter > 0:
        web_dir = Path(f"{web_dir}_iter{opt.load_iter}")
    image_dir = web_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    webpage = None
    if opt.make_report:
        from util import html

        webpage = html.HTML(
            web_dir,
            f"Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.epoch}",
        )

    print(f"Inference dataset size = {dataset_size}; processing = {total_images}")
    print(f"Saving generated images to {image_dir}")
    emit_event(
        "run_started",
        task="inference",
        source_model=opt.source_model,
        dataset_mode=opt.dataset_mode,
        total=total_images,
        output_dir=str(web_dir.resolve()),
    )

    processed = 0
    for data in dataset:
        if processed >= total_images:
            break
        model.set_input(data)
        model.test()
        image_paths = model.get_image_paths()
        if isinstance(image_paths, (str, Path)):
            image_paths = [image_paths]
        batch_count = min(len(image_paths), total_images - processed)
        visuals = model.get_current_visuals() if (opt.save_inputs or webpage) else None

        for batch_index in range(batch_count):
            stem = Path(image_paths[batch_index]).stem
            result_adapter = getattr(model, "result_image", None)
            generated = (
                result_adapter(batch_index)
                if result_adapter is not None
                else util.tensor2im(model.fake[batch_index : batch_index + 1])
            )
            util.save_image(
                generated,
                image_dir / f"{stem}{opt.output_suffix}.png",
                aspect_ratio=opt.aspect_ratio,
            )

            if visuals is not None and opt.save_inputs:
                for label, tensor in visuals.items():
                    if label == "fake":
                        continue
                    image = util.tensor2im(tensor[batch_index : batch_index + 1])
                    util.save_image(
                        image,
                        image_dir / f"{stem}_{label}.png",
                        aspect_ratio=opt.aspect_ratio,
                    )

            if webpage is not None and visuals is not None:
                from util.visualizer import save_images

                batch_visuals = {
                    label: tensor[batch_index : batch_index + 1]
                    for label, tensor in visuals.items()
                }
                save_images(
                    webpage,
                    batch_visuals,
                    [image_paths[batch_index]],
                    aspect_ratio=opt.aspect_ratio,
                    width=opt.display_winsize,
                )

        processed += batch_count
        print(f"Processed {processed}/{total_images} images")
        emit_event("progress", task="inference", current=processed, total=total_images)

    if webpage is not None:
        webpage.save()
    emit_event("run_finished", task="inference", processed=processed)
    print("Inference complete!")


def main() -> None:
    global GUI_EVENTS
    arguments = sys.argv[1:]
    if not _has_option(arguments, "--model"):
        sys.argv.extend(["--model", "test"])
    opt = TestOptions().parse()
    GUI_EVENTS = bool(opt.gui_events)
    if opt.model != "test":
        raise ValueError("Unified inference requires --model test.")
    opt.device = util.init_ddp(opt.gpu_ids)
    try:
        run_inference(opt)
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
