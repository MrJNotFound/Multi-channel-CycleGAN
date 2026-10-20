"""Generate virtual H&E images from datasets/testA_BF and testA_AF."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from spif.data import SPIFDataset
from spif.networks import forward_generator
from spif.runtime import ROOT, experiment_directory, read_checkpoint, save_image, select_device


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataroot", default=str(ROOT / "datasets"), help="Root containing testA_BF and testA_AF.")
    parser.add_argument("--name", default="spif", help="Experiment name.")
    parser.add_argument("--checkpoints_dir", default=str(ROOT / "checkpoints"), help="Checkpoint root.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path; defaults to checkpoints/<name>/latest.pt.")
    parser.add_argument("--results_dir", default=str(ROOT / "results"), help="Generated image output root.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--cpu_threads", type=int, default=4, help="PyTorch CPU threads.")
    return parser


def run(args):
    device = select_device(args.device, args.cpu_threads)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else (
        experiment_directory(args.checkpoints_dir, args.name) / "latest.pt")
    checkpoint = read_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    generator = forward_generator(config["ngf"], config["n_blocks"])
    generator.load_state_dict(checkpoint["networks"]["G_A"], strict=True)
    trained_epoch = checkpoint["epoch"]
    del checkpoint
    generator.to(device).eval()
    dataset = SPIFDataset(args.dataroot)
    output = experiment_directory(args.results_dir, args.name)
    # Retain the source extension as part of the name to avoid a.jpg/a.png collisions.
    print(f"SPIF | device={device} | pairs={len(dataset)} | epoch={trained_epoch}", flush=True)
    records = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            source = sample["A"].unsqueeze(0).to(device)
            height, width = source.shape[-2:]
            # Pad only the right and bottom; restore the exact input dimensions.
            padded = F.pad(source, (0, (-width) % 4, 0, (-height) % 4), mode="reflect")
            generated = generator(padded)[0, :, :height, :width]
            relative = Path(sample["name"])
            target = output / relative.parent / (relative.name + "_HE.png")
            save_image(generated, target)
            records.append({"input": sample["name"], "output": target.relative_to(output).as_posix(),
                            "width": width, "height": height})
            print(f"[{index + 1}/{len(dataset)}] {target}", flush=True)
    report = {"checkpoint": str(checkpoint_path.resolve()), "epoch": trained_epoch, "images": records}
    (output / "inference.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(records)} virtual H&E images to {output}")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
