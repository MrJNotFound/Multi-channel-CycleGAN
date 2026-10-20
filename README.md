# SPIF: Structure-Preserving Image Fusion

Core code for virtual H&E staining from aligned brightfield (BF) and
autofluorescence (AF) images. This release contains the current **SPIF V2
single-stage competitive fusion** architecture and its structure-preserving
training objective.

The forward generator fuses BF/AF features using complementary spatial and
channel weights at half resolution. The reverse generator reconstructs BF/AF
with separate modality decoders. Training combines LSGAN, bidirectional cycle
consistency, an original-resolution Sobel gradient loss, and a coarse
Gaussian-filtered intensity correlation loss.

## Installation

Use Python 3.10 or newer in a dedicated environment:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

For GPU training, install a CUDA-enabled PyTorch build appropriate for your
system. Both entry points select CUDA automatically when available and
otherwise use CPU. A single GPU is supported. Use `--device cpu` or
`--device cuda:0` to select a device explicitly.

## Prepare the data

Place prepared, 8-bit image patches in these folders:

```text
datasets/
├── trainA_BF/    # Training brightfield images
├── trainA_AF/    # Training autofluorescence images
├── trainB/       # Training RGB H&E images
├── testA_BF/     # Test brightfield images
└── testA_AF/     # Test autofluorescence images
```

- Each BF image must have an AF image with the **same relative filename and
  dimensions**, including any subfolder path. BF/AF must already be spatially
  registered. Missing partners and mismatched dimensions produce an error.
- Prepare AF with the intensity inversion used by the SPIF preprocessing
  protocol, so its foreground/background brightness arrangement agrees with
  BF. These entry points use the provided intensities and normalize them to
  `[-1, 1]`.
- H&E training images are **unpaired** with BF/AF. They do not need matching
  names or counts. No H&E images are required for inference.
- Supported formats: PNG, JPEG, TIFF, and BMP. BF/AF are read as grayscale;
  H&E is read as RGB. Convert higher-bit-depth images to the intended 8-bit
  range before use.
- Training resizes to 286×286, then randomly crops to 256×256 and applies
  horizontal flips. BF/AF share the same spatial augmentation. Test images
  retain their original size and must be at least 16×16 pixels. Use prepared
  patches rather than whole-slide files that exceed device memory.

Only empty folder placeholders are included in Git; datasets and trained
weights are not distributed.

## Train

With the default folder layout, run:

```bash
python train.py
```

Output is saved under `checkpoints/spif/`:

```text
checkpoints/spif/
├── latest.pt       # Networks, optimizers, hyperparameters, replay pools, RNG state
├── config.json     # Training hyperparameters
├── losses.csv      # Sample-weighted epoch loss averages
└── samples/        # BF, AF, generated H&E, and real H&E previews per epoch
```

The defaults use 64 base channels, 9 residual blocks, instance normalization,
no dropout, a batch size of 1, Adam with learning rate `0.0002`, and
100 constant-rate epochs followed by 100 linear-decay epochs. Both cycle-loss
weights are 10. The SP weight is 1 and the relative intensity weight is 0.5.
Gradient and intensity guidance use separate frontloaded cosine schedules,
reaching zero at 10% and 2.5% of the total epochs, respectively.

Adjust runtime or training settings directly on the command line:

```bash
python train.py --name experiment1 --batch_size 8 --workers 4
python train.py --dataroot /path/to/data --name experiment2
```

Use `python train.py --help` to inspect SPIF hyperparameters. There are no
model selectors or external model configuration files. The two-channel
BF/AF input, RGB H&E output, SPIF architecture, LSGAN discriminators, and
training objective are fixed.

Resume the latest completed epoch with:

```bash
python train.py --resume
```

For a named experiment, also pass the same `--name` and data/output paths.
Resume restores the saved hyperparameters; conflicting overrides are
rejected. Checkpoints include optimizer, image replay, and random states.
Reproducing an uninterrupted run also requires the same dataset, worker
count, software, and device. A fresh run refuses to overwrite a nonempty
experiment folder.

`latest.pt` is written atomically after every completed epoch. Use
`--save_every 10` to additionally retain numbered checkpoints, or
`--stop_after_epoch 10` to stop a run without shortening its configured
learning-rate and SP-loss schedules.

## Test / inference

After training, populate `testA_BF/` and `testA_AF/`, then run:

```bash
python test.py
```

This loads `checkpoints/spif/latest.pt` and writes RGB PNG images to
`results/spif/`. For example, `testA_BF/sample.tif` produces
`results/spif/sample.tif_HE.png`. Relative subfolders and exact image
dimensions are preserved. Right/bottom reflection padding is removed after
inference. `inference.json` records the checkpoint and output files.

```bash
python test.py --name experiment1
python test.py --checkpoint checkpoints/experiment1/epoch_0010.pt --name epoch10
```

Architecture dimensions are read from the checkpoint, so inference does not
require repeating training flags. Repeating inference for the same experiment
replaces outputs with matching filenames. This release uses a self-contained
checkpoint format produced by `train.py`; legacy multi-model checkpoints
are not accepted.

## Repository layout

```text
train.py           # SPIF training and resume entry point
test.py            # BF/AF to H&E inference entry point
spif/
├── data.py        # Paired BF/AF loading and synchronized augmentation
├── networks.py    # SPIF generators and PatchGAN discriminators
├── losses.py      # Fine-gradient and coarse-intensity losses
├── model.py       # SPIF optimization and checkpoint state
└── runtime.py     # Device, reproducibility, and output I/O
tests/             # SPIF and end-to-end regression checks
requirements.txt
LICENSE
```

## Verification

The tests use temporary synthetic images and CPU-sized networks. They check
fusion gradients, structure losses, BF/AF alignment, checkpoint restoration,
and the train-to-inference workflow:

```bash
python -m unittest discover -s tests -v
```

## Acknowledgments and license

The cycle-consistency framework, residual blocks, and PatchGAN design derive
from the CycleGAN/pix2pix implementation by Jun-Yan Zhu and Taesung Park.
The original license notices are retained in [LICENSE](LICENSE).
