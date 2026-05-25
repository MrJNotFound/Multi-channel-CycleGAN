# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Training

```bash
# UTOM model (dual-channel BF+AF to H&E, the primary use case)
python train.py \
    --dataroot ./datasets/mouse_kidney_dual_BF_AF_HE \
    --name experiment_name \
    --model utom \
    --dataset_mode dual_channel \
    --input_nc 2 \
    --output_nc 3 \
    --lambda_identity 0 \
    --batch_size 8 \
    --n_epochs 100 \
    --load_size 286 \
    --crop_size 256

# Standard CycleGAN (single channel or RGB)
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```

Training checkpoints and logs are saved to `./checkpoints/<name>/`. Loss curves go to `loss_log.txt`, options to `train_opt.txt`.

### Testing / Inference

```bash
python test.py \
    --dataroot <test_data_path> \
    --name experiment_name \
    --model utom \
    --input_nc 2 \
    --output_nc 3 \
    --no_dropout \
    --preprocess none \
    --epoch latest
```

Results are saved to `./results/<name>/test_latest/`. Use `--num_test` to limit images, `--results_dir` to redirect output.

## Architecture

This is a **plugin-based image-to-image translation framework** built on the pytorch-CycleGAN-and-pix2pix scaffold. Models, datasets, and options are discoverable by name — e.g., `--model utom` dynamically imports `models/utom_model.py` and instantiates `UTOMModel`.

### Model Hierarchy

```
BaseModel (ABC)
├── CycleGANModel — standard CycleGAN with cycle-consistency + identity loss
├── UTOMModel(CycleGAN) — CycleGAN + gradient-domain structure-preserving content loss
├── Pix2PixModel
└── TestModel — inference-only, single-direction, uses `--model_suffix` to pick which generator to load
```

Key design pattern: each model class defines:
- `self.loss_names` — loss keys for logging
- `self.model_names` — network names for save/load (e.g., `["G_A", "G_B", "D_A", "D_B"]`)
- `self.visual_names` — tensors to display/save
- `self.optimizers` — PyTorch optimizers (BaseModel.setup creates schedulers from them)

Training loop (`train.py`) calls `model.set_input(data)` → `model.optimize_parameters()` → `model.compute_visuals()` each iteration. Test loop calls `model.set_input(data)` → `model.test()` (which wraps `forward()` + `compute_visuals()` in `no_grad`).

### Dual-Branch Generator (`networks.py`)

When `input_nc=2`, both `CycleGANModel` and `UTOMModel` automatically switch from `resnet_9blocks` to `dual_resnet_9blocks`, which instantiates `DualBranchResnetGenerator`:

```
BF (ch 0) → Head_BF (7×7 conv, ngf) ─┐
                                      ├→ CrossChannelFusion → Shared Backbone → Output
AF (ch 1) → Head_AF (7×7 conv, ngf) ─┘
```

`CrossChannelFusion`: each branch's features are modulated by channel attention derived from the other branch's context (Squeeze-and-Excitation), then concatenated, projected, and added residually.

The shared backbone is a standard ResNet generator: 2× downsampling → N ResNet blocks → 2× upsampling → Tanh output.

### UTOM Content Loss (`utom_model.py:210`)

UTOM adds a structure-preserving loss in the gradient domain:
1. Compute Sobel gradient magnitude from channel-mean inputs and outputs
2. Match gradients with L1 loss
3. Weight decays as `15 * exp(-counter / data_size)` — structure guidance dominates early training, then cedes to CycleGAN objectives

`opt.counter` is the running total of training samples seen (incremented by `batch_size` each iteration in `train.py:70`). `opt.data_size` is the dataset size.

### Dataset Plugin System

Same pattern as models: `--dataset_mode dual_channel` imports `data/dual_channel_dataset.py` → `DualChannelDataset`.

`DualChannelDataset` loads from `trainA_BF/` and `trainA_AF/` directories, matches files by name, applies identical spatial transforms to both channels, and stacks them as `(2, H, W)`. Domain B (H&E) uses a random unpaired index per sample for CycleGAN training.

Required directory structure:
```
dataroot/
├── trainA_BF/    # brightfield (grayscale)
├── trainA_AF/    # autofluorescence (grayscale, same filenames as BF)
├── trainB/       # H&E RGB
├── testA_BF/
└── testA_AF/
```

### DDP Support

`util/util.py` provides `init_ddp()` and `cleanup_ddp()`. When `LOCAL_RANK` is set in the environment, training uses `DistributedSampler` and NCCL backend. Use `--norm syncbatch` for multi-GPU training (InstanceNorm is incompatible with DDP).

### Visualization Pipeline

Training uses `Visdom` for live loss plotting and saves an HTML results page. Images are written to `./checkpoints/<name>/web/`. `util/visualizer.py` handles display; `util/html.py` generates the HTML index.

## Tool Pipeline

The `tool/` directory provides preprocessing/postprocessing for whole-slide images:

1. **`patch_generate_multi_mode.py`** — Extract patches from large WSI images with background filtering (color/mask/random modes)
2. **`slide_windows.py`** — Sliding-window patch extraction with configurable overlap
3. **`patch_stitching.py`** — Reassemble patches into a full WSI using Hann/Gaussian weighted blending; parses patch coordinates from filenames like `patch_y000000_x000000.png`
4. **`paired_img_regist.py`** — Register unpaired images (rigid/affine)
5. **`color_normalization.py`** — H&E stain normalization
6. **`threshold_seg.py`** — Threshold-based segmentation
7. **`auto_freq_filter.py`** — Frequency-domain filtering

All tool scripts have hardcoded config variables near the top — edit the paths/modes directly in the script.

## Critical Behaviors

- **`input_nc=2` triggers dual-channel mode everywhere**: generator type auto-switches, visualizer splits channels for display, dataset expects two subdirectories
- **Identity loss is auto-disabled** when `input_nc != output_nc` (prints a warning)
- **Checkpoints use `weights_only=True`** since PyTorch 2.6 — don't change this
- **`train.py` has hardcoded `sys.argv`** that overrides command-line args; remove or edit the list for actual use
- **Test scripts hardcode `batch_size=1`, `serial_batches=True`, `no_flip=True`** — these are required for correct output
