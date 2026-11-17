"""Fixed source-only previews; deliberately isolated from optimizer and RNG."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw
import torch

from data.base_dataset import get_transform
from models.cycle_gan_structure_model import CoarseIntensityDistributionLoss
from training.state import restore_rng, rng_state


def load_source(record, manifest):
    opt = SimpleNamespace(preprocess='resize_and_crop',
                          load_size=manifest['load_size'], crop_size=manifest['crop_size'],
                          no_flip=True)
    transform = get_transform(opt, {'crop_pos': tuple(record['crop_pos']), 'flip': False},
                              grayscale=True)
    tensors = []
    for domain in ('BF', 'AF'):
        with Image.open(record[domain]) as image:
            tensors.append(transform(image.convert('L')))
    return torch.cat(tensors, dim=0)


def tensor_image(tensor):
    array = ((tensor.detach().cpu().float().clamp(-1, 1) + 1) * 127.5).round().byte()
    if array.shape[0] == 1:
        return Image.fromarray(array[0].numpy()).convert('RGB')
    return Image.fromarray(array.permute(1, 2, 0).numpy())


def save_fixed_preview(model, opt, epoch):
    manifest = json.loads(Path(opt.fixed_preview_manifest).read_text(encoding='utf-8'))
    destination = Path(opt.checkpoints_dir) / opt.name / 'fixed_previews' / f'epoch_{epoch:03d}'
    destination.mkdir(parents=True, exist_ok=True)
    # Save every submodule flag, including any deliberately frozen/eval parts.
    network = model.netG_A
    modes = [(module, module.training) for module in network.modules()]
    saved_rng = rng_state()
    try:
        network.eval()
        criterion = CoarseIntensityDistributionLoss().to(model.device)
        size = manifest['crop_size']
        sheet = Image.new('RGB', (3 * size, (size + 22) * len(manifest['locations'])), 'white')
        draw = ImageDraw.Draw(sheet)
        rows = []
        with torch.inference_mode():
            for index, record in enumerate(manifest['locations']):
                source = load_source(record, manifest).unsqueeze(0).to(model.device)
                generated = network(source)
                if not torch.isfinite(generated).all():
                    raise RuntimeError(f'Non-finite fixed preview at {record["id"]}')
                result = tensor_image(generated[0])
                result.save(destination / f'{record["id"]}.png')
                top = index * (size + 22)
                draw.text((4, top + 3), f'{record["id"]}    BF / AF / virtual HE', fill='black')
                for column, image in enumerate((tensor_image(source[0, :1]),
                                                 tensor_image(source[0, 1:2]), result)):
                    sheet.paste(image, (column * size, top + 22))
                correlation = criterion.coarse_correlation(source, generated).item()
                rows.append({'roi': record['id'], 'coarse_correlation': correlation})
        sheet.save(destination / 'contact_sheet.jpg', quality=95)
        with (destination / 'diagnostics.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['roi', 'coarse_correlation'])
            writer.writeheader()
            writer.writerows(rows)
    finally:
        for module, training in modes:
            module.training = training
        restore_rng(saved_rng)
    return destination
