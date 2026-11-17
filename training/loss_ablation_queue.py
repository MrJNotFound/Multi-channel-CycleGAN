"""Prepare and run the fixed-SCCF loss study through training/train.py.

Defaults: A1/A2 x seeds 101..120 (stop at 5); B0..B6 x 101..103 (200).
No existing checkpoints are used when starting a study. Reuse --resume only
for interrupted runs in the same immutable output directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.experiment_queue import (command_for, fingerprint, run_jobs,
                                       timestamp, write_json)

# condition: (gradient coefficient, intensity coefficient, gradient schedule,
#             intensity schedule). All use exactly the same model composition.
CONDITIONS = {
    'A1': (10., 0., 'cosine', 'cosine'),
    'A2': (10., 5., 'cosine', 'cosine'),
    'B0': (0., 0., 'cosine', 'cosine'),
    'B1': (10., 0., 'cosine', 'cosine'),
    'B2': (0., 5., 'cosine', 'cosine'),
    'B3': (10., 5., 'cosine', 'cosine'),
    'B4': (10., 5., 'constant', 'cosine'),
    'B5': (10., 5., 'cosine', 'constant'),
    'B6': (10., 5., 'constant', 'constant'),
}
FOLDERS = {'BF': 'trainA_BF', 'AF': 'trainA_AF', 'HE': 'trainB'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(path, fields, rows):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_jobs(dataroot, output, workers=4, gpu_ids='0', selected=None):
    dataroot, output = Path(dataroot).resolve(), Path(output).resolve()
    common = dict(
        model='cycle_gan_dual_fusion_structure', spif_version='v2',
        dataset_mode='dual_channel', dataroot=str(dataroot),
        bf_dir=str(dataroot / FOLDERS['BF']), af_dir=str(dataroot / FOLDERS['AF']),
        target_dir=str(dataroot / FOLDERS['HE']), direction='AtoB',
        input_nc=2, output_nc=3, netG='resnet_9blocks', netD='basic',
        ngf=64, ndf=64, n_layers_D=3, norm='instance', no_dropout=True,
        init_type='normal', init_gain=0.02, gan_mode='lsgan',
        lambda_A=10., lambda_B=10., lambda_identity=0., pool_size=50,
        lambda_content=1., lambda_sp=10., lambda_intensity=0.5,
        sp_grad_decay_fraction=0.1, sp_intensity_decay_fraction=0.025,
        grad_channel_weights=[0.5, 0.5], intensity_pool_size=8,
        batch_size=8, preprocess='resize_and_crop', load_size=286, crop_size=256,
        serial_batches=False, no_flip=False, n_epochs=100, n_epochs_decay=100,
        lr=0.0002, lr_policy='linear', beta1=0.5, beta2=0.999, epoch_count=1,
        num_threads=workers, gpu_ids=gpu_ids, deterministic=False, cyclegan_runtime=True,
        complete_epoch_state=False, weights_only=True, record_initial_hashes=True,
        save_epoch_freq=5, save_latest_freq=5000,
        display_freq=0, save_visuals_epoch_freq=0, no_html=True, print_freq=200,
        gui_events=True, checkpoints_dir=str(output / 'checkpoints'),
        fixed_preview_manifest=str(output / 'diagnostic_manifest.json'),
        pause_file=str(output / 'PAUSE'),
    )
    jobs = []
    for stage, seeds, conditions, endpoint, previews in (
        ('A', range(101, 121), ('A1', 'A2'), 5, [1, 2, 3, 4, 5]),
        ('B', range(101, 104), tuple(f'B{i}' for i in range(7)), 200, [1, 5, 10, 20, 50, 100, 200]),
    ):
        for seed in seeds:
            for condition in conditions:
                grad, intensity, grad_mode, intensity_mode = CONDITIONS[condition]
                name = f'{condition}_seed{seed}'
                values = dict(common, seed=seed, name=name, stop_after_epoch=endpoint,
                              sp_grad_weight=grad, sp_intensity_weight=intensity,
                              sp_grad_schedule=grad_mode, sp_intensity_schedule=intensity_mode,
                              fixed_preview_epochs=previews)
                jobs.append(dict(id=name, label=condition, stage=stage, seed=seed, values=values))
    if selected:
        allowed = set(CONDITIONS) | {job['id'] for job in jobs}
        if set(selected) - allowed:
            raise ValueError(f'Unknown selections: {set(selected) - allowed}')
        jobs = [j for j in jobs if j['id'] in selected or j['label'] in selected]
    return jobs


def audit_dataset(dataroot):
    from PIL import Image
    from data.image_folder import make_dataset
    files, records = {}, []
    for domain, folder in FOLDERS.items():
        directory = dataroot / folder
        paths = sorted(Path(p) for p in make_dataset(str(directory)))
        if not paths:
            raise ValueError(f'Empty training domain: {directory}')
        files[domain] = {p.name: p for p in paths}
        if len(files[domain]) != len(paths):
            raise ValueError(f'Duplicate basenames in {directory}')
        for path in paths:
            with Image.open(path) as image:
                image.load()
                if image.size != (512, 512):
                    raise ValueError(f'Expected original 512 x 512 patch: {path}: {image.size}')
                records.append(dict(domain=domain, file=str(path.relative_to(dataroot)),
                                    width=image.width, height=image.height, mode=image.mode,
                                    sha256=digest(path)))
        print(f'Audit {domain}: {len(paths)} valid 512 x 512 images', flush=True)
    if set(files['BF']) != set(files['AF']):
        raise ValueError('BF/AF filename sets differ; refuse to silently discard unmatched images.')
    return files, records


def select_diagnostics(files, count=16):
    """Deterministic coverage/texture sampling; never a training mask or endpoint."""
    import numpy as np
    from PIL import Image
    from skimage.filters import threshold_otsu
    candidates = []
    thumbnails = []
    for name, path in sorted(files['BF'].items()):
        with Image.open(path) as image:
            # Match the fixed training-scale center crop before thumbnail statistics.
            crop = image.convert('L').resize((286, 286), Image.Resampling.BICUBIC).crop((15, 15, 271, 271))
            array = np.asarray(crop.resize((64, 64)), dtype=np.uint8)
        thumbnails.append((name, array))
    threshold = float(threshold_otsu(np.concatenate([x.ravel() for _, x in thumbnails])))
    for name, array in thumbnails:
        coverage = float((array < threshold).mean())
        texture = float(np.abs(np.diff(array.astype(float), axis=0)).mean()
                        + np.abs(np.diff(array.astype(float), axis=1)).mean())
        candidates.append(dict(name=name, coverage=coverage, texture=texture))
    # Eight partial-coverage locations plus eight positions spanning texture.
    boundary = sorted((x for x in candidates if 0.10 <= x['coverage'] <= 0.90),
                      key=lambda x: (x['coverage'], x['name']))
    order = []
    for pool, amount in ((boundary, min(8, count)),
                         (sorted(candidates, key=lambda x: (x['texture'], x['name'])), count)):
        if pool:
            for index in np.linspace(0, len(pool)-1, min(amount, len(pool)), dtype=int):
                if pool[index]['name'] not in [x['name'] for x in order]:
                    order.append(pool[index])
                if len(order) == min(count, len(candidates)):
                    break
        if len(order) == min(count, len(candidates)):
            break
    for record in sorted(candidates, key=lambda x: x['name']):
        if len(order) >= min(count, len(candidates)):
            break
        if record['name'] not in [x['name'] for x in order]:
            order.append(record)
    return {
        'role': 'training_diagnostics_only_not_independent_validation',
        'load_size': 286, 'crop_size': 256, 'flip': False,
        'selection': 'fixed before training; BF coverage/texture strata; visual review required',
        'selection_threshold_BF': threshold,
        'locations': [dict(id=f'ROI{i+1:02d}', crop_pos=[15, 15],
                           BF=str(files['BF'][r['name']].resolve()), AF=str(files['AF'][r['name']].resolve()),
                           BF_sha256=digest(files['BF'][r['name']]), AF_sha256=digest(files['AF'][r['name']]),
                           coverage_proxy=r['coverage'], texture_proxy=r['texture'])
                      for i, r in enumerate(order)],
    }


def source_contact_sheet(manifest, output):
    from PIL import Image, ImageDraw
    from training.diagnostics import load_source, tensor_image
    sheet = Image.new('RGB', (4 * 256, 8 * (256 + 22)), 'white')
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(manifest['locations']):
        source = load_source(record, manifest)
        x, y = (index % 2) * 512, (index // 2) * 278
        draw.text((x+4, y+4), record['id'] + '  BF / AF', fill='black')
        sheet.paste(tensor_image(source[:1]), (x, y+22))
        sheet.paste(tensor_image(source[1:]), (x+256, y+22))
    sheet.save(output / 'diagnostic_sources.jpg', quality=95)


def code_manifest():
    files = []
    for directory in ('training', 'models', 'data', 'options', 'util', 'gui'):
        files.extend(sorted((ROOT / directory).rglob('*.py')))
    return {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in files}


def export_review(job, output):
    mapping = json.loads((output / 'review_key.json').read_text(encoding='utf-8'))
    source = Path(job['values']['checkpoints_dir']) / job['values']['name'] / 'fixed_previews'
    for epoch in job['values']['fixed_preview_epochs']:
        image = source / f'epoch_{epoch:03d}' / 'contact_sheet.jpg'
        if image.is_file():
            destination = output / 'blinded_review' / mapping[job['id']]
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(image, destination / f'epoch_{epoch:03d}.jpg')


def write_materials(jobs, output):
    from models.cycle_gan_structure_model import frontloaded_cosine_decay
    rows = []
    for job in jobs:
        v = job['values']
        for epoch in range(1, 201):
            grad = 1 if v['sp_grad_schedule'] == 'constant' else frontloaded_cosine_decay(epoch, 200, .1)
            intensity = 1 if v['sp_intensity_schedule'] == 'constant' else frontloaded_cosine_decay(epoch, 200, .025)
            rows.append(dict(run_id=job['id'], epoch=epoch,
                             gradient_weight=v['sp_grad_weight']*grad,
                             intensity_weight=v['sp_intensity_weight']*intensity,
                             learning_rate=.0002*(1-max(0, epoch-100)/101),
                             executed=epoch <= v['stop_after_epoch']))
    write_csv(output / 'weight_schedule.csv', list(rows[0]), rows)
    order = list(range(1, len(jobs)+1))
    random.Random(42671).shuffle(order)
    mapping = {job['id']: f'R{number:03d}' for job, number in zip(jobs, order)}
    write_json(output / 'review_key.json', mapping)
    review = output / 'blinded_review'
    review.mkdir(exist_ok=True)
    fields = ['blind_id', 'epoch5_inversion_yes_no_uncertain', 'first_epoch',
              'ever_in_first5', 'persistent', 'affected_roi_ids', 'other_failure', 'notes']
    write_csv(review / 'scores.csv', fields,
              sorted((dict(blind_id=mapping[j['id']]) for j in jobs if j['stage'] == 'A'),
                     key=lambda r: r['blind_id']))
    (review / 'README.txt').write_text(
        'Judge at the run level, not at the ROI/epoch level. Primary endpoint: epoch 5\n'
        'clear foreground/background inversion: yes / no / uncertain. Disorganized\n'
        'outputs and ROIs without a clear background relationship are not automatically\n'
        'inversions. Record other failures separately. Review all the same ROIs.\n'
        'Correlation is auxiliary only. Do not open ../review_key.json during grading.\n'
        'These images are training diagnostics, NOT independent validation/test data.\n', encoding='utf-8')
    fields = ['seed', 'A1_blind_id', 'A2_blind_id', 'A1_epoch5', 'A2_epoch5']
    ids = {j['id'] for j in jobs}
    write_csv(output / 'paired_inversion_results.csv', fields,
              [dict(seed=seed, A1_blind_id=mapping.get(f'A1_seed{seed}', ''),
                    A2_blind_id=mapping.get(f'A2_seed{seed}', ''))
               for seed in range(101, 121) if f'A1_seed{seed}' in ids or f'A2_seed{seed}' in ids])
    fields = ['run_id', 'seed', 'epoch', 'SSIM', 'PSNR', 'LPIPS', 'FID', 'failure_notes']
    write_csv(output / 'final_quality_results.csv', fields,
              [dict(run_id=j['id'], seed=j['seed'], epoch=200) for j in jobs if j['stage'] == 'B'])


def prepare(args, jobs):
    output, dataroot = args.output.resolve(), args.dataroot.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / 'study_plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
        if plan['jobs'] != jobs:
            raise ValueError('Study options/selection changed; choose a new output directory.')
        if plan['code_sha256'] != code_manifest():
            raise ValueError('Training code changed since preparation; do not mix versions in this study.')
        if digest(output / 'diagnostic_manifest.json') != plan['diagnostic_manifest_sha256']:
            raise ValueError('Fixed diagnostic locations changed.')
        with (output / 'dataset_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
            records = list(csv.DictReader(stream))
        if fingerprint(records) != plan['dataset']['fingerprint']:
            raise ValueError('Dataset manifest changed.')
        expected = {str(Path(row['file'])) for row in records}
        from data.image_folder import make_dataset
        actual = {str(Path(p).relative_to(dataroot)) for folder in FOLDERS.values()
                  for p in make_dataset(str(dataroot / folder))}
        if actual != expected:
            raise ValueError('Training image set changed since preparation.')
        for row in records:
            if digest(dataroot / row['file']) != row['sha256']:
                raise ValueError(f'Training image changed: {row["file"]}')
        return plan
    if (output / 'queue_state.json').exists():
        raise ValueError('Existing queue without study_plan.json; refusing to overwrite it.')
    files, records = audit_dataset(dataroot)
    write_csv(output / 'dataset_manifest.csv', list(records[0]), records)
    # Normalize types to exactly what CSV reload returns.
    records = [{key: str(value) for key, value in r.items()} for r in records]
    if args.diagnostics:
        manifest = json.loads(args.diagnostics.read_text(encoding='utf-8'))
    else:
        manifest = select_diagnostics(files)
    if manifest['load_size'] != 286 or manifest['crop_size'] != 256 or not manifest['locations']:
        raise ValueError('Diagnostics must use fixed resize 286 and crop 256.')
    write_json(output / 'diagnostic_manifest.json', manifest)
    source_contact_sheet(manifest, output)
    write_materials(jobs, output)
    plan = dict(created=timestamp(), jobs=jobs,
                dataset=dict(root=str(dataroot), counts={d: len(p) for d, p in files.items()},
                             fingerprint=fingerprint(records)),
                total_runs=len(jobs), total_epochs=sum(j['values']['stop_after_epoch'] for j in jobs),
                code_sha256=code_manifest(), plan_sha256=digest(args.plan),
                diagnostic_manifest_sha256=digest(output / 'diagnostic_manifest.json'),
                precision='float32; cuDNN TF32/benchmark on; matmul TF32 off; strict determinism off; single GPU',
                checkpoint_policy='G/D weights only; every 5 epochs and latest every 5000 samples; no training_state',
                execution_override='User requested the original CycleGAN runtime and weight-only checkpoint policy.',
                evaluation='independent registered HE evaluation data not supplied; result tables unscored')
    shutil.copyfile(args.plan, output / 'source_experiment_plan.md')
    snapshot = output / 'code_snapshot'
    for relative in plan['code_sha256']:
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    import torch
    write_json(output / 'environment.json', dict(
        recorded=timestamp(), python=sys.version, executable=sys.executable,
        platform=platform.platform(), torch=str(torch.__version__), cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        workers=args.workers, batch_size=8, precision=plan['precision']))
    for filename, command in (
        ('environment.freeze.txt', [sys.executable, '-m', 'pip', 'freeze']),
        ('git_commit.txt', ['git', 'rev-parse', 'HEAD']),
        ('git_status.txt', ['git', 'status', '--short']),
        ('git_diff.patch', ['git', 'diff', '--binary']),
        ('nvidia_smi.txt', ['nvidia-smi']),
    ):
        result = subprocess.run(command, cwd=ROOT, capture_output=True)
        (output / filename).write_bytes(result.stdout + result.stderr)
    write_json(plan_path, plan)
    return plan


def preflight(args, jobs):
    """Real CUDA forward/backward/weight-save/preview checks, outside study runs."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the batch-8 preflight.')
    directory = args.preflight_dir.resolve() if args.preflight_dir else args.output / 'preflight'
    if directory.exists():
        raise ValueError(f'Preflight output already exists: {directory}; keep it and use a new output.')
    directory.mkdir(parents=True)
    manifest = json.loads((args.output / 'diagnostic_manifest.json').read_text(encoding='utf-8'))
    manifest['locations'] = manifest['locations'][:2]
    write_json(directory / 'diagnostic_manifest.json', manifest)
    report = dict(status='running', excluded_from_experiment_results=True, checks=[])
    write_json(directory / 'report.json', report)

    def invoke(condition, name, stop, resume=False):
        job = next(j for j in jobs if j['label'] == condition)
        values = dict(job['values'], name=name, checkpoints_dir=str(directory / 'checkpoints'),
                      max_dataset_size=8, stop_after_epoch=stop, fixed_preview_epochs=[1, 2],
                      fixed_preview_manifest=str(directory / 'diagnostic_manifest.json'),
                      pause_file=str(directory / 'PAUSE'), num_threads=args.workers)
        if resume:
            values.update(continue_train=True, epoch='latest')
        command = command_for(values, sys.executable)
        log_path = directory / f'{name}_{stop}.log'
        with log_path.open('w', encoding='utf-8') as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                    env={**os.environ, 'PYTHONIOENCODING': 'utf-8',
                                         'PYTHONHASHSEED': str(values['seed'])})
        if result.returncode:
            report.update(status='failed', failed_log=str(log_path), returncode=result.returncode)
            write_json(directory / 'report.json', report)
            raise RuntimeError(f'Preflight failed for {name}; inspect {log_path}')
        text = log_path.read_text(encoding='utf-8')
        events = [json.loads(line[len('@@MLCGUI '):]) for line in text.splitlines()
                  if line.startswith('@@MLCGUI ')]
        if not any(e['type'] == 'run_finished' for e in events):
            raise RuntimeError(f'Preflight did not finish: {name}')
        checkpoint = directory / 'checkpoints' / name
        from training.experiment_queue import verify_completion
        verify_completion(values, checkpoint)
        if list(checkpoint.glob('*training_state.pth')):
            raise RuntimeError('Weight-only preflight unexpectedly saved training state.')
        if not (checkpoint / 'fixed_previews' / f'epoch_{stop:03d}' / 'ROI01.png').exists():
            raise RuntimeError('Preflight preview is missing.')
        report['checks'].append(dict(condition=condition, resume=resume, endpoint=stop,
                                     metrics=[e['values'] for e in events if e['type'] == 'metrics'],
                                     timing=[e for e in events if e['type'] == 'epoch_end']))
        write_json(directory / 'report.json', report)
        print(f'Preflight passed: {name}, endpoint={stop}, resume={resume}', flush=True)
        return checkpoint

    for condition in (f'B{i}' for i in range(7)):
        invoke(condition, condition, 1)
    invoke('B6', 'B6', 2, resume=True)
    hashes = [json.loads((directory / 'checkpoints' / f'B{i}' / 'initial_network_hashes.json')
                         .read_text(encoding='utf-8')) for i in range(7)]
    if any(value != hashes[0] for value in hashes[1:]):
        raise RuntimeError('Same seed has different initial networks between conditions.')
    report.update(status='passed', checkpoint_policy='weights only',
                  weight_only_continuation_checked=True, same_initial_networks_all_conditions=True)
    write_json(directory / 'report.json', report)
    print('Preflight complete: 7 conditions, weight-only saving/continuation and fixed previews passed.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataroot', type=Path, default=ROOT / 'datasets/train_mouse_kidney_dual_BF_AF_HE_512')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/mouse_kidney_loss_ablation_cyclegan')
    parser.add_argument('--plan', type=Path, default=ROOT / 'LOSS_ABLATION_EXPERIMENT_PLAN.md')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--gpu-ids', default='0')
    parser.add_argument('--select', nargs='+', help='condition IDs or exact run IDs; omitted means all 61')
    parser.add_argument('--diagnostics', type=Path, help='optional prepared fixed-location manifest')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--export-review', action='store_true')
    parser.add_argument('--preflight', action='store_true',
                        help='after preparation, verify 7 conditions and weight-only continuation on 8 real images')
    parser.add_argument('--preflight-dir', type=Path, help='separate output for a new preflight attempt')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.status:
        path = args.output / 'queue_state.json'
        print(path.read_text(encoding='utf-8') if path.exists() else 'Queue not started.')
        return
    if args.export_review:
        plan = json.loads((args.output / 'study_plan.json').read_text(encoding='utf-8'))
        for job in plan['jobs']:
            export_review(job, args.output)
        return
    if args.gpu_ids != '0':
        raise ValueError('This study launcher is configured for one CUDA GPU (--gpu-ids 0).')
    jobs = build_jobs(args.dataroot, args.output, args.workers, args.gpu_ids, args.select)
    if args.preflight:
        preflight(args, jobs)
        return
    for index, job in enumerate(jobs, 1):
        command_for(job['values'], sys.executable)
        print(f'{index:02d} {job["id"]} -> epoch {job["values"]["stop_after_epoch"]}', flush=True)
    plan = prepare(args, jobs)
    print(f'Prepared {len(jobs)} runs, {plan["total_epochs"]} epochs. Output: {args.output}', flush=True)
    if args.run:
        run_jobs(jobs, args.output, plan['dataset'], sys.executable, resume=args.resume,
                 on_job_complete=lambda job: export_review(job, args.output))


if __name__ == '__main__':
    main()
