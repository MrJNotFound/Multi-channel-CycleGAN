"""Select and run experiment presets sequentially, with durable completion state.

This orchestrator uses training/train.py for every actual training job. It does
not implement another training loop. Configuration files contain experiments
with id, label, input modalities and train_options, plus dataset root/folders.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def select_experiments(plan, selected=None):
    experiments = {e['id']: e for e in plan['experiments']}
    if len(experiments) != len(plan['experiments']):
        raise ValueError('Experiment IDs must be unique.')
    order = plan.get('execution_order', list(experiments))
    if len(order) != len(set(order)) or set(order) != set(experiments):
        raise ValueError('execution_order must contain every ID exactly once.')
    wanted = set(selected) if selected else set(order)
    if selected and len(selected) != len(wanted):
        raise ValueError('Repeated experiment selections are not allowed.')
    if wanted - set(order):
        raise ValueError(f'Unknown selections: {sorted(wanted-set(order))}')
    result = [experiments[key] for key in order if key in wanted]
    # The full SPIF run is always last, including user-selected subsets.
    return sorted(result, key=lambda e: e.get('label', '').lower() == 'spif')


def training_values(plan, experiment, output, seed, gpu_ids, workers):
    values = {**plan.get('common_options', {}), **experiment['train_options']}
    source = Path(plan['dataset']['root']).resolve()
    folders = plan['dataset']['folders']
    name = f"{experiment['id']}_{experiment['label'].replace(' ', '_')}_seed{seed}"
    values.update(dataroot=str(source), name=name,
                  checkpoints_dir=str(Path(output).resolve() / 'checkpoints'),
                  seed=seed, gpu_ids=gpu_ids, num_threads=workers,
                  target_dir=str(source / folders['HE']),
                  gui_events=True, save_visuals_epoch_freq=5)
    if experiment['input'] == ['BF', 'AF']:
        values.update(bf_dir=str(source / folders['BF']), af_dir=str(source / folders['AF']))
    else:
        values['source_dir'] = str(source / folders[experiment['input'][0]])
    return values


def command_for(values, python):
    from gui.launcher import build_command, build_option_parser
    parser, _ = build_option_parser('train', values['model'], values['dataset_mode'], values)
    known = {action.dest for action in parser._actions}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f'Unknown model options: {sorted(unknown)}')
    command = build_command('train', values['model'], values['dataset_mode'], values,
                            python_executable=python)
    return [command[0], '-u', *command[1:]]


def verify_dataset(plan, config_path):
    """Check actual files against the threshold-calibration manifest, if present."""
    root = Path(plan['dataset']['root'])
    files = {}
    for domain, folder in plan['dataset']['folders'].items():
        directory = root / folder
        files[domain] = {p.name: p for p in directory.glob('*.jpg')}
        expected = plan['dataset']['counts'][domain]
        if len(files[domain]) != expected:
            raise ValueError(f'{domain}: expected {expected} images, found {len(files[domain])}')
    if set(files['BF']) != set(files['AF']):
        raise ValueError('BF/AF filename sets do not match.')
    manifest = Path(config_path).parent / 'per_image_statistics.csv'
    hashes = {}
    if manifest.is_file():
        with manifest.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                if row['domain'] not in files:
                    continue
                source = files[row['domain']].get(row['file'])
                if source is None:
                    raise ValueError(f'Missing calibrated image: {row["file"]}')
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                if digest != row['sha256']:
                    raise ValueError(f'Image changed since calibration: {source}')
                hashes[f'{row["domain"]}/{row["file"]}'] = digest
        if len(hashes) != sum(map(len, files.values())):
            raise ValueError('Calibration manifest does not cover every training image.')
    return {'counts': {key: len(value) for key, value in files.items()},
            'sha256': fingerprint(hashes) if hashes else None}


@contextlib.contextmanager
def queue_lock(output):
    path = Path(output) / 'queue.lock'
    with path.open('a+b') as handle:
        if path.stat().st_size == 0:
            handle.write(b' ')
            handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('Another process owns this experiment queue.') from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def initial_state(jobs, dataset):
    return {'created': timestamp(), 'updated': timestamp(), 'status': 'pending',
            'fingerprint': fingerprint(jobs), 'dataset': dataset,
            'jobs': {job['id']: {'status': 'pending', 'label': job['label'],
                               'attempts': 0} for job in jobs}}


def run_jobs(jobs, output, dataset, python, resume=False):
    """Stop on failure; completed jobs never run again in this output directory."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with queue_lock(output):
        state_path = output / 'queue_state.json'
        state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else initial_state(jobs, dataset)
        if state['fingerprint'] != fingerprint(jobs) or state['dataset'] != dataset:
            raise ValueError('Queue/data changed. Use a new output directory for a different experiment.')
        state['pid'] = os.getpid()
        state['status'] = 'running'
        write_json(output / 'queue_config.json', jobs)
        write_json(state_path, state)
        for job in jobs:
            item = state['jobs'][job['id']]
            if item['status'] == 'completed':
                print(f"Skip completed: {job['id']} {job['label']}", flush=True)
                continue
            values = dict(job['values'])
            checkpoint_dir = Path(values['checkpoints_dir']) / values['name']
            if item['attempts']:
                if not resume:
                    raise RuntimeError(f"{job['id']} was interrupted/failed. Inspect its log and use --resume.")
                if not (checkpoint_dir / 'latest_training_state.pth').is_file():
                    raise RuntimeError(f"No checkpoint for {job['id']}; refusing to silently restart it.")
                values['continue_train'] = True
                values['epoch'] = 'latest'
            elif checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
                raise RuntimeError(f'Refusing to overwrite existing experiment: {checkpoint_dir}')
            command = command_for(values, python)
            log_path = output / 'logs' / f"{job['id']}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            item.update(status='running', started=timestamp(), attempts=item['attempts']+1,
                        command=command, log=str(log_path), checkpoint_dir=str(checkpoint_dir))
            state['current'] = job['id']
            write_json(state_path, state)
            print(f"Starting {job['id']}: {job['label']}", flush=True)
            env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf-8'}
            process = None
            finished = False
            last_update = 0.0
            try:
                with log_path.open('a', encoding='utf-8') as log:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                               stderr=subprocess.STDOUT, text=True,
                                               encoding='utf-8', errors='replace', env=env,
                                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    item['pid'] = process.pid
                    write_json(state_path, state)
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        if line.startswith('@@MLCGUI '):
                            event = json.loads(line[len('@@MLCGUI '):])
                            kind = event.get('type')
                            if kind == 'run_finished':
                                finished = True
                            if kind in {'progress', 'epoch_end', 'metrics', 'checkpoint'}:
                                item['progress'] = {**item.get('progress', {}), **event}
                            now = time.monotonic()
                            if now-last_update >= 5 or kind in {'epoch_end','run_finished'}:
                                state['updated'] = timestamp()
                                write_json(state_path, state)
                                last_update = now
                            if kind == 'epoch_end':
                                print(f"{job['id']} epoch {event['epoch']}/{event['final_epoch']} ({event['duration_seconds']:.0f}s)", flush=True)
                    code = process.wait()
                    if code != 0 or not finished:
                        raise RuntimeError(f"{job['id']} exited with code {code}; see {log_path}")
                final_epoch = values['n_epochs'] + values['n_epochs_decay']
                if not (checkpoint_dir / f'{final_epoch}_training_state.pth').is_file():
                    raise RuntimeError(f'{job["id"]} did not save its final training state.')
                item.update(status='completed', finished=timestamp(), returncode=code)
                state['updated'] = timestamp()
                write_json(state_path, state)
            except BaseException as exc:
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                item.update(status='failed', error=str(exc), finished=timestamp())
                state.update(status='failed', updated=timestamp())
                write_json(state_path, state)
                raise
        state.update(status='completed', current=None, updated=timestamp())
        write_json(state_path, state)
        print('All selected experiments completed.', flush=True)


def smoke_one(values):
    """One optimization step on real images; never writes model checkpoints."""
    import random
    import numpy as np
    import torch
    from data import create_dataset
    from models import create_model
    from models.base_model import BaseModel
    from options.train_options import TrainOptions
    from util.util import init_ddp

    values = {**values, 'max_dataset_size': values['batch_size'], 'num_threads': 0,
              'no_html': True}
    command = command_for(values, sys.executable)
    sys.argv = command[2:]  # strip python and -u, retaining train.py as argv[0]
    opt = TrainOptions().parse()
    opt.device = init_ddp(opt.gpu_ids)
    if opt.device.type != 'cuda':
        raise RuntimeError('A CUDA GPU is required for this full-size preflight.')
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    dataset = create_dataset(opt)
    dataset.set_epoch(1)
    batch = next(iter(dataset))
    opt.data_size, opt.counter = 1000, opt.batch_size
    model = create_model(opt)
    if type(model).data_dependent_initialize is not BaseModel.data_dependent_initialize:
        model.data_dependent_initialize(batch)
    model.setup(opt)
    if hasattr(model, 'set_current_epoch'):
        model.set_current_epoch(1)
    model.set_input(batch)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    model.optimize_parameters()
    torch.cuda.synchronize()
    losses = model.get_current_losses()
    if not all(np.isfinite(value) for value in losses.values()):
        raise RuntimeError(f'Non-finite losses: {losses}')
    if tuple(model.fake_B.shape) != (opt.batch_size, 3, opt.crop_size, opt.crop_size):
        raise RuntimeError(f'Unexpected generated shape: {tuple(model.fake_B.shape)}')
    result = {'status':'passed', 'losses':losses, 'step_seconds':time.monotonic()-start,
              'peak_gpu_gib':torch.cuda.max_memory_allocated()/1024**3}
    write_json(Path(opt.checkpoints_dir)/opt.name/'smoke_result.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'runs'/'experiment_queue')
    parser.add_argument('--select', nargs='+', help='selected experiment IDs; each runs once')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--gpu-ids', default='0')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--python', default=sys.executable)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--smoke-test', action='store_true')
    mode.add_argument('--status', action='store_true')
    parser.add_argument('--resume', action='store_true', help='resume interrupted jobs; completed jobs stay skipped')
    parser.add_argument('--_smoke-one', help=argparse.SUPPRESS)
    args = parser.parse_args()
    plan = json.loads(args.config.read_text(encoding='utf-8'))
    experiments = select_experiments(plan, args.select)
    jobs = [{'id':e['id'], 'label':e['label'],
             'values':training_values(plan,e,args.output,args.seed,args.gpu_ids,args.workers)} for e in experiments]
    if args.status:
        path = args.output/'queue_state.json'
        print(path.read_text(encoding='utf-8') if path.exists() else 'Queue has not started.')
        return
    if args._smoke_one:
        job = next(j for j in jobs if j['id'] == args._smoke_one)
        job['values']['checkpoints_dir'] = str(args.output.resolve()/'smoke')
        smoke_one(job['values'])
        return
    for index,job in enumerate(jobs,1):
        print(f"{index:02d}. {job['id']} {job['label']} (seed={args.seed})", flush=True)
    if not args.run and not args.smoke_test:
        return
    dataset = verify_dataset(plan,args.config)
    for job in jobs:
        command_for(job['values'],args.python)  # validate every job before first launch
    if args.smoke_test:
        (args.output/'smoke').mkdir(parents=True,exist_ok=True)
        for job in jobs:
            command = [args.python,'-u',str(Path(__file__).resolve()),'--config',str(args.config.resolve()),
                       '--output',str(args.output.resolve()),'--seed',str(args.seed),'--gpu-ids',args.gpu_ids,
                       '--_smoke-one',job['id']]
            with (args.output/'smoke'/f"{job['id']}.log").open('w',encoding='utf-8') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,
                               env={**os.environ,'PYTHONIOENCODING':'utf-8'})
            print(f"Preflight passed: {job['id']} {job['label']}",flush=True)
        return
    run_jobs(jobs,args.output,dataset,args.python,resume=args.resume)


if __name__ == '__main__':
    main()
