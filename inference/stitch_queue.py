"""Run trained experiment configurations through infer.py and patch_stitching.py.

Source tiles and checkpoints are read-only. Generated tiles are temporary and
are removed only after all stitched images for that experiment are verified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.experiment_queue import fingerprint, queue_lock, timestamp, write_json

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
COORDINATES = re.compile(r'^(.*?)_?patch_y(\d+)_x(\d+)$')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def audit_tiles(root):
    """Reject ambiguous pairing/coordinates, corrupt tiles and incomplete grids."""
    from PIL import Image
    root = Path(root).resolve()
    domains = {}
    for domain in ('BF', 'AF'):
        files = sorted(p for p in (root / f'testA_{domain}').rglob('*')
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        names = {p.name: p for p in files}
        if not files or len(names) != len(files):
            raise ValueError(f'{domain}: empty dataset or duplicate filenames')
        domains[domain] = names
    if domains['BF'].keys() != domains['AF'].keys():
        raise ValueError('BF and AF filename sets must match exactly.')
    groups, hashes, stems = {}, {}, set()
    for name, bf in domains['BF'].items():
        match = COORDINATES.fullmatch(bf.stem)
        if match is None or bf.stem in stems:
            raise ValueError(f'Ambiguous or missing tile coordinates: {name}')
        stems.add(bf.stem)
        label, y, x = match.groups()
        label = label or 'stitched'
        dimensions = []
        for domain in ('BF', 'AF'):
            path = domains[domain][name]
            with Image.open(path) as im:
                dimensions.append(im.size)
                im.verify()
            hashes[f'{domain}/{name}'] = sha256(path)
        if dimensions[0] != dimensions[1]:
            raise ValueError(f'BF/AF dimensions differ: {name}')
        w, h = dimensions[0]
        if w % 256 or h % 256:
            raise ValueError(f'Tiles must be divisible by 256 for U-Net without resizing: {name}')
        groups.setdefault(label, []).append(dict(name=name, x=int(x), y=int(y), width=w, height=h))
    summaries = {}
    for label, tiles in sorted(groups.items()):
        tiles.sort(key=lambda t: (t['y'], t['x']))
        sizes = {(t['width'], t['height']) for t in tiles}
        positions = {(t['x'], t['y']) for t in tiles}
        xs, ys = sorted({t['x'] for t in tiles}), sorted({t['y'] for t in tiles})
        if len(sizes) != 1 or len(positions) != len(tiles):
            raise ValueError(f'{label}: inconsistent sizes or duplicate coordinates')
        w, h = next(iter(sizes))
        if (xs[0] != 0 or ys[0] != 0 or len(tiles) != len(xs) * len(ys)
                or any(b-a > w for a, b in zip(xs, xs[1:]))
                or any(b-a > h for a, b in zip(ys, ys[1:]))):
            raise ValueError(f'{label}: grid has uncovered areas; supply a complete tile set')
        summaries[label] = dict(width=xs[-1]+w, height=ys[-1]+h, tiles=tiles)
    return dict(root=str(root), count=len(stems), sha256=fingerprint(hashes),
                groups=summaries, dimensions_source='tile coordinate extents; may include edge padding')


def inference_values(job, tiles, output, epoch, gpu_ids, batch_size, workers):
    from gui.launcher import build_option_parser
    original = job['values']
    source_model = original['model']
    parser, _ = build_option_parser('inference', source_model, 'single', original)
    known = {action.dest for action in parser._actions}
    values = {key: value for key, value in original.items() if key in known}
    root = Path(tiles['root'])
    if original['input_nc'] == 2:
        dataroot = root / 'testA'  # SingleDataset reads testA_BF and testA_AF.
    elif original['input_nc'] == 1:
        modality = Path(original['source_dir']).name.upper()
        if modality not in {'BF', 'AF'}:
            raise ValueError(f'Cannot identify single-channel source: {original["source_dir"]}')
        dataroot = root / f'testA_{modality}'
    else:
        raise ValueError('Expected a BF, AF or BF+AF experiment.')
    values.update(model='test', source_model=source_model, dataset_mode='single',
                  dataroot=str(dataroot), phase='test', epoch=str(epoch), load_iter=0,
                  results_dir=str(Path(output).resolve() / '.patches' / job['id']),
                  preprocess='none', no_flip=True, serial_batches=True, eval=True,
                  batch_size=batch_size, num_threads=workers, gpu_ids=gpu_ids,
                  num_test=tiles['count'], max_dataset_size=tiles['count'],
                  save_inputs=False, make_report=False, output_suffix='_fake',
                  png_compress_level=1, aspect_ratio=1.0, suffix='', gui_events=True)
    return values


def make_jobs(config, tiles, output, epoch, gpu_ids, batch_size, workers, selected=None):
    from gui.launcher import build_command
    from models.registry import model_spec
    source = json.loads(Path(config).read_text(encoding='utf-8'))
    ids = [j['id'] for j in source]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', key) for key in ids):
        raise ValueError('Experiment IDs must be unique safe directory names.')
    if selected and (len(selected) != len(set(selected)) or set(selected) - set(ids)):
        raise ValueError('Unknown or repeated experiment selections.')
    source = [j for j in source if not selected or j['id'] in selected]
    source.sort(key=lambda j: j['label'].lower() == 'spif')
    jobs = []
    for job in source:
        values = inference_values(job, tiles, output, epoch, gpu_ids, batch_size, workers)
        suffix = '_A' if model_spec(values['source_model']).architecture == 'cycle_gan' else ''
        values['model_suffix'] = suffix
        checkpoint = Path(values['checkpoints_dir']) / values['name'] / f'{epoch}_net_G{suffix}.pth'
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        command = build_command('inference', values['source_model'], 'single', values,
                                python_executable=sys.executable)
        jobs.append(dict(id=job['id'], label=job['label'], values=values,
                         checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                         command=[command[0], '-u', *command[1:]]))
    if not jobs:
        raise ValueError('No experiments selected.')
    return jobs


def generated_directory(job):
    values = job['values']
    return Path(values['results_dir']) / values['name'] / f"test_{values['epoch']}" / 'images'


def verify_generated(directory, tiles):
    from PIL import Image
    expected = {Path(t['name']).stem+'_fake.png': (t['width'], t['height'])
                for group in tiles['groups'].values() for t in group['tiles']}
    actual = {p.name for p in Path(directory).iterdir() if p.is_file()}
    if actual != expected.keys():
        raise ValueError(f'Generated tile set is incomplete or contains extras: {directory}')
    for name, dimensions in expected.items():
        with Image.open(Path(directory) / name) as im:
            if im.size != dimensions or im.mode != 'RGB':
                raise ValueError(f'Generated tile geometry/mode differs: {name}')
            im.verify()


def remove_temporary_tree(path, output):
    """Only delete a direct experiment directory under output/.patches."""
    path, output = Path(path).absolute(), Path(output).resolve()
    parent = output / '.patches'
    if path.parent != parent or path.resolve().parent != parent.resolve() or parent.is_symlink():
        raise ValueError(f'Refusing cleanup outside temporary experiment directory: {path}')
    if not path.exists():
        return
    for candidate in [parent, path, *path.rglob('*')]:
        attributes = getattr(candidate.lstat(), 'st_file_attributes', 0)
        if (candidate.is_symlink() or attributes & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0)
                or not candidate.resolve().is_relative_to(output)):
            raise ValueError(f'Refusing cleanup through a filesystem link: {candidate}')
    shutil.rmtree(path)


def stitch_outputs(directory, destination, tiles):
    import cv2
    from PIL import Image
    from tool.patch_stitching import stitch_group
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    for label, group in tiles['groups'].items():
        items = [(t['x'], t['y'], Path(directory) / (Path(t['name']).stem+'_fake.png'))
                 for t in group['tiles']]
        stitched, _, _, sx, sy, width, height = stitch_group(
            items, group['width'], group['height'], 'hann', False, (255, 255, 255), 3)
        path = destination / f'{label}.png'
        temporary = path.with_name(path.stem+'.partial.png')
        if not cv2.imwrite(str(temporary), stitched, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise OSError(f'Failed to write {temporary}')
        del stitched
        with Image.open(temporary) as im:
            if im.size != (width, height) or im.mode != 'RGB':
                raise ValueError(f'Stitched image failed geometry verification: {temporary}')
            im.verify()
        digest = sha256(temporary)
        temporary.replace(path)
        results.append(dict(path=str(path), width=width, height=height,
                            tiles=len(items), stride=[sx, sy], sha256=digest,
                            bytes=path.stat().st_size))
        print(f'  Stitched {label}: {width}x{height}, {len(items)} tiles', flush=True)
    return results


def verify_outputs(results):
    for result in results:
        path = Path(result['path'])
        if not path.is_file() or sha256(path) != result['sha256']:
            raise ValueError(f'Completed output is missing or changed: {path}')


def run_inference_process(job, log_path, progress):
    child = None
    finished = None
    last_update = 0
    with Path(log_path).open('a', encoding='utf-8') as log:
        try:
            child = subprocess.Popen(job['command'], cwd=ROOT, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                     errors='replace', env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            for line in child.stdout:
                log.write(line)
                log.flush()
                if not line.startswith('@@MLCGUI '):
                    continue
                event = json.loads(line[len('@@MLCGUI '):])
                if event['type'] == 'run_finished':
                    finished = event.get('processed')
                if event['type'] == 'progress' and time.monotonic() - last_update >= 5:
                    progress(event, child.pid)
                    last_update = time.monotonic()
            code = child.wait()
            if code or finished != job['values']['num_test']:
                raise RuntimeError(f'{job["id"]}: inference incomplete (exit={code}); see {log_path}')
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                child.wait()


def run_queue(jobs, tiles, output, resume=False):
    output = Path(output).resolve()
    for source in [Path(tiles['root']), *[Path(j['values']['checkpoints_dir']) for j in jobs]]:
        source = source.resolve()
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError('Output must be separate from source tiles and checkpoint directories.')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'logs').mkdir(exist_ok=True)
    with queue_lock(output):
        state_path = output / 'inference_state.json'
        signature = fingerprint(dict(jobs=jobs, tiles=tiles, blending='hann', ignore_padding=False))
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding='utf-8'))
            if state['fingerprint'] != signature:
                raise ValueError('Inputs, weights or configuration changed; use a new output directory.')
        else:
            if (output / 'stitched').exists() or (output / '.patches').exists():
                raise ValueError('Output already contains untracked images; use a new output directory.')
            state = dict(created=timestamp(), fingerprint=signature,
                         jobs={j['id']: dict(status='pending', label=j['label']) for j in jobs})
        state.update(status='running', pid=os.getpid())
        write_json(output / 'inference_plan.json', dict(jobs=jobs, tiles=tiles, blending='hann',
                                                       ignore_padding=False, precision='float32'))

        def save():
            state['updated'] = timestamp()
            write_json(state_path, state)

        save()
        try:
            for job in jobs:
                key = job['id']
                item = state['jobs'][key]
                temporary = output / '.patches' / key
                if item['status'] == 'completed':
                    verify_outputs(item['outputs'])
                    remove_temporary_tree(temporary, output)
                    print(f'Skip completed: {key}', flush=True)
                    continue
                if item['status'] != 'pending' and not resume:
                    raise RuntimeError(f'{key} was interrupted; use --resume after inspecting its log.')
                state['current'] = key
                item.update(status='running', started=timestamp())
                save()
                print(f'[{key}] {job["label"]}: inference and stitching', flush=True)

                def progress(event, pid):
                    item.update(progress=event, child_pid=pid)
                    save()

                if item.get('phase') != 'stitching':
                    remove_temporary_tree(temporary, output)
                    item['phase'] = 'inference'
                    save()
                    run_inference_process(job, output / 'logs' / f'{key}.log', progress)
                verify_generated(generated_directory(job), tiles)
                item['phase'] = 'stitching'
                save()
                results = stitch_outputs(generated_directory(job), output / 'stitched' / job['values']['name'], tiles)
                verify_outputs(results)
                item.update(status='completed', phase='completed', finished=timestamp(), outputs=results)
                save()  # Durable output record precedes deleting any generated tiles.
                remove_temporary_tree(temporary, output)
                item['temporary_tiles_removed'] = True
                save()
                print(f'[{key}] Completed {len(results)} images; temporary tiles removed.', flush=True)
            if (output / '.patches').exists():
                (output / '.patches').rmdir()  # Only remove an empty directory.
            state.update(status='completed', current=None, finished=timestamp())
            save()
        except BaseException as exc:
            state.update(status='failed', error=str(exc))
            if state.get('current'):
                item = state['jobs'][state['current']]
                if item['status'] != 'completed':
                    item.update(status='failed', error=str(exc))
            save()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='training queue_config.json')
    parser.add_argument('--input', type=Path, required=True, help='contains testA_BF and testA_AF')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--gpu-ids', default='0')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--select', nargs='+')
    parser.add_argument('--run', action='store_true', help='otherwise only audit and list the plan')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error('batch-size must be positive and workers nonnegative')
    tiles = audit_tiles(args.input)
    jobs = make_jobs(args.config, tiles, args.output, args.epoch, args.gpu_ids,
                     args.batch_size, args.workers, args.select)
    print(f'{tiles["count"]} paired tiles; {len(tiles["groups"])} regions; {len(jobs)} experiments', flush=True)
    for label, group in tiles['groups'].items():
        print(f'  {label}: {group["width"]}x{group["height"]}', flush=True)
    print('Order: '+', '.join(j['id'] for j in jobs), flush=True)
    if args.run:
        run_queue(jobs, tiles, args.output, args.resume)


if __name__ == '__main__':
    main()
