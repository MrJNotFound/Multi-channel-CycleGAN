import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest

from inference import stitch_queue as queue


def paired_tiles(root, coordinates=((0, 0),)):
    for domain in ('BF', 'AF'):
        directory = root / f'testA_{domain}'
        directory.mkdir(parents=True, exist_ok=True)
        for x, y in coordinates:
            Image.new('RGB', (256, 256), (150, 150, 150)).save(
                directory / f'region_patch_y{y:06}_x{x:06}.png')
    return queue.audit_tiles(root)


def test_audit_rejects_missing_pair_and_coordinate_gap(tmp_path):
    source = tmp_path / 'inputs'
    paired_tiles(source, ((0, 0), (128, 0)))
    next((source / 'testA_AF').glob('*.png')).unlink()
    with pytest.raises(ValueError, match='filename sets'):
        queue.audit_tiles(source)
    with pytest.raises(ValueError, match='uncovered areas'):
        paired_tiles(tmp_path / 'gaps', ((0, 0), (512, 0)))


def test_stitch_geometry_and_color_are_preserved(tmp_path):
    pytest.importorskip('cv2')
    tiles = paired_tiles(tmp_path / 'inputs', ((0, 0), (128, 0), (0, 128), (128, 128)))
    generated = tmp_path / 'generated'
    generated.mkdir()
    colors = [(200, 20, 30), (30, 200, 20), (20, 30, 200), (80, 90, 100)]
    for tile, color in zip(tiles['groups']['region']['tiles'], colors):
        Image.new('RGB', (256, 256), color).save(generated / (Path(tile['name']).stem+'_fake.png'))
    queue.verify_generated(generated, tiles)
    outputs = queue.stitch_outputs(generated, tmp_path / 'stitched', tiles)
    queue.verify_outputs(outputs)
    with Image.open(outputs[0]['path']) as im:
        assert im.size == (384, 384)
        pixels = np.asarray(im)
        np.testing.assert_allclose(pixels[32, 32], colors[0], atol=1)
        np.testing.assert_allclose(pixels[-32, -32], colors[3], atol=1)
        assert pixels[190, 190].min() > 20
    Image.new('RGB', (128, 128)).save(next(generated.glob('*.png')))
    with pytest.raises(ValueError, match='geometry'):
        queue.verify_generated(generated, tiles)


def test_cleanup_cannot_delete_source_or_output_root(tmp_path):
    output = tmp_path / 'output'
    generated = output / '.patches' / 'C01'
    generated.mkdir(parents=True)
    (generated / 'tile.png').write_bytes(b'temporary')
    source = tmp_path / 'source'
    source.mkdir()
    sentinel = source / 'original.png'
    sentinel.write_bytes(b'original')
    for unsafe in (source, output, output / '.patches'):
        with pytest.raises(ValueError, match='Refusing cleanup'):
            queue.remove_temporary_tree(unsafe, output)
    assert sentinel.read_bytes() == b'original'
    queue.remove_temporary_tree(generated, output)
    assert not generated.exists()


def test_stitch_failure_keeps_tiles_resume_removes_only_after_success(tmp_path):
    pytest.importorskip('cv2')
    tiles = paired_tiles(tmp_path / 'inputs')
    output = tmp_path / 'output'
    values = dict(name='example', epoch='200', results_dir=str(output / '.patches' / 'C01'),
                  checkpoints_dir=str(tmp_path / 'checkpoints'), num_test=1)
    jobs = [dict(id='C01', label='example', values=values)]

    def generate(job, log, progress):
        directory = queue.generated_directory(job)
        directory.mkdir(parents=True)
        name = tiles['groups']['region']['tiles'][0]['name']
        Image.new('RGB', (256, 256), 'pink').save(directory / (Path(name).stem+'_fake.png'))

    with patch.object(queue, 'run_inference_process', side_effect=generate) as infer:
        with patch.object(queue, 'stitch_outputs', side_effect=OSError('disk unavailable')):
            with pytest.raises(OSError, match='disk unavailable'):
                queue.run_queue(jobs, tiles, output)
        assert list(queue.generated_directory(jobs[0]).glob('*.png'))
        queue.run_queue(jobs, tiles, output, resume=True)
        queue.run_queue(jobs, tiles, output)
        assert infer.call_count == 1
    state = json.loads((output / 'inference_state.json').read_text(encoding='utf-8'))
    assert state['status'] == 'completed'
    assert state['jobs']['C01']['temporary_tiles_removed']
    assert not (output / '.patches').exists()
    assert len(list((tmp_path / 'inputs').rglob('*.png'))) == 2
