"""Original CycleGAN runtime and weight-only checkpoints without heavy state."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from training.experiment_queue import (latest_weight_epoch, run_jobs,
                                       verify_completion, verify_weight_checkpoint)
from training.train import _prepare_resume, _save_checkpoint, configure_runtime


def test_weight_save_skips_all_training_state_and_ignores_existing_state_on_resume(tmp_path):
    net = torch.nn.Linear(2, 2)
    opt = SimpleNamespace(weights_only=True, epoch='2', load_iter=0, epoch_count=1)
    model = SimpleNamespace(opt=opt, save_dir=tmp_path, model_names=['G_A'], netG_A=net)
    model.save_networks = lambda suffix: torch.save(net.state_dict(), tmp_path / f'{suffix}_net_G_A.pth')
    with patch('training.train._save_state', side_effect=AssertionError('Must not save optimizer/RNG state')):
        _save_checkpoint(model, '2', 2, 4000, epoch_boundary=True)
        _save_checkpoint(model, 'latest', 3, 5000, epoch_boundary=False)
    assert not list(tmp_path.glob('*training_state.pth'))
    assert latest_weight_epoch(tmp_path) == 2
    metadata = verify_weight_checkpoint(tmp_path, '2')
    assert metadata['total_iters'] == 4000
    # Even an old full-state file must not be read in the user-selected weight-only mode.
    (tmp_path / '2_training_state.pth').write_bytes(b'not a training state')
    with patch('training.train._read_state', side_effect=AssertionError('Must not load training state')):
        state, total = _prepare_resume(opt, model, tmp_path, 2000, [])
    assert state is None and total == 4000 and opt.epoch_count == 3
    opt.epoch = 'latest'
    with pytest.raises(ValueError, match='partial latest'):
        _prepare_resume(opt, model, tmp_path, 2000, [])
    (tmp_path / '2_net_G_A.pth').unlink()
    with pytest.raises(RuntimeError, match='Missing network'):
        verify_weight_checkpoint(tmp_path, '2')


def test_original_runtime_restores_expected_backend_flags():
    before = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark,
              torch.backends.cudnn.deterministic, torch.backends.cudnn.allow_tf32,
              torch.backends.cuda.matmul.allow_tf32)
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.allow_tf32 = False
        actual = configure_runtime(SimpleNamespace(cyclegan_runtime=True, deterministic=False,
                                                   preprocess='resize_and_crop', weights_only=True))
        assert actual == dict(deterministic_algorithms=False, cudnn_benchmark=True,
                              cudnn_deterministic=False, cudnn_allow_tf32=True,
                              matmul_allow_tf32=False, weights_only=True)
        with pytest.raises(ValueError):
            configure_runtime(SimpleNamespace(cyclegan_runtime=True, deterministic=True))
    finally:
        torch.use_deterministic_algorithms(before[0])
        torch.backends.cudnn.benchmark = before[1]
        torch.backends.cudnn.deterministic = before[2]
        torch.backends.cudnn.allow_tf32 = before[3]
        torch.backends.cuda.matmul.allow_tf32 = before[4]


def test_queue_pauses_continues_and_completes_without_any_training_state(tmp_path):
    child = tmp_path / 'child.py'
    child.write_text('''import json,sys
from pathlib import Path
p=Path(sys.argv[1]); p.mkdir(parents=True,exist_ok=True)
continued=sys.argv[2]=='True'
if continued: assert sys.argv[3:] == ['1','2']
epoch=5 if continued else 1
names=['G_A','G_B','D_A','D_B']
for name in names: (p/f'{epoch}_net_{name}.pth').write_text('weights')
(p/f'{epoch}_checkpoint.json').write_text(json.dumps(dict(epoch=epoch,next_epoch=epoch+1,
 epoch_boundary=True,total_iters=epoch*2000,networks=names)))
print('@@MLCGUI '+json.dumps({'type':'run_finished' if continued else 'run_paused'}),flush=True)
''', encoding='utf-8')
    values = dict(name='short', checkpoints_dir=str(tmp_path / 'checkpoints'),
                  n_epochs=100, n_epochs_decay=100, stop_after_epoch=5, weights_only=True)
    jobs = [dict(id='short', label='short', values=values)]

    def command(v, python):
        return [python, str(child), str(Path(v['checkpoints_dir']) / v['name']),
                str(v.get('continue_train', False)), str(v.get('epoch', 'latest')), str(v.get('epoch_count', 1))]

    with patch('training.experiment_queue.command_for', side_effect=command):
        run_jobs(jobs, tmp_path, {}, sys.executable)
        assert json.loads((tmp_path / 'queue_state.json').read_text())['status'] == 'paused'
        run_jobs(jobs, tmp_path, {}, sys.executable, resume=True)
    state = json.loads((tmp_path / 'queue_state.json').read_text())
    assert state['status'] == 'completed'
    assert state['jobs']['short']['attempts'] == 2
    assert 'reset' in state['jobs']['short']['continuation']
    assert not list(tmp_path.rglob('*training_state.pth'))
    verify_completion(values, tmp_path / 'checkpoints/short')
