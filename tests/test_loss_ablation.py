"""Contracts that matter for comparing losses rather than changing experiments."""
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.cycle_gan_structure_model import SPIFModel
from models.networks import get_scheduler
from training.diagnostics import save_fixed_preview
from training.experiment_queue import final_epoch_for
from training.loss_ablation_queue import build_jobs, CONDITIONS
from training.state import rng_state, restore_rng
from training.train import _save_state, _restore_state
from util.image_pool import ImagePool


def weight_model(grad=None, intensity=None, grad_mode='cosine', int_mode='cosine'):
    model = SPIFModel.__new__(SPIFModel)
    model.device = torch.device('cpu')
    model.opt = SimpleNamespace(n_epochs=100, n_epochs_decay=100, epoch_count=1,
                               lambda_content=1., lambda_sp=10., lambda_intensity=.5,
                               sp_grad_weight=grad, sp_intensity_weight=intensity,
                               sp_grad_schedule=grad_mode, sp_intensity_schedule=int_mode,
                               sp_grad_decay_fraction=.1, sp_intensity_decay_fraction=.025,
                               grad_channel_weights=[.5, .5], input_nc=2)
    model._init_content_loss_options(model.opt)
    return model


@pytest.mark.parametrize('condition', list(CONDITIONS))
def test_independent_schedules_and_endpoints(condition):
    grad, intensity, grad_mode, int_mode = CONDITIONS[condition]
    model = weight_model(grad, intensity, grad_mode, int_mode)
    assert model._effective_sp_weights() == (grad, intensity)
    model.set_current_epoch(5)
    g5, i5 = model._effective_sp_weights()
    assert g5 == pytest.approx(grad if grad_mode == 'constant' else grad * .8945702546981968)
    assert i5 == (intensity if int_mode == 'constant' else 0)
    model.set_current_epoch(200)
    assert model._effective_sp_weights() == (
        grad if grad_mode == 'constant' else 0, intensity if int_mode == 'constant' else 0)


def test_intensity_only_has_real_gradients_and_no_gradient_term():
    model = weight_model(0, 5)
    model.real_A, model.real_B = torch.randn(2, 2, 32, 32), torch.randn(2, 3, 32, 32)
    model.fake_A = torch.randn(2, 2, 32, 32, requires_grad=True)
    model.fake_B = torch.randn(2, 3, 32, 32, requires_grad=True)
    loss = model.content_loss()
    loss.backward()
    assert model.loss_sp_grad_A == 0
    assert model.loss_sp_grad_B == 0
    assert torch.count_nonzero(model.fake_A.grad) > 0
    assert torch.count_nonzero(model.fake_B.grad) > 0
    assert loss.item() == pytest.approx(5 * (model.loss_sp_int_A + model.loss_sp_int_B).item())


def test_legacy_coefficients_are_unchanged():
    assert weight_model()._effective_sp_weights() == (10, 5)


def test_regular_pooling_preserves_adaptive_pooling_loss_and_gradient():
    from unittest.mock import patch
    from models.cycle_gan_structure_model import CoarseIntensityDistributionLoss
    import torch.nn.functional as F
    loss = CoarseIntensityDistributionLoss()
    source = torch.randn(1, 2, 256, 256)
    generated = torch.randn(1, 3, 256, 256, requires_grad=True)
    regular = loss(source, generated)
    regular_grad, = torch.autograd.grad(regular, generated)
    # Independently use the former adaptive operator for the same bins.
    with patch('models.cycle_gan_structure_model.F.avg_pool2d',
               side_effect=lambda x, kernel_size: F.adaptive_avg_pool2d(x, (8, 8))):
        adaptive = loss(source, generated)
        adaptive_grad, = torch.autograd.grad(adaptive, generated)
    torch.testing.assert_close(regular, adaptive)
    torch.testing.assert_close(regular_grad, adaptive_grad)


def test_plan_has_61_fresh_runs_and_unchanged_short_run_horizon(tmp_path):
    jobs = build_jobs(tmp_path / 'data', tmp_path / 'out')
    assert len(jobs) == len({j['id'] for j in jobs}) == 61
    assert sum(final_epoch_for(j['values']) for j in jobs) == 4400
    for job in jobs:
        v = job['values']
        assert v['n_epochs'] + v['n_epochs_decay'] == 200
        assert not v.get('pretrained_name') and not v.get('continue_train')
        assert v['model'] == 'cycle_gan_dual_fusion_structure'
        assert v['spif_version'] == 'v2'
        assert v['cyclegan_runtime'] and not v['deterministic']
        assert v['weights_only'] and not v['complete_epoch_state']
        assert v['save_epoch_freq'] == 5 and v['save_latest_freq'] == 5000
        assert v['seed'] in (range(101, 121) if job['stage'] == 'A' else (101, 102, 103))
        assert final_epoch_for(v) == (5 if job['stage'] == 'A' else 200)
    selected = build_jobs(tmp_path, tmp_path / 'out', selected=['B2', 'A1_seed101'])
    assert [j['id'] for j in selected] == ['A1_seed101', 'B2_seed101', 'B2_seed102', 'B2_seed103']


def test_resume_scheduler_does_not_double_count_epochs():
    opt = SimpleNamespace(lr_policy='linear', epoch_count=1, n_epochs=100, n_epochs_decay=100)
    p = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.Adam([p], lr=.0002)
    scheduler = get_scheduler(optimizer, opt)
    for _ in range(120):
        optimizer.step()
        scheduler.step()
    state = scheduler.state_dict()
    resumed_opt = SimpleNamespace(**{**vars(opt), 'epoch_count': 121, 'scheduler_epoch_count': 1})
    resumed = get_scheduler(optimizer, resumed_opt)
    resumed.load_state_dict(state)
    for _ in range(80):
        optimizer.step()
        scheduler.step()
        resumed.step()
        assert scheduler.get_last_lr() == resumed.get_last_lr()


def test_checkpoint_roundtrip_restores_rng_and_replay(tmp_path):
    net = torch.nn.Conv2d(2, 2, 1)
    optimizer = torch.optim.Adam(net.parameters(), lr=.001)
    net(torch.randn(1, 2, 8, 8)).sum().backward()
    optimizer.step()
    model = SimpleNamespace(opt=SimpleNamespace(epoch_count=1), save_dir=tmp_path,
                            model_names=['G_A'], netG_A=net, device=torch.device('cpu'),
                            optimizers=[optimizer], schedulers=[], fake_A_pool=ImagePool(2))
    model.fake_A_pool.query(torch.randn(2, 2, 8, 8))
    _save_state(model, 'latest', 3, 16, include_networks=True)
    state = torch.load(tmp_path / 'latest_training_state.pth', weights_only=True)
    expected = (random.random(), np.random.rand(), torch.rand(4))
    model.fake_A_pool.images.clear()
    model.fake_A_pool.num_imgs = 0
    _restore_state(model, state)
    actual = (random.random(), np.random.rand(), torch.rand(4))
    assert expected[:2] == actual[:2]
    assert torch.equal(expected[2], actual[2])
    assert model.fake_A_pool.num_imgs == 2
    assert torch.equal(model.fake_A_pool.images[0], state['replay_pools']['fake_A_pool']['images'][0])
    assert state['next_epoch'] == 3 and state['epoch_boundary']
    assert 'G_A' in state['networks']


def test_fixed_preview_preserves_rng_training_modes_and_existing_model_tensors(tmp_path):
    from PIL import Image
    source = tmp_path / 'source.png'
    Image.fromarray(np.arange(64*64, dtype=np.uint8).reshape(64, 64)).save(source)
    manifest = dict(load_size=40, crop_size=32,
                    locations=[dict(id='ROI01', BF=str(source), AF=str(source), crop_pos=[4, 4])])
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    net = torch.nn.Sequential(torch.nn.Conv2d(2, 3, 1), torch.nn.Dropout(.5), torch.nn.Tanh())
    net.train()
    net[1].eval()
    source_tensor = torch.randn(1, 2, 32, 32)
    model = SimpleNamespace(netG_A=net, device=torch.device('cpu'), real_A=source_tensor)
    opt = SimpleNamespace(fixed_preview_manifest=str(path), checkpoints_dir=str(tmp_path), name='test')
    before = rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(3))
    restore_rng(before)
    save_fixed_preview(model, opt, 1)
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert expected[:2] == actual[:2] and torch.equal(expected[2], actual[2])
    assert net.training and not net[1].training
    assert model.real_A is source_tensor
    assert (tmp_path / 'test/fixed_previews/epoch_001/ROI01.png').is_file()
