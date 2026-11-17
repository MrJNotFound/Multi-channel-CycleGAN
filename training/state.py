"""Serializable RNG, replay buffers and atomic epoch recovery bundles."""
from __future__ import annotations

import random
import hashlib

import numpy as np
import torch

from util.image_pool import ImagePool


def rng_state():
    numpy = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': (numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state['python'])
    numpy = state['numpy']
    np.random.set_state((numpy[0], np.array(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def unwrap(net):
    if hasattr(net, 'module'):
        net = net.module
    return getattr(net, '_orig_mod', net)


def network_states(model):
    return {name: unwrap(getattr(model, 'net' + name)).state_dict()
            for name in model.model_names if isinstance(name, str)}


def network_hashes(model):
    result = {}
    for name, state in network_states(model).items():
        digest = hashlib.sha256()
        for key, tensor in state.items():
            digest.update(key.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        result[name] = digest.hexdigest()
    return result


def replay_states(model):
    return {name: {'pool_size': pool.pool_size,
                   'images': [image.detach().cpu() for image in getattr(pool, 'images', [])]}
            for name, pool in vars(model).items() if isinstance(pool, ImagePool)}


def restore_replay(model, states):
    for name, state in states.items():
        pool = getattr(model, name)
        if pool.pool_size != state['pool_size']:
            raise ValueError(f'Replay pool size changed: {name}')
        pool.images = [image.to(model.device) for image in state['images']]
        pool.num_imgs = len(pool.images)
