"""PatchNCE remains differentiable after being embedded in the CUT module."""

from types import SimpleNamespace

import pytest
import torch

from models.cut_model import PatchNCELoss


@pytest.mark.parametrize("all_negatives", [False, True])
def test_patchnce_matches_explicit_logits_and_detaches_keys(all_negatives):
    opt = SimpleNamespace(batch_size=2, nce_T=0.07,
                          nce_includes_all_negatives_from_minibatch=all_negatives)
    generator = torch.Generator().manual_seed(123)
    query = torch.randn(6, 4, generator=generator, requires_grad=True)
    key = torch.randn(6, 4, generator=generator, requires_grad=True)
    loss = PatchNCELoss(opt)(query, key)
    expected = []
    group_size = 6 if all_negatives else 3
    for index in range(6):
        start = (index // group_size) * group_size
        logits = [(query[index] * key[index].detach()).sum()]
        for other in range(start, start + group_size):
            logits.append(query.new_tensor(-10.) if index == other
                          else (query[index] * key[other].detach()).sum())
        expected.append(-torch.log_softmax(torch.stack(logits) / opt.nce_T, dim=0)[0])
    torch.testing.assert_close(loss, torch.stack(expected))
    loss.mean().backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert key.grad is None
