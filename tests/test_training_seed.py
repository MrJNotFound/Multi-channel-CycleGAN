import random
import numpy as np
import torch
from data import EpochSeededDataset


class RandomDataset:
    def __len__(self):
        return 3

    def __getitem__(self,index):
        return (random.random(),float(np.random.rand()),float(torch.rand(())))


def test_data_randomness_is_independent_of_model_randomness_and_epoch_specific():
    dataset=EpochSeededDataset(RandomDataset(),2026)
    first=dataset[0]
    random.seed(888); np.random.seed(999); torch.manual_seed(1000)
    torch.randn(500)
    assert dataset[0] == first
    dataset.epoch=2
    assert dataset[0] != first
    dataset.epoch=1
    assert dataset[0] == first
    assert dataset[1] != first


def test_sample_does_not_change_caller_random_state():
    dataset=EpochSeededDataset(RandomDataset(),2026)
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    expected=RandomDataset()[0]
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    dataset[0]
    assert RandomDataset()[0] == expected
