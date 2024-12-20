from itertools import chain
from typing import Iterator

import numpy as np
from torch.utils.data import Sampler


class SeededSampler(Sampler):
    """A :class:`SeededSampler` samples a dataset in a randomly seeded fashion"""

    def __init__(self, N: int, seed: int):
        if seed is None:
            raise ValueError("arg 'seed' was `None`! A SeededSampler must be seeded!")

        self.idxs = np.arange(N)
        self.rg = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[int]:
        self.rg.shuffle(self.idxs)

        return iter(self.idxs)

    def __len__(self) -> int:
        return len(self.idxs)


class ClassBalanceSampler(Sampler):
    """A :class:`ClassBalanceSampler` samples a dataset such that
    positive and negative classes are equally sampled

    Parameters
    ----------
    Y : np.ndarray
        an array of shape ``n x t`` containing the dataset labels, where ``n`` is the dataset size
        and ``t`` is the number of targets
    seed : int | None, default=None
        the random seed to use for shuffling (only used when :attr:`shuffle` is ``True``)
    shuffle : bool, default=False
        whether to shuffle the data during sampling
    """

    def __init__(self, Y: np.ndarray, seed: int | None = None, shuffle: bool = False):
        self.shuffle = shuffle
        self.rg = np.random.default_rng(seed)

        idxs = np.arange(len(Y))
        actives = Y.any(1)

        self._pos_idxs = idxs[actives]
        self._neg_idxs = idxs[~actives]

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            self.rg.shuffle(self._pos_idxs)
            self.rg.shuffle(self._neg_idxs)

        return chain(*zip(self._pos_idxs, self._neg_idxs))

    def __len__(self) -> int:
        return 2 * min(len(self._pos_idxs), len(self._neg_idxs))
