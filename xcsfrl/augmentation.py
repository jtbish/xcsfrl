import abc

import numpy as np

from .hyperparams import get_hyperparam as get_hp


def make_aug_strat(poly_order):
    if poly_order == 1:
        return LinearAugmentation()
    elif poly_order == 2:
        return QuadraticAugmentation()
    else:
        assert False


class AugmentationStratABC(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, obs):
        raise NotImplementedError


class LinearAugmentation(AugmentationStratABC):
    def __call__(self, obs):
        return np.concatenate(([get_hp("x_nought")], obs))


class QuadraticAugmentation(AugmentationStratABC):
    def __call__(self, obs):
        aug_obs = []
        for elem in obs:
            aug_obs.append(elem)
            aug_obs.append(elem**2)
        return np.concatenate(([get_hp("x_nought")], aug_obs))
