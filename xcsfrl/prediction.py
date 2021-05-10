import abc

import numpy as np

from .classifier import LinearPredClassifier, QuadraticPredClassifier
from .hyperparams import get_hyperparam as get_hp


class PredictionStrategyABC(metaclass=abc.ABCMeta):
    def make_classifier(self, condition, action, time_step):
        return self._CLFR_CLS(condition, action, time_step)

    @abc.abstractmethod
    def aug_obs(self, obs):
        raise NotImplementedError


class LinearPrediction(PredictionStrategyABC):
    _CLFR_CLS = LinearPredClassifier

    def aug_obs(self, obs):
        return np.concatenate(([get_hp("x_nought")], obs))


class QuadraticPrediction(PredictionStrategyABC):
    _CLFR_CLS = QuadraticPredClassifier

    def aug_obs(self, obs):
        aug_obs = []
        for elem in obs:
            aug_obs.append(elem)
            aug_obs.append(elem**2)
        return np.concatenate(([get_hp("x_nought")], aug_obs))
