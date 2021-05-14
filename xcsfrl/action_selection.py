import abc

from .hyperparams import get_hyperparam as get_hp
from .rng import get_rng
from .util import filter_null_prediction_arr_entries


class ActionSelectionStrategyABC(metaclass=abc.ABCMeta):
    def __init__(self, action_space):
        self._action_space = action_space

    @abc.abstractmethod
    def __call__(self, prediction_arr, time_step=None):
        raise NotImplementedError


class FixedEpsilonGreedy(ActionSelectionStrategyABC):
    def __call__(self, prediction_arr, time_step=None):
        epsilon = get_hp("p_explr")
        should_explore = get_rng().random() < epsilon
        if should_explore:
            return get_rng().choice(self._action_space)
        else:
            prediction_arr = filter_null_prediction_arr_entries(prediction_arr)
            return max(prediction_arr, key=prediction_arr.get)
