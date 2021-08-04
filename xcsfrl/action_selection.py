import abc
from collections import OrderedDict
from enum import Enum

from .hyperparams import get_hyperparam as get_hp
from .rng import get_rng

NULL_ACTION = -1

ActionSelectionModes = Enum("ActionSelectionModes", ["explore", "exploit"])


def choose_action_selection_mode():
    # 50/50 chance of either explore/exploit on episode start
    return get_rng().choice(list(iter(ActionSelectionModes)))


def greedy_action_selection(prediction_arr):
    prediction_arr = filter_null_prediction_arr_entries(prediction_arr)
    return max(prediction_arr, key=prediction_arr.get)


def filter_null_prediction_arr_entries(prediction_arr):
    return OrderedDict(
        {a: p
         for (a, p) in prediction_arr.items() if p is not None})


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
            return greedy_action_selection(prediction_arr)
