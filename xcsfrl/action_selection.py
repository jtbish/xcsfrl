import abc
from collections import OrderedDict

from .rng import get_rng


class ActionSelectionStrategyABC(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, prediction_arr, action_space):
        raise NotImplementedError

    def _filter_null_entries(self, prediction_arr):
        return OrderedDict({a: p for (a, p) in prediction_arr.items() if p is
                           not None})


class BalancedExploreExploit(ActionSelectionStrategyABC):
    """Epsilon greedy with epsilon fixed at 0.5"""
    def __call__(self, prediction_arr, action_space):
        should_exploit = get_rng().random() < 0.5
        if should_exploit:
            prediction_arr = self._filter_null_entries(prediction_arr)
            return max(prediction_arr, key=prediction_arr.get)
        else:
            return get_rng().choice(action_space)
