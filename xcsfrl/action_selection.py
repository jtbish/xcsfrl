import abc

from .rng import get_rng
from .util import filter_null_prediction_arr_entries


class ActionSelectionStrategyABC(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, prediction_arr, action_space):
        raise NotImplementedError


class BalancedExploreExploit(ActionSelectionStrategyABC):
    """Epsilon greedy with epsilon fixed at 0.5"""
    def __call__(self, prediction_arr, action_space):
        should_exploit = get_rng().random() < 0.5
        if should_exploit:
            prediction_arr = filter_null_prediction_arr_entries(prediction_arr)
            return max(prediction_arr, key=prediction_arr.get)
        else:
            return get_rng().choice(action_space)
