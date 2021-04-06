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


class BalancedExploreExploit(ActionSelectionStrategyABC):
    """Epsilon greedy with epsilon fixed at 0.5"""
    def __call__(self, prediction_arr, time_step=None):
        should_exploit = get_rng().random() < 0.5
        if should_exploit:
            prediction_arr = filter_null_prediction_arr_entries(prediction_arr)
            return max(prediction_arr, key=prediction_arr.get)
        else:
            return get_rng().choice(self._action_space)


class LinearDecayEpsilonGreedy(ActionSelectionStrategyABC):
    """Epsilon greedy with epsilon starting at 1.0, decaying to epsilon_min
    with gradient specified by decay_factor."""
    _EPSILON_MAX = 1.0

    def __init__(self, action_space):
        super().__init__(action_space)
        self._epsilon = self._EPSILON_MAX

    def __call__(self, prediction_arr, time_step):
        self._epsilon = self._decay_epsilon(time_step)
        should_exploit = get_rng().random() < self._epsilon
        if should_exploit:
            prediction_arr = filter_null_prediction_arr_entries(prediction_arr)
            return max(prediction_arr, key=prediction_arr.get)
        else:
            return get_rng().choice(self._action_space)

    def _decay_epsilon(self, time_step):
        decayed_val = self._EPSILON_MAX - \
            get_hp("e_greedy_decay_factor")*time_step
        return max(decayed_val, get_hp("e_greedy_epsilon_min"))
