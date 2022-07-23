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


def _epsilon_greedy(epsilon, action_space, prediction_arr):
    should_explore = get_rng().random() < epsilon
    if should_explore:
        return get_rng().choice(action_space)
    else:
        return greedy_action_selection(prediction_arr)


class ActionSelectionStrategyABC(metaclass=abc.ABCMeta):
    def __init__(self, action_space):
        self._action_space = action_space

    @abc.abstractmethod
    def __call__(self, prediction_arr, num_ga_calls=None):
        raise NotImplementedError


class FixedEpsilonGreedy(ActionSelectionStrategyABC):
    def __call__(self, prediction_arr, num_ga_calls=None):
        epsilon = get_hp("p_explr")
        return _epsilon_greedy(epsilon, self._action_space, prediction_arr)


class LinearDecayEpsilonGreedy(ActionSelectionStrategyABC):
    """Epsilon greedy policy in which epsilon is initially set to _EPSILON_INIT
    then linearly decays to a value of _EPSILON_MIN over *half* of the total
    number of GA calls to be trained for."""

    _EPSILON_INIT = 1.0
    _EPSILON_MIN = 0.1

    def __init__(self, action_space, total_num_ga_calls):
        super().__init__(action_space)
        self._epsilon = self._EPSILON_INIT
        self._total_num_ga_calls = total_num_ga_calls
        self._decay_grad = self._calc_decay_grad(self._total_num_ga_calls)

    def _calc_decay_grad(self, total_num_ga_calls):
        (x1, y1) = (0, self._EPSILON_INIT)
        (x2, y2) = ((total_num_ga_calls / 2), self._EPSILON_MIN)
        m = (y2 - y1) / (x2 - x1)
        assert m < 0
        return m

    def __call__(self, prediction_arr, num_ga_calls):
        self._epsilon = self._decay_epsilon(num_ga_calls)
        return _epsilon_greedy(self._epsilon, self._action_space,
                               prediction_arr)

    def _decay_epsilon(self, num_ga_calls):
        # y = mx + c
        new_epsilon = ((self._decay_grad * num_ga_calls) + self._EPSILON_INIT)
        new_epsilon = max(new_epsilon, self._EPSILON_MIN)
        return new_epsilon
