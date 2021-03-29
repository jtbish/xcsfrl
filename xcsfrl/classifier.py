from .hyperparams import get_hyperparam as get_hp
import numpy as np
from .id import get_next_clfr_id

_EXPERIENCE_MIN = 0
_ACTION_SET_SIZE_MIN = 1
_NUMEROSITY_MIN = 1
_TIME_STAMP_MIN = 0
_ERROR_MIN = 0.0
_FITNESS_MIN = 0.0
_FITNESS_MAX = 1.0


class Classifier:
    def __init__(self, condition, action, time_step):
        self._condition = condition
        self._action = action
        self._weight_vec = self._init_weight_vec(self._condition)
        self.error = get_hp("epsilon_I")
        self.fitness = get_hp("fitness_I")
        self.experience = 0
        self.time_stamp = time_step
        self.action_set_size = 1
        self.numerosity = 1
        self._id = get_next_clfr_id()

    @property
    def condition(self):
        return self._condition

    @condition.setter
    def condition(self, val):
        self._condition = val

    @property
    def action(self):
        return self._action

    @action.setter
    def action(self, val):
        self._action = val

    @property
    def weight_vec(self):
        return self._weight_vec

    @weight_vec.setter
    def weight_vec(self, val):
        self._weight_vec = val

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, val):
        assert val >= _ERROR_MIN
        self._error = val

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, val):
        assert _FITNESS_MIN <= val <= _FITNESS_MAX
        self._fitness = val

    @property
    def experience(self):
        return self._experience

    @experience.setter
    def experience(self, val):
        assert val >= _EXPERIENCE_MIN
        self._experience = val

    @property
    def time_stamp(self):
        return self._time_stamp

    @time_stamp.setter
    def time_stamp(self, val):
        assert val >= _TIME_STAMP_MIN
        self._time_stamp = val

    @property
    def action_set_size(self):
        return self._action_set_size

    @action_set_size.setter
    def action_set_size(self, val):
        assert val >= _ACTION_SET_SIZE_MIN
        self._action_set_size = val

    @property
    def numerosity(self):
        return self._numerosity

    @numerosity.setter
    def numerosity(self, val):
        assert val >= _NUMEROSITY_MIN
        self._numerosity = val

    def _init_weight_vec(self, condition):
        # linear prediction requires n+1 weights, where n is number of
        # predicates in condition
        n = len(condition)
        return np.zeros(shape=n+1, dtype=np.float32)

    def does_match(self, obs):
        return self._condition.does_match(obs)

    def prediction(self, obs):
        raise NotImplementedError

    def __eq__(self, other):
        return self._id == other._id
