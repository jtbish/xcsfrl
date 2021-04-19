import numpy as np

from .hyperparams import get_hyperparam as get_hp
from .rng import get_rng

_EXPERIENCE_MIN = 0
_ACTION_SET_SIZE_MIN = 1
_NUMEROSITY_MIN = 1
_TIME_STAMP_MIN = 0
_ATTR_EQ_REL_TOL = 1e-10


class Classifier:
    def __init__(self, condition, action, time_step):
        """Only used by covering."""
        self._condition = condition
        self._action = action
        self._num_features = len(condition)
        self._weight_vec = self._init_weight_vec(self._num_features)
        self._cov_mat = self._init_cov_mat(self._num_features)
        self._niche_min_error = get_hp("mu_I")
        self._error = get_hp("epsilon_I")
        self._fitness = get_hp("fitness_I")
        self._experience = 0
        self._time_stamp = time_step
        self._action_set_size = 1
        self._numerosity = 1

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
    def cov_mat(self):
        return self._cov_mat

    @cov_mat.setter
    def cov_mat(self, val):
        self._cov_mat = val

    @property
    def niche_min_error(self):
        return self._niche_min_error

    @niche_min_error.setter
    def niche_min_error(self, val):
        self._niche_min_error = val

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, val):
        self._error = val

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, val):
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

    def _init_weight_vec(self, num_features):
        # linear prediction requires n+1 weights, n = num features
        low = get_hp("weight_I_min")
        high = get_hp("weight_I_max")
        assert low <= high
        return get_rng().uniform(low, high, size=(num_features+1))

    def _init_cov_mat(self, num_features):
        # cov mat is shape (n+1)x(n+1), n = num features
        return np.identity(n=(num_features + 1),
                           dtype=np.float32) * get_hp("delta_rls")

    def does_match(self, obs):
        return self._condition.does_match(obs)

    def prediction(self, aug_obs):
        return np.dot(aug_obs, self._weight_vec)

    def reset_cov_mat(self):
        self._cov_mat = self._init_cov_mat(self._num_features)

    def is_more_general(self, other):
        my_generality = self._condition.calc_generality()
        other_generality = other._condition.calc_generality()
        return ((my_generality > other_generality)
                and self._condition.does_subsume(other._condition))

    def __eq__(self, other):
        return (self._condition == other._condition
                and self._action == other._action
                and self._weight_vec_is_close(other)
                and self._cov_mat_is_close(other) and np.isclose(
                    self._error, other._error, rtol=_ATTR_EQ_REL_TOL)
                and np.isclose(
                    self._fitness, other._fitness, rtol=_ATTR_EQ_REL_TOL)
                and self._experience == other._experience
                and self._time_stamp == other._time_stamp
                and np.isclose(self._action_set_size,
                               other._action_set_size,
                               rtol=_ATTR_EQ_REL_TOL)
                and self._numerosity == other._numerosity)

    def _weight_vec_is_close(self, other):
        return np.all(
            np.isclose(self._weight_vec,
                       other._weight_vec,
                       rtol=_ATTR_EQ_REL_TOL))

    def _cov_mat_is_close(self, other):
        return np.all(
            np.isclose(self._cov_mat, other._cov_mat, rtol=_ATTR_EQ_REL_TOL))
