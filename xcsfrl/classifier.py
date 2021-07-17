import numpy as np

from .hyperparams import get_hyperparam as get_hp
from .rng import get_rng

_EXPERIENCE_MIN = 0
_ACTION_SET_SIZE_MIN = 1
_NUMEROSITY_MIN = 1
_TIME_STAMP_MIN = 0
_ATTR_EQ_REL_TOL = 1e-10


class ClassifierBase:
    def __init__(self, condition, action, time_step, poly_order):
        """Only used by covering."""
        self._condition = condition
        self._action = action
        self._num_features = len(condition)
        self._poly_order = poly_order
        self._weight_vec = self._init_weight_vec(self._num_features,
                                                 self._poly_order)
        self._niche_min_error = get_hp("mu_I")
        self._error = get_hp("epsilon_I")
        self._fitness = get_hp("fitness_I")
        self._experience = 0
        self._time_stamp = time_step
        self._action_set_size = 1
        self._numerosity = 1

        # "reactive"/calculated params for deletion
        self._deletion_vote = self._calc_deletion_vote(self._action_set_size,
                                                       self._numerosity)
        self._deletion_has_sufficient_exp = \
            self._calc_deletion_has_sufficient_exp(self._experience)
        self._deletion_numerosity_scaled_fitness = \
            self._calc_deletion_numerosity_scaled_fitness(self._fitness,
                                                          self._numerosity)

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
        self._deletion_numerosity_scaled_fitness = \
            self._calc_deletion_numerosity_scaled_fitness(self._fitness,
                                                          self._numerosity)

    @property
    def experience(self):
        return self._experience

    @experience.setter
    def experience(self, val):
        assert val >= _EXPERIENCE_MIN
        self._experience = val
        self._deletion_has_sufficient_exp = \
            self._calc_deletion_has_sufficient_exp(self._experience)

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
        self._deletion_vote = self._calc_deletion_vote(self._action_set_size,
                                                       self._numerosity)

    @property
    def numerosity(self):
        return self._numerosity

    @numerosity.setter
    def numerosity(self, val):
        assert val >= _NUMEROSITY_MIN
        self._numerosity = val
        self._deletion_vote = self._calc_deletion_vote(self._action_set_size,
                                                       self._numerosity)
        self._deletion_numerosity_scaled_fitness = \
            self._calc_deletion_numerosity_scaled_fitness(self._fitness,
                                                          self._numerosity)

    @property
    def deletion_vote(self):
        return self._deletion_vote

    @property
    def deletion_has_sufficient_exp(self):
        return self._deletion_has_sufficient_exp

    @property
    def deletion_numerosity_scaled_fitness(self):
        return self._deletion_numerosity_scaled_fitness

    def _init_weight_vec(self, num_features, poly_order):
        # weight vec is of len k*n+1, k = poly order, n = num features
        low = get_hp("weight_I_min")
        high = get_hp("weight_I_max")
        assert low <= high
        return get_rng().uniform(low,
                                 high,
                                 size=(poly_order * num_features + 1)).astype(
                                     np.float32)

    def _calc_deletion_vote(self, action_set_size, numerosity):
        return action_set_size * numerosity

    def _calc_deletion_has_sufficient_exp(self, experience):
        return experience > get_hp("theta_del")

    def _calc_deletion_numerosity_scaled_fitness(self, fitness, numerosity):
        return fitness / numerosity

    def does_match(self, obs):
        return self._condition.does_match(obs)

    def does_subsume(self, other):
        return self._condition.does_subsume(other._condition)

    def is_more_general(self, other):
        return self._condition.generality > other._condition.generality

    def prediction(self, aug_obs):
        return np.dot(aug_obs, self._weight_vec)

    def __eq__(self, other):
        # Fast version of eq: (condition, action) pair must be unique for all
        # macroclassifiers. Sufficient for removal checks.
        return (self._action == other._action) and (self._condition
                                                    == other._condition)

    def full_eq(self, other):
        # full version of eq: check all non-calculated params
        return (self._condition == other._condition
                and self._action == other._action
                and self._weight_vec_is_close(other)
                and np.isclose(self._niche_min_error,
                               other._niche_min_error,
                               rtol=_ATTR_EQ_REL_TOL) and
                np.isclose(self._error, other._error, rtol=_ATTR_EQ_REL_TOL)
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


class RLSClassifier(ClassifierBase):
    """Classifier for Recursive Least Squares prediction. In addition to
    standard weight vec, also has cov mat."""
    def __init__(self, condition, action, time_step, poly_order):
        super().__init__(condition, action, time_step, poly_order)
        self._cov_mat = self._init_cov_mat(self._num_features,
                                           self._poly_order)

    @property
    def cov_mat(self):
        return self._cov_mat

    @cov_mat.setter
    def cov_mat(self, val):
        self._cov_mat = val

    def _init_cov_mat(self, num_features, poly_order):
        # cov mat is of shape (k*n+1)x(k*n+1), k = poly order, n = num features
        return np.identity(n=(poly_order * num_features + 1),
                           dtype=np.float32) * get_hp("delta_rls")

    def reset_cov_mat(self):
        self._cov_mat = self._init_cov_mat(self._num_features,
                                           self._poly_order)

    def full_eq(self, other):
        return super().full_eq(other) and self._cov_mat_is_close(other)

    def _cov_mat_is_close(self, other):
        return np.all(
            np.isclose(self._cov_mat, other._cov_mat, rtol=_ATTR_EQ_REL_TOL))


class NLMSClassifier(ClassifierBase):
    pass
