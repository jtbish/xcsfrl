import abc

import numpy as np

from .augmentation import make_aug_strat
from .classifier import NLMSClassifier, RLSClassifier
from .hyperparams import get_hyperparam as get_hp

np.seterr(divide="raise", over="raise", invalid="raise")


class PredictionStrategyABC(metaclass=abc.ABCMeta):
    def __init__(self, poly_order):
        poly_order = int(poly_order)
        assert poly_order >= 1
        self._poly_order = poly_order
        self._aug_strat = make_aug_strat(self._poly_order)

    def make_classifier(self, condition, action, time_step):
        return self._CLFR_CLS(condition, action, time_step, self._poly_order)

    def aug_obs(self, obs, x_nought):
        return self._aug_strat(obs, x_nought)

    @abc.abstractmethod
    def process_aug_obs(self, aug_obs):
        raise NotImplementedError

    @abc.abstractmethod
    def update_prediction(self, clfr, payoff, aug_obs, proc_obs):
        raise NotImplementedError


class RecursiveLeastSquaresPrediction(PredictionStrategyABC):
    _CLFR_CLS = RLSClassifier

    def process_aug_obs(self, aug_obs):
        return np.reshape(aug_obs, (1, len(aug_obs)))  # row vector

    def update_prediction(self, clfr, payoff, aug_obs, proc_obs):
        tau_rls = get_hp("tau_rls")
        cov_mat_resets_allowed = (tau_rls > 0)
        if cov_mat_resets_allowed:
            should_reset_cov_mat = (clfr.experience % tau_rls == 0)
            if should_reset_cov_mat:
                clfr.reset_cov_mat()

        # optimal matrix parenthesisations pre-calced via DP
        # lambda_rls inclusion as per Butz et al. '08 Function approximation
        # with XCS: Hyperellipsoidal Conditions, Recursive Least Squares and
        # Compaction
        x = proc_obs
        lambda_rls = get_hp("lambda_rls")
        beta_rls = lambda_rls + (x @ (clfr.cov_mat @ x.T))
        clfr.cov_mat = (1 / lambda_rls) * (clfr.cov_mat - (1 / beta_rls) * (
            (clfr.cov_mat @ x.T) @ (x @ clfr.cov_mat)))
        gain_vec = np.dot(clfr.cov_mat, x.T)
        gain_vec = gain_vec.T
        assert gain_vec.shape == x.shape
        gain_vec = gain_vec[0]
        assert gain_vec.shape == clfr.weight_vec.shape

        error = payoff - clfr.prediction(aug_obs)
        clfr.weight_vec += (gain_vec * error)


class NormalisedLeastMeanSquaresPrediction(PredictionStrategyABC):
    _CLFR_CLS = NLMSClassifier

    def process_aug_obs(self, aug_obs):
        return np.sum(np.square(aug_obs))

    def update_prediction(self, clfr, payoff, aug_obs, proc_obs):
        """See Lanzi et al. '06 Generalistaion in the XCSF Classifier System:
        Analysis, Improvement, and Extension (ECJ) - Algorithm 2 for best
        description."""
        norm = proc_obs
        error = payoff - clfr.prediction(aug_obs)
        correction = (get_hp("eta") / norm) * error
        clfr.weight_vec += (aug_obs * correction)
