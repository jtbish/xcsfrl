import numpy as np

from .hyperparams import get_hyperparam as get_hp
from .subsumption import action_set_subsumption
from .util import augment_obs_vec, calc_num_micros

_MAX_ACC = 1.0


def update_action_set(action_set, payoff, obs, pop):
    _update_experience(action_set)
    _update_prediction(action_set, payoff, obs)
    _update_error(action_set, payoff, obs)
    _update_action_set_size(action_set)
    _update_fitness(action_set)
    if get_hp("do_as_subsumption"):
        action_set_subsumption(action_set, pop)


def _update_experience(action_set):
    for clfr in action_set:
        clfr.experience += 1


def _update_prediction(action_set, payoff, obs):
    """RLS prediction update."""
    x = augment_obs_vec(obs)
    x = np.reshape(x, (1, len(x)))  # (1, n+1) row vector, n = num features

    for clfr in action_set:
        should_reset_cov_mat = (clfr.experience % get_hp("tau_rls") == 0)
        if should_reset_cov_mat:
            clfr.reset_cov_mat()

        beta_rls = 1 + np.linalg.multi_dot((x, clfr.cov_mat, x.T))
        clfr.cov_mat -= (1 / beta_rls) * np.linalg.multi_dot(
            (clfr.cov_mat, x.T, x, clfr.cov_mat))
        gain_vec = np.dot(clfr.cov_mat, x.T)
        gain_vec = gain_vec.T
        assert gain_vec.shape == x.shape
        gain_vec = gain_vec.flatten()
        assert gain_vec.shape == clfr.weight_vec.shape

        error = payoff - clfr.prediction(obs)
        for i in range(0, len(clfr.weight_vec)):
            clfr.weight_vec[i] += gain_vec[i] * error


def _update_error(action_set, payoff, obs):
    beta = get_hp("beta")
    for clfr in action_set:
        payoff_diff = (abs(payoff - clfr.prediction(obs)) - clfr.error)
        if clfr.experience < (1 / beta):
            clfr.error += (payoff_diff / clfr.experience)
        else:
            clfr.error += (beta * payoff_diff)


def _update_action_set_size(action_set):
    as_num_micros = calc_num_micros(action_set)
    beta = get_hp("beta")
    for clfr in action_set:
        as_size_diff = (as_num_micros - clfr.action_set_size)
        if clfr.experience < (1 / beta):
            clfr.action_set_size += (as_size_diff / clfr.experience)
        else:
            clfr.action_set_size += (beta * as_size_diff)


def _update_fitness(action_set):
    acc_sum = 0
    acc_vec = []
    e_nought = get_hp("epsilon_nought")
    for clfr in action_set:
        if clfr.error < e_nought:
            acc = _MAX_ACC
        else:
            acc = (get_hp("alpha") *
                   (clfr.error / e_nought)**(-1 * get_hp("nu")))
        acc_vec.append(acc)
        acc_sum += (acc * clfr.numerosity)

    for (clfr, acc) in zip(action_set, acc_vec):
        relative_acc = (acc * clfr.numerosity / acc_sum)
        clfr.fitness += (get_hp("beta") * (relative_acc - clfr.fitness))
