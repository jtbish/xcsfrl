import numpy as np

from .hyperparams import get_hyperparam as get_hp
from .subsumption import action_set_subsumption
from .util import calc_num_micros

_MAX_ACC = 1.0


def update_action_set(action_set, payoff, obs, pop, pop_ops_history,
                      pred_strat):
    _update_experience(action_set)
    aug_obs = pred_strat.aug_obs(obs)
    _update_prediction(action_set, payoff, aug_obs)
    use_niche_min_error = (get_hp("beta_epsilon") != 0)
    if use_niche_min_error:
        _update_niche_min_error(action_set)
    _update_error(action_set, payoff, aug_obs, use_niche_min_error)
    _update_action_set_size(action_set)
    _update_fitness(action_set)
    if get_hp("do_as_subsumption"):
        action_set_subsumption(action_set, pop, pop_ops_history)


def _update_experience(action_set):
    for clfr in action_set:
        clfr.experience += 1


def _update_prediction(action_set, payoff, aug_obs):
    """RLS prediction update."""
    x = np.reshape(aug_obs, (1, len(aug_obs)))  # row vector

    tau_rls = get_hp("tau_rls")
    cov_mat_resets_allowed = (tau_rls > 0)
    for clfr in action_set:
        if cov_mat_resets_allowed:
            should_reset_cov_mat = (clfr.experience % tau_rls == 0)
            if should_reset_cov_mat:
                clfr.reset_cov_mat()

        beta_rls = 1 + (x @ clfr.cov_mat @ x.T)
        clfr.cov_mat -= (1 /
                         beta_rls) * (clfr.cov_mat @ x.T @ x @ clfr.cov_mat)
        gain_vec = np.dot(clfr.cov_mat, x.T)
        gain_vec = gain_vec.T
        assert gain_vec.shape == x.shape
        gain_vec = gain_vec[0]
        assert gain_vec.shape == clfr.weight_vec.shape

        error = payoff - clfr.prediction(aug_obs)
        clfr.weight_vec += (gain_vec * error)


def _update_niche_min_error(action_set):
    min_error_as = min([clfr.error for clfr in action_set])
    for clfr in action_set:
        min_error_diff = (min_error_as - clfr.niche_min_error)
        clfr.niche_min_error += (get_hp("beta_epsilon") * min_error_diff)


def _update_error(action_set, payoff, aug_obs, use_niche_min_error):
    beta = get_hp("beta")
    for clfr in action_set:
        payoff_diff = abs(payoff - clfr.prediction(aug_obs))
        if use_niche_min_error:
            if (payoff_diff - clfr.niche_min_error) >= 0:
                error_target = (payoff_diff - clfr.niche_min_error -
                                clfr.error)
            else:
                error_target = (get_hp("epsilon_nought") - clfr.error)
        else:
            error_target = (payoff_diff - clfr.error)

        if clfr.experience < (1 / beta):
            clfr.error += (error_target / clfr.experience)
        else:
            clfr.error += (beta * error_target)


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
