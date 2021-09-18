import numpy as np

from .hyperparams import get_hyperparam as get_hp
from .subsumption import action_set_subsumption
from .util import calc_num_micros

np.seterr(all="raise")

_MAX_ACC = 1.0


def update_action_set(action_set, payoff, obs, pop, pred_strat):
    use_niche_min_error = (get_hp("beta_epsilon") != 0)
    if use_niche_min_error:
        min_error_as = min([clfr.error for clfr in action_set])
    else:
        min_error_as = None

    as_num_micros = calc_num_micros(action_set)
    aug_obs = pred_strat.aug_obs(obs)
    proc_obs = pred_strat.process_aug_obs(aug_obs)

    for clfr in action_set:
        _update_experience(clfr)
        if use_niche_min_error:
            _update_niche_min_error(clfr, min_error_as)
            _update_error_with_mu(clfr, payoff, aug_obs)
        else:
            _update_error(clfr, payoff, aug_obs)
        pred_strat.update_prediction(clfr, payoff, aug_obs, proc_obs)
        _update_action_set_size(clfr, as_num_micros)
    _update_fitness(action_set)

    if get_hp("do_as_subsumption"):
        action_set_subsumption(action_set, pop)


def _update_experience(clfr):
    clfr.experience += 1


def _update_niche_min_error(clfr, min_error_as):
    beta_epsilon = get_hp("beta_epsilon")
    min_error_diff = (min_error_as - clfr.niche_min_error)
    if clfr.experience < (1 / beta_epsilon):
        clfr.niche_min_error += (min_error_diff / clfr.experience)
    else:
        clfr.niche_min_error += (beta_epsilon * min_error_diff)


def _update_error_with_mu(clfr, payoff, aug_obs):
    beta = get_hp("beta")
    payoff_diff = abs(payoff - clfr.prediction(aug_obs))
    # use scheme described in Lanzi '99 An Extension to XCS for Stochastic
    # Environments
    if (payoff_diff - clfr.niche_min_error) >= 0:
        error_target = (payoff_diff - clfr.niche_min_error - clfr.error)
    else:
        error_target = (get_hp("epsilon_nought") - clfr.error)

    if clfr.experience < (1 / beta):
        clfr.error += (error_target / clfr.experience)
    else:
        clfr.error += (beta * error_target)


def _update_error(clfr, payoff, aug_obs):
    beta = get_hp("beta")
    payoff_diff = abs(payoff - clfr.prediction(aug_obs))
    error_target = (payoff_diff - clfr.error)
    if clfr.experience < (1 / beta):
        clfr.error += (error_target / clfr.experience)
    else:
        clfr.error += (beta * error_target)


def _update_action_set_size(clfr, as_num_micros):
    beta = get_hp("beta")
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
