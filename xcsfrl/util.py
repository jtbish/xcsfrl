from collections import OrderedDict
import numpy as np

from .hyperparams import get_hyperparam as get_hp


def filter_null_prediction_arr_entries(prediction_arr):
    return OrderedDict({a: p for (a, p) in prediction_arr.items() if p is
                       not None})


def augment_obs_vec(obs):
    return np.concatenate(([get_hp("x_nought")], obs))


def calc_num_micros(clfr_set):
    return sum([clfr.numerosity for clfr in clfr_set])


def calc_num_macros(clfr_set):
    return len(clfr_set)
