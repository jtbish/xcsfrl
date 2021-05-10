from collections import OrderedDict


def filter_null_prediction_arr_entries(prediction_arr):
    return OrderedDict({a: p for (a, p) in prediction_arr.items() if p is
                       not None})


def calc_num_micros(clfr_set):
    return sum([clfr.numerosity for clfr in clfr_set])


def calc_num_macros(clfr_set):
    return len(clfr_set)
