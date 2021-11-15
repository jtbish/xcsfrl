def calc_num_macros(clfr_set):
    return len(clfr_set)


def calc_num_micros(clfr_set):
    return sum([clfr.numerosity for clfr in clfr_set])


def is_empty(clfr_set):
    return calc_num_macros(clfr_set) == 0
