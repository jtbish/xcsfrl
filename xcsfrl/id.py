_curr_clfr_id = 0


def get_next_clfr_id():
    global _curr_clfr_id
    _curr_clfr_id += 1
    return _curr_clfr_id
