from .rng import get_rng


def calc_num_unique_actions(match_set):
    return len(set([clfr.action for clfr in match_set]))


def gen_covering_classifier(obs, encoding, match_set, action_space, time_step,
                            pred_strat):
    condition = encoding.gen_covering_condition(obs)
    actions_to_cover = _find_actions_to_cover(match_set, action_space)
    action = get_rng().choice(actions_to_cover)
    return pred_strat.make_classifier(condition, action, time_step)


def _find_actions_to_cover(match_set, action_space):
    actions_covered_in_m = set([clfr.action for clfr in match_set])
    actions_to_cover = list(set(action_space) - actions_covered_in_m)
    return actions_to_cover
