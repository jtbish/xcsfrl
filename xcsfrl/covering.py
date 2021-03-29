from .classifier import Classifier


def find_actions_to_cover(match_set, action_space):
    # don't use theta_mna, instead make sure all actions are covered
    actions_covered_in_m = set([clfr.action for clfr in match_set])
    actions_to_cover = set(action_space) - actions_covered_in_m
    return actions_to_cover


def gen_covering_classifier(obs, encoding, action, time_step):
    condition = encoding.gen_covering_condition(obs)
    return Classifier(condition, action, time_step)
