from .hyperparams import get_hyperparam as get_hp
import copy


def action_set_subsumption(action_set, pop):
    # find most general clfr in [A]
    most_general_clfr = None
    for clfr in action_set:
        if could_subsume(clfr):
            if (most_general_clfr is None or
                    clfr.is_more_general(most_general_clfr)):
                most_general_clfr = clfr

    # do the subsumptions if possible
    if most_general_clfr is not None:
        # iter over copy of [A] so can remove subsumees from actual [A] within
        # loop
        for clfr in copy.deepcopy(action_set):
            if most_general_clfr.is_more_general(clfr):
                most_general_clfr.numerosity += clfr.numerosity
                action_set.remove(clfr)
                pop.remove(clfr)


def does_subsume(subsumer, subsumee):
    """Determines if subsumer clfr really does subsume subsumee clfr."""
    return (subsumer.action == subsumee.action and could_subsume(subsumer)
            and subsumer.is_more_general(subsumee))


def could_subsume(clfr):
    return (clfr.experience > get_hp("theta_sub")
            and clfr.error < get_hp("epsilon_nought"))
