import copy

from .hyperparams import get_hyperparam as get_hp


def action_set_subsumption(action_set, pop):
    # find most general clfr in [A]
    most_general_clfr = None
    for clfr in action_set:
        if could_subsume(clfr):
            if (most_general_clfr is None
                    or clfr.is_more_general(most_general_clfr)):
                most_general_clfr = clfr

    # do the subsumptions if possible
    if most_general_clfr is not None:
        # iter over copy of [A] so can remove subsumees from actual [A] within
        # loop
        for clfr in copy.deepcopy(action_set):
            if most_general_clfr.does_subsume(clfr):
                num_micros_subsumed = clfr.numerosity
                pop.alter_numerosity(most_general_clfr,
                                     delta=num_micros_subsumed,
                                     op="as_subsumption")
                action_set.remove(clfr)
                pop.remove(clfr)


def does_subsume(subsumer, subsumee):
    """Determines if subsumer clfr really does subsume subsumee clfr."""
    return (could_subsume(subsumer) and subsumer.action == subsumee.action
            and subsumer.does_subsume(subsumee))


def could_subsume(clfr):
    return (clfr.experience > get_hp("theta_sub")
            and clfr.error < get_hp("epsilon_nought"))
