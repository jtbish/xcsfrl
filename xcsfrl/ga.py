import copy
import logging

from .classifier import RLSClassifier
from .condition import Condition
from .deletion import deletion
from .hyperparams import get_hyperparam as get_hp
from .rng import get_rng
from .subsumption import does_subsume

_ERROR_CUTDOWN = 0.25
_NICHE_MIN_ERROR_CUTDOWN = _ERROR_CUTDOWN
_FITNESS_CUTDOWN = 0.1


def run_ga(action_set, pop, time_step, encoding, action_space):
    for clfr in action_set:
        clfr.time_stamp = time_step

    parent_a = _tournament_selection(action_set)
    parent_b = _tournament_selection(action_set)
    child_a = copy.deepcopy(parent_a)
    child_b = copy.deepcopy(parent_b)
    child_a.numerosity = 1
    child_b.numerosity = 1
    child_a.experience = 0
    child_b.experience = 0
    for child in (child_a, child_b):
        if isinstance(child, RLSClassifier):
            child.reset_cov_mat()

    do_crossover = get_rng().random() < get_hp("chi")
    if do_crossover:
        _uniform_crossover(child_a, child_b, encoding)

        avg_parent_niche_min_error = (parent_a.niche_min_error +
                                      parent_b.niche_min_error) / 2
        child_a.niche_min_error = avg_parent_niche_min_error
        child_b.niche_min_error = avg_parent_niche_min_error

        avg_parent_error = (parent_a.error + parent_b.error) / 2
        child_a.error = avg_parent_error
        child_b.error = avg_parent_error

        avg_parent_fitness = (parent_a.fitness + parent_b.fitness) / 2
        child_a.fitness = avg_parent_fitness
        child_b.fitness = avg_parent_fitness

    for child in (child_a, child_b):
        child.niche_min_error *= _NICHE_MIN_ERROR_CUTDOWN
        child.error *= _ERROR_CUTDOWN
        child.fitness *= _FITNESS_CUTDOWN
        _mutation(child, encoding, action_space)

        if get_hp("do_ga_subsumption"):
            if does_subsume(parent_a, child):
                pop.alter_numerosity(parent_a, delta=1, op="ga_subsumption")
            elif does_subsume(parent_b, child):
                pop.alter_numerosity(parent_b, delta=1, op="ga_subsumption")
            else:
                _insert_in_pop(pop, child)
        else:
            _insert_in_pop(pop, child)
        deletion(pop)


def _tournament_selection(action_set):
    """From Butz book 'Rule Based Evolutionary Online Learning Systems' SELECT
    OFFSPRING function in Appendix B."""
    best = None
    while best is None:
        max_fitness = 0
        for clfr in action_set:
            if clfr.numerosity_scaled_fitness > max_fitness:
                for _ in range(clfr.numerosity):
                    if get_rng().random() < get_hp("tau"):
                        best = clfr
                        max_fitness = clfr.numerosity_scaled_fitness
                        break
    return best


def _uniform_crossover(child_a, child_b, encoding):
    """Uniform crossover on condition allele seqs."""
    a_cond_alleles = child_a.condition.alleles
    b_cond_alleles = child_b.condition.alleles
    assert len(a_cond_alleles) == len(b_cond_alleles)
    n = len(a_cond_alleles)

    def _swap(seq_a, seq_b, idx):
        seq_a[idx], seq_b[idx] = seq_b[idx], seq_a[idx]

    for idx in range(0, n):
        if get_rng().random() < get_hp("upsilon"):
            _swap(a_cond_alleles, b_cond_alleles, idx)

    a_new_cond = Condition(a_cond_alleles, encoding)
    b_new_cond = Condition(b_cond_alleles, encoding)
    child_a.condition = a_new_cond
    child_b.condition = b_new_cond


def _mutation(child, encoding, action_space):
    _mutate_condition(child, encoding)
    _mutate_action(child, action_space)


def _mutate_condition(child, encoding):
    mut_cond_alleles = encoding.mutate_condition_alleles(
        child.condition.alleles)
    # make and set new Condition obj so phenotypes are properly pre-calced
    # and cached
    new_cond = Condition(mut_cond_alleles, encoding)
    child.condition = new_cond


def _mutate_action(child, action_space):
    should_mut_action = get_rng().random() < get_hp("mu")
    if should_mut_action:
        other_actions = list(set(action_space) - {child.action})
        mut_action = get_rng().choice(other_actions)
        child.action = mut_action


def _insert_in_pop(pop, child):
    for clfr in pop:
        # check action first to potentially short circuit more expensive
        # condition check
        if (clfr.action == child.action and clfr.condition == child.condition):
            pop.alter_numerosity(clfr, delta=1, op="absorption")
            return
    pop.add_new(child, op="insertion")
