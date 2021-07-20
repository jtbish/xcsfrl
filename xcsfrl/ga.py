import copy
import math

from .condition import Condition
from .deletion import deletion
from .hyperparams import get_hyperparam as get_hp
from .rng import get_rng
from .subsumption import does_subsume
from .util import calc_num_macros, calc_num_micros

_ERROR_CUTDOWN = 0.25
_FITNESS_CUTDOWN = 0.1


def run_ga(action_set, pop, time_step, encoding, action_space,
           active_clfr_set):
    avg_time_stamp_in_as = sum(
        [clfr.time_stamp * clfr.numerosity
         for clfr in action_set]) / calc_num_micros(action_set)
    should_apply_ga = ((time_step - avg_time_stamp_in_as) > get_hp("theta_ga"))
    if should_apply_ga:
        _run_ga(action_set, pop, time_step, encoding, action_space,
                active_clfr_set)


def _run_ga(action_set, pop, time_step, encoding, action_space,
            active_clfr_set):
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

    do_crossover = get_rng().random() < get_hp("chi")
    if do_crossover:
        _uniform_crossover(child_a, child_b, encoding)

        child_error = _ERROR_CUTDOWN * (parent_a.error + parent_b.error) / 2
        child_a.error = child_error
        child_b.error = child_error

        child_fitness = _FITNESS_CUTDOWN * (parent_a.fitness +
                                            parent_b.fitness) / 2
        child_a.fitness = child_fitness
        child_b.fitness = child_fitness

        child_niche_min_error = _ERROR_CUTDOWN * (parent_a.niche_min_error +
                                                  parent_b.niche_min_error) / 2
        child_a.niche_min_error = child_niche_min_error
        child_b.niche_min_error = child_niche_min_error

    for child in (child_a, child_b):
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
        deletion(pop, active_clfr_set)


def _tournament_selection(action_set):
    def _select_random(action_set):
        idx = get_rng().randint(0, len(action_set))
        return action_set[idx]

    as_size = calc_num_macros(action_set)
    tourn_size = math.ceil(get_hp("tau") * as_size)
    assert 1 <= tourn_size <= as_size

    best = _select_random(action_set)
    for _ in range(2, (tourn_size + 1)):
        clfr = _select_random(action_set)
        if clfr.fitness > best.fitness:
            best = clfr
    return best


def _two_point_crossover(child_a, child_b, encoding):
    """Two point crossover on condition allele seqs."""
    a_cond_alleles = child_a.condition.alleles
    b_cond_alleles = child_b.condition.alleles
    assert len(a_cond_alleles) == len(b_cond_alleles)
    n = len(a_cond_alleles)

    first = get_rng().randint(0, n + 1)
    second = get_rng().randint(0, n + 1)
    cut_start_idx = min(first, second)
    cut_end_idx = max(first, second)

    for idx in range(cut_start_idx, cut_end_idx):
        _swap(a_cond_alleles, b_cond_alleles, idx)

    # make and set new Condition objs so phenotypes are properly pre-calced
    # and cached
    a_new_cond = Condition(a_cond_alleles, encoding)
    b_new_cond = Condition(b_cond_alleles, encoding)
    child_a.condition = a_new_cond
    child_b.condition = b_new_cond


def _uniform_crossover(child_a, child_b, encoding):
    """Uniform crossover on condition allele seqs."""
    a_cond_alleles = child_a.condition.alleles
    b_cond_alleles = child_b.condition.alleles
    assert len(a_cond_alleles) == len(b_cond_alleles)
    n = len(a_cond_alleles)

    for idx in range(0, n):
        if get_rng().random() < get_hp("upsilon"):
            _swap(a_cond_alleles, b_cond_alleles, idx)

    a_new_cond = Condition(a_cond_alleles, encoding)
    b_new_cond = Condition(b_cond_alleles, encoding)
    child_a.condition = a_new_cond
    child_b.condition = b_new_cond


def _swap(seq_a, seq_b, idx):
    seq_a[idx], seq_b[idx] = seq_b[idx], seq_a[idx]


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
