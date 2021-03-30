from .hyperparams import get_hyperparam as get_hp
from .rng import get_rng
from .util import calc_num_micros, calc_num_macros

_MIN_NUM_MACROS = 1


def deletion(pop):
    max_pop_size = get_hp("N")
    pop_size = calc_num_micros(pop)
    num_to_delete = max(0, (pop_size - max_pop_size))
    for _ in range(num_to_delete):
        _delete_single_microclfr(pop)
    assert calc_num_macros(pop) >= _MIN_NUM_MACROS
    assert calc_num_micros(pop) <= max_pop_size


def _delete_single_microclfr(pop):
    # since pop is ordered seq. can get away with caching votes at start and
    # iterating in order to do roulette spin
    votes = [_deletion_vote(clfr, pop) for clfr in pop]
    choice_point = get_rng().random() * sum(votes)
    vote_sum = 0
    for (clfr, vote) in zip(pop, votes):
        vote_sum += vote
        if vote_sum > choice_point:
            if clfr.numerosity > 1:
                clfr.numerosity -= 1
            elif clfr.numerosity == 1:
                pop.remove(clfr)
            else:
                # not possible
                assert False
            break


def _deletion_vote(clfr, pop):
    vote = clfr.action_set_size * clfr.numerosity
    avg_fitness_in_pop = sum([clfr.fitness
                              for clfr in pop]) / calc_num_micros(pop)
    has_sufficient_exp = clfr.experience > get_hp("theta_del")
    scaled_fitness = clfr.fitness / clfr.numerosity
    has_low_fitness = scaled_fitness < (get_hp("delta") * avg_fitness_in_pop)
    should_increase_vote = has_sufficient_exp and has_low_fitness
    if should_increase_vote:
        vote *= (avg_fitness_in_pop / scaled_fitness)
    return vote
