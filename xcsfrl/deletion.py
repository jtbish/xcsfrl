from .hyperparams import get_hyperparam as get_hp
from .rng import get_rng

_MIN_NUM_MACROS = 1


def deletion(pop):
    max_pop_size = get_hp("N")
    pop_size = pop.num_micros
    num_to_delete = max(0, (pop_size - max_pop_size))
    if num_to_delete > 0:
        for _ in range(num_to_delete):
            _delete_single_microclfr(pop)
        assert pop.num_macros >= _MIN_NUM_MACROS
        assert pop.num_micros <= max_pop_size


def _delete_single_microclfr(pop):
    avg_fitness_in_pop = sum([clfr.fitness
                              for clfr in pop]) / pop.num_micros
    # since pop is ordered seq. can get away with caching votes at start and
    # iterating in order to do roulette spin
    votes = [_deletion_vote(clfr, avg_fitness_in_pop) for clfr in pop]
    choice_point = get_rng().random() * sum(votes)
    vote_sum = 0
    clfr_to_remove = None
    for (clfr, vote) in zip(pop, votes):
        vote_sum += vote
        if vote_sum > choice_point:
            if clfr.numerosity > 1:
                pop.alter_numerosity(clfr, delta=-1, op="deletion")
            elif clfr.numerosity == 1:
                clfr_to_remove = clfr
            else:
                # not possible
                assert False
            break

    if clfr_to_remove is not None:
        pop.remove(clfr_to_remove, op="deletion")


def _deletion_vote(clfr, avg_fitness_in_pop):
    vote = clfr.action_set_size * clfr.numerosity
    has_sufficient_exp = clfr.experience > get_hp("theta_del")
    scaled_fitness = clfr.fitness / clfr.numerosity
    has_low_fitness = scaled_fitness < (get_hp("delta") * avg_fitness_in_pop)
    should_increase_vote = has_sufficient_exp and has_low_fitness
    if should_increase_vote:
        vote *= (avg_fitness_in_pop / scaled_fitness)
    return vote
