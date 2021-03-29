import numpy as np

_rng = np.random.RandomState()
_has_been_seeded = False


def seed_rng(seed):
    seed = int(seed)
    _rng.seed(seed)
    global _has_been_seeded
    _has_been_seeded = True


def get_rng():
    assert _has_been_seeded
    return _rng
