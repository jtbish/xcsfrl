_hyperparams_registry = {}


def register_hyperparams(hyperparams_dict):
    global _hyperparams_registry
    _hyperparams_registry = {**_hyperparams_registry, **hyperparams_dict}


def get_hyperparam(name):
    return _hyperparams_registry[name]
