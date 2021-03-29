from collections import OrderedDict

from .covering import find_actions_to_cover, gen_covering_classifier
from .deletion import deletion
from .hyperparams import register_hyperparams


class XCSF:
    def __init__(self, env, encoding, action_selection_strat,
                 hyperparams_dict):
        self._env = env
        self._encoding = encoding
        self._action_selection_strat = action_selection_strat
        register_hyperparams(hyperparams_dict)

        self._pop = []
        self._prev_action_set = None
        self._prev_reward = None
        self._prev_obs = None
        self._time_step = 0

    def run_episode():
        pass

    def _gen_match_set(self, obs):
        match_set = [clfr for clfr in self._pop if clfr.does_match(obs)]
        actions_to_cover = find_actions_to_cover(match_set,
                                                 self._env.action_space)
        for action in actions_to_cover:
            clfr = gen_covering_classifier(obs, self._encoding, action,
                                           self._time_step)
            self._pop.append(clfr)
            deletion(self._pop)
            match_set.append(clfr)
        return match_set

    def _gen_prediction_arr(self, match_set, obs):
        prediction_arr = OrderedDict(
            {action: None
             for action in self._env.action_space})
        actions_reprd_in_m = set([clfr.action for clfr in match_set])
        for a in actions_reprd_in_m:
            prediction_arr[a] = 0.0

        fitness_sum_arr = OrderedDict(
            {action: 0.0
             for action in self._env.action_space})

        for clfr in match_set:
            a = clfr.action
            prediction_arr[a] += clfr.prediction(obs) * clfr.fitness
            fitness_sum_arr[a] += clfr.fitness

        for a in self._env.action_space:
            if fitness_sum_arr[a] != 0.0:
                prediction_arr[a] /= fitness_sum_arr[a]
        return prediction_arr

    def _select_action(self, prediction_arr):
        self._action_selection_strat(prediction_arr, self._env.action_space)
