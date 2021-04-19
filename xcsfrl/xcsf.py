import logging
from collections import OrderedDict

from .covering import find_actions_to_cover, gen_covering_classifier
from .deletion import deletion
from .error import NoActionError
from .ga import run_ga
from .hyperparams import get_hyperparam as get_hp
from .hyperparams import register_hyperparams
from .param_update import update_action_set
from .rng import seed_rng
from .util import augment_obs_vec, filter_null_prediction_arr_entries


class XCSF:
    def __init__(self, env, encoding, action_selection_strat,
                 hyperparams_dict):
        self._env = env
        self._encoding = encoding
        self._action_selection_strat = action_selection_strat
        register_hyperparams(hyperparams_dict)
        seed_rng(get_hp("seed"))

        self._pop = []
        self._prev_action_set = None
        self._prev_reward = None
        self._prev_obs = None
        self._time_step = 0

    def train(self, num_steps):
        # restart episode or resume where left off
        if self._env.is_terminal():
            obs = self._env.reset()
        else:
            assert self._prev_obs is not None
            obs = self._prev_obs

        steps_done = 0
        while steps_done < num_steps:
            obs = self._run_step(obs)
            if self._env.is_terminal():
                obs = self._env.reset()
            steps_done += 1

        return self._pop

    def _run_step(self, obs):
        match_set = self._gen_match_set(obs)
        prediction_arr = self._gen_prediction_arr(match_set, obs)
        action = self._select_action(prediction_arr)
        action_set = self._gen_action_set(match_set, action)
        (next_obs, reward, is_terminal) = self._env.step(action)
        if self._prev_action_set is not None:
            assert self._prev_reward is not None
            assert self._prev_obs is not None
            prediction_arr = filter_null_prediction_arr_entries(prediction_arr)
            payoff = self._prev_reward + get_hp("gamma") * \
                max(prediction_arr.values())
            update_action_set(self._prev_action_set, payoff, self._prev_obs,
                              self._pop)
            run_ga(self._prev_action_set, self._pop, self._time_step,
                   self._encoding, self._env.action_space)
        if is_terminal:
            payoff = reward
            update_action_set(action_set, payoff, obs, self._pop)
            run_ga(action_set, self._pop, self._time_step, self._encoding,
                   self._env.action_space)
            self._prev_action_set = None
        else:
            self._prev_action_set = action_set
            self._prev_reward = reward
            self._prev_obs = obs
        self._time_step += 1
        return next_obs

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
        aug_obs = augment_obs_vec(obs)

        prediction_arr = OrderedDict(
            {action: None
             for action in self._env.action_space})
        actions_reprd_in_m = set([clfr.action for clfr in match_set])
        for a in actions_reprd_in_m:
            # to bootstap sum below
            prediction_arr[a] = 0

        fitness_sum_arr = OrderedDict(
            {action: 0
             for action in self._env.action_space})

        for clfr in match_set:
            a = clfr.action
            prediction_arr[a] += clfr.prediction(aug_obs) * clfr.fitness
            fitness_sum_arr[a] += clfr.fitness

        for a in self._env.action_space:
            if fitness_sum_arr[a] != 0:
                prediction_arr[a] /= fitness_sum_arr[a]
        return prediction_arr

    def _select_action(self, prediction_arr):
        return self._action_selection_strat(prediction_arr, self._time_step)

    def _gen_action_set(self, match_set, action):
        return [clfr for clfr in match_set if clfr.action == action]

    def select_action(self, obs):
        """Action selection for testing - always exploit"""
        match_set = [clfr for clfr in self._pop if clfr.does_match(obs)]
        if len(match_set) > 0:
            prediction_arr = self._gen_prediction_arr(match_set, obs)
            prediction_arr = filter_null_prediction_arr_entries(prediction_arr)
            # greedy action selection
            return max(prediction_arr, key=prediction_arr.get)
        else:
            raise NoActionError

    def gen_prediction_arr(self, obs):
        """Q-value calculation for outside probing."""
        match_set = [clfr for clfr in self._pop if clfr.does_match(obs)]
        return self._gen_prediction_arr(match_set, obs)
