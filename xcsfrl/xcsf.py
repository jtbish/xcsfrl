import logging
from collections import OrderedDict

from .action_selection import NULL_ACTION, filter_null_prediction_arr_entries
from .covering import calc_num_unique_actions, gen_covering_classifier
from .deletion import deletion
from .ga import run_ga
from .hyperparams import get_hyperparam as get_hp
from .hyperparams import register_hyperparams
from .param_update import update_action_set
from .population import Population
from .rng import seed_rng


class XCSF:
    def __init__(self, env, encoding, action_selection_strat, pred_strat,
                 hyperparams_dict):
        self._env = env
        self._encoding = encoding
        self._action_selection_strat = action_selection_strat
        self._pred_strat = pred_strat
        register_hyperparams(hyperparams_dict)
        seed_rng(get_hp("seed"))

        self._pop = Population()
        self._prev_action_set = None
        self._prev_reward = None
        self._prev_obs = None
        self._curr_obs = None
        self._time_step = 0

    @property
    def pop(self):
        return self._pop

    def train(self, num_steps):
        # restart episode or resume where left off
        # prime the current obs
        if self._curr_obs is None:
            assert self._env.is_terminal()
            self._curr_obs = self._env.reset()

        steps_done = 0
        while steps_done < num_steps:
            self._run_step()
            if self._env.is_terminal():
                assert self._curr_obs is None
                self._curr_obs = self._env.reset()
            steps_done += 1

    def _run_step(self):
        obs = self._curr_obs
        match_set = self._gen_match_set_and_cover(obs)
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
                              self._pop, self._pred_strat)
            run_ga(self._prev_action_set,
                   self._pop,
                   self._time_step,
                   self._encoding,
                   self._env.action_space,
                   active_clfr_set=action_set)  # might need to delete from [A]
        if is_terminal:
            payoff = reward
            update_action_set(action_set, payoff, obs, self._pop,
                              self._pred_strat)
            run_ga(action_set,
                   self._pop,
                   self._time_step,
                   self._encoding,
                   self._env.action_space,
                   active_clfr_set=None)  # no point in deleting from [A]
            self._prev_action_set = None
            self._prev_reward = None
            self._prev_obs = None
            self._curr_obs = None
        else:
            self._prev_action_set = action_set
            self._prev_reward = reward
            self._prev_obs = obs
            self._curr_obs = next_obs
        self._time_step += 1

    def _gen_match_set_and_cover(self, obs):
        match_set = self._gen_match_set(obs)
        theta_mna = len(self._env.action_space)  # always cover all actions
        while (calc_num_unique_actions(match_set) < theta_mna):
            clfr = gen_covering_classifier(obs, self._encoding, match_set,
                                           self._env.action_space,
                                           self._time_step, self._pred_strat)
            self._pop.add_new(clfr, op="covering")
            # might also need to delete from [M]
            deletion(self._pop, active_clfr_set=match_set)
            match_set.append(clfr)
        return match_set

    def _gen_match_set(self, obs):
        return [clfr for clfr in self._pop if clfr.does_match(obs)]

    def _gen_prediction_arr(self, match_set, obs):
        aug_obs = self._pred_strat.aug_obs(obs)

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
        """Action selection for training - use action selection strat."""
        return self._action_selection_strat(prediction_arr, self._time_step)

    def _gen_action_set(self, match_set, action):
        return [clfr for clfr in match_set if clfr.action == action]

    def select_action(self, obs):
        """Action selection for testing - always exploit"""
        match_set = self._gen_match_set(obs)
        if len(match_set) > 0:
            prediction_arr = self._gen_prediction_arr(match_set, obs)
            prediction_arr = filter_null_prediction_arr_entries(prediction_arr)
            # greedy action selection
            return max(prediction_arr, key=prediction_arr.get)
        else:
            return NULL_ACTION

    def gen_prediction_arr(self, obs):
        """Q-value calculation for outside probing."""
        match_set = self._gen_match_set(obs)
        return self._gen_prediction_arr(match_set, obs)
