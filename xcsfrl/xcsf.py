import logging
from collections import OrderedDict

from .action_selection import (NULL_ACTION, ActionSelectionModes,
                               choose_action_selection_mode,
                               filter_null_prediction_arr_entries,
                               greedy_action_selection)
from .covering import calc_num_unique_actions, gen_covering_classifier
from .deletion import deletion
from .ga import run_ga
from .hyperparams import get_hyperparam as get_hp
from .hyperparams import register_hyperparams
from .param_update import update_action_set
from .population import Population
from .rng import seed_rng
from .util import calc_num_micros


class XCSF:
    def __init__(self, env, encoding, action_selection_strat, pred_strat,
                 hyperparams_dict):
        self._env = env
        self._encoding = encoding
        self._action_selection_strat = action_selection_strat
        self._pred_strat = pred_strat
        self._hyperparams_dict = hyperparams_dict
        register_hyperparams(self._hyperparams_dict)
        seed_rng(get_hp("seed"))
        # cache x_nought so can use it after pickling to do predictions without
        # re-registering hyperparams
        self._x_nought = get_hp("x_nought")

        self._pop = Population()
        self._prev_action_set = None
        self._prev_reward = None
        self._prev_obs = None
        self._curr_obs = None
        self._time_step = 0
        self._episodes_trained = 0
        self._num_ga_calls = 0

    @property
    def pop(self):
        return self._pop

    def train_for_time_steps(self, num_steps):
        # restart episode or resume where left off
        # prime the current obs
        if self._curr_obs is None:
            assert self._env.is_terminal()
            self._curr_obs = self._env.reset()
            self._action_selection_mode = choose_action_selection_mode()

        steps_done = 0
        while steps_done < num_steps:
            self._run_step()
            if self._env.is_terminal():
                assert self._curr_obs is None
                self._curr_obs = self._env.reset()
                self._action_selection_mode = choose_action_selection_mode()
            steps_done += 1

    def train_for_episodes(self, num_episodes):
        # should always be in terminal state when starting this func
        assert self._curr_obs is None
        assert self._env.is_terminal()

        for _ in range(num_episodes):
            self._curr_obs = self._env.reset()
            self._action_selection_mode = choose_action_selection_mode()
            while not self._env.is_terminal():
                self._run_step()
            self._episodes_trained += 1

    def train_for_ga_calls(self, num_ga_calls):
        # restart episode or resume where left off
        # prime the current obs
        if self._curr_obs is None:
            assert self._env.is_terminal()
            self._curr_obs = self._env.reset()
            self._action_selection_mode = choose_action_selection_mode()

        curr_num_ga_calls = self._num_ga_calls
        target_num_ga_calls = (curr_num_ga_calls + num_ga_calls)
        while (self._num_ga_calls < target_num_ga_calls):
            self._run_step()
            if self._env.is_terminal():
                assert self._curr_obs is None
                self._curr_obs = self._env.reset()
                self._action_selection_mode = choose_action_selection_mode()

    def _run_step(self):
        obs = self._curr_obs
        match_set = self._gen_match_set_and_cover(obs)
        prediction_arr = self._gen_prediction_arr(match_set, obs)
        action = self._select_action(prediction_arr)
        action_set = self._gen_action_set(match_set, action)
        (next_obs, reward, is_terminal, _) = self._env.step(action)
        if self._prev_action_set is not None:
            assert self._prev_reward is not None
            assert self._prev_obs is not None
            prediction_arr = filter_null_prediction_arr_entries(prediction_arr)
            payoff = self._prev_reward + get_hp("gamma") * \
                max(prediction_arr.values())
            update_action_set(self._prev_action_set, payoff, self._prev_obs,
                              self._pop, self._pred_strat, self._x_nought)
            self._try_run_ga(self._prev_action_set, self._pop, self._time_step,
                             self._encoding, self._env.action_space)
        if is_terminal:
            payoff = reward
            update_action_set(action_set, payoff, obs, self._pop,
                              self._pred_strat, self._x_nought)
            self._try_run_ga(action_set, self._pop, self._time_step,
                             self._encoding, self._env.action_space)
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
            deletion(self._pop)
            match_set.append(clfr)
        return match_set

    def _gen_match_set(self, obs):
        return [clfr for clfr in self._pop if clfr.does_match(obs)]

    def _gen_prediction_arr(self, match_set, obs):
        aug_obs = self._pred_strat.aug_obs(obs, self._x_nought)

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
        if self._action_selection_mode == ActionSelectionModes.explore:
            return self._action_selection_strat(
                prediction_arr, num_ga_calls=self._num_ga_calls)
        elif self._action_selection_mode == ActionSelectionModes.exploit:
            return greedy_action_selection(prediction_arr)
        else:
            assert False

    def _gen_action_set(self, match_set, action):
        return [clfr for clfr in match_set if clfr.action == action]

    def _try_run_ga(self, action_set, pop, time_step, encoding, action_space):
        # GA can only be active on exploration episodes/"problems"
        if self._action_selection_mode == ActionSelectionModes.explore:
            avg_time_stamp_in_as = sum(
                [clfr.time_stamp * clfr.numerosity
                 for clfr in action_set]) / calc_num_micros(action_set)
            should_apply_ga = ((time_step - avg_time_stamp_in_as) >
                               get_hp("theta_ga"))
            if should_apply_ga:
                run_ga(action_set, pop, time_step, encoding, action_space)
                self._num_ga_calls += 1

    def select_action(self, obs):
        """Action selection for outside testing - always exploit"""
        match_set = self._gen_match_set(obs)
        if len(match_set) > 0:
            prediction_arr = self._gen_prediction_arr(match_set, obs)
            return greedy_action_selection(prediction_arr)
        else:
            return NULL_ACTION

    def gen_prediction_arr(self, obs):
        """Q-value calculation for outside probing."""
        match_set = self._gen_match_set(obs)
        return self._gen_prediction_arr(match_set, obs)
