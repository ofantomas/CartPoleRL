# Environment definitions for tabular CCA
from abc import ABC, abstractmethod
import random
import numpy as np


# Parent class for tabular mdps
class AbstractTabEnv(ABC):
    def __init__(self, n_states, n_actions, initial_pos):
        self.n_states = n_states
        self.n_actions = n_actions
        self.initial_pos = initial_pos
        self.current_pos = initial_pos

        self.transitions = None
        self.rewards = None
        self.dones = None
        self.possible_r_values = None
        self.init_env_data()

    @abstractmethod
    def init_env_data(self):
        pass

    def reset(self, new_initial_pos=None):
        if new_initial_pos is not None:
            self.initial_pos = new_initial_pos
            self.current_pos = new_initial_pos
        else:
            self.current_pos = int(self.initial_pos)  # Cast to avoid shallow copy pitfalls
        return self.current_pos

    def is_done(self):
        return self.dones[self.current_pos]

    # TODO consider allowing noisy transitions
    def step(self, act):
        if self.transitions[self.current_pos, act] is not None:
            self.current_pos = self.transitions[self.current_pos, act]
        reward = self.rewards[self.current_pos]

        return self.current_pos, reward, self.is_done()


class AbstractGridEnv(AbstractTabEnv):
    @abstractmethod
    def __init__(self, n_states, n_actions, initial_pos):
        self.state_map = None
        self.done_map = None
        self.reward_map = None
        super().__init__(n_states, n_actions, initial_pos)

    @abstractmethod
    def init_maps(self):
        pass

    @staticmethod
    def apply_action(i, j, action):
        if action == 0:
            return i - 1, j
        if action == 1:
            return i, j + 1
        if action == 2:
            return i + 1, j
        if action == 3:
            return i, j - 1
        raise Exception(f"No such action: {action}!")

    def init_env_data(self):
        self.init_maps()
        assert set(self.state_map.flatten().tolist()) == set(range(self.n_states))
        self.transitions = np.zeros((self.n_states, self.n_actions), dtype='int32')
        self.rewards = np.zeros((self.n_states,), dtype='int32')
        self.dones = np.zeros((self.n_states,), dtype='bool')
        for i in range(self.state_map.shape[0]):
            for j in range(self.state_map.shape[1]):
                s1 = self.state_map[i, j]
                self.dones[s1] = self.done_map[i, j]
                self.rewards[s1] = self.reward_map[i, j]
                for action in range(self.n_actions):
                    (i2, j2) = self.apply_action(i, j, action)
                    i2 = max(0, min(i2, self.state_map.shape[0]-1))
                    j2 = max(0, min(j2, self.state_map.shape[1] - 1))
                    s2 = self.state_map[i2, j2]
                    self.transitions[s1, action] = s2
        self.possible_r_values = list(set(self.reward_map.flatten().tolist()))


# Classic tabular MDP, consists of a 4x4 grid with "holes" that end the episode in failure and stochastic transitions
class FrozenLakeEnv(AbstractGridEnv):
    def __init__(self, slip_rate=0.1):
        super().__init__(16, 4, 0)
        # distribution P(a|s, Z=1) of size episode_len x n_actions x n_states
        self.pt_a_sz_success = None
        # distribution P(a|s, Z=0) of size episode_len x n_actions x n_states
        self.pt_a_sz_fail = None
        # distribution P(Z=1|s) of size episode_len x n_actions x n_states
        self.pt_z_s = None
        # distribution P(Z=1|s, a) of size episode_len x n_actions x n_states x n_actions
        self.pt_z_sa = None
        # transition probabilities
        self.p_s_sa = None
        # state visitation distribution of size n_states
        self.ds = None
        #slip rate
        assert (slip_rate < 0.5) and (slip_rate >= 0.0), "Slip rate must be between in [0, 0.5)"
        self.slip_rate = slip_rate
        # compute transition probabilities
        self.compute_transition_probs()

    def init_maps(self):
        self.state_map = np.array(
            [[0, 1, 2, 3],
             [4, 5, 6, 7],
             [8, 9, 10, 11],
             [12, 13, 14, 15]], dtype='int32')

        self.reward_map = np.zeros((4, 4), dtype='int32')
        self.reward_map[3, 3] = 1
        self.done_map = np.zeros((4, 4), dtype='bool')
        self.done_states = [5, 7, 11, 12, 15]
        for i in range(4):
            for j in range(4):
                if self.state_map[i, j] in self.done_states:
                    self.done_map[i, j] = True

    # Step env with noisy transitions
    def step(self, act):
        rnd = random.random()
        # 10% chance of action proceeding the one selected
        if rnd < self.slip_rate:
            act = (act - 1) % 4
        # Another 10% chance of the action following the one selected
        elif rnd < 2 * self.slip_rate:
            act = (act + 1) % 4

        return super().step(act)

    
    def compute_transition_probs(self):
        # compute transition probabilities
        self.p_s_sa = np.zeros((self.n_states, self.n_states, self.n_actions))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                # according to policy
                next_s = self.transitions[s, a]
                self.p_s_sa[next_s, s, a] += 1 - 2 * self.slip_rate
                for shift in (-1, 1):
                    changed_a = (a + shift) % 4
                    next_s = self.transitions[s, changed_a]
                    self.p_s_sa[next_s, s, a] += self.slip_rate

            # recalculate for terminal states
            if s in self.done_states:
                self.p_s_sa[:, s, :] = 0.
                self.p_s_sa[s, s, :] = 1.

    
    def initialize_hindsight(self, episode_length, pi_a_s):
        # compute state to state probabilities
        p_s_s = (self.p_s_sa * pi_a_s.T).sum(2)

        # compute state visitation distribution
        ds_unnormalized = np.zeros(self.n_states)
        p_s_current = np.zeros(self.n_states)
        p_s_current[0] = 1.
        for t in range(episode_length):
            ds_unnormalized += p_s_current
            p_s_current = p_s_s@p_s_current
        ds_unnormalized[self.done_states] = 0.
        self.ds = ds_unnormalized / ds_unnormalized.sum()


        # only (terminal) state giving reward
        success_state = np.flatnonzero(self.reward_map.flatten()).item()

        # compute foresight distributions
        pt_z_sa = np.zeros((episode_length, self.n_states, self.n_actions))
        pt_z_s = np.zeros((episode_length, self.n_states))
        p_s_s_power = np.eye(self.n_states)
        for t in range(episode_length - 1, -1, -1):
            pt_z_sa[t] = (p_s_s_power[:, :, None, None] * self.p_s_sa).sum(1)[success_state]
            pt_z_s[t] = (pt_z_sa[t] * pi_a_s.T).sum(1)
            p_s_s_power = p_s_s@p_s_s_power

        self.pt_z_s = pt_z_s
        self.pt_z_sa = pt_z_sa

        # compute hindsight distributions, nan where they are not defined
        pt_a_sz_success = np.full((episode_length, self.n_actions, self.n_states), np.nan)
        pt_a_sz_fail = np.full((episode_length, self.n_actions, self.n_states), np.nan)
        for t in range(episode_length):
            nominator = pt_z_sa[t].T * pi_a_s
            denominator = pt_z_s[t]
            np.divide(nominator, denominator, out=pt_a_sz_success[t], where=(denominator != 0))

            nominator = (1 - pt_z_sa[t].T) * pi_a_s
            denominator = 1 - pt_z_s[t]
            np.divide(nominator, denominator, out=pt_a_sz_fail[t], where=(denominator != 0))

        self.pt_a_sz_success = pt_a_sz_success
        self.pt_a_sz_fail = pt_a_sz_fail

    def get_hindsight_probability(self, states, actions, ts, successes):
        if self.pt_a_sz_success is None:
            raise Exception("Initialize with `initialize_hindsight` method first!")

        states, actions, ts = (np.array(l, dtype='int') for l in (states, actions, ts))
        successes = np.array(successes, dtype='bool')

        hindsight = np.empty_like(states, dtype='float')
        for mask, pt_a_sz in zip((successes, ~successes), (self.pt_a_sz_success, self.pt_a_sz_fail)):
            hindsight[mask] = pt_a_sz[ts[mask], actions[mask], states[mask]]

        return hindsight

    def compute_hindsight_probabilities_analytically(self, pi_a_s):
        A = np.zeros((self.n_states, self.n_states), dtype='float32')
        b = np.zeros((self.n_states,), dtype='float32')
        b[15] = 1
        for s in range(self.n_states):
            A[s, s] += 1
            if s in self.done_states:
                continue
            for a in range(self.n_actions):
                for a_delta, p_sprime_sa in zip((-1, 0, 1), (self.slip_rate, 1 - 2 * self.slip_rate, self.slip_rate)):
                    morphed_a = (a + a_delta) % self.n_actions
                    sprime = self.transitions[s, morphed_a]
                    A[s, sprime] -= pi_a_s[a, s] * p_sprime_sa
        self.pt_z_s = np.linalg.solve(A, b) 

        self.pt_z_sa = np.zeros((self.n_states, self.n_actions), dtype='float32')
        for s in range(self.n_states):
            if s in self.done_states:
                continue
            for a in range(self.n_actions):
                for a_delta, p in zip((-1, 0, 1), (self.slip_rate, 1 - 2 * self.slip_rate, self.slip_rate)):
                    morphed_a = (a + a_delta) % self.n_actions
                    self.pt_z_sa[s, a] += p * self.pt_z_s[self.transitions[s, morphed_a]]
        self.pt_z_sa[15, :] = 1

    def get_state_value(self, states, ts=None):
        if self.pt_z_s is None:
            raise Exception(f"Initialize with `initialize_hindsight` method first!")

        if len(self.pt_z_s.shape) == 1:
            values = self.pt_z_s[states]
        else:
            states, ts = (np.array(l, dtype='int') for l in (states, ts))
            values = self.pt_z_s[ts, states]
        return values

    def get_state_all_values(self, ts=None):
        if self.pt_z_s is None:
            raise Exception(f"Initialize with `initialize_hindsight` method first!")
        if len(self.pt_z_s.shape) == 1:
            values = self.pt_z_s
        else:
            ts = np.array(ts)
            values = self.pt_z_s[ts, :]

        return values

    def get_state_action_value(self, states, actions, ts=None):
        if self.pt_z_sa is None:
            raise Exception(f"Initialize with `initialize_hindsight` method first!")

        if len(self.pt_z_sa.shape) == 2:
            values = self.pt_z_sa[states, actions]
        else:
            states, actions, ts = (np.array(l, dtype='int') for l in (states, actions, ts))
            values = self.pt_z_sa[ts, states, actions]

        return values
    
    def get_state_all_action_values(self, states, ts=None):
        if self.pt_z_sa is None:
            raise Exception(f"Initialize with `initialize_hindsight` method first!")
        
        if len(self.pt_z_sa.shape) == 2:
            values = self.pt_z_sa[states, :]
        else:
            states, ts = (np.array(l, dtype='int') for l in (states, ts))
            values = self.pt_z_sa[ts, states, :]

        return values
    
    def get_transition_probs(self, states, actions):
        if self.p_s_sa is None:
            raise Exception(f"Initialize with `compute_transition_probs` method first!")

        states, actions = ((np.array(l, dtype='int') for l in (states, actions)))
        transition_probs = self.p_s_sa[:, states, actions]

        return transition_probs



class TabEnv:
    def __init__(self, n_states, n_actions, initial_pos):
        # Transition matrix of possible transitions between states
        self.n_states = n_states
        self.n_actions = n_actions
        # Each element is the state transitioned to from [state, action]
        self.transitions = np.zeros((n_states, n_actions), dtype=np.int)
        self.rewards = np.zeros((n_states,))  # Reward as function of state, for now
        self.initial_pos = initial_pos
        self.current_pos = initial_pos

        self.possible_r_values = [0]  # Array of reward values (rationals only) for credit computation

        # Other setup to be done by child classes

    def reset(self, new_initial_pos=None):
        if new_initial_pos is not None:
            self.initial_pos = new_initial_pos
            self.current_pos = new_initial_pos
        else:
            self.current_pos = int(self.initial_pos)  # Cast to avoid shallow copy pitfalls
        return self.current_pos

    # Return if done
    def is_done(self):
        return False  # Child classes extend this

    # Step env
    # TODO consider allowing noisy transitions
    def step(self, act):
        if self.transitions[self.current_pos, act] is not None:
            self.current_pos = self.transitions[self.current_pos, act]
        reward = self.rewards[self.current_pos]

        return self.current_pos, reward, self.is_done()


# Simple 3-state MDP to test
class TestEnv(TabEnv):
    def __init__(self):
        super().__init__(3, 3, 0)
        # Setup transitions, each state connects to self, next, previous, goal is to reach state 2
        self.transitions[0,0] = 0
        self.transitions[0,1] = 1
        self.transitions[0,2] = 0
        self.transitions[1,0] = 0
        self.transitions[1,1] = 1
        self.transitions[1,2] = 2
        self.transitions[2,0] = 2
        self.transitions[2,1] = 2
        self.transitions[2,2] = 2
        # Only state 2 provides reward
        self.rewards[2] = 1

        self.possible_r_values = [0, 1]

    def is_done(self):
        return self.current_pos == 2

# Shortcut env as described by HCA with n = 5
class ShortcutEnv(TabEnv):
    def __init__(self):
        super().__init__(5, 2, 0)
        self.transitions[0,0] = 4
        self.transitions[0,1] = 1
        self.transitions[1,0] = 4
        self.transitions[1,1] = 2
        self.transitions[2,0] = 4
        self.transitions[2,1] = 3
        self.transitions[3,0] = 4
        self.transitions[3,1] = 4
        self.transitions[4,0] = 4
        self.transitions[4,1] = 4

        self.rewards[0] = -1
        self.rewards[1] = -1
        self.rewards[2] = -1
        self.rewards[3] = -1
        self.rewards[4] = 1

        self.possible_r_values = [-1, 1]

        self.epsilon_to_goal = 0.1

    def is_done(self):
        return self.current_pos == 4

    def step(self, act):
        # only the action from the first step matters
        rand = np.random.rand()
        if rand < self.epsilon_to_goal:
            self.current_pos = 4
            reward = self.rewards[self.current_pos]

            return self.current_pos, reward, self.is_done()
        else:
            return super().step(act)


# Delayed effect env as described by HCA
class DelayedEffectEnv(TabEnv):
    def __init__(self, n_states_in_chain = 3, intermediate_reward_noise_std = 0):
        super().__init__(n_states_in_chain+2, 2, 0)
        #self.transitions[0,0] = 1
        #self.transitions[0,1] = 1
        for s in range(n_states_in_chain+1):
            self.transitions[s,0] = s+1
            self.transitions[s,1] = s+1

        self.transitions[n_states_in_chain+1,0] = n_states_in_chain+1
        self.transitions[n_states_in_chain+1,1] = n_states_in_chain+1

        self.possible_r_values = [-1, 1]

        self.middle_state_std = intermediate_reward_noise_std

    def is_done(self):
        return self.current_pos == self.n_states-1

    def step(self, act):
        # only the action from the first step matters
        if self.current_pos == 0:
            if act == 0:
                self.rewards[self.n_states-1] = 1
            else:
                self.rewards[self.n_states-1] = -1

        s,r,d = super().step(act)

        if not self.is_done() and self.middle_state_std != 0:
             r = np.random.normal(0, self.middle_state_std)
        return s, r, d

# Ambiguous bandit env as described by HCA
# TODO make compatible with finite reward value counts
class AmbiguousBanditEnv(TabEnv):
    def __init__(self):
        super().__init__(3, 2, 0)
        self.transitions[0,0] = 1
        self.transitions[0,1] = 2
        self.transitions[1,0] = 1
        self.transitions[1,1] = 1
        self.transitions[2,0] = 2
        self.transitions[2,1] = 2

        self.possible_r_values = [1, 1.5, 2]

        self.mean_std_for_action_from_state_0 = [[1, 1.5], [2, 1.5]]

        self.crossover_epsilon = 0.1

    def is_done(self):
        return self.current_pos == 1 or self.current_pos == 2

    def step(self, act):
        # only the action from the first step matters
        if self.current_pos == 0:
            self.rewards[act + 1] = np.random.normal(
                self.mean_std_for_action_from_state_0[act][0],
                self.mean_std_for_action_from_state_0[act][1]
            )

        # Random chance of visiting the other state
        if random.random() < self.crossover_epsilon:
            act = (act + 1) % 2

        return super().step(act)


# Simple 9-state grid mpd, easier to visualize
class SmallGridEnv(TabEnv):
    def __init__(self):
        super().__init__(9, 5, 0)

        # Grid looks like:
        # 0 1 2
        # 3 4 5
        # 6 7 8
        # Each state connects to neighbors and to self
        # Setup transitions, each state connects to self, next, previous, goal is to reach state 2
        self.transitions[0, 0] = 0
        self.transitions[0, 1] = 0
        self.transitions[0, 2] = 1
        self.transitions[0, 3] = 3
        self.transitions[0, 4] = 0

        self.transitions[1, 0] = 1
        self.transitions[1, 1] = 1
        self.transitions[1, 2] = 2
        self.transitions[1, 3] = 4
        self.transitions[1, 4] = 0

        self.transitions[2, 0] = 2
        self.transitions[2, 1] = 2
        self.transitions[2, 2] = 2
        self.transitions[2, 3] = 5
        self.transitions[2, 4] = 1

        self.transitions[3, 0] = 3
        self.transitions[3, 1] = 0
        self.transitions[3, 2] = 4
        self.transitions[3, 3] = 6
        self.transitions[3, 4] = 3

        self.transitions[4, 0] = 4
        self.transitions[4, 1] = 1
        self.transitions[4, 2] = 5
        self.transitions[4, 3] = 7
        self.transitions[4, 4] = 3

        self.transitions[5, 0] = 5
        self.transitions[5, 1] = 2
        self.transitions[5, 2] = 5
        self.transitions[5, 3] = 8
        self.transitions[5, 4] = 4

        self.transitions[6, 0] = 6
        self.transitions[6, 1] = 3
        self.transitions[6, 2] = 7
        self.transitions[6, 3] = 6
        self.transitions[6, 4] = 6

        self.transitions[7, 0] = 7
        self.transitions[7, 1] = 4
        self.transitions[7, 2] = 8
        self.transitions[7, 3] = 7
        self.transitions[7, 4] = 6

        self.transitions[8, 0] = 8
        self.transitions[8, 1] = 5
        self.transitions[8, 2] = 8
        self.transitions[8, 3] = 8
        self.transitions[8, 4] = 7

        # Only state 8 provides reward
        self.rewards[8] = 1

        self.possible_r_values = [0, 1]

    def is_done(self):
        return self.current_pos == 8


# Variant of the above that adds extra self-transition actions, credit should do better here?
class SmallGridExtraActionsEnv(TabEnv):
    def __init__(self):
        super().__init__(9, 9, 0)

        # Grid looks like:
        # 0 1 2
        # 3 4 5
        # 6 7 8
        # Each state connects to neighbors and to self
        # Setup transitions, each state connects to self, next, previous, goal is to reach state 2
        self.transitions[0, 0] = 0
        self.transitions[0, 1] = 0
        self.transitions[0, 2] = 1
        self.transitions[0, 3] = 3
        self.transitions[0, 4] = 0

        self.transitions[1, 0] = 1
        self.transitions[1, 1] = 1
        self.transitions[1, 2] = 2
        self.transitions[1, 3] = 4
        self.transitions[1, 4] = 0

        self.transitions[2, 0] = 2
        self.transitions[2, 1] = 2
        self.transitions[2, 2] = 2
        self.transitions[2, 3] = 5
        self.transitions[2, 4] = 1

        self.transitions[3, 0] = 3
        self.transitions[3, 1] = 0
        self.transitions[3, 2] = 4
        self.transitions[3, 3] = 6
        self.transitions[3, 4] = 3

        self.transitions[4, 0] = 4
        self.transitions[4, 1] = 1
        self.transitions[4, 2] = 5
        self.transitions[4, 3] = 7
        self.transitions[4, 4] = 3

        self.transitions[5, 0] = 5
        self.transitions[5, 1] = 2
        self.transitions[5, 2] = 5
        self.transitions[5, 3] = 8
        self.transitions[5, 4] = 4

        self.transitions[6, 0] = 6
        self.transitions[6, 1] = 3
        self.transitions[6, 2] = 7
        self.transitions[6, 3] = 6
        self.transitions[6, 4] = 6

        self.transitions[7, 0] = 7
        self.transitions[7, 1] = 4
        self.transitions[7, 2] = 8
        self.transitions[7, 3] = 7
        self.transitions[7, 4] = 6

        self.transitions[8, 0] = 8
        self.transitions[8, 1] = 5
        self.transitions[8, 2] = 8
        self.transitions[8, 3] = 8
        self.transitions[8, 4] = 7

        # Add additional self-transitions
        for i in range(9):
            self.transitions[i,5:] = i

        # Only state 8 provides reward
        self.rewards[8] = 1

        self.possible_r_values = [0, 1]

    def is_done(self):
        return self.current_pos == 8


# Similar to the grids above, but all actions result in a state transition
# Actions that would move off the grid result in "wrap around" to the other side of the grid
# This means the optimal path is now 2 steps rather than 4
class SmallGridNoNoOpEnv(TabEnv):
    def __init__(self):
        super().__init__(9, 4, 0)

        # Grid looks like:
        # 0 1 2
        # 3 4 5
        # 6 7 8
        # Each state connects to neighbors and to self
        # Setup transitions, each state connects to self, next, previous, goal is to reach state 2
        self.transitions[0, 0] = 6
        self.transitions[0, 1] = 1
        self.transitions[0, 2] = 3
        self.transitions[0, 3] = 2

        self.transitions[1, 0] = 7
        self.transitions[1, 1] = 2
        self.transitions[1, 2] = 4
        self.transitions[1, 3] = 0

        self.transitions[2, 0] = 8
        self.transitions[2, 1] = 0
        self.transitions[2, 2] = 5
        self.transitions[2, 3] = 1

        self.transitions[3, 0] = 0
        self.transitions[3, 1] = 4
        self.transitions[3, 2] = 6
        self.transitions[3, 3] = 5

        self.transitions[4, 0] = 1
        self.transitions[4, 1] = 5
        self.transitions[4, 2] = 7
        self.transitions[4, 3] = 3

        self.transitions[5, 0] = 2
        self.transitions[5, 1] = 3
        self.transitions[5, 2] = 8
        self.transitions[5, 3] = 4

        self.transitions[6, 0] = 3
        self.transitions[6, 1] = 7
        self.transitions[6, 2] = 0
        self.transitions[6, 3] = 8

        self.transitions[7, 0] = 4
        self.transitions[7, 1] = 8
        self.transitions[7, 2] = 1
        self.transitions[7, 3] = 6

        self.transitions[8, 0] = 5
        self.transitions[8, 1] = 6
        self.transitions[8, 2] = 2
        self.transitions[8, 3] = 7

        # Only state 8 provides reward
        self.rewards[8] = 1

        self.possible_r_values = [0, 1]

    def is_done(self):
        return self.current_pos == 8


# As smallgrid, but there is no "done" signal, so the agent must learn to remain in state 8 to receive further reward.
class SmallGridNotDoneEnv(SmallGridEnv):
    def is_done(self):
        return False

# Classic tabular MDP, consists of a 4x4 grid with "holes" that end the episode in failure and stochastic transitions
class FrozenLakeEnvLegacy(TabEnv):
    def __init__(self):
        super().__init__(16, 4, 0)

        # Grid looks like:
        # 0 1 2 3
        # 4 5 6 7
        # 8 9 10 11
        # 12 13 14 15
        # Each state connects to neighbors and to self
        # States 5, 7, 11, and 12 are holes (set done)
        # Setup transitions, each state connects to self, next, previous, goal is to reach state 15
        self.transitions[0, 0] = 0
        self.transitions[0, 1] = 1
        self.transitions[0, 2] = 4
        self.transitions[0, 3] = 0

        self.transitions[1, 0] = 1
        self.transitions[1, 1] = 2
        self.transitions[1, 2] = 5
        self.transitions[1, 3] = 0

        self.transitions[2, 0] = 2
        self.transitions[2, 1] = 3
        self.transitions[2, 2] = 6
        self.transitions[2, 3] = 1

        self.transitions[3, 0] = 3
        self.transitions[3, 1] = 3
        self.transitions[3, 2] = 7
        self.transitions[3, 3] = 2

        self.transitions[4, 0] = 0
        self.transitions[4, 1] = 5
        self.transitions[4, 2] = 8
        self.transitions[4, 3] = 4

        self.transitions[5, 0] = 1
        self.transitions[5, 1] = 6
        self.transitions[5, 2] = 9
        self.transitions[5, 3] = 4

        self.transitions[6, 0] = 2
        self.transitions[6, 1] = 7
        self.transitions[6, 2] = 10
        self.transitions[6, 3] = 5

        self.transitions[7, 0] = 3
        self.transitions[7, 1] = 7
        self.transitions[7, 2] = 11
        self.transitions[7, 3] = 6

        self.transitions[8, 0] = 4
        self.transitions[8, 1] = 9
        self.transitions[8, 2] = 12
        self.transitions[8, 3] = 8

        self.transitions[9, 0] = 5
        self.transitions[9, 1] = 10
        self.transitions[9, 2] = 13
        self.transitions[9, 3] = 8

        self.transitions[10, 0] = 6
        self.transitions[10, 1] = 11
        self.transitions[10, 2] = 14
        self.transitions[10, 3] = 9

        self.transitions[11, 0] = 7
        self.transitions[11, 1] = 11
        self.transitions[11, 2] = 15
        self.transitions[11, 3] = 10

        self.transitions[12, 0] = 8
        self.transitions[12, 1] = 13
        self.transitions[12, 2] = 12
        self.transitions[12, 3] = 12

        self.transitions[13, 0] = 9
        self.transitions[13, 1] = 14
        self.transitions[13, 2] = 13
        self.transitions[13, 3] = 12

        self.transitions[14, 0] = 10
        self.transitions[14, 1] = 15
        self.transitions[14, 2] = 14
        self.transitions[14, 3] = 13

        self.transitions[15, 0] = 11
        self.transitions[15, 1] = 15
        self.transitions[15, 2] = 15
        self.transitions[15, 3] = 14

        # Only state 15 provides reward
        self.rewards[15] = 1
        # Let's try adding penalties for failing
        #self.rewards[5] = -1
        #self.rewards[7] = -1
        #self.rewards[11] = -1
        #self.rewards[12] = -1

        self.possible_r_values = [0, 1]

    def is_done(self):
        return (self.current_pos == 5 or
                self.current_pos == 7 or
                self.current_pos == 11 or
                self.current_pos == 12 or
                self.current_pos == 15)

    # Step env with noisy transitions
    def step(self, act):
        rnd = random.random()
        # 10% chance of action proceeding the one selected
        if rnd < 0.1:
            act = (act - 1) % 4
        # Another 10% chance of the action following the one selected
        elif rnd < 0.2:
            act = (act + 1) % 4

        # Flat 10% chance of a random action
        #if random.random() < 0.1:
        #    act = random.randint(0,3)

        return super().step(act)

# Testing a possible degenerate case
class CounterexampleBanditEnv(TabEnv):
    def __init__(self):
        super().__init__(3, 2, 0)
        self.transitions[0,0] = 1
        self.transitions[0,1] = 2
        self.transitions[1,0] = 1
        self.transitions[1,1] = 1
        self.transitions[2,0] = 2
        self.transitions[2,1] = 2

        self.rewards[1] = 1
        self.rewards[2] = 2

        self.possible_r_values = [1, 2]

    def is_done(self):
        return self.current_pos == 1 or self.current_pos == 2


# Testing another possible degenerate case
class CounterexampleBandit2Env(TabEnv):
    def __init__(self):
        super().__init__(3, 2, 0)
        self.transitions[0,0] = 1
        self.transitions[0,1] = 2
        self.transitions[1,0] = 1
        self.transitions[1,1] = 1
        self.transitions[2,0] = 2
        self.transitions[2,1] = 2

        self.rewards[1] = 0
        self.rewards[2] = 0

        self.possible_r_values = [-1, 1, 2, -6]

    def is_done(self):
        return self.current_pos == 1 or self.current_pos == 2

    # TODO Needs to be 4 different rewards, expected value needs to be 0, 3 different probabilities among rewards
    def step(self, act):
        if act == 1:
            if random.random() < 0.5:
                self.rewards[2] = 1
            else:
                self.rewards[2] = -1
        elif act == 0:
            if random.random() < 0.25:
                self.rewards[1] = -6
            else:
                self.rewards[1] = 2
        return super().step(act)
