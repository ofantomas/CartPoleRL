from itertools import groupby
import numpy as np
import torch
from variance_estimation import rollout


class RandomAgent:
    '''
    Performs actions randomly
    Args:
        n_actions (int): number of possible actions
    '''
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def select_action(self, state, inference):
        return np.random.randint(0, self.n_actions)

    def update(self, states, actions, rewards, dones):
        pass


# Basic reinforce
class ReinforceAgent:
    '''
    Implements REINFORCE algorithm. Policy is trained with GD.
    Args:
        env_shape (tuple): [num_states x num_actions],
        alpha (float): learning rate for policy,
        gamma (float): MDP's gamma
    '''
    def __init__(self, env_shape: tuple, alpha: float, gamma: float):
        self.alpha = alpha
        self.gamma = gamma
        self.env_shape = env_shape
        self.pi = torch.ones(self.env_shape, dtype=torch.float, requires_grad=True)
        self.pi_opt = torch.optim.SGD(params=[self.pi], lr=self.alpha)

    def select_action(self, state, inference):
        probs = self.pi[state].softmax(0)
        if inference:  # Take maxprob
            act = probs.argmax()
        else:
            cat = torch.distributions.Categorical(probs=probs)
            act = cat.sample()
        return act

    # Accumulate rewards across time
    def accumulate_rewards(self, rewards, dones):
        acc_rewards = np.zeros_like(rewards)
        for idx, r in reversed(list(enumerate(rewards))):
            if idx < len(rewards) - 1:
                acc_rewards[idx] += self.gamma * acc_rewards[idx + 1] * (1 - dones[idx])
            acc_rewards[idx] += r
        return acc_rewards

    # Computes the eligibility given states/actions/advantages. Does not update the policy
    def compute_eligibility(self, states, actions, advantage):
        logits = self.pi[states]
        policy = torch.distributions.Categorical(logits=logits)
        log_probs = policy.log_prob(actions)
        loss = - log_probs * advantage.detach()
        return loss, policy

    def compute_var_per_sa_pair(self, states, actions, loss):
        # Compute mean variance of losses per sa pair
        tuple_list = list(zip(states.numpy(), actions.numpy(), loss.detach().numpy()))
        tuple_list.sort(key=lambda x: x[:2])  # groupby needs it
        grouped = groupby(tuple_list, lambda x: x[:2])
        variances = [np.var([x[2] for x in lst]) for k, lst in grouped]
        mean_var = np.mean(variances)
        return mean_var

    # Does what it says, for easy re-use across methods
    # Returns policy gradient/loss values and the parametrized policy distribution, for logging
    def make_pg_step(self, states, actions, advantage):
        loss, policy = self.compute_eligibility(states, actions, advantage)
        mean_var = self.compute_var_per_sa_pair(states, actions, loss)
        self.pi_opt.zero_grad()
        loss.mean().backward()
        self.pi_opt.step()  # SGD step

        return loss, policy, mean_var

    # Given a trajectory for one episode, update the policy
    def update(self, states, actions, rewards, next_states, dones):
        states = torch.LongTensor(states)
        actions = torch.LongTensor(actions)
        rewards = np.asarray(rewards)
        dones = torch.LongTensor(dones)

        # First compute cumulative rewards
        cum_rewards = torch.Tensor(self.accumulate_rewards(rewards, dones))

        # Next, compute elegibility
        loss, policy, mean_var = self.make_pg_step(states, actions, cum_rewards)

        return loss.mean().item(), mean_var, policy.entropy().mean().item(), cum_rewards.mean()


# Extension of reinforce to include a value function baseline
class ValueBaselineAgent(ReinforceAgent):
    '''
    Implements REINFORCE algorithm with value baseline.
    Policy and V function are trained with GD.
    Args:
        env_shape (tuple): [num_states x num_actions],
        alpha (float): learning rate for policy,
        beta (float): learning rate for V function,
        gamma (float): MDP's gamma
    '''
    def __init__(self, env_shape: tuple, alpha: float, beta: float, gamma: float):
        super().__init__(env_shape, alpha, gamma)
        self.value = torch.zeros(env_shape[0], dtype=torch.float, requires_grad=True)
        self.beta = beta
        self.value_opt = torch.optim.SGD(params=[self.value], lr=self.beta)

    def update_values(self, states, cum_rewards):
        vals = self.value[states]
        val_loss = (vals - cum_rewards) ** 2 / 2

        self.value_opt.zero_grad()
        val_loss.mean().backward()
        self.value_opt.step()

        return val_loss

    # Compute value-baseline advantage values, and update the value function
    def compute_advantage(self, states, cum_rewards, dones):
        # Subtract value baseline
        vals = self.value[states]
        advantage = cum_rewards - vals.detach()
        return advantage

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.LongTensor(states)
        actions = torch.LongTensor(actions)
        rewards = np.asarray(rewards)

        # First compute cumulative rewards
        cum_rewards = torch.Tensor(self.accumulate_rewards(rewards, dones))

        # Subtract value baseline
        adv = self.compute_advantage(states, cum_rewards, dones)
        self.update_values(states, cum_rewards)

        # Update the policy
        loss, cats, mean_var = self.make_pg_step(states, actions, adv)

        return loss.mean().item(), mean_var, cats.entropy().mean().item(), adv.mean()


# Extension of reinforce to include a value function baseline
class PerfectValueBaselineAgent(ValueBaselineAgent):
    '''
    Implements REINFORCE algorithm with perfect value baseline.
    Policy is trained with GD. Perfect V function is provided
    by the environment.
    Args:
        env_shape (tuple): [num_states x num_actions],
        alpha (float): learning rate for policy,
        beta (float): learning rate for Q function,
        gamma (float): MDP's gamma
        env: environment,
        episode_length (int): max length of the episode,
        analytical (bool): whether to compute Q function
            assuming infinite episode_length
    '''
    def __init__(self, env_shape: tuple, alpha: float, beta:float,
                 gamma: float, env, episode_length: int, analytical: bool):
        super().__init__(env_shape, alpha, beta, gamma)
        self.env = env
        self.episode_length = episode_length
        self.analytical = analytical

    # Compute value-baseline advantage values, and update the value function
    def compute_advantage(self, states, cum_rewards, dones):
        pi_a_s = torch.softmax(self.pi, 1) #num_states x num_actions
        pi_a_s = pi_a_s.detach().numpy().T

        if self.analytical is True:
            self.env.compute_hindsight_probabilities_analytically(pi_a_s)
            vals = torch.FloatTensor(self.env.get_state_value(states=states))
        else:
            # here we assume that data is sequential and begins with the start of the episode
            self.env.initialize_hindsight(self.episode_length, pi_a_s)
            extended_done_ids = np.flatnonzero([True] + dones)
            ts = []
            for i in range(len(extended_done_ids) - 1):
                ts.extend(list(range(extended_done_ids[i + 1] - extended_done_ids[i])))
            vals = torch.FloatTensor(self.env.get_state_value(states=states, ts=ts))

        advantage = cum_rewards - vals
        return advantage


class OptimalStateBaselineAgent(ReinforceAgent):
    '''
    Implements REINFORCE algorithm with optimal state baseline.
    Policy is trained with GD. Optimal state baseline is evaluated
    over 10000 episodes.
    Args:
        env_shape (tuple): [num_states x num_actions],
        alpha (float): learning rate for policy,
        gamma (float): MDP's gamma
        env: environment,
        episode_length (int): max length of the episode
    '''
    def __init__(self, env_shape: tuple, alpha: float, gamma: float,
                 env, episode_length: int):
        super().__init__(env_shape, alpha, gamma)
        self.env = env
        self.episode_length = episode_length
        self.optimal_baseline = None
        self.delta_var = None

    def estimate_optimal_baseline(self, n_episodes=10000):
        # compute grad log pi
        ns = self.env.n_states
        na = self.env.n_actions
        log_prob = torch.log_softmax(self.pi, 1)
        grad_log_prob = torch.zeros(ns, na, ns, na)
        for i in range(ns):
            for j in range(na):
                grad_log_prob[i, j], = torch.autograd.grad(outputs=[log_prob[i, j]],
                                                           inputs=[self.pi],
                                                           retain_graph=True)
        l2_norms_squared = np.empty(n_episodes)
        returns = np.empty(n_episodes)
        for i in range(n_episodes):
            ep_states, ep_acts, ep_rewards, _, _ = rollout(self.env,
                                                                               self,
                                                                               self.episode_length)
            l2_norms_squared[i] = (grad_log_prob[ep_states, ep_acts].sum(0) ** 2).sum(0).sum(0)
            returns[i] = sum(ep_rewards)
        self.optimal_baseline = (returns * l2_norms_squared).mean() / l2_norms_squared.mean()
        self.delta_var = self.optimal_baseline ** 2 * l2_norms_squared.mean()

    # Compute value-baseline advantage values, and update the value function
    def compute_advantage(self, cum_rewards):
        advantage = cum_rewards - self.optimal_baseline
        return advantage

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.LongTensor(states)
        actions = torch.LongTensor(actions)
        rewards = np.asarray(rewards)

        # First compute cumulative rewards
        cum_rewards = torch.Tensor(self.accumulate_rewards(rewards, dones))

        # Subtract value baseline
        self.estimate_optimal_baseline()
        adv = self.compute_advantage(cum_rewards)

        # Update the policy
        loss, cats, mean_var = self.make_pg_step(states, actions, adv)

        return loss.mean().item(), mean_var, cats.entropy().mean().item(), adv.mean()


# Similar to ValueBaselineAgent, but with a multiplier on the value baseline
#  (so it can be made bigger or added instead)
class ValueModAgent(ValueBaselineAgent):
    '''
    Similar to ValueBaselineAgent, but with a multiplier on the value baseline
    (so it can be made bigger or added instead).
    Args:
        env_shape (tuple): [num_states x num_actions],
        alpha (float): learning rate for policy,
        beta (float): learning rate for Q function,
        gamma (float): MDP's gamma,
        value_mult: float -- multiplication rate
    '''
    def __init__(self, env_shape: tuple, alpha: float, gamma: float,
                 beta:float, value_mult: float = 1):
        super().__init__(env_shape, alpha, beta, gamma)
        self.value_mult = value_mult

    # Like the parent, but with a multiplier
    def compute_update_value_advantage(self, states, cum_rewards):
        # Subtract value baseline
        vals = torch.index_select(self.value, 0, states)
        # Use original diff to fit value function
        val_loss = cum_rewards - vals

        adv = (cum_rewards - vals * self.value_mult).detach()

        # Compute manual value loss
        for s in range(val_loss.size(0)):
            self.value[states[s]] += val_loss[s] * (self.alpha / val_loss.size(0))

        return adv, val_loss


class ActionStateBaselineAgent(ReinforceAgent):
    '''
    Implements REINFORCE algorithm with Q function baseline.
    Policy and Q function are trained with GD.
    Args:
        env_shape (tuple): [num_states x num_actions],
        alpha (float): learning rate for policy,
        beta (float): learning rate for Q function,
        gamma (float): MDP's gamma
    '''
    def __init__(self, env_shape: tuple, alpha: float, beta: float, gamma: float):
        super().__init__(env_shape, alpha, gamma)

        self.Q = torch.zeros(env_shape, dtype=torch.float, requires_grad=True)
        self.beta = beta
        self.q_opt = torch.optim.SGD(params=[self.Q], lr=self.beta)

    # Compute q-baseline advantage values, and update the q function
    # Unlike other methods, the policy target here is the Q-value directly
    def update_q_values(self, states, actions, cum_rewards):
        # Subtract Q baseline
        q_s = self.Q[states, actions].squeeze()
        q_loss = (cum_rewards - q_s) ** 2 / 2
        self.q_opt.zero_grad()
        q_loss.mean().backward()
        self.q_opt.step()

        return q_loss

    def compute_eligibility(self, states, actions, advantage, expectation):
        policy = torch.distributions.Categorical(logits=self.pi[states])
        #compute expecation over action~pi of grad_log_pi(a|s) * Q(a, s)
        log_probs = policy.log_prob(actions)
        loss = -log_probs * advantage.detach() - expectation
        return loss, policy

    # Compute value-baseline advantage values, and update the value function
    def compute_advantage(self, states, actions, cum_rewards, dones):
        policy = torch.distributions.Categorical(logits=self.pi[states])
        expectation = (policy.probs * self.Q[states]).sum(dim=1).squeeze()
        q_a_s = self.Q[states, actions].squeeze()
        # Subtract action-state baseline and add
        # expectation over actions to keep the estimate unbiased
        advantage = cum_rewards - q_a_s
        return advantage, expectation

    def make_pg_step(self, states, actions, advantage, expectation):
        loss, cats = self.compute_eligibility(states, actions, advantage, expectation)
        mean_var = self.compute_var_per_sa_pair(states, actions, loss)
        self.pi_opt.zero_grad()
        loss.mean().backward()
        self.pi_opt.step()
        return loss, cats, mean_var

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.LongTensor(states)
        actions = torch.LongTensor(actions)
        rewards = np.asarray(rewards)

        # First compute cumulative rewards
        cum_rewards = torch.Tensor(self.accumulate_rewards(rewards, dones))

        # Update Q values and get targets
        _ = self.update_q_values(states, actions, cum_rewards)
        adv, expectation = self.compute_advantage(states, actions, cum_rewards, dones)

        # Update the policy
        loss, cats, mean_var = self.make_pg_step(states, actions, adv, expectation)

        return loss.mean().item(), mean_var, cats.entropy().mean().item(), adv.mean()



class PerfectActionStateBaselineAgent(ActionStateBaselineAgent):
    '''
    Implements REINFORCE algorithm with perfect Q function baseline.
    Policy is trained with GD. Perfect Q function is provided
    by the environment.
    Args:
        env_shape (tuple): [num_states x num_actions],
        alpha (float): learning rate for policy,
        beta (float): learning rate for Q function,
        gamma (float): MDP's gamma
        env: environment,
        episode_length (int): max length of the episode,
        analytical (bool): whether to compute Q function
            assuming infinite episode_length
    '''
    def __init__(self, env_shape: tuple, alpha: float, beta: float, gamma: float,
                 env, episode_length, analytical: bool):
        super().__init__(env_shape, alpha, beta, gamma)
        self.env = env
        self.episode_length = episode_length
        self.analytical = analytical


    def compute_advantage(self, states, actions, cum_rewards, dones):
        pi_a_s = torch.softmax(self.pi, 1) #num_states x num_actions
        pi_a_s = pi_a_s.detach().numpy().T

        if self.analytical is True:
            self.env.compute_hindsight_probabilities_analytically(pi_a_s)
            q_s = torch.FloatTensor(self.env.get_state_all_action_values(states=states))
            q_a_s = torch.FloatTensor(self.env.get_state_action_value(states=states,
                                                                      actions=actions))
        else:
            self.env.initialize_hindsight(self.episode_length, pi_a_s)
            # here we assume that data is sequential and begins with the start of the episode
            extended_done_ids = np.flatnonzero([True] + dones)
            ts = []
            for i in range(len(extended_done_ids) - 1):
                ts.extend(list(range(extended_done_ids[i + 1] - extended_done_ids[i])))
            q_s = torch.FloatTensor(self.env.get_state_all_action_values(states=states, ts=ts))
            q_a_s = torch.FloatTensor(self.env.get_state_action_value(states=states,
                                                                      actions=actions, ts=ts))

        policy = torch.distributions.Categorical(logits=self.pi[states])
        expectation = (policy.probs * q_s).sum(dim=1).squeeze()

        advantage = cum_rewards - q_a_s
        return advantage, expectation


class TrajectoryCVAgent(ActionStateBaselineAgent):
    '''
    Implements REINFORCE algorithm with TrajCV baseline.
    Policy and Q function are trained with GD.
    Args:
        env_shape (tuple): [num_states x num_actions],
        alpha (float): learning rate for policy,
        beta (float): learning rate for Q function,
        gamma (float): MDP's gamma
    '''
    def __init__(self, env_shape: tuple, alpha: float, beta: float, gamma: float):
        super().__init__(env_shape, alpha, beta, gamma)

    def accumulate(self, values):
        return torch.flip(torch.cumsum(torch.flip(values, dims=[0]), 0), dims=[0])

    # Compute value-baseline advantage values, and update the value function
    def compute_advantage(self, states, actions, cum_rewards, dones):
        # Subtract action-state baseline and add
        # expectation over actions to keep the estimate unbiased
        policy = torch.distributions.Categorical(logits=self.pi[states])
        expectation = (policy.probs * self.Q[states]).sum(dim=1).squeeze()
        #Compute expectations of future Q functions to make estimate unbiased
        future_expectation = self.accumulate(torch.cat((expectation[1:], torch.FloatTensor([0]))))
        # Compute sum of future Q functions
        q_a_s = self.Q[states, actions].squeeze()
        q_a_s = self.accumulate(q_a_s)

        advantage = cum_rewards - q_a_s + future_expectation
        return advantage, expectation

class PerfectTrajectoryCVAgent(TrajectoryCVAgent):
    '''
    Implements REINFORCE algorithm with perfect TrajCV function baseline.
    Policy is trained with GD. Perfect Q function is provided
    by the environment.
    Args:
        env_shape (tuple): [num_states x num_actions],
        alpha (float): learning rate for policy,
        beta (float): learning rate for Q functions,
        gamma (float): MDP's gamma
        env: environment,
        episode_length (int): max length of the episode,
        analytical (bool): whether to compute Q function
            assuming infinite episode_length
    '''
    def __init__(self, env_shape: tuple, alpha: float, beta: float, gamma: float,
                 env, episode_length, analytical: bool):
        super().__init__(env_shape, alpha, beta, gamma)
        self.env = env
        self.episode_length = episode_length
        self.analytical = analytical

    def compute_advantage(self, states, actions, cum_rewards, dones):
        pi_a_s = torch.softmax(self.pi, 1)
        pi_a_s = pi_a_s.detach().numpy().T

        if self.analytical is True:
            self.env.compute_hindsight_probabilities_analytically(pi_a_s)
            q_s = torch.FloatTensor(self.env.get_state_all_action_values(states=states))
            q_a_s = torch.FloatTensor(self.env.get_state_action_value(states=states,
                                                                      actions=actions))
        else:
            self.env.initialize_hindsight(self.episode_length, pi_a_s)
            # here we assume that data is sequential and begins with the start of the episode
            extended_done_ids = np.flatnonzero([True] + dones)
            ts = []
            for i in range(len(extended_done_ids) - 1):
                ts.extend(list(range(extended_done_ids[i + 1] - extended_done_ids[i])))
            q_s = torch.FloatTensor(self.env.get_state_all_action_values(states=states, ts=ts))
            q_a_s = torch.FloatTensor(self.env.get_state_action_value(states=states,
                                                                      actions=actions, ts=ts))

        policy = torch.distributions.Categorical(logits=self.pi[states])
        expectation = (policy.probs * q_s).sum(dim=1).squeeze()
        future_expectation = self.accumulate(torch.cat((expectation[1:], torch.FloatTensor([0]))))

        # Compute sum of future Q functions
        q_a_s = self.accumulate(q_a_s)
        advantage = cum_rewards - q_a_s + future_expectation
        return advantage, expectation


class DynamicsTrajCVAgent(TrajectoryCVAgent):
    '''
    Implements REINFORCE algorithm with Dynamics TrajCV baseline.
    Policy, Q function and V functions are trained with GD.
    Transition function is estimated from rollouts.
    Args:
        env_shape (tuple): [num_states x num_actions],
        alpha (float): learning rate for policy,
        beta (float): learning rate for Q, V functions,
        gamma (float): MDP's gamma
    '''
    def __init__(self, env_shape: tuple, alpha: float, beta:float, gamma: float):
        super().__init__(env_shape, alpha, beta, gamma)

        self.p_s_sa = torch.zeros((env_shape[0], env_shape[0], env_shape[1]),
                                  dtype=torch.float, requires_grad=False)
        self.value = torch.zeros(env_shape[0], dtype=torch.float, requires_grad=True)
        self.beta = beta
        self.value_opt = torch.optim.SGD(params=[self.value], lr=self.beta)

    def update_values(self, states, cum_rewards):
        vals = self.value[states]
        val_loss = (vals - cum_rewards) ** 2 / 2

        self.value_opt.zero_grad()
        val_loss.mean().backward()
        self.value_opt.step()

        return val_loss

    def get_transition_probs(self, states, actions):
        transition_probs = self.p_s_sa[:, states, actions]
        transition_probs = transition_probs / (transition_probs.sum(dim=0) + 1e-8)
        return transition_probs

    def update_transition_probs(self, states, next_states, actions):
        self.p_s_sa[next_states, states, actions] += 1

    def compute_advantage(self, states, next_states, actions, cum_rewards, dones):
        # Subtract action-state baseline and add
        # expectation over actions to keep the estimate unbiased
        policy = torch.distributions.Categorical(logits=self.pi[states])
        expectation = (policy.probs * self.Q[states]).sum(dim=1).squeeze()
        future_expectation = self.accumulate(torch.cat((expectation[1:], torch.FloatTensor([0]))))

        # Get Q, V functions for episode
        v_s = self.value[next_states]
        q_a_s = self.Q[states, actions].squeeze()

        # Compute sum of future Q, V functions
        v_s = self.accumulate(torch.cat((v_s[1:], torch.FloatTensor([0]))))
        q_a_s = self.accumulate(q_a_s)

        transition_probs = self.get_transition_probs(states, actions)
        expectation_states = (transition_probs * self.value.unsqueeze(dim=1)).sum(dim=0).squeeze()
        future_expectation_states = self.accumulate(torch.cat((expectation_states[1:],
                                                               torch.FloatTensor([0]))))

        advantage = cum_rewards - q_a_s - v_s + future_expectation_states + future_expectation
        return advantage, expectation

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.LongTensor(states)
        actions = torch.LongTensor(actions)
        rewards = np.asarray(rewards)

        # First compute cumulative rewards
        cum_rewards = torch.Tensor(self.accumulate_rewards(rewards, dones))

        # Update V, Q values
        _ = self.update_values(states, cum_rewards)
        _ = self.update_q_values(states, actions, cum_rewards)

        #Update transition probs
        self.update_transition_probs(states, next_states, actions)

        #Get targets
        adv, expectation = self.compute_advantage(states, next_states, actions, cum_rewards, dones)

        # Update the policy
        loss, cats, mean_var = self.make_pg_step(states, actions, adv, expectation)

        return loss.mean().item(), mean_var, cats.entropy().mean().item(), adv.mean()


class PerfectDynamicsTrajCVAgent(DynamicsTrajCVAgent):
    '''
    Implements REINFORCE algorithm with Dynamics TrajCV baseline.
    Policy is trained with GD. Q function, V function and
    transition function are provided by the environment.
     Args:
        env_shape (tuple): [num_states x num_actions],
        alpha (float): learning rate for policy,
        beta (float): learning rate for Q, V functions,
        gamma (float): MDP's gamma
        env: environment,
        episode_length (int): max length of the episode,
        analytical (bool): whether to compute Q, V functions
            assuming infinite episode_length
    '''
    def __init__(self, env_shape: tuple, alpha: float, beta:float,
                 gamma: float, env, episode_length, analytical: bool):
        super().__init__(env_shape, alpha, beta, gamma)
        self.env = env
        self.episode_length = episode_length
        self.analytical = analytical

    def compute_advantage(self, states, next_states, actions, cum_rewards, dones):
        pi_a_s = torch.softmax(self.pi, 1)
        pi_a_s = pi_a_s.detach().numpy().T

        if self.analytical is True:
            self.env.compute_hindsight_probabilities_analytically(pi_a_s)
            q_s = torch.FloatTensor(self.env.get_state_all_action_values(states=states))
            q_a_s = torch.FloatTensor(self.env.get_state_action_value(states=states,
                                                                      actions=actions))
            v_all_s = torch.FloatTensor(self.env.get_state_all_values()).unsqueeze(dim=1)
            v_all_s[15, :] = 0
            v_s = torch.FloatTensor(self.env.get_state_value(states=next_states))
            v_s[next_states == 15] = 0
        else:
            self.env.initialize_hindsight(self.episode_length, pi_a_s)
            # here we assume that data is sequential and begins with the start of the episode
            extended_done_ids = np.flatnonzero([True] + dones)
            ts = []
            for i in range(len(extended_done_ids) - 1):
                ts.extend(list(range(extended_done_ids[i + 1] - extended_done_ids[i])))
            q_s = torch.FloatTensor(self.env.get_state_all_action_values(states=states, ts=ts))
            q_a_s = torch.FloatTensor(self.env.get_state_action_value(states=states,
                                                                      actions=actions, ts=ts))
            v_all_s = torch.FloatTensor(self.env.get_state_all_values(ts=ts).T)
            #tmp
            v_all_s[15, :] = 0
            v_s = torch.FloatTensor(self.env.get_state_value(states=next_states, ts=ts))
            v_s[next_states == 15] = 0



        transition_probs = torch.FloatTensor(self.env.get_transition_probs(states=states,
                                                                           actions=actions))
        # Compute Value functions in next states
        policy = torch.distributions.Categorical(logits=self.pi[states])
        expectation = (policy.probs * q_s).sum(dim=1).squeeze()
        future_expectation = self.accumulate(torch.cat((expectation[1:], torch.FloatTensor([0]))))
        expectation_states = (transition_probs * v_all_s).sum(dim=0).squeeze()
        future_expectation_states = self.accumulate(torch.cat((expectation_states[1:],
                                                               torch.FloatTensor([0]))))

        # Compute sum of future Q, V functions
        v_s = self.accumulate(torch.cat((v_s[1:], torch.FloatTensor([0]))))
        q_a_s = self.accumulate(q_a_s)

        advantage = cum_rewards - q_a_s - v_s  + future_expectation_states + future_expectation
        return advantage, expectation


class PerfectDynamicsEstQVTrajCVAgent(DynamicsTrajCVAgent):
    '''
    Implements REINFORCE algorithm with Dynamics TrajCV baseline.
    Policy, Q function, V function are trained with GD.
    Transition function is provided by the environment.
    Args:
        env_shape (tuple): [num_states x num_actions],
        alpha (float): learning rate for policy,
        beta (float): learning rate for Q, V functions,
        gamma (float): MDP's gamma
        env: environment,
        episode_length (int): max length of the episode
        analytical (bool): whether to compute Q, V functions
            assuming infinite episode_length
    '''
    def __init__(self, env_shape: tuple, alpha: float, beta:float,
                 gamma: float, env):
        super().__init__(env_shape, alpha, beta, gamma)
        self.env = env

    def compute_advantage(self, states, next_states, actions, cum_rewards, dones):
        # Subtract action-state baseline and add
        # expectation over actions to keep the estimate unbiased
        policy = torch.distributions.Categorical(logits=self.pi[states])
        expectation = (policy.probs * self.Q[states]).sum(dim=1).squeeze()
        future_expectation = self.accumulate(torch.cat((expectation[1:], torch.FloatTensor([0]))))

        # Get Q, V functions for episode
        v_s = self.value[next_states]
        q_a_s = self.Q[states, actions].squeeze()

        # Compute sum of future Q, V functions
        v_s = self.accumulate(torch.cat((v_s[1:], torch.FloatTensor([0]))))
        q_a_s = self.accumulate(q_a_s)

        transition_probs = torch.FloatTensor(self.env.get_transition_probs(states=states,
                                                                           actions=actions))
        pi_a_s = torch.softmax(self.pi, 1)
        pi_a_s = pi_a_s.detach().numpy().T
        self.env.compute_hindsight_probabilities_analytically(pi_a_s)
        expectation_states = (transition_probs * self.value.unsqueeze(dim=1)).sum(dim=0).squeeze()
        future_expectation_states = self.accumulate(torch.cat((expectation_states[1:],
                                                               torch.FloatTensor([0]))))

        advantage = cum_rewards - q_a_s - v_s + future_expectation_states + future_expectation
        return advantage, expectation