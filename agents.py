from itertools import groupby
from variance_estimation import rollout
import numpy as np
import torch


class RandomAgent:
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def select_action(self, state, inference):
        return np.random.randint(0, self.n_actions)

    def update(self, states, actions, rewards, dones):
        pass


# Basic reinforce
class ReinforceAgent:
    def __init__(self, env_shape: tuple, alpha: float, gamma: float, possible_rs: np.array = None):
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

        # Compute mean variance of losses per sa pair
        tuple_list = list(zip(states.numpy(), actions.numpy(), loss.detach().numpy()))
        tuple_list.sort(key=lambda x: x[:2])  # groupby needs it
        grouped = groupby(tuple_list, lambda x: x[:2])
        variances = [np.var([x[2] for x in lst]) for k, lst in grouped]
        mean_var = np.mean(variances)
        return loss, policy, mean_var

    # Does what it says, for easy re-use across methods
    # Returns policy gradient/loss values and the parametrized policy distribution, for logging
    def make_pg_step(self, states, actions, advantage):
        loss, cats, mean_var = self.compute_eligibility(states, actions, advantage)
        # TODO consider doing this manually?
        self.pi_opt.zero_grad()
        loss.mean().backward()
        self.pi_opt.step()  # SGD step

        return loss, cats, mean_var

    # Given a trajectory for one episode, update the policy
    def update(self, states, actions, rewards, next_states, dones):
        states = torch.LongTensor(states)
        actions = torch.LongTensor(actions)
        rewards = np.asarray(rewards)
        dones = torch.LongTensor(dones)

        # First compute cumulative rewards
        cum_rewards = torch.Tensor(self.accumulate_rewards(rewards, dones))

        # Next, compute elegibility
        loss, cats, mean_var = self.make_pg_step(states, actions, cum_rewards)

        return loss.mean().item(), mean_var, cats.entropy().mean().item(), cum_rewards.mean()


# Extension of reinforce to include a value function baseline
class ValueBaselineAgent(ReinforceAgent):
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
    def compute_advantage(self, states, cum_rewards):
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
        adv = self.compute_advantage(states, cum_rewards)
        self.update_values(states, cum_rewards)

        # Update the policy
        loss, cats, mean_var = self.make_pg_step(states, actions, adv)

        return loss.mean().item(), mean_var, cats.entropy().mean().item(), adv.mean()


# Extension of reinforce to include a value function baseline
class PerfectValueBaselineAgent(ReinforceAgent):
    def __init__(self, env_shape: tuple, alpha: float, gamma: float,
                 env, episode_length):
        super().__init__(env_shape, alpha, gamma)
        self.env = env
        self.episode_length = episode_length

    # Compute value-baseline advantage values, and update the value function
    def compute_advantage(self, states, cum_rewards, dones):
        pi_a_s = torch.softmax(self.pi, 1)
        pi_a_s = pi_a_s.detach().numpy().T

        self.env.initialize_hindsight(self.episode_length, pi_a_s)

        # here we assume that data is sequential and begins with the start of the episode
        extended_done_ids = np.flatnonzero([True] + dones)
        ts = []
        for i in range(len(extended_done_ids) - 1):
            ts.extend(list(range(extended_done_ids[i + 1] - extended_done_ids[i])))

        vals = self.env.get_state_value(states=states, ts=ts)
        vals = torch.FloatTensor(vals)
        advantage = cum_rewards - vals
        return advantage

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.LongTensor(states)
        actions = torch.LongTensor(actions)
        rewards = np.asarray(rewards)

        # First compute cumulative rewards
        cum_rewards = torch.Tensor(self.accumulate_rewards(rewards, dones))

        # Subtract value baseline
        adv = self.compute_advantage(states, cum_rewards, dones)

        # Update the policy
        loss, cats, mean_var = self.make_pg_step(states, actions, adv)

        return loss.mean().item(), mean_var, cats.entropy().mean().item(), adv.mean()


class OptimalStateBaselineAgent(ReinforceAgent):
    def __init__(self, env_shape: tuple, alpha: float, gamma: float,
                 env, episode_length):
        super().__init__(env_shape, alpha, gamma)
        self.env = env
        self.episode_length = episode_length
        self.optimal_baseline = None

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
            ep_states, ep_acts, ep_rewards, ep_next_states, ep_dones = rollout(self.env,
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


# Similar to ValueBaselineAgent, but with a multiplier on the value baseline (so it can be made bigger or added instead)
class ValueModAgent(ValueBaselineAgent):
    def __init__(self, env_shape: tuple, alpha: float, gamma: float, value_mult: float = 1):
        super().__init__(env_shape, alpha, gamma)
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


# Extension of reinforce to include a Q function reward target
class QTargetAgent(ReinforceAgent):
    def __init__(self, env_shape: tuple, alpha: float, gamma: float):
        super().__init__(env_shape, alpha, gamma)

        self.Q = torch.zeros(env_shape, dtype=torch.float)

    # Compute q-baseline advantage values, and update the q function
    # Unlike other methods, the policy target here is the Q-value directly
    def compute_update_q_values(self, states, actions, cum_rewards):
        # Subtract Q baseline
        qs = torch.index_select(self.Q, 0, states)
        qs = torch.gather(qs, 1, actions.unsqueeze(1)).squeeze()
        q_loss = cum_rewards - qs
        adv = qs.detach()

        # Compute manual value loss
        for s in range(q_loss.size(0)):
            self.Q[states[s], actions[s]] += q_loss[s] * (self.alpha/q_loss.size(0))

        return adv, q_loss

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.LongTensor(states)
        actions = torch.LongTensor(actions)
        rewards = np.asarray(rewards)

        # First compute cumulative rewards
        cum_rewards = torch.Tensor(self.accumulate_rewards(rewards, dones))

        # Update Q values and get targets
        adv, q_loss = self.compute_update_q_values(states, actions, cum_rewards)

        # Update the policy
        loss, cats, mean_var = self.make_pg_step(states, actions, adv)

        return loss.mean().item(), mean_var, cats.entropy().mean().item(), adv.mean()


class QAdvantageAgent(ValueBaselineAgent, QTargetAgent):
    def __init__(self, env_shape: tuple, alpha: float, gamma: float):
        super(ValueBaselineAgent, self).__init__(env_shape, alpha, gamma)

        self.Q = torch.zeros(env_shape, dtype=torch.float)
        self.value = torch.zeros(env_shape[0], dtype=torch.float)

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.LongTensor(states)
        actions = torch.LongTensor(actions)
        rewards = np.asarray(rewards)

        # First compute cumulative rewards
        cum_rewards = torch.Tensor(self.accumulate_rewards(rewards, dones))

        # Update Q and get targets
        qs, _ = self.compute_update_q_values(states, actions, cum_rewards)

        # Next subtract value baseline from qs to get advantage
        adv, _ = self.compute_update_value_advantage(states, qs)

        # Update the policy
        loss, cats, mean_var = self.make_pg_step(states, actions, adv)

        return loss.mean().item(), mean_var, cats.entropy().mean().item(), adv.mean()


class ActionStateBaselineAgent(ReinforceAgent):
    def __init__(self, env_shape: tuple, alpha: float, beta: float, gamma: float):
        super().__init__(env_shape, alpha, gamma)

        self.Q = torch.zeros(env_shape, dtype=torch.float, requires_grad=True)
        self.beta = beta
        self.q_opt = torch.optim.SGD(params=[self.Q], lr=self.beta)

    # Compute q-baseline advantage values, and update the q function
    # Unlike other methods, the policy target here is the Q-value directly
    def update_q_values(self, states, actions, cum_rewards):
        # Subtract Q baseline
        qs = torch.gather(self.Q[states], 1, actions.unsqueeze(1)).squeeze()
        q_loss = (cum_rewards - qs) ** 2 / 2

        # Compute manual value loss
        self.q_opt.zero_grad()
        q_loss.mean().backward()
        self.q_opt.step()

        return q_loss

    def compute_eligibility(self, states, actions, advantage):
        logits = self.pi[states]
        policy = torch.distributions.Categorical(logits=logits)
        #compute expecation over action~pi of grad_log_pi(a|s) * Q(a, s)
        E_pi_a_s_q = (policy.probs * self.Q[states]).sum(dim=1).squeeze()
        log_probs = policy.log_prob(actions)
        loss = - log_probs * advantage.detach() - E_pi_a_s_q
        

        # Compute mean variance of losses per sa pair
        tuple_list = list(zip(states.numpy(), actions.numpy(), loss.detach().numpy()))
        tuple_list.sort(key=lambda x: x[:2])  # groupby needs it
        grouped = groupby(tuple_list, lambda x: x[:2])
        variances = [np.var([x[2] for x in lst]) for k, lst in grouped]
        mean_var = np.mean(variances)
        return loss, policy, mean_var
    
    # Compute value-baseline advantage values, and update the value function
    def compute_advantage(self, states, actions, cum_rewards):
        # Subtract action-state baseline and add expectation over actions to keep the estimate unbiased
        q_a_s = torch.gather(self.Q[states], 1, actions.unsqueeze(1)).squeeze()
        advantage = cum_rewards - q_a_s.detach()
        return advantage 

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.LongTensor(states)
        actions = torch.LongTensor(actions)
        rewards = np.asarray(rewards)

        # First compute cumulative rewards
        cum_rewards = torch.Tensor(self.accumulate_rewards(rewards, dones))

        # Update Q values and get targets
        _ = self.update_q_values(states, actions, cum_rewards)
        adv = self.compute_advantage(states, actions, cum_rewards)

        # Update the policy
        loss, cats, mean_var = self.make_pg_step(states, actions, adv)

        return loss.mean().item(), mean_var, cats.entropy().mean().item(), adv.mean()


class PerfectActionStateBaselineAgent(ReinforceAgent):
    def __init__(self, env_shape: tuple, alpha: float, gamma: float, 
                 env, episode_length):
        super().__init__(env_shape, alpha, gamma)
        self.env = env
        self.episode_length = episode_length


    def compute_eligibility(self, states, actions, advantage, expectation):
        logits = self.pi[states]
        policy = torch.distributions.Categorical(logits=logits)
        #compute expecation over action~pi of grad_log_pi(a|s) * Q(a, s)
        log_probs = policy.log_prob(actions)
        loss = - log_probs * advantage - expectation

        # Compute mean variance of losses per sa pair
        tuple_list = list(zip(states.numpy(), actions.numpy(), loss.detach().numpy()))
        tuple_list.sort(key=lambda x: x[:2])  # groupby needs it
        grouped = groupby(tuple_list, lambda x: x[:2])
        variances = [np.var([x[2] for x in lst]) for k, lst in grouped]
        mean_var = np.mean(variances)
        return loss, policy, mean_var
    

    def compute_advantage(self, states, actions, cum_rewards, dones):
        pi_a_s = torch.softmax(self.pi, 1)
        pi_a_s = pi_a_s.detach().numpy().T

        self.env.initialize_hindsight(self.episode_length, pi_a_s)

        # here we assume that data is sequential and begins with the start of the episode
        extended_done_ids = np.flatnonzero([True] + dones)
        ts = []
        for i in range(len(extended_done_ids) - 1):
            ts.extend(list(range(extended_done_ids[i + 1] - extended_done_ids[i])))

        q_s = self.env.get_state_all_action_values(states=states, ts=ts)
        q_s = torch.FloatTensor(q_s)
        policy = torch.distributions.Categorical(logits=self.pi[states])
        expectation = (policy.probs * q_s).sum(dim=1).squeeze()

        q_a_s = self.env.get_state_action_value(states=states, actions=actions, ts=ts)
        q_a_s = torch.FloatTensor(q_a_s)
        advantage = cum_rewards - q_a_s
        return advantage, expectation

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.LongTensor(states)
        actions = torch.LongTensor(actions)
        rewards = np.asarray(rewards)

        # First compute cumulative rewards
        cum_rewards = torch.Tensor(self.accumulate_rewards(rewards, dones))

        # Update Q values and get targets
        adv, expectation = self.compute_advantage(states, actions, cum_rewards, dones)

        # Update the policy
        loss, cats, mean_var = self.make_pg_step(states, actions, adv, expectation)

        return loss.mean().item(), mean_var, cats.entropy().mean().item(), adv.mean()

    def make_pg_step(self, states, actions, advantage, expectation):
        loss, cats, mean_var = self.compute_eligibility(states, actions, advantage, expectation)
        # TODO consider doing this manually?
        self.pi_opt.zero_grad()
        loss.mean().backward()
        self.pi_opt.step()  # SGD step

        return loss, cats, mean_var


class TrajectoryCVAgent(ActionStateBaselineAgent):
    def __init__(self, env_shape: tuple, alpha: float, beta: float, gamma: float):
        super().__init__(env_shape, alpha, beta, gamma)

    def accumulate(self, values):
        return torch.flip(torch.cumsum(torch.flip(values, dims=[0]), 0), dims=[0])
    
    def compute_eligibility(self, states, actions, advantage):
        logits = self.pi[states]
        policy = torch.distributions.Categorical(logits=logits)
        # Compute expectation over action~pi of grad_log_pi(a|s) * Q(a, s)
        expectation = (policy.probs * self.Q[states]).sum(dim=1).squeeze()
        log_probs = policy.log_prob(actions)
        #C ompute expectations of future Q functions to make estimate unbiased

        future_expectation = self.accumulate(torch.cat((expectation[1:], torch.FloatTensor([0]))))
        loss = - log_probs * (advantage.detach() + future_expectation.detach()) - expectation
        
        # Compute mean variance of losses per sa pair
        tuple_list = list(zip(states.numpy(), actions.numpy(), loss.detach().numpy()))
        tuple_list.sort(key=lambda x: x[:2])  # groupby needs it
        grouped = groupby(tuple_list, lambda x: x[:2])
        variances = [np.var([x[2] for x in lst]) for k, lst in grouped]
        mean_var = np.mean(variances)
        return loss, policy, mean_var
    
    # Compute value-baseline advantage values, and update the value function
    def compute_advantage(self, states, actions, cum_rewards):
        # Subtract action-state baseline and add expectation over actions to keep the estimate unbiased
        q_a_s = torch.gather(self.Q[states], 1, actions.unsqueeze(1)).squeeze()
        # Compute sum of future Q functions 
        q_a_s = self.accumulate(q_a_s)
        advantage = cum_rewards - q_a_s
        return advantage

class PerfectTrajectoryCVAgent(PerfectActionStateBaselineAgent):
    def __init__(self, env_shape: tuple, alpha: float, gamma: float, 
                 env, episode_length):
        super().__init__(env_shape, alpha, gamma, env, episode_length)

    def accumulate(self, values):
        return torch.flip(torch.cumsum(torch.flip(values, dims=[0]), 0), dims=[0])

    def compute_eligibility(self, states, actions, advantage, expectation):
        logits = self.pi[states]
        policy = torch.distributions.Categorical(logits=logits)
        #compute expecation over action~pi of grad_log_pi(a|s) * Q(a, s)
        log_probs = policy.log_prob(actions)
        loss = -log_probs * advantage.detach()  - expectation

        # Compute mean variance of losses per sa pair
        tuple_list = list(zip(states.numpy(), actions.numpy(), loss.detach().numpy()))
        tuple_list.sort(key=lambda x: x[:2])  # groupby needs it
        grouped = groupby(tuple_list, lambda x: x[:2])
        variances = [np.var([x[2] for x in lst]) for k, lst in grouped]
        mean_var = np.mean(variances)
        return loss, policy, mean_var
    

    def compute_advantage(self, states, actions, cum_rewards, dones):
        pi_a_s = torch.softmax(self.pi, 1)
        pi_a_s = pi_a_s.detach().numpy().T

        self.env.initialize_hindsight(self.episode_length, pi_a_s)

        # here we assume that data is sequential and begins with the start of the episode
        extended_done_ids = np.flatnonzero([True] + dones)
        ts = []
        for i in range(len(extended_done_ids) - 1):
            ts.extend(list(range(extended_done_ids[i + 1] - extended_done_ids[i])))

        q_s = self.env.get_state_all_action_values(states=states, ts=ts)
        q_s = torch.FloatTensor(q_s)
        policy = torch.distributions.Categorical(logits=self.pi[states])
        expectation = (policy.probs * q_s).sum(dim=1).squeeze()
        future_expectation = self.accumulate(torch.cat((expectation[1:], torch.FloatTensor([0]))))

        q_a_s = self.env.get_state_action_value(states=states, actions=actions, ts=ts)
        q_a_s = torch.FloatTensor(q_a_s)
        # Compute sum of future Q functions 
        q_a_s = self.accumulate(q_a_s)
        advantage = cum_rewards - q_a_s + future_expectation
        return advantage, expectation


class PerfectDynamicsTrajCVAgent(PerfectTrajectoryCVAgent):
    def __init__(self, env_shape: tuple, alpha: float, gamma: float, 
                 env, episode_length):
        super().__init__(env_shape, alpha, gamma, env, episode_length)

    def compute_advantage(self, states, actions, cum_rewards, dones):
        pi_a_s = torch.softmax(self.pi, 1)
        pi_a_s = pi_a_s.detach().numpy().T

        self.env.initialize_hindsight(self.episode_length, pi_a_s)

        # here we assume that data is sequential and begins with the start of the episode
        extended_done_ids = np.flatnonzero([True] + dones)
        ts = []
        for i in range(len(extended_done_ids) - 1):
            ts.extend(list(range(extended_done_ids[i + 1] - extended_done_ids[i])))

        # Compute Value functions in next states
        
        q_s = self.env.get_state_all_action_values(states=states, ts=ts)
        q_s = torch.FloatTensor(q_s)
        policy = torch.distributions.Categorical(logits=self.pi[states])
        expectation = (policy.probs * q_s).sum(dim=1).squeeze()
        future_expectation = self.accumulate(torch.cat((expectation[1:], torch.FloatTensor([0]))))

        v_all_s = self.env.get_state_all_values(ts=ts).T
        v_all_s = torch.FloatTensor(v_all_s)
        transition_probs = self.env.get_transition_probs(states=states, actions=actions)
        transition_probs = torch.FloatTensor(transition_probs)
        expectation_states = (transition_probs * v_all_s).sum(dim=0).squeeze()
        future_expectation_states = self.accumulate(torch.cat((expectation_states[1:], torch.FloatTensor([0]))))

        v_s = self.env.get_state_value(states=states, ts=ts)
        v_s = torch.FloatTensor(v_s)
        # Compute sum of future V functions 
        v_s = self.accumulate(torch.cat((v_s[1:], torch.FloatTensor([0]))))

        q_a_s = self.env.get_state_action_value(states=states, actions=actions, ts=ts)
        q_a_s = torch.FloatTensor(q_a_s)
        # Compute sum of future Q functions 
        q_a_s = self.accumulate(q_a_s)
        advantage = cum_rewards - q_a_s - v_s + future_expectation + future_expectation_states
        return advantage, expectation    


class DynamicsTrajCVAgent(TrajectoryCVAgent):
    def __init__(self, env_shape: tuple, alpha: float, gamma: float, beta: float):
        super().__init__(env_shape, alpha, beta, gamma)

        self.p_s_sa = torch.zeros((env_shape[0], env_shape[0], env_shape[1]), dtype=torch.float, requires_grad=False)
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
    
    def update_transition_probs(self, states, actions):
        for next_state, state, action in zip(states[1:], states[:-1], actions[:-1]):
            self.p_s_sa[next_state, state, action] += 1
    
    def compute_advantage(self, states, actions, cum_rewards, dones):
        # Subtract action-state baseline and add expectation over actions to keep the estimate unbiased
        q_a_s = torch.gather(self.Q[states], 1, actions.unsqueeze(1)).squeeze()
        v_s = self.value[states]
        # Compute sum of future Q functions
        v_s = self.accumulate(torch.cat((v_s[1:], torch.FloatTensor([0]))))
        q_a_s = self.accumulate(q_a_s)

        transition_probs = self.get_transition_probs(states, actions)
        expectation_states = (transition_probs * self.value.unsqueeze(dim=1)).sum(dim=0).squeeze()
        future_expectation_states = self.accumulate(torch.cat((expectation_states[1:], torch.FloatTensor([0]))))

        advantage = cum_rewards - q_a_s - v_s + future_expectation_states
        return advantage

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
        self.update_transition_probs(states, actions)

        #Get targets
        adv = self.compute_advantage(states, actions, cum_rewards, dones)

        # Update the policy
        loss, cats, mean_var = self.make_pg_step(states, actions, adv)

        return loss.mean().item(), mean_var, cats.entropy().mean().item(), adv.mean()


class PerfectDynamicsEstQVTrajCVAgent(DynamicsTrajCVAgent):
    def __init__(self, env_shape: tuple, alpha: float, gamma: float, beta: float, 
                 env, episode_length):
        super().__init__(env_shape, alpha, beta, gamma)

        self.env = env
        self.episode_length = episode_length

    def compute_advantage(self, states, actions, cum_rewards, dones):
        pi_a_s = torch.softmax(self.pi, 1)
        pi_a_s = pi_a_s.detach().numpy().T

        self.env.initialize_hindsight(self.episode_length, pi_a_s)

        # here we assume that data is sequential and begins with the start of the episode
        extended_done_ids = np.flatnonzero([True] + dones)
        ts = []
        for i in range(len(extended_done_ids) - 1):
            ts.extend(list(range(extended_done_ids[i + 1] - extended_done_ids[i])))

        # Compute Value functions in next states
        
        q_s_a = self.Q[states, actions]
        q_s_a = self.accumulate(q_s_a)
        v_s = self.value[states]
        v_s = self.accumulate(torch.cat((v_s[1:], torch.zeros(1))))

        transition_probs = self.env.get_transition_probs(states=states, actions=actions)
        transition_probs = torch.FloatTensor(transition_probs)
        expectation_states = (transition_probs * self.value.unsqueeze(1)).sum(dim=0).squeeze()
        future_expectation_states = self.accumulate(torch.cat((expectation_states[1:], torch.FloatTensor([0]))))

        #v_s = self.env.get_state_value(states=states, ts=ts)
        #v_s = torch.FloatTensor(v_s)
        # Compute sum of future V functions 
        #v_s = self.accumulate(torch.cat((v_s[1:], torch.FloatTensor([0]))))

        # Compute sum of future Q functions 
        advantage = cum_rewards - q_s_a - v_s + future_expectation_states
        return advantage

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.LongTensor(states)
        actions = torch.LongTensor(actions)
        rewards = np.asarray(rewards)

        # First compute cumulative rewards
        cum_rewards = torch.Tensor(self.accumulate_rewards(rewards, dones))

        # Update V, Q values
        _ = self.update_values(states, cum_rewards)
        _ = self.update_q_values(states, actions, cum_rewards)
    
        #Get targets
        adv = self.compute_advantage(states, actions, cum_rewards, dones)

        # Update the policy
        loss, cats, mean_var = self.make_pg_step(states, actions, adv)

        return loss.mean().item(), mean_var, cats.entropy().mean().item(), adv.mean() 
