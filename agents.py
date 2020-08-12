from itertools import groupby
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


# Parent credit agent class, contains credit-baseline algorithm plus a lot of functions for other extensions
# Extensions should inherit this class and overload update+add functions as needed
# PSA: Avoid additional layers of inheritance!
class CreditBaselineAgent(ReinforceAgent):
    def __init__(self, env_shape: tuple, alpha: float, gamma: float, possible_rs: np.array):
        super().__init__(env_shape, alpha, gamma)

        self.n_state_actions = env_shape
        self.n_states = env_shape[0]
        self.possible_rs = possible_rs

        self.s_counts = None
        self.sa_counts = None
        self.s_r_counts = None
        self.sa_r_counts = None
        self.r_index = None
        self.reset_r_counts()

    # Reset the credit counts to be more on-policy
    def reset_r_counts(self):
        self.s_counts = torch.zeros(self.n_states)
        self.sa_counts = torch.zeros(self.n_state_actions)
        # Index from possible reward values to state-action conditioned probabilities
        self.s_r_counts = torch.zeros(self.n_states, len(self.possible_rs))
        self.sa_r_counts = torch.zeros(self.n_state_actions + (len(self.possible_rs),))
        # Mapping from reward values to indices in the count tensors
        self.r_index = dict()
        c = 0
        for possible_r in self.possible_rs:
            self.r_index[possible_r] = c
            c += 1

    # Do a dict lookup for each reward and return same size vector of indices from self.r_index
    # TODO do this not the naive way
    def reward_lookup(self, reward_vec):
        return_vec = []
        for r in reward_vec:
            return_vec.append(self.r_index[r.item()])
        return torch.as_tensor(return_vec)

    # Get P(r|s)
    def get_prs(self, states, rewards):
        r_inds = self.reward_lookup(rewards)
        s_count = torch.index_select(self.s_counts, 0, states)
        s_r_count = torch.index_select(self.s_r_counts, 0, states)
        s_r_count = torch.gather(s_r_count, 1, r_inds.unsqueeze(1)).squeeze()

        prs = s_r_count / s_count
        # Just in case, shouldn't be needed unless we're off-policy for some reason
        prs[prs != prs] = 0

        return prs

    # Get P(r|s,a)
    def get_prsa(self, states, actions, rewards):
        r_inds = self.reward_lookup(rewards)
        sa_count = torch.index_select(self.sa_counts, 0, states)
        sa_count = torch.gather(sa_count, 1, actions.unsqueeze(1)).squeeze()
        sa_r_count = torch.index_select(self.sa_r_counts, 0, states)

        # Horrendous, vector indexing 3+D tensors sucks...
        sa_r_count = torch.gather(sa_r_count, 1,
                                  actions.unsqueeze(1).unsqueeze(2).expand(states.size(0), 1, sa_r_count.size(2)))
        sa_r_count = torch.gather(sa_r_count, 2, r_inds.unsqueeze(1).unsqueeze(2)).squeeze()

        prsa = sa_r_count / sa_count
        # For the case where action has not been sampled before, replace with prs
        if prsa.ndimension() == 0:
            prsa = prsa.unsqueeze(0)
        prsa[prsa != prsa] = self.get_prs(states, rewards)[prsa != prsa]

        return prsa

    # Update credit counts
    # CALL THIS BEFORE COMPUTING PRS/PRSA OR RISK DIVIDE BY 0!
    def update_credit_counts(self, states, actions, rewards):
        for t in range(states.size(0)):
            self.s_counts[states[t]] += 1
            self.sa_counts[states[t], actions[t]] += 1
            # TODO make this work with gamma <1 !
            r_ind = self.r_index[rewards[t].item()]
            self.s_r_counts[states[t], r_ind] += 1
            self.sa_r_counts[states[t], actions[t], r_ind] += 1

        for arr in [self.s_counts, self.sa_counts, self.s_r_counts, self.sa_r_counts]:
            arr *= 0.9

    # Computes credit, but with a mixture param
    def compute_credit_mixture(self, states, actions, cum_rewards, mix_ratio):

        # Now get the updated credit to use as a baseline
        prs = self.get_prs(states, cum_rewards)
        prsa = self.get_prsa(states, actions, cum_rewards)

        credit_ratio = prs / (mix_ratio * prsa + (1 - mix_ratio)*prs)  # Denominator mixture

        return credit_ratio

    # Computes credit only
    def compute_credit(self, states, actions, cum_rewards):

        # Now get the updated credit to use as a baseline
        prs = self.get_prs(states, cum_rewards)
        prsa = self.get_prsa(states, actions, cum_rewards)

        credit_ratio = prs / (prsa + 0.00001)

        return credit_ratio

    # Update credit counts and return the credit ratio for the current batch
    def compute_update_credit(self, states, actions, cum_rewards):

        self.update_credit_counts(states, actions, cum_rewards)

        # Now get the updated credit to use as a baseline
        prs = self.get_prs(states, cum_rewards)
        prsa = self.get_prsa(states, actions, cum_rewards)

        credit_ratio = prs / (prsa + 0.00001)

        return credit_ratio

    # As compute_update_credit, but with a mixture parameter between P(r|s) and P(r|s,a) in the denominator
    def compute_update_credit_mixture(self, states, actions, cum_rewards, mix_ratio):

        self.update_credit_counts(states, actions, cum_rewards)

        # Now get the updated credit to use as a baseline
        prs = self.get_prs(states, cum_rewards)
        prsa = self.get_prsa(states, actions, cum_rewards)

        credit_ratio = prs / (mix_ratio * prsa + (1 - mix_ratio)*prs)  # Denominator mixture

        return credit_ratio

    # Given a trajectory for one episode, update the policy
    def update(self, states, actions, rewards, next_states, dones):
        states = torch.LongTensor(states)
        actions = torch.LongTensor(actions)
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)

        # First compute cumulative rewards
        cum_rewards = torch.Tensor(self.accumulate_rewards(rewards, dones))

        # Update credit counts and compute credit ratio
        credit_ratio = self.compute_update_credit(states, actions, cum_rewards)

        advantage = cum_rewards * (1 - credit_ratio)

        # Next, compute eligibility
        loss, cats, mean_var = self.make_pg_step(states, actions, advantage)

        return loss.mean().item(), mean_var, cats.entropy().mean().item(), advantage.mean()


class PerfectCreditBaselineAgent(ReinforceAgent):
    def __init__(self, env_shape: tuple, alpha: float, gamma: float, possible_rs: np.array, env, episode_length):
        super().__init__(env_shape, alpha, gamma)

        self.env = env
        self.episode_length = episode_length
        self.n_state_actions = env_shape
        self.n_states = env_shape[0]
        self.possible_rs = possible_rs


    # Computes credit only
    def compute_credit(self, states, actions, rewards, dones):
        pi_a_s = torch.softmax(self.pi, 1)
        pi_a_s = pi_a_s.detach().numpy().T

        self.env.initialize_hindsight(self.episode_length, pi_a_s)

        # here we assume that data is sequential and begins with the start of the episode
        extended_done_ids = np.flatnonzero([True] + dones.tolist())
        ts = []
        for i in range(len(extended_done_ids) - 1):
            ts.extend(list(range(extended_done_ids[i + 1] - extended_done_ids[i])))

        # is part of rewarding trajectory
        successes = np.empty_like(states, dtype='bool')
        for i in range(len(extended_done_ids) - 1):
            start = extended_done_ids[i]
            end = extended_done_ids[i + 1]
            successes[start:end] = (rewards[end - 1] == 1)

        pt_a_sz = self.env.get_hindsight_probability(states=states, actions=actions, ts=ts, successes=successes)
        credit_ratio = pi_a_s[actions, states] / pt_a_sz

        return credit_ratio

    # Given a trajectory for one episode, update the policy
    def update(self, states, actions, rewards, next_states, dones):
        states = torch.LongTensor(states)
        actions = torch.LongTensor(actions)
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)

        # First compute cumulative rewards
        cum_rewards = torch.Tensor(self.accumulate_rewards(rewards, dones))

        # Get credit ratios from environment, mark undefined
        credit_ratio = self.compute_credit(states, actions, rewards, dones)
        credit_ratio = torch.FloatTensor(credit_ratio)

        advantage = cum_rewards * (1 - credit_ratio)

        # Next, compute eligibility
        loss, cats, mean_var = self.make_pg_step(states, actions, advantage)

        return loss.mean().item(), mean_var, cats.entropy().mean().item(), advantage.mean()

# =======================================================================================================
# =======================================================================================================


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


# Implements MICA, a.k.a. advantage=r*P(r|s,a)/P(r|s)
class MICAAgent(CreditBaselineAgent):
    def __init__(self, env_shape: np.array, alpha: float, gamma: float, possible_rs: np.array):
        super().__init__(env_shape, alpha, gamma, possible_rs)

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.LongTensor(states)
        actions = torch.LongTensor(actions)
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)

        # First compute cumulative rewards
        cum_rewards = torch.Tensor(self.accumulate_rewards(rewards, dones))

        # Update credit counts and compute credit ratio
        credit_ratio = self.compute_update_credit(states, actions, cum_rewards)

        advantage = cum_rewards * (1/credit_ratio - 1)

        # Next, compute eligibility
        loss, cats, mean_var = self.make_pg_step(states, actions, advantage)

        return loss.mean().item(), mean_var, cats.entropy().mean().item(), advantage.mean()


# Extension of MICA, computes advantage=r*P(r|s,a)/P(r|s) - V(s)
class MICAValueAgent(CreditBaselineAgent):
    def __init__(self, env_shape: tuple, alpha: float, gamma: float, possible_rs: np.array):
        super().__init__(env_shape, alpha, gamma, possible_rs)

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.LongTensor(states)
        actions = torch.LongTensor(actions)
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)

        # First compute cumulative rewards
        cum_rewards = torch.Tensor(self.accumulate_rewards(rewards, dones))

        # Update credit counts and compute credit ratio
        credit_ratio = self.compute_update_credit(states, actions, cum_rewards)

        # Weight rewards with credit
        advantage = cum_rewards * (1/credit_ratio - 1)

        # Compute value advantage
        advantage, val_loss = self.compute_update_value_advantage(states, advantage)

        # Next, compute eligibility
        loss, cats, mean_var = self.make_pg_step(states, actions, advantage)

        return loss.mean().item(), mean_var, cats.entropy().mean().item(), advantage.mean()


# As Credit baseline, but with a mixture of P(r|s,a)  and P(r|s) in the denominator
class CreditBaselineMixtureAgent(CreditBaselineAgent):
    def __init__(self, env_shape: tuple, alpha: float, gamma: float, possible_rs: np.array, mix_ratio):
        super().__init__(env_shape, alpha, gamma, possible_rs)
        self.mix_ratio = mix_ratio

    # Given a trajectory for one episode, update the policy
    def update(self, states, actions, rewards, next_states, dones):
        states = torch.LongTensor(states)
        actions = torch.LongTensor(actions)
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)

        # First compute cumulative rewards
        cum_rewards = torch.Tensor(self.accumulate_rewards(rewards, dones))

        # Update credit counts and compute credit ratio
        credit_ratio = self.compute_update_credit_mixture(states, actions, cum_rewards, self.mix_ratio)

        advantage = cum_rewards * (1 - credit_ratio)

        # Next, compute eligibility
        loss, cats, mean_var = self.make_pg_step(states, actions, advantage)

        return loss.mean().item(), mean_var, cats.entropy().mean().item(), advantage.mean()


# Like above, but with added counterfactual action training
class CreditBaselineMixtureCounterfactualAgent(CreditBaselineAgent):
    def __init__(self, env_shape: tuple, alpha: float, gamma: float, possible_rs: np.array, mix_ratio):
        super().__init__(env_shape, alpha, gamma, possible_rs)
        self.mix_ratio = mix_ratio

    # Given a trajectory for one episode, update the policy
    def update(self, states, actions, rewards, next_states, dones):
        states = torch.LongTensor(states)
        actions = torch.LongTensor(actions)
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)

        # First compute cumulative rewards
        cum_rewards = torch.Tensor(self.accumulate_rewards(rewards, dones))

        # Update credit counts and compute credit ratio
        credit_ratio = self.compute_update_credit_mixture(states, actions, cum_rewards, self.mix_ratio)

        # Sample a batch of alternative actions from the policy
        alt_actions = self.select_action(states, False)

        # If we have not seen this action before, we assume P(r|s,a)=P(r|s)
        alt_credit_ratio = self.compute_credit_mixture(states, alt_actions, cum_rewards, self.mix_ratio)

        advantage = cum_rewards * (1 - credit_ratio)

        alt_advantage = cum_rewards * (1 - alt_credit_ratio)

        # Next, compute eligibility for each set of actions
        loss, cats, mean_var = self.compute_eligibility(states, actions, advantage)
        alt_loss, alt_cats, alt_mean_var = self.compute_eligibility(states, alt_actions, alt_advantage)

        # Now update the combined objective
        self.pi_opt.zero_grad()
        (loss + alt_loss).mean().backward()
        self.pi_opt.step()  # SGD step

        return loss.mean().item(), mean_var, cats.entropy().mean().item(), advantage.mean()


class MixtureVBaselineAgent(CreditBaselineAgent):
    def __init__(self, env_shape: tuple, alpha: float, gamma: float, possible_rs: np.array, mix_ratio):
        super().__init__(env_shape, alpha, gamma, possible_rs)
        self.mix_ratio = mix_ratio

    # Given a trajectory for one episode, update the policy
    def update(self, states, actions, rewards, next_states, dones):
        states = torch.LongTensor(states)
        actions = torch.LongTensor(actions)
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)

        # First compute cumulative rewards
        cum_rewards = torch.Tensor(self.accumulate_rewards(rewards, dones))
        self.update_values(states, cum_rewards)
        vals = torch.index_select(self.value, 0, states)
        adv = cum_rewards - vals

        # Update credit counts and compute credit ratio
        credit_ratio = self.compute_update_credit_mixture(states, actions, cum_rewards, self.mix_ratio)

        # Sample a batch of alternative actions from the policy
        alt_actions = self.select_action(states, False)

        # If we have not seen this action before, we assume P(r|s,a)=P(r|s)
        alt_credit_ratio = self.compute_credit_mixture(states, alt_actions, cum_rewards, self.mix_ratio)

        advantage = adv * (1 - credit_ratio)

        alt_advantage = adv * (1 - alt_credit_ratio)

        # Next, compute eligibility for each set of actions
        loss, cats, mean_var = self.compute_eligibility(states, actions, advantage)
        alt_loss, alt_cats, alt_mean_var = self.compute_eligibility(states, alt_actions, alt_advantage)

        # Now update the combined objective
        self.pi_opt.zero_grad()
        (loss + alt_loss).mean().backward()
        self.pi_opt.step()  # SGD step

        return loss.mean().item(), mean_var, cats.entropy().mean().item(), advantage.mean()


# Like above, but with only counterfactual training and no mixture
class CreditBaselineCounterfactualAgent(CreditBaselineAgent):
    def __init__(self, env_shape: tuple, alpha: float, gamma: float, possible_rs: np.array):
        super().__init__(env_shape, alpha, gamma, possible_rs)

    # Given a trajectory for one episode, update the policy
    def update(self, states, actions, rewards, next_states, dones):
        states = torch.LongTensor(states)
        actions = torch.LongTensor(actions)
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)

        # First compute cumulative rewards
        cum_rewards = torch.Tensor(self.accumulate_rewards(rewards, dones))

        # Update credit counts and compute credit ratio
        credit_ratio = self.compute_update_credit(states, actions, cum_rewards)

        # Sample a batch of alternative actions from the policy
        alt_actions = self.select_action(states, False)

        # If we have not seen this action before, we assume P(r|s,a)=P(r|s)
        alt_credit_ratio = self.compute_credit(states, alt_actions, cum_rewards)

        advantage = cum_rewards * (1 - credit_ratio)

        alt_advantage = cum_rewards * (1 - alt_credit_ratio)

        # Next, compute eligibility for each set of actions
        loss, cats, mean_var = self.compute_eligibility(states, actions, advantage)
        alt_loss, alt_cats, alt_mean_var = self.compute_eligibility(states, alt_actions, alt_advantage)

        # Now update the combined objective
        self.pi_opt.zero_grad()
        (loss + alt_loss).mean().backward()
        self.pi_opt.step()  # SGD step

        return loss.mean().item(), mean_var, cats.entropy().mean().item(), advantage.mean()


# Alternative to the credit baseline mixture model devised by Arsenii
# Does contrastive/negative credit sampling for training the policy
# May require more complex probability training than counts, not sure yet
# TODO make this not assume mix=0.5
class MICAMixtureCounterfactualAgent(CreditBaselineAgent):
    def __init__(self, env_shape: tuple, alpha: float, gamma: float, possible_rs: np.array, mix_ratio):
        super().__init__(env_shape, alpha, gamma, possible_rs)
        self.mix_ratio = mix_ratio

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.LongTensor(states)
        actions = torch.LongTensor(actions)
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)

        # First compute cumulative rewards
        cum_rewards = torch.Tensor(self.accumulate_rewards(rewards, dones))

        # Sample a batch of alternative actions from the policy
        alt_actions = self.select_action(states, False)

        # Update credit counts and compute credit ratio
        credit_ratio = self.compute_update_credit_mixture(states, actions, cum_rewards, self.mix_ratio)

        # If we have not seen this action before, we assume P(r|s,a)=P(r|s)
        alt_credit_ratio = self.compute_credit_mixture(states, alt_actions, cum_rewards, self.mix_ratio)

        advantage = cum_rewards * (1/credit_ratio - 1)

        alt_advantage = cum_rewards * (1/alt_credit_ratio - 1)

        # Next, compute eligibility for each set of actions
        loss, cats, mean_var = self.compute_eligibility(states, actions, advantage)
        alt_loss, alt_cats, alt_mean_var = self.compute_eligibility(states, alt_actions, alt_advantage)

        # Now update the combined objective
        self.pi_opt.zero_grad()
        (loss + alt_loss).mean().backward()
        self.pi_opt.step()  # SGD step

        return loss.mean().item(), mean_var, cats.entropy().mean().item(), advantage.mean()


# Mixture credit, but for MICA formulation
class MICAMixtureAgent(CreditBaselineAgent):
    def __init__(self, env_shape: tuple, alpha: float, gamma: float, possible_rs: np.array, mix_ratio):
        super().__init__(env_shape, alpha, gamma, possible_rs)
        self.mix_ratio = mix_ratio

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.LongTensor(states)
        actions = torch.LongTensor(actions)
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)

        # First compute cumulative rewards
        cum_rewards = torch.Tensor(self.accumulate_rewards(rewards, dones))

        # Update credit counts and compute credit ratio
        credit_ratio = self.compute_update_credit_mixture(states, actions, cum_rewards, self.mix_ratio)

        advantage = cum_rewards * (1/credit_ratio - 1)

        # Next, compute eligibility for each set of actions
        loss, cats, mean_var = self.make_pg_step(states, actions, advantage)

        return loss.mean().item(), mean_var, cats.entropy().mean().item(), advantage.mean()


# MICA with counterfactual training
# Does contrastive/negative credit sampling for training the policy
class MICACounterfactualAgent(CreditBaselineAgent):
    def __init__(self, env_shape: tuple, alpha: float, gamma: float, possible_rs: np.array):
        super().__init__(env_shape, alpha, gamma, possible_rs)

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.LongTensor(states)
        actions = torch.LongTensor(actions)
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)

        # First compute cumulative rewards
        cum_rewards = torch.Tensor(self.accumulate_rewards(rewards, dones))

        # Sample a batch of alternative actions from the policy
        alt_actions = self.select_action(states, False)

        # Update credit counts and compute credit ratio
        credit_ratio = self.compute_update_credit(states, actions, cum_rewards)  # p(r|s,a)/p(r|s)

        # If we have not seen this action before, we assume P(r|s,a)=P(r|s)
        alt_credit_ratio = self.compute_credit(states, alt_actions, cum_rewards)

        advantage = cum_rewards * (1/credit_ratio - 1)

        alt_advantage = cum_rewards * (1/alt_credit_ratio - 1)

        # Next, compute eligibility for each set of actions
        loss, cats, mean_var = self.compute_eligibility(states, actions, advantage)
        alt_loss, alt_cats, alt_mean_var = self.compute_eligibility(states, alt_actions, alt_advantage)

        # Now update the combined objective
        self.pi_opt.zero_grad()
        (loss + alt_loss).mean().backward()
        self.pi_opt.step()  # SGD step

        return loss.mean().item(), mean_var, cats.entropy().mean().item(), advantage.mean()


# 1 Step Credit with added value function baseline
# TODO refactor to unify with the other 1-step credit alg, since they are very similar now.
# TODO Refactor WIP, delete this when done
class OneStepCreditWithValueAgent(CreditBaselineAgent):
    def __init__(self, env_shape: tuple, alpha: float, gamma: float,
                 possible_rs: np.array, credit_type: str = "mica"):
        super().__init__(env_shape, alpha, gamma, possible_rs)

        self.n_state_actions = env_shape
        self.n_states = env_shape[0]
        self.possible_rs = [-1, 0, 1]
        self.credit_type = credit_type

        self.reset_r_counts()

        self.value = torch.zeros(self.n_states, dtype=torch.float)

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.LongTensor(states)
        actions = torch.LongTensor(actions)
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)

        # First compute cumulative rewards
        cum_rewards = torch.Tensor(self.accumulate_rewards(rewards, dones))

        # Subtract value function baseline
        advantage, val_loss = self.compute_update_value_advantage(states, cum_rewards)

        # TODO hack
        advantage = cum_rewards
        adv_for_credit = cum_rewards

        # Update credit counts and compute credit ratio
        credit_ratio = self.compute_update_credit(states, actions, adv_for_credit)
        # Weight the rewards post-baseline subtraction
        if self.credit_type == "mica":
            advantage = advantage * (1 / credit_ratio)
        else:
            advantage = advantage * (1 - credit_ratio)

        # Next, compute eligibility
        loss, cats, mean_var = self.make_pg_step(states, actions, advantage)
        return loss.mean().item(), mean_var, cats.entropy().mean().item(), advantage.mean()


# Control case, instead of using credit, simply multiply rewards by a random value >1
class RandomMultWithValueAgent(ValueBaselineAgent):
    def __init__(self, env_shape: tuple, alpha: float, gamma: float, possible_rs: np.array):
        super().__init__(env_shape, alpha, gamma, possible_rs)

        n_states = env_shape[0]

        self.value = torch.zeros(n_states, dtype=torch.float)

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.LongTensor(states)
        actions = torch.LongTensor(actions)
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)

        # First compute cumulative rewards
        cum_rewards = torch.Tensor(self.accumulate_rewards(rewards, dones))

        # Fake credit values, randomly upweight rewards
        credit_weighted_rewards = cum_rewards * (1 + np.random.random(cum_rewards.size())*1.5)

        # Subtract value function baseline
        advantage, val_loss = self.compute_update_value_advantage(states, credit_weighted_rewards)

        # Next, compute eligibility
        loss, cats, mean_var = self.make_pg_step(states, actions, advantage)

        return loss.mean().item(), mean_var, cats.entropy().mean().item(), advantage.mean()


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
        qs, q_loss = self.compute_update_q_values(states, actions, cum_rewards)

        # Next subtract value baseline from qs to get advantage
        adv, val_loss = self.compute_update_value_advantage(states, qs)

        # Update the policy
        loss, cats, mean_var = self.make_pg_step(states, actions, adv)

        return loss.mean().item(), mean_var, cats.entropy().mean().item(), adv.mean()


class HCAStateConditionalQAgent(ValueBaselineAgent, QTargetAgent):
    def __init__(self, env_shape: tuple, alpha: float, gamma: float, episode_length: int):
        super(ValueBaselineAgent, self).__init__(env_shape, alpha, gamma)

        self.r_hat = torch.zeros(env_shape, dtype=torch.float)
        self.value = torch.zeros(env_shape[0], dtype=torch.float)

        n_states = env_shape[0]

        # state visitation
        self.s_counts = torch.zeros(n_states)

        # state action pair visitations
        self.sa_counts = torch.zeros(env_shape)

        # action visitations for every state-state pair
        self.ss_a_counts = torch.zeros((n_states, n_states, env_shape[1]))

        # precompute gamma factors for a bit of efficiency
        self.gammas = torch.FloatTensor([self.gamma ** i for i in range(episode_length)])

    def update_state_action_visits(self, states, actions, next_states, dones):
        for t in range(states.size(0)):
            self.s_counts[states[t]] += 1
            self.sa_counts[states[t], actions[t]] += 1

            future_t = t
            while True:
                self.ss_a_counts[states[t], next_states[future_t], actions[t]] += 1
                if dones[future_t]:
                    break
                # print(f"t: {t}, future_t: {future_t}")
                future_t += 1

    def get_hca_ratios(self, a, s, s_futures):
        # this is h(a|x, y) (eqn 2 from the HCA paper)
        actions_from_states = self.ss_a_counts[s, s_futures]
        h = actions_from_states[:, a] / torch.sum(actions_from_states, dim=1)

        # policy
        pi = torch.softmax(self.pi[s], 0)[a]
        return h / pi

    # TODO make sure this doesn't accumulate the single-step rewards so we can do counterfactual reward assignment
    def hca_accumulate_rewards(self, states, actions, rewards, next_states, dones):
        acc_rewards = np.zeros_like(rewards)

        curr_episode_length = 0
        for idx in range(len(rewards)-1, -1, -1):
            # end of episode
            if dones[idx]:
                curr_episode_length = 0

            curr_episode_length += 1

            future_rs = rewards[idx: idx + curr_episode_length]
            discount_factors = self.gammas[0: curr_episode_length]

            hca_ratios = self.get_hca_ratios(
                actions[idx], states[idx], next_states[idx: idx + curr_episode_length]
            )
            discounted_future_rs = future_rs * discount_factors * hca_ratios
            acc_rewards[idx] += torch.sum(discounted_future_rs)
        return acc_rewards

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.LongTensor(states)
        actions = torch.LongTensor(actions)
        rewards_torch = torch.FloatTensor(rewards)
        dones = np.asarray(dones)

        # Train hindsight distribution
        self.update_state_action_visits(states, actions, next_states, dones)

        # First compute cumulative rewards
        hca_cum_rewards = torch.Tensor(self.hca_accumulate_rewards(states, actions, rewards_torch, next_states, dones))

        # Update r_hat
        rs = torch.index_select(self.r_hat, 0, states)
        rs = torch.gather(rs, 1, actions.unsqueeze(1)).squeeze()
        rhat_loss = rewards_torch - rs
        for s in range(rhat_loss.size(0)):
            self.r_hat[states[s], actions[s]] += rhat_loss[s] * (self.alpha / rhat_loss.size(0))    

        # TODO Now do counterfactual policy updates using hca_cum_rewards and r_hat

        adv = hca_cum_rewards

        # Update the policy
        loss, cats, mean_var = self.make_pg_step(states, actions, adv)

        return loss.mean().item(), mean_var, cats.entropy().mean().item(), adv.mean()


# Similar to 1stepcreditagent, but multi-step. Credits rewards out to the max time horizon
class MultiStepCreditAgent(ReinforceAgent):
    def __init__(self, env_shape: tuple, alpha: float, gamma: float, possible_rs: np.array,
                 time_horizon: int, credit_type: str = "baseline"):
        super().__init__(env_shape, alpha, gamma)

        # Override for credit with value pre-subtraction
        possible_rs = [-1, 0, 1]
        n_states = env_shape[0]
        self.time_horizon = time_horizon
        self.value_updates = None
        self.s_counts = torch.zeros(n_states)
        self.sa_counts = torch.zeros(env_shape)
        # Index from possible reward values to state-action conditioned probabilities
        self.s_r_counts = torch.zeros(n_states, time_horizon, len(possible_rs))
        self.sa_r_counts = torch.zeros(env_shape + (time_horizon, len(possible_rs)))
        # Mapping from reward values to indices in the count tensors
        self.r_index = dict()
        c = 0
        for possible_r in possible_rs:
            self.r_index[possible_r] = c
            c += 1

        # Multi-step value estimates too
        self.value_multistep = torch.zeros(n_states, time_horizon)

        self.credit_type = credit_type

    # Do a dict lookup for each reward and return same size vector of indices from self.r_index
    # TODO do this not the naive way
    def reward_lookup(self, reward_vec):
        return_vec = []
        for r in reward_vec:
            return_vec.append(self.r_index[r.item()])
        return torch.as_tensor(return_vec)

    # Compute credit lookup for a single timestep
    def compute_credit(self, state, action, reward, t):
        # Now get the updated credit to use as a baseline
        r_ind = self.r_index[reward]
        sa_count = self.sa_counts[state, action]
        sa_r_count = self.sa_r_counts[state, action, t, r_ind]

        prsa = sa_r_count / sa_count  # sa_count should always be >0 here since we updated it before calling...

        s_count = self.s_counts[state]
        s_r_count = self.s_r_counts[state, t, r_ind]

        prs = s_r_count / s_count  # Same for s_count >0

        credit_ratio = (prs + 0.000001) / (prsa + 0.000001)

        return credit_ratio

    # Accumulate rewards across time, with credit per timestep
    def accumulate_credited_rewards(self, states, actions, rewards, dones):
        acc_rewards = np.zeros_like(rewards)
        # Iterate over current timesteps, computing their uniquely credited future reward values
        for from_ind in range(rewards.size):
            curr_sum = 0
            curr_state = states[from_ind]
            curr_action = actions[from_ind]
            curr_horizon = min(from_ind+self.time_horizon, rewards.size)

            for to_ind in range(curr_horizon-1, from_ind-1, -1):
                curr_r = rewards[to_ind]
                curr_value = self.value_multistep[curr_state, to_ind - from_ind]
                curr_adv = (curr_r - curr_value).detach()
                curr_credit = self.compute_credit(curr_state, curr_action, curr_adv.sign().item(), to_ind - from_ind)

                curr_sum *= self.gamma * (1 - dones[to_ind])  # Mask out future episodes
                # Subtract value baseline

                # variant crediting schemes here
                if self.credit_type == "mica":
                    curr_sum += curr_adv * (1/curr_credit)
                else:
                    curr_sum += curr_adv * (1 - curr_credit)

            acc_rewards[from_ind] = curr_sum
        return acc_rewards

    # Update credit counts
    # Also update multi-step value estimates
    def update_credit(self, states, actions, rewards, dones):

        # Compute value updates here for efficiency, apply them after advantage computation to avoid 0 advantages
        self.value_updates = torch.zeros_like(self.value_multistep)
        # Update probabilities for credit
        for t in range(states.size(0)):
            self.s_counts[states[t]] += 1
            self.sa_counts[states[t], actions[t]] += 1

            r_ind = self.r_index[rewards[t].item()]
            to_ind = t
            # Credit current timestep
            self.s_r_counts[states[t], to_ind - t, r_ind] += 1
            self.sa_r_counts[states[t], actions[t], to_ind - t, r_ind] += 1
            # Credit future timesteps until done
            # Should apply credit for timestep when done is set, but not past
            while not dones[to_ind]:
                to_ind += 1
                # Get advantage to update credit counts with
                advantage = (rewards[to_ind] - self.value_multistep[states[t], to_ind - t]).detach()
                adv_sign = advantage.sign()
                future_r_ind = self.r_index[adv_sign.item()]
                self.s_r_counts[states[t], to_ind-t, future_r_ind] += 1
                self.sa_r_counts[states[t], actions[t], to_ind-t, future_r_ind] += 1
                self.value_updates[states[t]] += advantage

    # Given a trajectory for one episode, update the policy
    def update(self, states, actions, rewards, next_states, dones):
        states = torch.LongTensor(states)
        actions = torch.LongTensor(actions)
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)

        # First update credit counts
        self.update_credit(states, actions, rewards, dones)

        # Next, credit and accumulate rewards
        credited_rewards = torch.Tensor(self.accumulate_credited_rewards(states, actions, rewards, dones))

        # Finally, compute eligibility
        loss, cats, mean_var = self.make_pg_step(states, actions, credited_rewards)

        # Apply value updates computed in update_credit
        self.value_multistep += self.value_updates.detach() * (self.alpha / (states.size(0)*self.time_horizon))

        return loss.mean().item(), mean_var, cats.entropy().mean().item(), credited_rewards.mean()
