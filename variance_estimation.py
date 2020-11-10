import numpy as np
from scipy import special
import torch
from envs import FrozenLakeEnv


GOOD_POLICY = [[-1.2901, -0.8693,  6.6912, -0.5318],
        [ 0.7621, -0.4571, -1.1712,  4.8661],
        [ 0.6781,  0.4751,  1.6692,  1.1776],
        [ 1.1136,  0.9038,  0.7627,  1.2199],
        [-1.1953, -1.7602,  6.4497,  0.5059],
        [ 1.0000,  1.0000,  1.0000,  1.0000],
        [ 0.6780,  0.3094,  2.4584,  0.5541],
        [ 1.0000,  1.0000,  1.0000,  1.0000],
        [-1.1832,  7.0186, -1.7066, -0.1288],
        [-1.5505, -0.8033,  7.4297, -1.0758],
        [-0.8577, -1.1986,  6.2037, -0.1474],
        [ 1.0000,  1.0000,  1.0000,  1.0000],
        [ 1.0000,  1.0000,  1.0000,  1.0000],
        [-1.0185,  6.6443,  0.3716, -1.9975],
        [-1.6142,  6.7572, -0.0748, -1.0683],
        [ 1.0000,  1.0000,  1.0000,  1.0000]]


WEAK_POLICY = [[0.8338, 1.4808, 0.9949, 0.6905],
        [1.0216, 1.4151, 0.8332, 0.7301],
        [1.0282, 0.8370, 1.3207, 0.8142],
        [1.0976, 0.9931, 0.9162, 0.9931],
        [0.6420, 0.7122, 1.5614, 1.0844],
        [1.0000, 1.0000, 1.0000, 1.0000],
        [0.7792, 0.7792, 1.6625, 0.7792],
        [1.0000, 1.0000, 1.0000, 1.0000],
        [0.9057, 1.6872, 0.6676, 0.7394],
        [0.6432, 1.3616, 1.0075, 0.9877],
        [0.6351, 0.6643, 1.8829, 0.8177],
        [1.0000, 1.0000, 1.0000, 1.0000],
        [1.0000, 1.0000, 1.0000, 1.0000],
        [0.8467, 1.4666, 0.8967, 0.7900],
        [0.7994, 1.7828, 0.7231, 0.6946],
        [1.0000, 1.0000, 1.0000, 1.0000]]


MEDIOCRE_POLICY = [[ 6.8629e-02,  2.3531e-01,  3.0754e+00,  6.2067e-01],
        [ 7.0793e-01,  9.0863e-01,  6.2107e-01,  1.7624e+00],
        [ 9.7688e-01,  8.7125e-01,  1.2806e+00,  8.7125e-01],
        [ 1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00],
        [ 1.8608e-03, -2.3329e-01,  3.4957e+00,  7.3571e-01],
        [ 1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00],
        [ 8.2319e-01,  1.0352e+00,  1.2876e+00,  8.5407e-01],
        [ 1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00],
        [ 5.2751e-01,  2.9770e+00, -4.4948e-02,  5.4045e-01],
        [-2.6165e-01,  3.0104e+00,  1.0429e+00,  2.0842e-01],
        [ 2.1755e-01,  1.0170e-01,  3.2170e+00,  4.6372e-01],
        [ 1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00],
        [ 1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00],
        [ 6.6965e-01,  1.8689e+00,  1.2782e+00,  1.8327e-01],
        [ 3.1208e-01,  2.9747e+00,  7.2924e-01, -1.6064e-02],
        [ 1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00]]


def rollout(env, agent, ep_length, s0=None, a0=None):
    states = []
    acts = []
    rewards = []
    next_states = []
    dones = []

    s = env.reset(s0)
    for t in range(ep_length):
        if not a0 is None and t == 0:
            a = a0
        else:
            a = agent.select_action(s, inference=False).item()
        ns, r, d = env.step(a)

        if not d and t == ep_length - 1:
            d = True

        states.append(s)
        acts.append(a)
        rewards.append(r)
        next_states.append(ns)
        dones.append(d)

        s = ns

        if d:
            break

    return states, acts, rewards, next_states, dones


def estimate_pg_variance(env, agent, policy, ep_length, n_episodes=10000):
    # substitute new policy
    agent.pi = torch.FloatTensor(np.array(policy))
    agent.pi.requires_grad = True

    # forbid updates in agent
    agent.pi_opt = torch.optim.SGD(params=[agent.pi], lr=agent.alpha)
    pi_opt_step = agent.pi_opt.step
    agent.pi_opt.step = lambda *args: None
    pi_opt_zero_grad = agent.pi_opt.zero_grad
    agent.pi_opt.zero_grad = lambda *args: None

    # extract new policy
    pi_a_s = torch.softmax(agent.pi, 1)
    pi_a_s = pi_a_s.detach().numpy().T

    # estimate necessary distributions, prevent reestimation on each iteration
    env.initialize_hindsight(ep_length, pi_a_s)
    env_initialize_hindsight = env.initialize_hindsight
    env.initialize_hindsight = lambda *args: None

    if hasattr(agent, 'estimate_optimal_baseline'):
        agent.estimate_optimal_baseline()
        agent.estimate_optimal_baseline = lambda *args: None

    pi_grads = np.empty((n_episodes, env.n_states, env.n_actions))

    states = []
    acts = []
    rewards = []
    next_states = []
    dones = []

    for i in range(n_episodes):
        ep_states, ep_acts, ep_rewards, ep_next_states, ep_dones = rollout(env, agent, ep_length)
        agent.update(ep_states, ep_acts, ep_rewards, ep_next_states, ep_dones)
        pi_grads[i] = agent.pi._grad
        pi_opt_zero_grad()

        states.append(ep_states)
        acts.append(ep_acts)
        rewards.append(ep_rewards)
        next_states.append(ep_next_states)
        dones.append(ep_dones)

    agent.pi_opt.step = pi_opt_step
    agent.pi_opt.zero_grad = pi_opt_zero_grad
    env.initialize_hindsight = env_initialize_hindsight

    return pi_grads

def estimate_grad_update(env, agent, grad, lr, ep_length):
    new_pi = agent.pi.detach().numpy() - grad * lr
    pi_a_s = special.softmax(new_pi, 1).T
    value = estimate_policy_analytically(env=env, pi_a_s=pi_a_s, ep_length=ep_length)
    return value


def estimate_agent_analytically(env, agent, ep_length, inference=False):
    pi_a_s = special.softmax(agent.pi.detach().numpy(), 1).T
    if inference:  # make deterministic
        dim_a = pi_a_s.shape[0]
        pi_a_s = np.eye(dim_a)[np.argmax(pi_a_s, 0)].T
    value = estimate_policy_analytically(env=env, pi_a_s=pi_a_s, ep_length=ep_length)
    return value


def estimate_policy_analytically(env, pi_a_s, ep_length):
    env.initialize_hindsight(ep_length, pi_a_s)
    value = env.get_state_value([0], [0])[0]
    return value