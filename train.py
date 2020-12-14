# Core training/evaluation loop for tabular CCA
import math
import numpy as np
from tqdm import tqdm
from variance_estimation import estimate_agent_analytically,\
                                estimate_pg_variance


def train(agent, env, n_eps, ep_length, eps_per_train, log_freq: int=100,
          logger=None, estimate_variance: bool=False):
    episode_train_rewards = []
    episode_validation_rewards = []
    losses = []
    total_n_frames = 0

    for i in tqdm(range(math.ceil(n_eps / eps_per_train))):
        total_r_train = 0
        # Store data for episode
        # TODO consider using something more efficient than appending arrays
        states = []
        acts = []
        rewards = []
        next_states = []
        dones = []

        ts = []

        # Roll out a training episode, with updates.
        for _ in range(eps_per_train):
            # Reset the environment.
            s = env.reset()
            for t in range(ep_length):
                a = agent.select_action(s, inference=False).item()
                ns, r, d, _ = env.step(a)
                total_n_frames += 1
                # Set done if out of time
                if not d and t == ep_length - 1:
                    d = True
                states.append(s)
                acts.append(a)
                rewards.append(r)
                next_states.append(ns)
                dones.append(d)

                s = ns
                total_r_train += r

                if d:
                    ts.append(t+1)
                    break

        ts = np.array(ts)
        mean_t = ts.mean()

        # Once episode is over, train
        loss_mean, entropy, mean_advantage = agent.update(states, acts, rewards,
                                                                    next_states, dones)

        # Collect a validation trajectory.
        total_r_inference = validate(agent, env, ep_length)

        # Logging.
        if i % log_freq == 0 or i >= n_eps / eps_per_train:
            logging_dict = {'training_steps': i, 'episodes': i*eps_per_train,
                            'train_reward': total_r_train/eps_per_train,
                            'test_reward': total_r_inference, 'policy_gradient_mean': loss_mean,
                            'policy_entropy': entropy, 'timesteps': mean_t,
                            'mean_advantage': mean_advantage, 'total_n_frames': total_n_frames}
            if estimate_variance is True:
                pi_grad = estimate_pg_variance(env, agent, policy=None, ep_length=ep_length)
                grad_var_trace = np.var(pi_grad, axis=0).sum()
                logging_dict.update({'grad_var_trace': grad_var_trace})
            # WandB logging
            logger(logging_dict)

        # Aggregate.
        episode_train_rewards.append(total_r_train/eps_per_train)
        episode_validation_rewards.append(total_r_inference)
        losses.append(loss_mean)
    return episode_train_rewards, episode_validation_rewards, losses


def validate(agent, env, episode_length: int):
    total_r_inference = 0
    s = env.reset()
    for _ in range(episode_length):
        a = agent.select_action(s, inference=True)
        ns, r, d, _ = env.step(a)
        s = ns
        total_r_inference += r
        if d:
            break

    return total_r_inference
