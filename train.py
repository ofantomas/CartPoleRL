# Core training/evaluation loop for tabular CCA
import math

import numpy as np
from tqdm import tqdm


def train(agent, env, n_eps, ep_length, eps_per_train, log_freq: int = 100, logger = None):
    states_visited = np.zeros(env.transitions.shape, int)
    episode_train_rewards = []
    episode_validation_rewards = []
    losses = []
    loss_vars = []

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
        for batch_n in range(eps_per_train):
            # Reset the environment.
            s = env.reset()
            for t in range(ep_length):
                states_visited[s] += 1
                a = agent.select_action(s, inference=False).item()
                #print(s, a)
                ns, r, d = env.step(a)
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
        loss_mean, loss_var, entropy, mean_advantage = agent.update(states, acts, rewards, next_states, dones)


        # Collect a validation trajectory.
        trajectory, total_r_inference = validate(agent, env, ep_length)

        # Logging.
        if i % log_freq == 0 or i >= n_eps / eps_per_train:
            print(
                f"Iteration {i}: mean training reward: {total_r_train/eps_per_train}, inf reward: {total_r_inference}"
            )

            # WandB logging
            logger({'training_steps': i, 'episodes': i * eps_per_train, 'train_reward': total_r_train / eps_per_train, 'test_reward': total_r_inference, 'policy_gradient_mean': loss_mean, 'policy_gradient_var': loss_var, 'policy_entropy': entropy, 'timesteps': mean_t, 'mean_advantage': mean_advantage})

        # TODO support visualization logging
        # Log what we should log.
        #if log_freq != 0 and i % log_freq == 0:
        #    logging.log_trajectory("validation_traj", env.reward_grid, trajectory, i)
        #    logging.log_values("visited", states_visited, i)
        #    logging.log_values("ever_visited", (states_visited > 0).astype(int), i)
        #    logging.log_q_values(
        #        "q_val", agent.q, i, ["left", "right", "up", "down", "stay"]
        #    )

        # Aggregate.
        episode_train_rewards.append(total_r_train/eps_per_train)
        episode_validation_rewards.append(total_r_inference)
        losses.append(loss_mean)
        loss_vars.append(loss_var)
    return episode_train_rewards, episode_validation_rewards, states_visited, losses, loss_vars


def validate(agent, env, episode_length: int):
    total_r_inference = 0
    states = []
    s = env.reset()
    states.append(s)
    for _ in range(episode_length):
        a = agent.select_action(s, inference=True)
        ns, r, d = env.step(a)
        s = ns
        states.append(s)
        total_r_inference += r
        if d:
            break

    return states, total_r_inference
