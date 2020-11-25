import matplotlib.pyplot as plt
import numpy as np
from scipy import special


def plot_policy(env, policy, episode_length, output_path):
    # policy -- np.array [16 x 4]
    # env -- env
    plt.figure(figsize=(5, 5))
    h, w = env.state_map.shape

    pi_a_s = special.softmax(np.array(policy), 1).T
    pi = np.argmax(pi_a_s, 0).reshape(w, h)
    env.initialize_hindsight(episode_length, pi_a_s)
    V = env.get_state_all_values(ts=[0])
    plt.imshow(V.reshape(w, h), cmap='gray', interpolation='none', clim=(0, 1))

    ax = plt.gca()
    ax.set_xticks(np.arange(h)-.5)
    ax.set_yticks(np.arange(w)-.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    #Y, X = np.mgrid[0:4, 0:4]
    a2uv = {3: (-1, 0), 2: (0, -1), 1: (1, 0), 0: (0, 1)}
    for y in range(h):
        for x in range(w):
            if not env.done_map[y, x]:
                text = 'F'
            else:
                if env.reward_map[y, x]:
                    text = 'G'
                else:
                    text = 'H'
            plt.text(x, y, text,
                     color='g', size=24,  verticalalignment='center',
                     horizontalalignment='center', fontweight='bold')
            a = pi[y, x]
            if env.done_map[y, x]:
                continue
            u, v = a2uv[a]
            plt.arrow(x, y, u * 0.3, -v * 0.3, color='m',
                      head_width=0.15, head_length=0.15)

    plt.grid(color='b', lw=2, ls='-')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close('all')
