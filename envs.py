# Environment definitions for tabular CCA
import numpy as np
import gym


def get_sticky_actions_gym_env(env_name, sticky_action_prob=0.25):
    env = gym.make(env_name).unwrapped
    env = StickyActionEnv(env, sticky_action_prob)
    return env


class StickyActionEnv(gym.Wrapper):
    def __init__(self, env, p=0.25):
        super(StickyActionEnv, self).__init__(env)
        self.p = p
        self.last_action = 0

    def reset(self):
        self.last_action = 0
        return self.env.reset()

    def step(self, action):
        if self.unwrapped.np_random.uniform() < self.p:
            action = self.last_action
        self.last_action = action
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info