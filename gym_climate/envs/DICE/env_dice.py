import gym
from gym import spaces
import numpy as np
from gym_climate.envs.DICE.model.DICErun import DICE

class EnvDICE(gym.Env):
    def __init__(self):
        self.DICE = DICE()
        self.t_max = 99
        self.t = 1 # first observation is at t=0, first action at t=1 
        self.done = False
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32)

    def step(self, action):
        assert action in self.action_space, f"Error: {action} is an invalid action"
        self.action = (action+1) / 2
        self.reward = self.DICE.integrate(self.action, self.t)
        self.state = self.DICE.get_obs(self.t)
        self.t += 1
        if self.t > self.t_max:
            self.done = True

        return self.state, self.reward, self.done, {}

    def reset(self):
        self.DICE = DICE()
        self.t = 1
        self.done = False
        self.state = self.DICE.get_obs(self.t)

        return self.state

    def render(self, mode='human'):
        pass

    def close(self):
        pass
