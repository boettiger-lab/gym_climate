import gym
from gym import spaces
import numpy as np
from gym_climate.envs.DICE.model.DICErun import DICE

class EnvDICE(gym.Env):
    def __init__(self):
        self.DICE = DICE()
        self.t_max = 100
        self.t = 1 # first observation is at t=0, first action at t=1 
        self.done = False
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        # Skeleton obs space, need to think more about what to do here
        self.observation_space = spaces.Box(low=-10**(-6), high=10**(6), shape=(30,), dtype=np.float32)

    def step(self, action):
        assert action in self.action_space, f"Error: {action} is an invalid action"
        self.unnormalized_action = (action+1) / 2

        self.reward = self.DICE.integrate(self.unnormalized_action, self.t)
        self.t += 1
        self.state = self.DICE.get_obs(self.t)
        if self.t == 101:
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
