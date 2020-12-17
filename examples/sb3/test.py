import gym
import gym_climate
import numpy as np

env = gym.make('dice-v0')
env.step(np.array([0, 0]))
