import gym
import gym_climate
import numpy as np
env = gym.make('dice-v0')
obs = env.reset()
env.step(np.array([-0.9, -0.5]))
env.step(np.array([-0.9, -0.5]))
env.step(np.array([-0.9, -0.5]))
