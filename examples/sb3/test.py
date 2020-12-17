import gym
import gym_climate
import numpy as np
import pdb; pdb.set_trace()
env = gym.make('dice-v0')
obs = env.reset()
print(obs)
env.step(np.array([0, 0]))
