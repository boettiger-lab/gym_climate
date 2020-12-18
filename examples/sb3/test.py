import gym
import gym_climate
import numpy as np
from stable_baselines3.common.env_checker import check_env

env = gym.make('dice-v0')
check_env(env)
obs = env.reset()
for i in range(99):
    x,y,z,q = env.step(np.array([-0.9, -0.5]))
    print(y)
