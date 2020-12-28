import gym
import gym_climate
import numpy as np
from stable_baselines3.common.env_checker import check_env

env = gym.make('dice-v0')
obs = env.reset()
rewards = 0
for i in range(99):
    state,reward,done,_ = env.step(np.array([-1, -1]))
    rewards += reward
    print(reward)
print(rewards)
