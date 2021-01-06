import gym
import gym_climate
import numpy as np

env = gym.make('dice-v0')

obs = env.reset()
rewards = 0
for i in range(99):
    state,reward,done,_ = env.step(np.array([-0.5, -0.5]))
    rewards += reward
    print(reward)
print(rewards)
