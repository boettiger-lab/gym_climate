import gym
import gym_climate
from stable_baselines3.common.env_checker import check_env

if __name__=="__main__":
    env = gym.make("ays-v0")
    check_env(env)

