Here I have implemented the [AYS earth systems model](https://arxiv.org/pdf/1706.04542.pdf). Following [Strnad et al.](https://aip.scitation.org/doi/pdf/10.1063/1.5124673), one can access the ays environment as  follows:

`env = gym.make("ays-v0", reward_type="survive", random_reset=False)`

For `reward_type`, one can input either `"survive"` or `"distance"`. `"survive"` denotes a reward function that rewards the agent for staying within some planetary boundary. `"distance"` denotes a reward function that rewards the agent proportional to the distance to the planetary boundaries.
`random_reset` allows there to be a random reset every episode, otherwise the system resets to [A, Y, S] = [240, 7e13, 5e11]. Note the observation space is rescaled so that this default state is [a,y,s] = [0.5, 0.5, 0.5]. 
