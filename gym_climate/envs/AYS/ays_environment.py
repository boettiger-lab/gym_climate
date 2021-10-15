import gym
from gym import spaces
import numpy as np
from gym_climate.envs.AYS.ays_model import ays_rescaled_rhs
from scipy.integrate import odeint
from pandas import DataFrame
import matplotlib.pyplot as plt


class AYSEnvironment(gym.Env):
    def __init__(self, reward_type="survive", random_reset=False):
        # Initializing stuff relevant to the environment
        self.Tmax = 99
        self.random_reset = random_reset
        self.t = 0
        self.dt = 1
        self.done = False
        self.final_radius = 0.05
        self.init_state = np.array([0.5, 0.5, 0.5])
        self.state = self.init_state
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32,
        )
        self.reward_type = reward_type
        self.reward_function = self._reward_function(self.reward_type)

        # AYS model parameters
        self.beta = 0.03
        self.beta_lg = self.beta / 2
        self.epsilon = 147
        self.rho = 2
        self.phi = 47e9
        self.sigma = 4e12
        self.sigma_et = self.sigma * 0.5 ** self.rho
        self.tau_A = 50
        self.tau_S = 50
        self.theta = self.beta / 350

        # Planetary boundary parameters
        self.AYS_mid = [240, 7e13, 5e11]
        self.a_PB = self._compactification(345, self.AYS_mid[0])
        self.y_SF = self._compactification(4e13, self.AYS_mid[1])
        self.s_limit = 0.0

        # Fixed points -- p.18 of Kittel
        self.green_fp = [0.0, 1.0, 1.0]
        self.brown_fp = [0.6, 0.4, 0.0]

    def step(self, action):
        next_t = self.t + self.dt
        # Solve the ode's
        self._evolve_system(action, next_t)
        self.t = next_t

        # Get the reward
        reward = self.reward_function()

        # Check end conditions
        if self._arrived_at_final_state():
            self.done = True
        if self.t >= self.Tmax:
            self.done = True
        if not self._inside_planetary_boundaries():
            self.done = True
            reward = 0
        return self.state, reward, self.done, {}

    def _evolve_system(self, action, next_t):
        # Solves the ODE system until the specified time, `next_t`
        parameters = self._get_parameters(action)
        trajectory = odeint(
            ays_rescaled_rhs,
            self.state,
            [self.t, next_t],
            args=parameters[0],
            mxstep=50000,
        )

        self.state = np.array([trajectory[:, i][-1] for i in range(3)])

    def _reward_function(self, name):
        # This function collects the different reward functions that are
        # available
        def reward_survive():
            if self._inside_planetary_boundaries():
                reward = 1.0
            else:
                reward = -0.0000000000000001
            return reward

        def reward_distance_PB():
            # Gives a reward based on the distance to the planetary boundaries
            a, y, s = self.state
            norm = np.linalg.norm(self.state - self.PB)
            if self._inside_planetary_boundaries():
                reward = 1.0
            else:
                reward = 0.0
            reward *= norm
            return reward

        if self.reward_type == "survive":
            return reward_survive
        elif self.reward_type == "distance":
            return reward_distance_PB

    def _get_parameters(self, action):
        parameter_list = [
            (
                self.beta_lg if action == 1 or action == 3 else self.beta,
                self.epsilon,
                self.phi,
                self.rho,
                self.sigma_et if action == 2 or action == 3 else self.sigma,
                self.tau_A,
                self.tau_S,
                self.theta,
            )
        ]
        return parameter_list

    def _compactification(self, x, x_mid):
        # Scales from A,Y,S to a,y,s
        if x == 0:
            return 0.0
        if x == np.infty:
            return 1.0
        return x / (x + x_mid)

    def _inv_compactification(self, y, x_mid):
        # Scales from a,y,s to A,Y,S
        if y == 0:
            return 0.0
        if np.allclose(y, 1):
            return np.infty

        return x_mid * y / (1 - y)

    def _inside_planetary_boundaries(self):
        # See p. 18 desirable states discussion Kittel
        a, y, s = self.state
        is_inside = True
        if a > self.a_PB or y < self.y_SF or s < self.s_limit:
            is_inside = False

        return is_inside

    def _arrived_at_final_state(self):
        # This function checks if the state is proximal to the green/sustainable
        # or brown/unsustainable fixed points
        a, y, s = self.state
        if (
            np.abs(a - self.green_fp[0]) < self.final_radius
            and np.abs(y - self.green_fp[1]) < self.final_radius
            and np.abs(s - self.green_fp[2]) < self.final_radius
        ):
            return True
        elif (
            np.abs(a - self.brown_fp[0]) < self.final_radius
            and np.abs(y - self.brown_fp[1]) < self.final_radius
            and np.abs(s - self.brown_fp[2]) < self.final_radius
        ):
            return True
        else:
            return False

    def reset(self):
        if self.random_reset:
            self.state = np.array([random.random() for i in range(3)])
        else:
            self.state = self.init_state
        self.t = 0
        self.done = False
        return self.state

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def simulate_mdp(self, model, reps=1):
        row = []
        for rep in range(reps):
            obs = self.reset()
            action = 0
            reward = 0.0
            for t in range(self.Tmax):
                # record
                row.append([t, self.state, action, reward, int(rep)])

                # Predict and implement action
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, done, info = self.step(action)

                if done:
                    break
            row.append([t, self.state, action, reward, int(rep)])
        df = DataFrame(
            row, columns=["time", "state", "action", "reward", "rep"]
        )
        return df

    def plot_mdp(self, df, output="results.png"):
        # Converting multibinary actions to strings
        df.loc[df.action == 0, "action"] = "Null"
        df.loc[df.action == 1, "action"] = "LG"
        df.loc[df.action == 2, "action"] = "ET"
        df.loc[df.action == 3, "action"] = "LG+ET"

        fig, axs = plt.subplots(3, 1)
        for i in np.unique(df.rep):
            results = df[df.rep == i]
            episode_reward = np.cumsum(results.reward)
            axs[0].plot(
                results.time,
                results.state.apply(lambda x: x[0]),
                color="black",
                alpha=0.3,
            )
            axs[0].plot(
                results.time,
                results.state.apply(lambda x: x[1]),
                color="green",
                alpha=0.3,
            )
            axs[0].plot(
                results.time,
                results.state.apply(lambda x: x[2]),
                color="blue",
                alpha=0.3,
            )
            axs[1].plot(results.time, results.action, color="blue", alpha=0.3)
            axs[2].plot(results.time, episode_reward, color="blue", alpha=0.3)

        axs[0].set_ylabel("state")
        axs[1].set_ylabel("action")
        axs[2].set_ylabel("reward")
        fig.tight_layout()
        plt.savefig(output)
        plt.close("all")
