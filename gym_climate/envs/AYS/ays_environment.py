import gym
from gym import spaces
import numpy as np
from ays_model import ays_rescaled_rhs
from scipy.integrate import odeint


class AYSEnvironment(gym.Env):
    def __init__(self, reward_type="survive"):
        self.t_max = 99
        self.t = 0
        self.dt = 1
        self.done = 0
        self.final_radius = 0.05
        self.init_state = np.array([0.5, 0.5, 0.5])
        self.state = self.init_state
        self.action_space = spaces.MultiBinary(2)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32,
        )
        self.reward_type = reward_type
        self.reward_function = self._reward_function(self.reward_type)

        # Parameters (https://github.com/timkittel/ays-model/blob/master/ays_model.py)
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

        # Planetary boundary oarameters
        self.AYS_mid = [240, 7e13, 5e11]
        self.a_PB = self._compactification(345, self.AYS_mid[0])
        self.y_SF = self._compactification(4e13, self.AYS_mid[1])
        self.s_limit = 0.0

        # Fixed points -- p.18 of Kittel
        self.green_fp = [0.0, 1.0, 1.0]
        self.brown_fp = [0.6, 0.4, 0.0]

    def step(self, action):
        next_t = self.t + self.dt
        self._evolve_system(action, next_t)
        self.t = next_t
        if self._arrived_at_final_state():
            self.done = True
        reward = self.reward_function()
        if self.t >= self.t_max:
            self.done = True
        if not self._inside_planetary_boundaries():
            self.final_state = True
            reward = 0

        return self.state, reward, self.done, {}

    def _evolve_system(self, action, next_t):
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
        
        def reward_survive():
            if self._inside_planetary_boundaries():
                reward = 1.0
            else:
                reward = -0.0000000000000001
            return reward
    
        def reward_distance_PB():
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
                self.beta_lg if action[0] else self.beta,
                self.epsilon,
                self.phi,
                self.rho,
                self.sigma_et if action[1] else self.sigma,
                self.tau_A,
                self.tau_S,
                self.theta,
            )
        ]
        return parameter_list

    def _compactification(self, x, x_mid):
        if x == 0:
            return 0.0
        if x == np.infty:
            return 1.0
        return x / (x + x_mid)

    def _inv_compactification(self, y, x_mid):
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
        self.state = self.init_state
        return self.init_state

    def render(self, mode="human"):
        pass

    def close(self):
        pass
