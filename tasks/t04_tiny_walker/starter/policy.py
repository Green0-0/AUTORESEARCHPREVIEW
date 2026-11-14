"""Starter policy: linear map from the raw 24-d observation to the 4 joint torques, tanh-squashed.
params["theta"] (float32, 100) = W (4x24, row-major) followed by b (4)."""
import numpy as np


class Policy:
    def __init__(self, params):
        th = params["theta"].astype(np.float64)
        n_act = th.size // 25                       # 100 = 4 * (24 + 1)
        self.W = th[: -n_act].reshape(n_act, -1)
        self.b = th[-n_act:]

    def reset(self):
        pass

    def act(self, obs):
        return np.tanh(self.W @ obs + self.b)
