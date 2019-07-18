"""Inverted Double Pendulum Environment"""
import gym.envs.mujoco
import numpy as np

from pilco import moment_map

from .utils import INF_LENGTH_SCALE


class InvertedDoublePendulumEnv(gym.envs.mujoco.InvertedDoublePendulumEnv):
    def __init__(self):
        super().__init__()

        # State-Action Elements
        # [0] = qpos[0] ~ Unif(-0.1, 0.1)
        # [1:3] = sin(qpos[1:]) ~= qpos[0] ~ Unif(-0.1, 0.1)
        # [3:5] = cos(qpos[1:]) ~= 1 - qpos[0]^2/2; Var ~= Var(qpos)^2
        # [5:8] = qvel ~ N(0, 0.01)
        # [8:-1] = qfrc_constraint; Don't know what that is but it's always 0
        # [-1] = action

        # TODO: More accurate variance for cos(qpos[1:])
        initial_state_mean = np.zeros(self.observation_space.shape)
        initial_state_mean[3:5] = 1
        initial_state_variances = np.zeros(self.observation_space.shape)
        uniform_variance = 0.1 ** 2 / 3
        initial_state_variances[:3] = uniform_variance
        initial_state_variances[3:5] = uniform_variance ** 2
        initial_state_variances[5:-3] = 0.1 ** 2
        self.metadata["initial_state.mean"] = initial_state_mean
        self.metadata["initial_state.covariance"] = np.diag(initial_state_variances)

        def reward_moment_map(**kwargs):
            return moment_map.gp.DeterministicGaussianProcessMomentMap(
                inducing_points=initial_state_mean[:8],
                coefficients=np.ones((1, 1)),
                signal_variance=1,
                length_scale=np.array(
                    [[0.1, 0.1, 0.1, 0.01, 0.01, INF_LENGTH_SCALE, 1.0, 1.0]]
                ),
                **kwargs
            ).compose(moment_map.core.IndexMomentMap(slice(None, 8), **kwargs))

        self.metadata["reward.moment_map"] = reward_moment_map

    def step(self, action):
        observation, reward, done, info = super().step(action)
        # Re-scale reward to have maximum of 1
        return observation, reward / 10, done, info
