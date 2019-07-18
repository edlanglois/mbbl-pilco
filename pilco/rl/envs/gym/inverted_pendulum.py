"""Inverted Pendulum Environment"""
import gym.envs.mujoco
import numpy as np

from pilco import moment_map


class InvertedPendulumEnv(gym.envs.mujoco.InvertedPendulumEnv):
    def __init__(self):
        super().__init__()

        # Initial state drawn from uniform(-+initial_state_unif_bound)
        initial_state_unif_bound = 0.01
        self.metadata["initial_state.mean"] = np.zeros(self.observation_space.shape)
        self.metadata["initial_state.covariance"] = (
            np.eye(np.prod(self.observation_space.shape))
            * initial_state_unif_bound ** 2
            / 3
        )
        reward_threshold = 0.2  # Reward = 1 if abs(state[1]) < reward_threshold else 0

        def reward_moment_map(**kwargs):
            # State-Action Elements
            # [0:2] = qpos
            # [2:4] = qvel
            # [4]   = action
            return moment_map.gp.DeterministicGaussianProcessMomentMap(
                inducing_points=np.zeros((1, 2)),
                coefficients=np.ones((1, 1)),
                signal_variance=1,
                length_scale=np.array([reward_threshold]),
                **kwargs
            ).compose(moment_map.core.IndexMomentMap(slice(0, 2), **kwargs))

        self.metadata["reward.moment_map"] = reward_moment_map
