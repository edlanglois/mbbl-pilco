"""Inverted Pendulum Environment"""
import gym.envs.mujoco
import numpy as np

from pilco import moment_map


class SwimmerEnv(gym.envs.mujoco.SwimmerEnv):
    def __init__(self):
        super().__init__()

        # Initial state drawn from uniform(-+initial_state_unif_bound)
        initial_state_unif_bound = 0.1
        self.metadata["initial_state.mean"] = np.zeros(self.observation_space.shape)
        self.metadata["initial_state.covariance"] = (
            np.eye(np.prod(self.observation_space.shape))
            * initial_state_unif_bound ** 2
            / 3
        )

        # Reward: vel - 0.0001 * action**2
        # Note: rewards do not currently depend on action so can only do velocity

        def reward_moment_map(**kwargs):
            # System velocity is closest to velocity of first component
            return moment_map.core.IndexMomentMap(3, **kwargs)

        self.metadata["reward.moment_map"] = reward_moment_map
