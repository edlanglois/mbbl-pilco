"""Register environments with OpenAI Gym."""
import functools
import logging

import gym.error
from gym.envs import registration


def _try_register(on_exist, *args, **kwargs):
    """Try to register an environment.

    Args:
        on_exist: Function called when the id already exists.
            Called from within an except clause.
        *args: Passed through to gym.envs.registration.register.
        **kwargs: Passed through to gym.envs.registration.register.
    """
    try:
        registration.register(*args, **kwargs)
    except gym.error.Error as e:
        if "Cannot re-register id" not in str(e):
            raise
        on_exist()


def register_all(exist="fail"):
    """Perform all environment registrations.

    Args:
        exist: What to do if the environment ID is already registered.
            'fail' - Raises an error.
            'warn' - Log a warning and continue.
            'ignore' - Ignore the error and continue.
    """

    def raise_():
        raise  # called in except so ok; pylint: disable=misplaced-bare-raise

    def pass_():
        pass

    on_exist = {
        "fail": raise_,
        "warn": lambda: logging.warning(
            "Environment registration failed", exc_info=True
        ),
        "ignore": pass_,
    }[exist]
    register = functools.partial(_try_register, on_exist=on_exist)

    register(
        id="ContinuousCartPole-v0",
        entry_point="pilco.rl.envs.gym:ContinuousCartPoleEnv",
        max_episode_steps=200,
        reward_threshold=195.0,
    )
    register(
        id="InvertedPendulumExtra-v2",
        entry_point="pilco.rl.envs.gym:InvertedPendulumEnv",
        max_episode_steps=1000,
        reward_threshold=950.0,
    )
    register(
        id="InvertedDoublePendulumExtra-v2",
        entry_point="pilco.rl.envs.gym:InvertedDoublePendulumEnv",
        max_episode_steps=1000,
        reward_threshold=9100.0,
    )
    register(
        id="SwimmerExtra-v2",
        entry_point="pilco.rl.envs.gym:SwimmerEnv",
        max_episode_steps=1000,
        reward_threshold=360.0,
    )
