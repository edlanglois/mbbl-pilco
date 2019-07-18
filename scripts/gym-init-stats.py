#!/usr/bin/env python
"""Initial state statistics for gym environments."""
import argparse
import shutil
import sys

import gym
import numpy as np
import tqdm
from mbbl.env import env_register

import pilco.rl.envs  # noqa: Registers environments
from pilco import utils


def parse_args(args=None):
    """Parse command-line arguments.

    Args:
        args: A list of argument strings to use instead of sys.argv.

    Returns:
        An `argparse.Namespace` object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-e",
        "--env",
        nargs="?",
        default="CartPole-v0",
        type=str,
        help="Environment ID.",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        default=50000,
        type=int,
        help="Number of initial state samples to draw.",
    )
    return parser.parse_args(args)


def main(args=None):
    """Run script.

    Args:
        args: A list of argument strings to use instead of sys.argv.
    """
    args = parse_args(args)
    try:
        env = gym.make(args.env)
    except gym.error.Error:
        env, _ = env_register.make_env(args.env, None, misc_info={"reset_type": "gym"})
    stats = utils.stats.OnlineMeanVariance()

    try:
        print("Observation Space Shape:", env.observation_space.shape)
    except AttributeError:
        pass
    try:
        print("Action Space Shape:", env.action_space.shape)
    except AttributeError:
        pass

    for _ in tqdm.trange(args.num_samples):
        state = env.reset()
        stats.add(np.asarray(state))

    print("State Num Dimensions", len(state))
    print("mean\n", stats.mean())
    print("var\n", stats.variance())
    print("stddev\n", np.sqrt(stats.variance()))

    try:
        print()
        print(env.metadata["initial_state.mean"])
        print(np.diag(env.metadata["initial_state.covariance"]))
        print(np.sqrt(np.diag(env.metadata["initial_state.covariance"])))
    except (AttributeError, KeyError):
        pass


if __name__ == "__main__":
    try:
        _np = sys.modules["numpy"]
    except KeyError:
        pass
    else:
        _np.set_printoptions(linewidth=shutil.get_terminal_size().columns)
    main()
