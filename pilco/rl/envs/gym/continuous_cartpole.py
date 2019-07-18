"""Continuous action space cart-pole environment.

Modification of the OpenAI gym CartPole environment.

=== License ===
The MIT License

Copyright (c) 2016 OpenAI (https://openai.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
===============

Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math

import gym.envs.classic_control
import numpy as np
from gym import logger
from gym import spaces

from pilco import moment_map


class ContinuousCartPoleEnv(gym.envs.classic_control.CartPoleEnv):
    """Cart-Pole environment with continuous actions."""

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(-1, 1, shape=())
        self.reset_state_bound = 0.05

        # Moments of the initial state distribution (uniform about 0)
        self.metadata["initial_state.mean"] = np.zeros((4,))
        self.metadata["initial_state.covariance"] = (
            np.eye(4) * self.reset_state_bound ** 2 / 3
        )

        def reward_moment_map(**kwargs):
            # observation_action: [x, dx, theta, dtheta, action]
            # Reward depends on x & theta, slice(0, 4, 2) selects these two.
            return moment_map.gp.DeterministicGaussianProcessMomentMap(
                inducing_points=np.zeros((1, 2)),
                coefficients=np.ones((1, 1)),
                signal_variance=1,
                length_scale=np.array([self.x_threshold, self.theta_threshold_radians]),
                **kwargs,
            ).compose(moment_map.core.IndexMomentMap(slice(0, 4, 2), **kwargs))

        # Reward moment-map function.
        self.metadata["reward.moment_map"] = reward_moment_map

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag * action
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (
            force + self.polemass_length * theta_dot * theta_dot * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = (x, x_dot, theta, theta_dot)
        done = (
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment "
                    "has already returned done = True. You should always call "
                    "'reset()' once you receive 'done = True' -- any further "
                    "steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(
            low=-self.reset_state_bound, high=self.reset_state_bound, size=(4,)
        )
        self.steps_beyond_done = None
        return np.array(self.state)
