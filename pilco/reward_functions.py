"""Reward function implementations for PILCO."""
import math

from pilco import moment_map

# Usage instructions:
# Call mbbl_walker_reward and pass the result to the `reward_moment_map_fn` argument
# of `PILCOAgent`.
# Alternatively, assign the result to env.metadata["reward.moment_map"]


def mbbl_walker_reward(
    velocity_index,
    height_index,
    target_height,
    height_coefficient,
    control_index,
    control_coefficient,
):
    """Creates the reward moment map for walker environments in Model-Based Baseline.

    https://github.com/WilsonWangTHU/model_based_baseline

    The reward is
        velocity
        - height_coefficient * (height - target_height) ** 2
        - control_coefficient * ||action||^2

    Args:
        velocity_index: Index of the agent's velocity in the state-action vector.
        height_index: Index of the agent's height in the state-action vector.
        target_height: The target height value.
        height_coefficient: Coefficient on the height reward in the total reward.
        control_index: The index or indices of the control (action) vector.
            May be an integer or a slice object.
        control_coefficient: Coefficient on the control reward in the total reward.

    Returns:
        A function with the interface of a MomentMap initializer that creates
        the reward moment map when called.
    """

    def reward_moment_map(**kwargs):
        velocity_reward = moment_map.IndexMomentMap(velocity_index, **kwargs)

        height = moment_map.IndexMomentMap(height_index, **kwargs)
        height_offset = moment_map.LinearMomentMap(offset=-target_height, **kwargs)(
            height
        )
        base_height_cost = moment_map.SumSquaredMomentMap(**kwargs)(height_offset)
        height_reward = moment_map.LinearMomentMap(scale=-height_coefficient, **kwargs)(
            base_height_cost
        )

        controls = moment_map.IndexMomentMap(control_index, **kwargs)
        base_control_cost = moment_map.SumSquaredMomentMap(**kwargs)(controls)
        control_reward = moment_map.LinearMomentMap(-control_coefficient, **kwargs)(
            base_control_cost
        )

        # The covariance on this will be wrong because the rewards are not actually
        # uncorrelated.
        # PILCO never uses the reward covariances for planning so that's not a problem.
        # It only uses the covariance when logging statistics.
        total_reward = moment_map.AddUncorrelatedMomentMap(
            [velocity_reward, height_reward, control_reward], **kwargs
        )
        return total_reward

    return reward_moment_map


def fswimmer_walker_reward(
    velocity_index,
    height_index,
    target_height,
    height_coefficient,
    control_index,
    control_coefficient,
):
    """Creates the reward moment map for walker environments in Model-Based Baseline.

    https://github.com/WilsonWangTHU/model_based_baseline

    The reward is
        velocity
        - height_coefficient * (height - target_height) ** 2
        - control_coefficient * ||action||^2

    Args:
        velocity_index: Index of the agent's velocity in the state-action vector.
        height_index: Index of the agent's height in the state-action vector.
        target_height: The target height value.
        height_coefficient: Coefficient on the height reward in the total reward.
        control_index: The index or indices of the control (action) vector.
            May be an integer or a slice object.
        control_coefficient: Coefficient on the control reward in the total reward.

    Returns:
        A function with the interface of a MomentMap initializer that creates
        the reward moment map when called.
    """

    def reward_moment_map(**kwargs):
        velocity_reward = moment_map.IndexMomentMap(velocity_index, **kwargs)

        height = moment_map.IndexMomentMap(height_index, **kwargs)
        height_offset = moment_map.LinearMomentMap(offset=-target_height, **kwargs)(
            height
        )
        base_height_cost = moment_map.SumSquaredMomentMap(**kwargs)(height_offset)
        height_reward = moment_map.LinearMomentMap(scale=-height_coefficient, **kwargs)(
            base_height_cost
        )

        controls = moment_map.IndexMomentMap(control_index, **kwargs)
        base_control_cost = moment_map.SumSquaredMomentMap(**kwargs)(controls)
        control_reward = moment_map.LinearMomentMap(-control_coefficient, **kwargs)(
            base_control_cost
        )

        # The covariance on this will be wrong because the rewards are not actually
        # uncorrelated.
        # PILCO never uses the reward covariances for planning so that's not a problem.
        # It only uses the covariance when logging statistics.
        total_reward = moment_map.AddUncorrelatedMomentMap(
            [velocity_reward, height_reward, control_reward], **kwargs
        )
        return total_reward

    return reward_moment_map


def cartpole_reward(position_index, position_coefficient, angle_index):
    """Reward for cartpole.

    Args:
        position_index: Index of position in the state-action vector.
        position_coefficient: Coefficient on the position penalty.
        angle_index: Index of angle (radians from vertical) in the state-action vector.
    """

    def reward_moment_map(**kwargs):
        x = moment_map.IndexMomentMap(position_index, **kwargs)
        angle = moment_map.IndexMomentMap(angle_index, **kwargs)

        cos_angle = moment_map.SinMomentMap(**kwargs)(
            moment_map.LinearMomentMap(offset=math.pi / 2, **kwargs)(angle)
        )
        up_reward = cos_angle

        x_squared = moment_map.SumSquaredMomentMap(**kwargs)(x)
        distance_penalty_reward = moment_map.LinearMomentMap(
            scale=-position_coefficient, **kwargs
        )(x_squared)

        total_reward = moment_map.AddUncorrelatedMomentMap(
            [up_reward, distance_penalty_reward], **kwargs
        )
        return total_reward

    return reward_moment_map


def mountain_car_reward():
    """Creates the reward moment map for mountain_car environments in Model-Based
    Baseline.

    https://github.com/WilsonWangTHU/model_based_baseline

    The reward is
        height (i.e. data_dict['start_state'][0])

    Args:
        None
    Returns:
        A function with the interface of a MomentMap initializer that creates
        the reward moment map when called.
    """

    def reward_moment_map(**kwargs):
        velocity_reward = moment_map.IndexMomentMap(0, **kwargs)

        total_reward = moment_map.AddUncorrelatedMomentMap([velocity_reward], **kwargs)
        return total_reward

    return reward_moment_map


def pendulum_reward(control_index):
    """Creates the reward moment map for walker environments in Model-Based Baseline.

    https://github.com/WilsonWangTHU/model_based_baseline

    The reward is

            action = data_dict['action']
            true_action = action * self._env.env.max_torque

            max_torque = self._env.env.max_torque
            torque = np.clip(true_action, -max_torque, max_torque)[0]

            y, x, thetadot = data_dict['start_state']

            costs = y + .1 * |x| + .1 * (thetadot ** 2) + .001 * (torque ** 2)

    Args:
        control_index: The index or indices of the control (action) vector.
            May be an integer or a slice object.

    Returns:
        A function with the interface of a MomentMap initializer that creates
        the reward moment map when called.
    """

    def reward_moment_map(**kwargs):
        y = moment_map.IndexMomentMap(0, **kwargs)
        x = moment_map.IndexMomentMap(1, **kwargs)
        thetadot = moment_map.IndexMomentMap(2, **kwargs)

        y_reward = moment_map.LinearMomentMap(scale=-1.0, **kwargs)(y)
        x_reward = moment_map.LinearMomentMap(scale=-0.1, **kwargs)(
            moment_map.AbsMomentMap(**kwargs)(x)
        )

        thetadot_reward = moment_map.LinearMomentMap(scale=-0.1, **kwargs)(
            moment_map.SumSquaredMomentMap(**kwargs)(thetadot)
        )
        controls = moment_map.IndexMomentMap(control_index, **kwargs)
        torque = moment_map.LinearMomentMap(scale=2.0, **kwargs)(controls)
        base_control_cost = moment_map.SumSquaredMomentMap(**kwargs)(torque)
        control_reward = moment_map.LinearMomentMap(-0.001, **kwargs)(base_control_cost)

        total_reward = moment_map.AddUncorrelatedMomentMap(
            [x_reward, y_reward, thetadot_reward, control_reward], **kwargs
        )
        return total_reward

    return reward_moment_map


def inverted_pendulum_reward():
    """Creates the reward moment map for mountain_car environments in Model-Based
    Baseline.

    https://github.com/WilsonWangTHU/model_based_baseline

    The reward is
        height (i.e. data_dict['start_state'][0])

    Args:
        None
    Returns:
        A function with the interface of a MomentMap initializer that creates
        the reward moment map when called.
    """

    def reward_moment_map(**kwargs):
        y = moment_map.IndexMomentMap(1, **kwargs)
        y_reward = moment_map.LinearMomentMap(scale=-1.0, **kwargs)(
            moment_map.SumSquaredMomentMap(**kwargs)(y)
        )

        total_reward = moment_map.AddUncorrelatedMomentMap([y_reward], **kwargs)
        return total_reward

    return reward_moment_map


def surrogate_reacher_reward(distance_index, action_index=None):
    """Surrogate reward function for the reacher environment.

    The moments of the true reward function could not be identified.
    In particular, E[||X||] for X ~ Normal(mu, Sigma)

    This has the same optimum but uses ||X||^2 instead of ||X||.

    Args:
        distance_index: Index of distance in the state-action vector.
            Optionally includes the action indices as well.
        action_index: Index of the actions in the state-action vector.
            If `None` then the actions are assumed to be included
            in `distance_index`. Doing so will produce a more accurate covariance.
    """

    def reward_moment_map(**kwargs):
        distance = moment_map.IndexMomentMap(distance_index, **kwargs)
        cost = moment_map.SumSquaredMomentMap(**kwargs)(distance)

        if action_index is not None:
            action = moment_map.IndexMomentMap(action_index, **kwargs)
            action_cost = moment_map.SumSquaredMomentMap(**kwargs)(action)
            cost = moment_map.AddUncorrelatedMomentMap([cost, action_cost], **kwargs)

        reward = moment_map.LinearMomentMap(scale=-1, **kwargs)(cost)
        return reward

    return reward_moment_map


def acrobot_reward():
    """Creates the reward moment map for the acrobot environment."""

    def reward_moment_map(**kwargs):
        x0 = moment_map.IndexMomentMap(0, **kwargs)

        x0x2 = moment_map.ElementProductMomentMap(i=0, j=2, **kwargs)
        x1x3 = moment_map.ElementProductMomentMap(i=1, j=3, **kwargs)
        neg_x1x3 = moment_map.LinearMomentMap(scale=-1, **kwargs)(x1x3)

        neg_reward = moment_map.AddUncorrelatedMomentMap([x0, x0x2, neg_x1x3], **kwargs)
        reward = moment_map.LinearMomentMap(scale=-1, **kwargs)(neg_reward)
        return reward

    return reward_moment_map
