"""Needed for the mbbl"""

import numpy as np
from mbbl.env import env_register

from pilco import reward_functions

ENV_LIST = [
    "gym_cheetah",
    "gym_walker2d",
    "gym_hopper",
    "gym_swimmer",
    "gym_ant",
    "gym_reacher",
    "gym_acrobot",
    "gym_cartpole",
    "gym_mountain",
    "gym_pendulum",
    "gym_invertedPendulum",
]


def get_env_info(env_name, env_seed, dtype):
    """ @brief: In this function return the reward moment function, initial
        state mean and initial state covariance.
    """

    env, env_info = env_register.make_env(
        env_name, env_seed, misc_info={"reset_type": "gym"}
    )

    # the reward moment function
    if env_name == "gym_walker2d":
        reward_moment_map = reward_functions.mbbl_walker_reward(
            8,  # velocity index
            0,  # height index
            1.3,  # target_height
            3.0,  # height_coefficient
            slice(17, 17 + 6),  # actions
            0.1,  # action coeff
        )

    elif env_name == "gym_hopper":
        reward_moment_map = reward_functions.mbbl_walker_reward(
            5,  # velocity index
            0,  # height index
            1.3,  # target_height
            3.0,  # height_coefficient
            slice(11, 11 + 3),  # actions
            0.1,  # action coeff
        )
    elif env_name == "gym_swimmer":
        reward_moment_map = reward_functions.mbbl_walker_reward(
            3,  # velocity index
            0,  # height index
            0.0,  # target_height
            0.0,  # height_coefficient
            slice(8, 10),  # actions
            0.0001,  # action coeff
        )
    elif env_name == "gym_fswimmer":
        reward_moment_map = reward_functions.fswimmer_walker_reward(
            8,  # velocity index
            0,  # height index
            0.0,  # target_height
            0.0,  # height_coefficient
            slice(8, 10),  # actions
            0.0001,  # action coeff
        )
    elif "gym_cheetah" in env_name:
        reward_moment_map = reward_functions.mbbl_walker_reward(
            8,  # velocity index
            0,  # height index
            0.0,  # target_height
            0.0,  # height_coefficient
            slice(17, 17 + 6),  # actions
            0.1,  # action coeff
        )
    elif env_name == "gym_ant":
        reward_moment_map = reward_functions.mbbl_walker_reward(
            13,  # velocity index
            0,  # height index
            0.57,  # target_height
            3.0,  # height_coefficient
            slice(27, 27 + 8),  # actions
            0.1,  # action coeff
        )
    elif env_name == "gym_reacher":
        # Action space has size 2. Distance index is [-3:len(state)]
        # So [-5:] gets both
        reward_moment_map = reward_functions.surrogate_reacher_reward(
            distance_index=slice(-5, None)
        )
    elif env_name == "gym_acrobot":
        reward_moment_map = reward_functions.acrobot_reward()

    elif "gym_cartpole" in env_name:
        reward_moment_map = reward_functions.cartpole_reward(
            position_index=0, position_coefficient=0.01, angle_index=2
        )

    elif env_name == "gym_mountain":
        reward_moment_map = reward_functions.mountain_car_reward()

    elif "gym_pendulum" in env_name:
        reward_moment_map = reward_functions.pendulum_reward(slice(3, 4))

    elif env_name == "gym_invertedPendulum":
        reward_moment_map = reward_functions.inverted_pendulum_reward()

    else:
        raise NotImplementedError

    # the mean of the initial state
    initial_state_mean = np.array(np.reshape(get_x0(env_name), [-1]), dtype=np.float64)

    # the cov of the initial state
    initial_state_covariance = get_covX0(env_name, env_info)

    return (
        env,
        env_info,
        np.array(initial_state_mean, dtype=dtype),
        np.array(initial_state_covariance, dtype=dtype),
        reward_moment_map,
    )


def get_covX0(env_name, env_info):
    min_cov = 0.05 ** 2 / 3
    cov = np.array(np.eye(env_info["ob_size"]), dtype=np.float64) * min_cov

    # get the x0
    if "gym_cheetah" in env_name:
        cov = np.array(np.eye(env_info["ob_size"]), dtype=np.float64) * (0.1 ** 2 / 3)
    elif env_name in ["gym_walker2d"]:
        cov = np.array(np.eye(env_info["ob_size"]), dtype=np.float64) * (0.005 ** 2 / 3)
    elif env_name in ["gym_hopper"]:
        cov = np.array(np.eye(env_info["ob_size"]), dtype=np.float64) * (0.005 ** 2 / 3)
    elif env_name in ["gym_swimmer"]:
        cov = np.array(np.eye(env_info["ob_size"]), dtype=np.float64) * (0.1 ** 2 / 3)
    elif env_name in ["gym_fswimmer"]:
        cov = np.array(np.eye(env_info["ob_size"]), dtype=np.float64) * (0.1 ** 2 / 3)
    elif env_name in ["gym_ant"]:
        cov = np.array(np.eye(env_info["ob_size"]), dtype=np.float64) * (0.1 ** 2 / 3)

    # the second type of envrionments
    # NOTE: the covariance is not correct though
    elif env_name in ["gym_acrobot"]:
        #  return np.array([cos(s[0]), np.sin(s[0]), cos(s[1]), sin(s[1]),
        #                   s[2], s[3]])
        cov[0:2, 0:2] = estimate_var_of_func_uniform([np.cos, np.sin], 0.1)
        cov[2:4, 2:4] = estimate_var_of_func_uniform([np.cos, np.sin], 0.1)
        cov[4, 4] = 0.1 ** 2 / 3
        cov[5, 5] = 0.1 ** 2 / 3
        # # estimate_var_of_func_uniform(func, max_val, num_points=1000):

    elif "gym_cartpole" in env_name:
        cov = np.array(np.eye(env_info["ob_size"]), dtype=np.float64) * (0.05 ** 2 / 3)

    elif env_name in ["gym_mountain"]:
        cov[0, 0] = 0.5 ** 2 / 3

    elif "gym_pendulum" in env_name:
        cov[0:2, 0:2] = estimate_var_of_func_uniform([np.cos, np.sin], 3.14)
        cov[2, 2] = 1.0 ** 2 / 3

    elif env_name in ["gym_invertedPendulum"]:
        cov = np.array(np.eye(env_info["ob_size"]), dtype=np.float64) * (0.01 ** 2 / 3)

    elif env_name in ["gym_petsReacher"]:
        pass
    elif env_name in ["gym_petsCheetah"]:
        pass
    elif env_name in ["gym_petsPusher"]:
        pass
    else:
        assert env_name == "gym_reacher"
        """ @brief:
            qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
            while True:
                self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
                if np.linalg.norm(self.goal) < 2:
                    break
            qpos[-2:] = self.goal
            qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
            qvel[-2:] = 0
            self.set_state(qpos, qvel)
        return self._get_obs()

        note that the initialization of reacher is as following
            np.cos(theta),  the theta -> 0 / 0,,  # theta_1: 0, 2; theta_2: 1, 3
            np.sin(theta),
            self.sim.data.qpos.flat[2:],  # 2
            self.sim.data.qvel.flat[:2],  # 2
            self.get_body_com("fingertip") - self.get_body_com("target")
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            qpos[-2:] = self.goal
            the mean of self.get_body_com("fingertip")
        """
        theta_cov = estimate_var_of_func_uniform([np.cos, np.sin], 0.1)
        cov[0, 0] = cov[1, 1] = theta_cov[0, 0]
        cov[0, 2] = cov[1, 3] = theta_cov[0, 1]
        cov[2, 0] = cov[3, 1] = theta_cov[1, 0]
        cov[2, 2] = cov[3, 3] = theta_cov[1, 1]

        cov[4, 4] = cov[5, 5] = 0.1 ** 2 / 3
        cov[6, 6] = cov[7, 7] = 0.005 ** 2 / 3
        cov[8:9] = 0.1 ** 2 / 3

    return cov


def get_x0(env_name):
    if env_name in [
        "gym_walker2d",
        "gym_hopper",
        "gym_swimmer",
        "gym_cheetah",
        "gym_ant",
    ]:
        from mbbl.env.gym_env.walker import env
    elif env_name == "gym_cheetah":
        from mbbl.env.gym_env.walker import env
    elif "gym_cheetah" in env_name:
        from mbbl.env.gym_env.noise_gym_cheetah import env
    elif "gym_fswimmer" in env_name:
        from mbbl.env.gym_env.fixed_swimmer import env
    elif env_name in ["gym_reacher"]:
        from mbbl.env.gym_env.reacher import env

    elif env_name in ["gym_acrobot"]:
        from mbbl.env.gym_env.acrobot import env
    elif env_name == "gym_cartpole":
        from mbbl.env.gym_env.cartpole import env
    elif "gym_cartpole" in env_name:
        from mbbl.env.gym_env.noise_gym_cartpole import env
    elif env_name in ["gym_mountain"]:
        from mbbl.env.gym_env.mountain_car import env
    elif env_name == "gym_pendulum":
        from mbbl.env.gym_env.pendulum import env
    elif "gym_pendulum" in env_name:
        from mbbl.env.gym_env.noise_gym_pendulum import env
    elif env_name in ["gym_invertedPendulum"]:
        from mbbl.env.gym_env.invertedPendulum import env
    elif env_name in ["gym_petsReacher", "gym_petsCheetah", "gym_petsPusher"]:
        from mbbl.env.gym_env.pets import env
    else:
        raise NotImplementedError

    gym_env = env(env_name, 1234, misc_info={})

    # get the x0
    if env_name in ["gym_cheetah", "gym_walker2d", "gym_hopper"]:
        x0 = np.concatenate(
            [gym_env._env.env.init_qpos[1:], gym_env._env.env.init_qvel]
        )
    elif "gym_cheetah" in env_name:
        x0 = np.concatenate(
            [gym_env._env.env.init_qpos[1:], gym_env._env.env.init_qvel]
        )
    elif env_name in ["gym_swimmer", "gym_ant"]:
        x0 = np.concatenate(
            [gym_env._env.env.init_qpos[2:], gym_env._env.env.init_qvel]
        )
    elif env_name in ["gym_fswimmer"]:
        x0 = np.concatenate(
            [gym_env._env.init_qpos[2:], gym_env._env.init_qvel, np.array([0])]
        )

    elif env_name in ["gym_acrobot"]:
        # np.array([cos(s[0]), np.sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3]])
        x0 = np.array([1, 0, 1, 0, 0, 0])
    elif "gym_cartpole" in env_name:
        x0 = np.zeros(4)
    elif env_name in ["gym_mountain"]:
        x0 = np.array([-0.5, 0])
    elif "gym_pendulum" in env_name:
        #  return np.array([np.cos(theta), np.sin(theta), thetadot])
        x0 = np.array([1, 0, 0])
    elif env_name in ["gym_invertedPendulum"]:
        x0 = np.concatenate([gym_env._env.env.init_qpos, gym_env._env.env.init_qvel])

    elif env_name in ["gym_petsReacher"]:
        qpos = np.copy(gym_env._env.init_qpos)
        qvel = np.copy(gym_env._env.init_qvel)
        qpos[-3:] = 0.0
        qvel[-3:] = 0.0
        raw_obs = np.concatenate([qpos, qvel[:-3]])
        EE_pos = np.reshape(gym_env._env.get_EE_pos(raw_obs[None]), [-1])

        x0 = np.concatenate([raw_obs, EE_pos])
    elif env_name in ["gym_petsCheetah"]:
        qpos = np.copy(gym_env._env.init_qpos)
        qvel = np.copy(gym_env._env.init_qvel)
        x0 = np.concatenate([np.array([0]), qpos[1:], qvel])
    elif env_name in ["gym_petsPusher"]:
        qpos = np.copy(gym_env._env.init_qpos)
        qvel = np.copy(gym_env._env.init_qvel)
        qpos[-4:-2] = [-0.25, 0.15]
        qpos[-2:] = [0.0]
        qvel[-4:] = 0
        qvel = np.copy(gym_env._env.init_qvel)
        other_obs = gym_env._env._get_obs()
        x0 = np.concatenate([qpos[:7], qvel[:7], other_obs[14:]])
    else:
        assert env_name == "gym_reacher"
        """ @brief: note that the initialization of reacher is as following
            np.cos(theta),  the theta -> 0 / 0
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            qpos[-2:] = self.goal
            the mean of self.get_body_com("fingertip")
        """
        x0 = np.concatenate(
            [
                [1, 1, 0, 0],
                [0, 0],  # gym_env.env.init_qpos[2:],
                gym_env._env.env.init_qvel[:2],
                gym_env._env.env.get_body_com("fingertip")
                - np.array([0, 0, 0]),  # gym_env.env.get_body_com("target")
            ]
        )

    return x0


def estimate_var_of_func_uniform(func, max_val, num_points=1000):
    if len(func) == 1:
        points = func[0](np.random.uniform(-max_val, max_val, num_points))
        return np.var(points)
    else:
        random_points = np.random.uniform(-max_val, max_val, num_points)
        y = []
        for i_fun in func:
            y.append(i_fun(random_points))

        y = np.array(y)
        return np.cov(y)


if __name__ == "__main__":
    for env_name in ENV_LIST:
        env, env_info, initial_state_mean, initial_state_covariance, _ = get_env_info(
            env_name, 1234
        )
        print("env_name:", env_name)
        print("initial_state_mean:", initial_state_mean)
        print("cov:", initial_state_covariance)
