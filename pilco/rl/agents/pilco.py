"""PILCO agent"""
import collections
import contextlib
import functools
import itertools
import logging
import math
import os
import time
import typing
from dataclasses import dataclass
from typing import Optional

import gym
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import sklearn.metrics
import tensorflow as tf

import pilco.gp.sklearn
from pilco import features
from pilco import moment_map
from pilco import utils
from pilco.utils import tf as tf_utils

from . import core

logger = logging.getLogger(__name__)

# TODO: standardsize input & output data before fitting the GP


class PILCOAgent(core.Agent):
    """A PILCO agent."""

    def __init__(
        self,
        horizon,
        reward_moment_map_fn=None,
        initial_state_mean=None,
        initial_state_covariance=None,
        policy_size=50,
        scipy_optimizer_method="l-bfgs-b",
        tensorflow_optimizer=None,
        policy_update_iterations=100,
        dynamics_regressor=None,
        initial_random_actions=False,
        max_history_buffer_size=None,
        dtype=np.float64,
        device=None,
        log_dir=None,
        full_logging=False,
        visualize=False,
        env=None,
        **kwargs,
    ):
        """Initialize a PILCOAgent

        Args:
            horizon: The horizon used when optimizing the policy in simulation.
                Either an integer or a callable with interface
                    horizon(None, None) => initial_horizon
                    horizon(current_horizon, episode_rewards) => new_horizon
                See `DynamicHorizon`.
            reward_moment_map_fn: A function creating the reward moment_map.
                Must accept `backend` and `dtype` as arguments.
                Derived from `env.metadata["reward.moment_map"] by default.
            initial_state_mean: Mean of the environment initial state
                distribution.
                Defaults to `env.metadata["initial_state.mean"]`.
            initial_state_covariance: Covariance of the environment initial
                state distribution.
                Defaults to `env.metadata["initial_state.covariance"]`.
            policy_size: Number of hidden units in the policy network.
                The policy net is a 1 hidden layer RBF network.
            scipy_optimizer_method: The scipy optimization method to use.
                See the `method` argument of `scipy.optimize.minimize`.
            tensorflow_optimizer: Use this TensorFlow optimizer instead of
                `scipy_optimizer`. An instance of `tf.train.Optimizer`.
            policy_update_iterations: Maximum number of policy updates to apply per
                dynamics update.
            dynamics_regressor: Dynamics gaussian process regressor.
                An instance of BaseRBFGaussianProcessRegressor.
                Defaults to SklearnRBFGaussianProcessRegressor.
            initial_random_actions: If True, take random uniform actions
                initially. If False, use a randomly initialized policy.
            max_history_buffer_size: Maximum number of observed transitions to store.
                The oldest data is discarded first. The dynamics are learned to
                completion each time so any discarded data is likely to be entirely
                forgotten by the dynamics model. The default is to keep all data.
            dtype: Data type used in the TensorFlow graph.
            device: Device on which most TF graph nodes are placed.
            log_dir: Log summaries into this directory.
            full_logging: Run with full logging enabled. Includes computationally costly
                metrics that are not run by default.k
            visualize: If True, produce a visualisation of the predicted
                dynamics.
            env: Environment on which the agent is to be run.
            **kwargs: Additional agent arguments. See `core.Agent`.
        """
        logger.info("Creating PILCO agent.")
        super().__init__(env=env, trainable=True, **kwargs)
        self.horizon = horizon
        self.scipy_optimizer_method = scipy_optimizer_method
        self.scipy_optimizer = None
        self.tensorflow_optimizer = tensorflow_optimizer
        self.policy_update_iterations = policy_update_iterations
        self.initial_random_actions = initial_random_actions
        self.dtype = dtype

        self._dynamic_horzon = callable(self.horizon)
        if self._dynamic_horzon:
            self.current_horizon = self.horizon(None, None)
        else:
            self.current_horizon = self.horizon

        if reward_moment_map_fn is None:
            try:
                reward_moment_map_fn = env.metadata["reward.moment_map"]
            except (AttributeError, KeyError):
                raise ValueError(
                    "Must specify `reward` or provide an environment "
                    "with the metadata key 'reward.moment_map'."
                )
        self._reward_moment_map_fn = reward_moment_map_fn

        if initial_state_mean is None:
            try:
                initial_state_mean = env.metadata["initial_state.mean"]
            except (AttributeError, KeyError):
                raise ValueError(
                    "Must specify `initial_state_mean` or provide an environment "
                    "with the metadata key 'initial_state.mean'."
                )
        self.initial_state_mean = initial_state_mean

        if initial_state_covariance is None:
            try:
                initial_state_covariance = env.metadata["initial_state.covariance"]
            except (AttributeError, KeyError):
                raise ValueError(
                    "Must specify `initial_state_covariance` or provide an environment"
                    "with the metadata key 'initial_state.covariance'."
                )
        self.initial_state_covariance = initial_state_covariance

        assert isinstance(self.action_space, gym.spaces.Box)

        self.observation_dim = np.prod(self.observation_space.shape, dtype=int)
        self.action_dim = np.prod(self.action_space.shape, dtype=int)
        self.oa_dim = self.observation_dim + self.action_dim

        if np.all(np.isfinite(self.action_space.high)):
            self._policy_scale = self.action_space.high
            assert np.all(np.array_equal(self.action_space.low, -self._policy_scale))
        elif not np.any(np.isfinite(self.action_space.high)):
            self._policy_scale = None
        else:
            raise NotImplementedError("Mixed finite/infinite actions not supported.")

        if dynamics_regressor is None:
            dynamics_regressor = pilco.gp.sklearn.SklearnRBFGaussianProcessRegressor(
                noise_variance_bounds=(1e-10, 1e5), shared_kernel=False
            )
        self.dynamics = dynamics_regressor
        logger.info("Dynamics model: %s", self.dynamics)

        if max_history_buffer_size is None:
            self._X = []
            self._y = []
        else:
            self._X = collections.deque(maxlen=max_history_buffer_size)
            self._y = collections.deque(maxlen=max_history_buffer_size)

        # Whether the dynamics model has been trained
        self._trained_dynamics = False

        self._episode_index = 0
        self._episode_length = 0
        # True total reward returned from environment
        self._episode_true_reward = 0
        # Total reward from applying `self.reward` to the true episode states.
        self._episode_surrogate_reward = 0

        self._policy_update_epoch = 0

        self._full_logging = full_logging
        self._visualize = visualize
        # Plots are produced and saved when full_logging=True, just not shown to screen.
        self._produce_plots = self._full_logging or self._visualize
        self._log_history = (
            self._full_logging or self._produce_plots or self._dynamic_horzon
        )
        logger.debug("Log History: %s", self._log_history)

        if self._log_history:
            # Dict mapping attribute name to list of values.
            self._episode_history = self._new_episode_history()
            self._episode_prediction = None

        if self._produce_plots:
            self._fig = None
            self._prediction_axes = None
            self._figure_dir = None
            if log_dir is not None:
                self._figure_dir = os.path.join(log_dir, "figures")
                os.makedirs(self._figure_dir, exist_ok=True)

        self.feature_net = features.FlatFeatures(self.observation_space)
        logger.info("Creating PILCO TensorFlow graph.")
        self.graph = tf.Graph()
        with self.graph.as_default():  # pylint: disable=not-context-manager
            session_config = tf.ConfigProto(allow_soft_placement=True)
            self.session = tf.Session(graph=self.graph, config=session_config)
            if device is not None:
                with tf.device(device):
                    self.net = self._build_net(policy_size)
            else:
                self.net = self._build_net(policy_size)

            logger.info("Done creating PILCO graph.")

            logger.info("Initializing variables.")
            self.session.run(tf.global_variables_initializer())
            logger.info("Done initializing variables.")

        if log_dir is not None:
            self.tf_writer = tf.summary.FileWriter(
                log_dir, graph=self.graph, session=self.session
            )
            self.tf_writer.flush()
        else:
            self.tf_writer = None
        logger.info("Done creating PILCO agent.")

    def _new_episode_history(self):
        return {"observation": [], "action": [], "reward": [], "surrogate_reward": []}

    def _policy_fn(self, noise_variance=None, backend="numpy"):
        """Get the policy function moment mapper.

        action = sin(policy(state + noise))
        where
            * sin is present iff self._policy_scale is not None,
            * noise is present iff noise_variance is not None.

        Args:
            noise_variance: If present, Gaussian white noise with this variance
                is added to the state feature vector before passing through the
                policy function.
            backend: Backend to use. "numpy" or "tensorflow"
        """
        gp_cls = moment_map.gp.DeterministicGaussianProcessMomentMap
        policy_fn = gp_cls.from_params(self._policy, backend=backend, dtype=self.dtype)

        if noise_variance is not None:
            noise_fn = moment_map.math.WhiteNoiseMomentMap(
                noise_variance=noise_variance,
                input_dim=self.observation_dim,
                dtype=self.dtype,
                backend=backend,
            )
            policy_fn = policy_fn.compose(noise_fn)

        # Squash the policy output with sin if the action domain is finite.
        if self._policy_scale is not None:
            sin_fn = moment_map.math.SinMomentMap(
                output_scale=self._policy_scale, backend=backend, dtype=self.dtype
            )
            policy_fn = sin_fn.compose(policy_fn)

        return policy_fn

    def _state_action_fn(self, noise_variance=None, backend="numpy"):
        """Moment map of state to joint state-action using the policy."""
        policy_fn = self._policy_fn(noise_variance=noise_variance, backend=backend)
        return moment_map.core.JointInputOutputMomentMap(
            policy_fn, backend=backend, dtype=self.dtype
        )

    def _reward_fn(self, backend="numpy"):
        return self._reward_moment_map_fn(backend=backend, dtype=self.dtype)

    def _ph_dynamics_fn(self):
        """Dynamics function on placeholder inputs."""
        params = self._dynamics_params
        ph_dynamics_fn = moment_map.gp.GaussianProcessMomentMap(
            inducing_points=params["inducing_points"].value,
            coefficients=params["coefficients"].value,
            gram_L=params["gram_L"].value,
            signal_variance=params["signal_variance"].value,
            length_scale=params["length_scale"].value,
            backend="tensorflow",
            dtype=self.dtype,
        )
        return ph_dynamics_fn, params["noise_variance"].value

    @contextlib.contextmanager
    def _dynamics_feed_dict(self):
        """Dynamics model parameters feed dict."""
        gp_params = self.dynamics.get_params()
        gp_param_handles = self.session.run(
            {name: t.assign_op for name, t in self._dynamics_params.items()},
            feed_dict={
                t.assign_ph: getattr(gp_params, name)
                for name, t in self._dynamics_params.items()
            },
        )
        yield {
            t.handle_ph: gp_param_handles[name].handle
            for name, t in self._dynamics_params.items()
        }

        for handle in gp_param_handles.values():
            handle.delete()

    def _build_net(self, policy_size):
        """Build agent networks in the TensorFlow graph.

        Args:
            policy_size: Number of inducing points in the policy.

        Returns:
            net: A dictionary of important tensors in the graph.
        """
        net = {}

        observation_ph, features_op = self.feature_net.build()
        features_op = tf.cast(features_op, dtype=self.dtype)
        net["observation"] = observation_ph

        # Policy is the mean of a GP with n inducing points
        state_size = int(features_op.shape[-1])
        with tf.variable_scope("policy"):
            self._policy = pilco.gp.RBFGaussianProcessParameters(
                inducing_points=tf.get_variable(
                    "inducing_points",
                    shape=[policy_size, state_size],
                    dtype=self.dtype,
                    initializer=tf.initializers.random_normal(),
                ),
                length_scale=tf.exp(
                    tf.get_variable(
                        "log_length_scale",
                        shape=[state_size],
                        dtype=self.dtype,
                        initializer=tf.initializers.constant(1),
                    )
                ),
                coefficients=tf.get_variable(
                    "coefficients",
                    shape=[self.action_dim, policy_size],
                    dtype=self.dtype,
                    initializer=tf.initializers.random_normal(),
                ),
            )

        # Action
        policy_fn = self._policy_fn(backend="tensorflow")
        net["action"] = policy_fn(
            features_op, return_cov=False, return_io_cov=False
        ).output_mean

        def _make_persistent_tensor(shape=None, name=None):
            return tf_utils.PersistentTensor(
                session=self.session, dtype=self.dtype, shape=shape, name=name
            )

        # Dynamics model parameters
        state_action_size = state_size + self.action_dim
        dynamics_num_kernels = 1 if self.dynamics.shared_kernel else state_size
        with tf.name_scope("dynamics"):
            self._dynamics_params = {
                "inducing_points": _make_persistent_tensor(
                    shape=[None, state_action_size], name="inducing_points"
                ),
                "coefficients": _make_persistent_tensor(
                    shape=[state_size, None], name="coefficients"
                ),
                "gram_L": _make_persistent_tensor(
                    shape=[dynamics_num_kernels, None, None], name="gram_L"
                ),
                "signal_variance": _make_persistent_tensor(
                    shape=[dynamics_num_kernels], name="signal_variance"
                ),
                "length_scale": _make_persistent_tensor(
                    shape=[dynamics_num_kernels, state_action_size], name="length_scale"
                ),
                "noise_variance": _make_persistent_tensor(
                    shape=[dynamics_num_kernels], name="noise_variance"
                ),
            }

        horizon = tf.placeholder_with_default(
            input=tf.constant(self.current_horizon, dtype=tf.int32),
            shape=(),
            name="horizon_",  # Let the summary node have name "horizon"
        )
        net["horizon"] = horizon

        initial_state_mean_const = tf.constant(
            self.initial_state_mean, dtype=features_op.dtype
        )
        initial_state_mean = tf.placeholder_with_default(
            initial_state_mean_const,
            shape=initial_state_mean_const.shape,
            name="initial_state_mean",
        )
        initial_state_covariance_const = tf.constant(
            self.initial_state_covariance, dtype=initial_state_mean.dtype
        )
        initial_state_covariance = tf.placeholder_with_default(
            initial_state_covariance_const,
            shape=initial_state_covariance_const.shape,
            name="initial_state_covariance",
        )
        net["initial_state_mean"] = initial_state_mean
        net["initial_state_covariance"] = initial_state_covariance

        # Predict reward
        predicted_total_reward, predictions = self._predict_dynamics_net(
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            horizon=horizon,
            return_predictions=self._log_history,
        )
        net["predictions"] = predictions

        net["predicted_total_reward"] = predicted_total_reward
        predicted_mean_step_reward = predicted_total_reward / tf.cast(
            horizon, dtype=predicted_total_reward.dtype
        )
        net["predicted_mean_step_reward"] = predicted_mean_step_reward
        with tf.device(None):
            global_step = tf.train.create_global_step()

        loss = -predicted_mean_step_reward
        if self.tensorflow_optimizer is not None:
            policy_update = self.tensorflow_optimizer.minimize(
                loss, global_step=global_step
            )
            net["policy_update"] = policy_update
        else:
            self.scipy_optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                loss=loss,
                method=self.scipy_optimizer_method,
                options={"maxiter": self.policy_update_iterations},
            )

        with tf.device(None):
            epoch_ph = tf.placeholder(dtype=tf.int32, shape=())
            net["epoch"] = epoch_ph
            net["global_step"] = global_step
            net["increment_global_step"] = tf.assign_add(
                global_step, 1, name="increment_global_step"
            )

            tf.summary.scalar("predicted_total_reward", predicted_total_reward)
            tf.summary.scalar("predicted_mean_step_reward", predicted_mean_step_reward)
            tf.summary.scalar("epoch", epoch_ph)
            tf.summary.scalar("horizon", horizon)
            net["summary"] = tf.summary.merge_all()

        return net

    def _predict_dynamics_net(
        self, initial_state_mean, initial_state_covariance, horizon, return_predictions
    ):
        """Build network that predicts future dynamics and reward.

        Args:
            initial_state_mean: Mean of the initial state distribution.
            initial_state_covariance: Covariance of the initial state
                distribution. May be None.
            horizon: Number of time steps to predict forward, including the
                initial state.
            return_predictions: Whether to return the dynamics predictions or just total
                reward.

        Returns:
            total_reward: A scalar tensor containing the expected total reward
                for the predicted dynamics.
            predictions: The episode prediction tensors. An _EpisodePrediction
                instance. Each element is a tensor whose first dimension is
                size `history`. Is `None` if `return_predictions` is False.
        """
        with tf.name_scope("predict_dynamics"):
            # Distribution over future states
            dynamics_fn, noise_variance = self._ph_dynamics_fn()
            # Map state to joint state-action
            state_action_fn = self._state_action_fn(
                noise_variance=noise_variance, backend="tensorflow"
            )
            reward_fn = self._reward_fn(backend="tensorflow")

            with tf.name_scope("initial_state"):
                (
                    initial_state_action_mean,
                    initial_state_action_covariance,
                ) = self._state_action_distribution(
                    state_action_fn=state_action_fn,
                    state_mean=initial_state_mean,
                    state_covariance=initial_state_covariance,
                )
                initial_reward = tf.squeeze(
                    reward_fn(
                        mean=initial_state_action_mean,
                        covariance=initial_state_action_covariance,
                        return_cov=False,
                        return_io_cov=False,
                    ).output_mean,
                    axis=-1,
                )

            # Create tf.TensorArrays to hold the history
            def _make_history_array_for(elem, name):
                array = tf.TensorArray(
                    dtype=elem.dtype,
                    size=horizon,
                    dynamic_size=False,
                    clear_after_read=True,
                    tensor_array_name=name,
                    infer_shape=False,
                    element_shape=elem.shape,
                )
                return array.write(0, elem)

            if return_predictions:
                initial_prediction = _EpisodePrediction(
                    state_action_means=_make_history_array_for(
                        initial_state_action_mean, "state_action_means"
                    ),
                    state_action_covariances=_make_history_array_for(
                        initial_state_action_covariance, "state_action_covariances"
                    ),
                    reward_means=_make_history_array_for(
                        initial_reward, "reward_means"
                    ),
                    reward_variances=_make_history_array_for(
                        tf.zeros_like(initial_reward), "reward_variances"
                    ),
                )
            else:
                # Placeholder tensor
                initial_prediction = tf.zeros(0)

            # Set up while loop condition & step
            def condition(
                i, state_action_mean, state_action_covariance, total_reward, prediction
            ):
                del state_action_mean, state_action_covariance, total_reward
                del prediction
                return i < horizon

            def dynamics_step(
                i, state_action_mean, state_action_covariance, total_reward, prediction
            ):
                with tf.name_scope("dynamics_step"):
                    (
                        next_state_mean,
                        next_state_covariance,
                    ) = self._next_state_distribution(
                        dynamics_fn=dynamics_fn,
                        state_action_mean=state_action_mean,
                        state_action_covariance=state_action_covariance,
                    )
                    (
                        next_state_action_mean,
                        next_state_action_covariance,
                    ) = self._state_action_distribution(
                        state_action_fn=state_action_fn,
                        state_mean=next_state_mean,
                        state_covariance=next_state_covariance,
                    )
                    next_reward = reward_fn(
                        next_state_action_mean,
                        next_state_action_covariance,
                        return_cov=True,
                        return_io_cov=False,
                    )
                    next_reward_mean = tf.squeeze(next_reward.output_mean, axis=-1)
                    next_reward_variance = tf.squeeze(
                        next_reward.output_covariance, axis=[-2, -1]
                    )
                    if return_predictions:
                        next_prediction = _EpisodePrediction(
                            state_action_means=prediction.state_action_means.write(
                                i, next_state_action_mean
                            ),
                            state_action_covariances=(
                                prediction.state_action_covariances.write(
                                    i, next_state_action_covariance
                                )
                            ),
                            reward_means=prediction.reward_means.write(
                                i, next_reward_mean
                            ),
                            reward_variances=prediction.reward_variances.write(
                                i, next_reward_variance
                            ),
                        )
                    else:
                        # Propagate placeholder
                        next_prediction = prediction
                    return (
                        i + 1,
                        next_state_action_mean,
                        next_state_action_covariance,
                        total_reward + next_reward_mean,
                        next_prediction,
                    )

            _, _, _, total_reward, final_prediction = tf.while_loop(
                cond=condition,
                body=dynamics_step,
                loop_vars=(
                    tf.constant(1),
                    initial_state_action_mean,
                    initial_state_action_covariance,
                    initial_reward,
                    initial_prediction,
                ),
                back_prop=True,
                swap_memory=False,
                return_same_structure=True,
            )

            # Convert tensorarray into tensor
            if return_predictions:
                final_prediction_tensors = _EpisodePrediction(
                    *[array.stack() for array in final_prediction]
                )
            else:
                final_prediction_tensors = None
            return total_reward, final_prediction_tensors

    def _state_action_distribution(self, state_action_fn, state_mean, state_covariance):
        """The state-action distribution."""
        with tf.name_scope("state_action_distribution"):
            state_action = state_action_fn(
                mean=state_mean,
                covariance=state_covariance,
                return_cov=True,
                return_io_cov=False,
            )
            return state_action.output_mean, state_action.output_covariance

    def _next_state_distribution(
        self, dynamics_fn, state_action_mean, state_action_covariance
    ):
        """Predicted next state distribution from state-action distribution."""
        with tf.name_scope("next_state_distribution"):
            state_delta = dynamics_fn(
                mean=state_action_mean,
                covariance=state_action_covariance,
                return_cov=True,
                return_io_cov=True,
            )
            state_size = int(state_delta.output_mean.shape[-1])

            # Cov(state, state delta)
            s_sdelta_cross_cov = state_delta.output_input_covariance[:, :state_size]

            next_state_mean = state_action_mean[:state_size] + state_delta.output_mean
            next_state_covariance = (
                state_action_covariance[:state_size, :state_size]
                + state_delta.output_covariance
                + s_sdelta_cross_cov
                + tf.matrix_transpose(s_sdelta_cross_cov)
            )
            return next_state_mean, next_state_covariance

    def _expected_reward_net(
        self, state_action_mean, state_action_covariance, state_size
    ):
        """Expected rewards given batched state-action distributions."""
        with tf.name_scope("expected_reward"):
            reward_fn = self._reward_fn(backend="tensorflow")
            reward_mean = reward_fn(
                state_action_mean,
                state_action_covariance,
                return_cov=False,
                return_io_cov=False,
            ).output_mean
            return reward_mean

    def _act_normal(self, observation):
        observation_features = self.feature_net.prepare((observation,))

        if (
            self._log_history
            and self._trained_dynamics  # Can't predict if no trained dynamics
            and self._episode_prediction is None
        ):
            self._episode_prediction = self._predict_episode(
                np.squeeze(observation_features, axis=0), horizon=self.current_horizon
            )

        if self.initial_random_actions and self._policy_update_epoch == 0:
            return self.action_space.sample()

        return self._policy_action(observation_features)

    def _policy_action(self, observation_features):
        """Select an action from the policy for the given observation."""
        action_op = self.net["action"]
        action = np.squeeze(
            self.session.run(
                action_op, {self.net["observation"]: observation_features}
            ),
            axis=0,
        )
        action = np.reshape(action, action.shape[:-1] + self.action_space.shape)
        return action

    def _predict_episode(self, observation_features, horizon):
        """Predict dynamics starting from the given observation."""
        num_features = observation_features.shape[-1]
        with self._dynamics_feed_dict() as dynamics_feed_dict:
            feed_dict = {
                self.net["initial_state_mean"]: observation_features,
                self.net["initial_state_covariance"]: np.zeros(
                    [num_features, num_features], dtype=observation_features.dtype
                ),
                self.net["epoch"]: self._policy_update_epoch,
                self.net["horizon"]: horizon,
                **dynamics_feed_dict,
            }
            return self.session.run(self.net["predictions"], feed_dict=feed_dict)

    def update(self, step_info):
        observation_features = np.squeeze(
            self.feature_net.prepare((step_info.observation,)), axis=0
        )
        action_features = np.asarray(step_info.action.flat)
        observation_action_features = np.concatenate(
            [observation_features, action_features], axis=0
        )
        self._X.append(observation_action_features)
        next_observation_features = np.squeeze(
            self.feature_net.prepare((step_info.next_observation,)), axis=0
        )
        self._y.append(next_observation_features - observation_features)
        self._episode_length += 1
        self._episode_true_reward += step_info.reward
        reward_fn = self._reward_fn(backend="numpy")
        surrogate_reward = np.squeeze(
            reward_fn(
                observation_action_features, return_cov=False, return_io_cov=False
            ).output_mean,
            axis=-1,
        )
        self._episode_surrogate_reward += surrogate_reward

        if self._log_history:
            self._episode_history["observation"].append(observation_features)
            self._episode_history["action"].append(action_features)
            self._episode_history["reward"].append(step_info.reward)
            self._episode_history["surrogate_reward"].append(surrogate_reward)

        if step_info.done:
            logger.info("======== Episode Complete ========")
            logger.info("Episode Index: %d", self._episode_index)
            logger.info("Episode Length: %d", self._episode_length)
            logger.info("Episode True Reward: %g", self._episode_true_reward)
            logger.info("Episode Surrogate Reward: %g", self._episode_surrogate_reward)
            summaries = [
                tf.Summary.Value(
                    tag="episode_length", simple_value=self._episode_length
                ),
                tf.Summary.Value(
                    tag="episode_true_reward", simple_value=self._episode_true_reward
                ),
                tf.Summary.Value(
                    tag="episode_surrogate_reward",
                    simple_value=self._episode_surrogate_reward,
                ),
            ]

            if self._full_logging and self._episode_index > 0:
                # Log dynamics model prediction accuracy on the episode that just
                # completed
                # The episode history arrays are all the same length aligned to the same
                # timestep. We want to predict the next timestep so the prediction
                # inputs are indexed by [:-1] and the targets are [1:].
                # TODO: These predictions seem unexpectedly bad, are the targets right?
                episode_observations = np.asarray(self._episode_history["observation"])
                episode_observation_actions = np.concatenate(
                    [episode_observations, np.asarray(self._episode_history["action"])],
                    axis=-1,
                )
                predicted_observations, predicted_obs_vars = self.dynamics.predict(
                    episode_observation_actions[:-1],
                    return_var=True,
                    predictive_noise=True,
                )
                summaries.extend(
                    _regression_evaluation_summaries(
                        "dynamics_metrics/episode_1step",
                        # Model predicts change in observations
                        y_true=episode_observations[1:] - episode_observations[:-1],
                        y_pred=predicted_observations,
                        y_pred_normal_var=predicted_obs_vars,
                    )
                )

                compare_len = min(
                    len(episode_observation_actions),  # true episode len,
                    len(self._episode_prediction.state_action_means),  # pred horizon
                )
                summaries.extend(
                    _regression_evaluation_summaries(
                        "dynamics_metrics/episode_fromstart",
                        y_true=episode_observation_actions[:compare_len],
                        y_pred=self._episode_prediction.state_action_means[
                            :compare_len
                        ],
                        y_pred_normal_cov=self._episode_prediction.state_action_covariances[
                            :compare_len
                        ],
                    )
                )

            if self._dynamic_horzon:
                original_horizon = self.current_horizon
                self.current_horizon = self.horizon(
                    self.current_horizon, self._episode_history["reward"]
                )
                logger.debug(
                    "Updated horizon: %d => %d", original_horizon, self.current_horizon
                )

            if self._produce_plots:
                self.plot_episode()
                if self._visualize:
                    plt.pause(0.1)
                if self._figure_dir is not None:
                    plt.savefig(
                        os.path.join(
                            self._figure_dir, f"episode{self._episode_index}.svg"
                        )
                    )

            if self._log_history:
                self._episode_history = self._new_episode_history()
                self._episode_prediction = None

            self._episode_index += 1
            self._episode_length = 0
            self._episode_true_reward = 0
            self._episode_surrogate_reward = 0

            summaries.extend(self._update_dynamics_model())
            summaries.extend(self._update_policy(self.policy_update_iterations))

            if self.tf_writer is not None:
                summaries = [summary for summary in summaries if summary is not None]
                self.tf_writer.add_summary(
                    tf.Summary(value=summaries), global_step=self._episode_index
                )
                self.tf_writer.flush()
            return

    def _update_dynamics_model(self):
        """Update the dynamics model from the recorded history.

        Returns:
            A list of tf.Summary.Value objects.
        """
        summaries = []
        logger.info("= Updating dynamics model =")
        summaries.append(_summarize_and_log("history_buffer_size", len(self._X), "%d"))

        start_time = time.monotonic()
        try:
            self.dynamics.fit(self._X, self._y)
        except (tf.errors.InvalidArgumentError, tf.errors.OpError):
            logging.exception("Dynamics training failed")

        end_time = time.monotonic()
        self._trained_dynamics = True

        elapsed_seconds = end_time - start_time
        logger.info("Updating dynamics model complete.")

        summaries.append(
            _summarize_and_log("dynamics_update_seconds", elapsed_seconds, "%.3f")
        )
        summaries.extend(_gp_parameter_summaries("dynamics_params", self.dynamics))
        if self._full_logging:
            y_pred, y_pred_var = self.dynamics.predict(
                self._X, return_var=True, predictive_noise=True
            )
            summaries.extend(
                _regression_evaluation_summaries(
                    "dynamics_metrics/train",
                    y_true=self._y,
                    y_pred=y_pred,
                    y_pred_normal_var=y_pred_var,
                )
            )

        return summaries

    def _update_policy(self, iterations):
        """Update policy given the current dynamics model."""
        summaries = []
        logger.info("= Updating policy. Epoch %d. =", self._policy_update_epoch)
        logger.debug("Horizon: %d", self.current_horizon)
        start_time = time.monotonic()

        with self._dynamics_feed_dict() as dynamics_feed_dict:
            feed_dict = {
                self.net["epoch"]: self._policy_update_epoch,
                self.net["horizon"]: self.current_horizon,
                **dynamics_feed_dict,
            }

            policy_update_op = self.net.get("policy_update")
            try:
                if policy_update_op is None:
                    predicted_total_reward = self._train_policy_scipy(feed_dict)
                else:
                    predicted_total_reward = self._train_policy_tensorflow(
                        policy_update_op, iterations, feed_dict
                    )
            except (tf.errors.InvalidArgumentError, tf.errors.OpError):
                logging.exception("Policy training failed")
                predicted_total_reward = None

        logger.info("Policy improvement complete.")
        summaries.append(
            _summarize_and_log(
                "policy_update_seconds", time.monotonic() - start_time, "%.3f"
            )
        )

        if predicted_total_reward is None:
            logger.info("No predicted reward (policy unchanged or training crashed)")
        else:
            logger.info("Predicted total reward: %f", predicted_total_reward)
            logger.info(
                "Predicted mean step reward: %f",
                predicted_total_reward / self.current_horizon,
            )
        self._policy_update_epoch += 1
        return summaries

    def _train_policy_scipy(self, feed_dict):
        """Train the policy using a scipy optimizer."""

        # loss_callback may be called several times per step as the line search
        # is performed.
        # step_callback is called after, with the new variable values when
        # a step is taken.
        #
        # We want to record summaries at each step, but we don't observe it on
        # step_callback. Instead, keep track of the last seen values from
        # loss_callback and save on step_callback.

        info = {}
        increment_global_step_op = self.net["increment_global_step"]
        total_reward = None

        def loss_callback(global_step, summary, total_reward):
            info["global_step"] = global_step
            info["summary"] = summary
            info["total_reward"] = total_reward

        def step_callback(*args):
            del args  # Unused
            if self.tf_writer is not None:
                self.tf_writer.add_summary(
                    info["summary"], global_step=info["global_step"]
                )
            nonlocal total_reward
            total_reward = info["total_reward"]
            self.session.run(increment_global_step_op)

        self.scipy_optimizer.minimize(
            session=self.session,
            feed_dict=feed_dict,
            fetches=[
                self.net["global_step"],
                self.net["summary"],
                self.net["predicted_total_reward"],
            ],
            step_callback=step_callback,
            loss_callback=loss_callback,
        )
        if self.tf_writer is not None:
            self.tf_writer.flush()
        return total_reward

    def _train_policy_tensorflow(self, policy_update_op, iterations, feed_dict):
        """Train the policy using a policy update op in TensorFlow."""
        predicted_total_reward = self.net["predicted_total_reward"]
        predicted_mean_step_reward = self.net["predicted_mean_step_reward"]
        summary_op = self.net["summary"]
        global_step_op = self.net["global_step"]

        with utils.cli.message_progress_bar(["reward"], iterations) as bar:
            for i in range(iterations):
                if i == 1 and self._policy_update_epoch == 0:
                    # Record execution stats. Do on 2nd update instead of 1st
                    # to avoid possible transient delays.
                    run_options = tf.RunOptions(
                        # pylint: disable=no-member
                        trace_level=tf.RunOptions.FULL_TRACE
                    )
                    run_metadata = tf.RunMetadata()
                else:
                    run_options = None
                    run_metadata = None

                (total_reward, mean_reward, _, summary, global_step) = self.session.run(
                    (
                        predicted_total_reward,
                        predicted_mean_step_reward,
                        policy_update_op,
                        summary_op,
                        global_step_op,
                    ),
                    feed_dict=feed_dict,
                    options=run_options,
                    run_metadata=run_metadata,
                )
                if self.tf_writer is not None:
                    self.tf_writer.add_summary(summary, global_step=global_step)
                    if run_metadata is not None:
                        self.tf_writer.add_run_metadata(
                            run_metadata, f"epoch{self._policy_update_epoch}"
                        )
                        self.tf_writer.flush()
                bar.update(i, reward=mean_reward)
        if self.tf_writer is not None:
            self.tf_writer.flush()
        return total_reward

    def plot_episode(self):
        """Plot episode trajectory and predictions."""
        if self._fig is None:
            self._fig = plt.figure()

        if self._prediction_axes is None:
            self._prediction_axes = self._fig.subplots(
                nrows=self.oa_dim + 1, ncols=1, sharex=True, sharey=False
            )

        for ax in self._prediction_axes:
            ax.clear()

        observation_axes = self._prediction_axes[: self.observation_dim]
        action_axes = self._prediction_axes[self.observation_dim : -1]
        reward_axis = self._prediction_axes[-1]

        for i, ax in enumerate(observation_axes):
            ax.set_title(f"Obs {i}")
        for i, ax in enumerate(action_axes):
            ax.set_title(f"Action {i}")
        reward_axis.set_title("Reward")
        reward_axis.set_xlabel("Step")
        xlim = self.current_horizon
        reward_axis.set_xlim(0, xlim)

        actual_state_actions = np.concatenate(
            [
                np.asarray(self._episode_history["observation"]),
                np.asarray(self._episode_history["action"]),
            ],
            axis=-1,
        )
        actual_rewards = np.asarray(self._episode_history["reward"])
        surrogate_rewards = np.asarray(self._episode_history["surrogate_reward"])

        if self._episode_prediction is None:
            plot_predictions(
                self._prediction_axes[: self.oa_dim],
                y_obs=actual_state_actions[:xlim, :],
            )
            plot_predictions([reward_axis], y_obs=surrogate_rewards[:xlim, None])
        else:
            plot_predictions(
                self._prediction_axes[: self.oa_dim],
                y_mean=self._episode_prediction.state_action_means[:xlim, :],
                y_var=self._episode_prediction.state_action_covariances[:xlim, :, :],
                y_obs=actual_state_actions,
            )
            plot_predictions(
                [reward_axis],
                y_mean=self._episode_prediction.reward_means[:xlim, None],
                y_var=self._episode_prediction.reward_variances[:xlim, None],
                y_obs=surrogate_rewards[:xlim, None],
            )
        reward_axis.plot(actual_rewards[:xlim], c="g")


@dataclass
class DynamicHorizon:
    """An dynamic horizon function that grows exponentially based on episode rewards.

    The horizon is expanded by a constant multiplicative factor if all of the following
    conditions are met:
        * the episode lenth is >= current_horizon
        * the average reward over [transient_steps:current_horizon]
            is >= average_reward_threshold (if not None)
        * the total reward over [transient_steps:current_horizon]
            is >= total_reward_threshold (if not None)

    Attributes:
        minimum_horizon: The minimum and initial horizon size.
        maximum_horizon: Optional maximum horizon size.
        expansion_factor: Horizon expanson factor.
        average_reward_threshold: Optional average per-step reward threshold.
        total_reward_threshold: Optional total reward threshold for expansion.
        transient_steps: Ignore this many steps at the start of the episode.
    """

    minimum_horizon: int = 10
    maximum_horizon: Optional[int] = None
    expansion_factor: float = 2.0
    average_reward_threshold: Optional[float] = 0.5
    total_reward_threshold: Optional[float] = None
    transient_steps: int = 0

    def __call__(
        self, current_horizon: int, episode_rewards: typing.List[float]
    ) -> int:
        """Get a new horizon size based on the current horizon and episode rewards.

        Args:
            current_horizon: The current horizon value.
                May be `None` to indicate that an initial reward should be returned.
                If `None`, the other arguments will be `None` as well.
            episode_rewards: A list of rewards produced by a single episode where
                planning used `current_horizon`.

        Returns:
            horizon: The new horizon size to use.
        """
        if current_horizon is None:
            return self.minimum_horizon

        valid_current_horizon = max(current_horizon, self.minimum_horizon)
        if self.maximum_horizon is not None:
            valid_current_horizon = min(valid_current_horizon, self.maximum_horizon)

        if len(episode_rewards) < current_horizon:
            return valid_current_horizon

        relevant_rewards = episode_rewards[self.transient_steps : current_horizon]
        total_reward = sum(relevant_rewards)
        if (
            self.total_reward_threshold is not None
            and total_reward < self.total_reward_threshold
        ):
            return valid_current_horizon

        if (
            self.average_reward_threshold is not None
            and total_reward < self.average_reward_threshold * len(relevant_rewards)
        ):
            return current_horizon

        horizon = int(math.ceil(current_horizon * self.expansion_factor))
        horizon = max(horizon, self.minimum_horizon)
        if self.maximum_horizon is not None:
            horizon = min(horizon, self.maximum_horizon)
        return horizon


def plot_predictions(axes, y_mean=None, y_var=None, y_obs=None, x=None, width_std=2):
    """Draw predictions on a list of axes.

    Args:
        axes: A list NUM_DIMS axes on which to plot the predictions.
        y_mean: Prediction means. An array of shape `[NUM_STEPS, NUM_DIMS]`.
        y_var: Prediction variances. An array of shape `[NUM_STEPS, NUM_DIMS]`
            or `[NUM_STEPS, NUM_DIMS, NUM_DIMS]`.
        y_obs: Optional observations. An array shape `[N, NUM_DIMS]`.
        x: Optional x values for y_mean. An array of shape `[NUM_STEPS]`.
            Defaults to range(NUM_STEPS).
        width_std: Width of the shaded uncertainty region in terms of standard
            deviations.
    """
    if y_mean is None or y_var is None:
        if y_obs is None:
            return  # Nothing to plot
        for ax, y_obs_col in zip(axes, y_obs.T):
            ax.plot(y_obs_col, color="k")
        return

    if x is None:
        x = np.arange(len(y_mean))

    if len(y_var.shape) > 2:
        y_var = np.diagonal(y_var, axis1=-2, axis2=-1)
    y_offset = np.sqrt(y_var) * width_std

    if y_obs is None:
        y_obs_T = itertools.repeat(None)
    else:
        y_obs_T = y_obs.T

    for ax, y_mean_col, y_offset_col, y_obs_col in zip(
        axes, y_mean.T, y_offset.T, y_obs_T
    ):
        if y_mean_col is not None and y_offset_col is not None:
            ax.fill_between(
                x, y_mean_col - y_offset_col, y_mean_col + y_offset_col, alpha=0.5
            )
        if y_obs_col is not None:
            ax.plot(y_obs_col, color="k")


class _EpisodePrediction(typing.NamedTuple):
    state_action_means: typing.Any
    state_action_covariances: typing.Any
    reward_means: typing.Any
    reward_variances: typing.Any = None


def _summarize_and_log(name, value, fmt="%g"):
    logger.debug(f"{name}: {fmt}", value)
    return tf.Summary.Value(tag=name, simple_value=value)


def _summarize_histogram(name, values):
    try:
        bin_counts, bin_edges = np.histogram(values, bins="auto")
    except ValueError:
        return None
    return tf.Summary.Value(
        tag=name,
        histo=tf.HistogramProto(
            min=np.min(values),
            max=np.max(values),
            num=np.size(values),
            sum=np.sum(values),
            sum_squares=np.sum(np.square(values)),
            bucket_limit=bin_edges[1:],
            bucket=bin_counts,
        ),
    )


_REGRESSION_METRICS = {
    "r2": functools.partial(sklearn.metrics.r2_score, multioutput="variance_weighted"),
    "r2_unweighted": sklearn.metrics.r2_score,
    "explained_variance": functools.partial(
        sklearn.metrics.explained_variance_score, multioutput="variance_weighted"
    ),
    "explained_variance_unweighted": sklearn.metrics.explained_variance_score,
    "mean_squared_error": sklearn.metrics.mean_squared_error,
}


def _regression_evaluation_summaries(
    tag, y_true, y_pred, y_pred_normal_var=None, y_pred_normal_cov=None
):
    """Produce a list of regression evaluation summaries.

    Args:
        tag: The summary tag prefix.
        y_true: The true target values.
            An array of shape `(num_points, num_dimensions)`.
        y_pred: The predicted values.
            An array of shape `(num_points, num_dimensions)`.
        y_pred_normal_var: Optional prediction variances assuming Gaussian
            predictive distributions.
            An array of shape `(num_points, num_dimensions)`.
        y_pred_normal_cov: Optional prediction covariances assuming multivariate
            Gaussian predictive distributions.
            An array of shape `(num_points, num_dimensions, num_dimensions)`.

    Returns:
        A list of tf.Summary.Value protobufs.
    """
    summaries = []

    for metric_name, metric_fn in _REGRESSION_METRICS.items():
        summaries.append(
            _summarize_and_log(tag + "/" + metric_name, metric_fn(y_true, y_pred))
        )

    log_pred_singular_values = None
    log_probabilities = None

    if y_pred_normal_cov is not None:
        log_pred_singular_values = np.log(
            np.linalg.svd(y_pred_normal_cov, compute_uv=False)
        )
        try:
            L = np.linalg.cholesky(y_pred_normal_cov)
        except np.linalg.LinAlgError as e:
            if "Matrix is not positive definite" in str(e):
                logger.warning("Could not calculate log probability:" + str(e))
            else:
                raise
        else:
            a = utils.numpy.batch_solve(L, (y_true - y_pred)[..., None]).squeeze(
                axis=-1
            )
            log_probabilities = (
                -0.5 * np.sum(np.square(a), axis=-1)
                - np.log(2 * np.pi) * a.shape[-1] / 2
                - np.sum(np.log(np.diagonal(L, axis1=-2, axis2=-1)), axis=-1)
            )

    elif y_pred_normal_var is not None:
        log_pred_singular_values = np.log(y_pred_normal_var)

        log_probabilities = scipy.stats.norm.logpdf(
            y_true, loc=y_pred, scale=np.sqrt(y_pred_normal_var)
        ).sum(axis=-1)

    if log_pred_singular_values is not None:
        min_log_pred_singular_values = np.min(log_pred_singular_values, axis=-1)
        max_log_pred_singular_values = np.max(log_pred_singular_values, axis=-1)
        log_pred_condition_numbers = (
            max_log_pred_singular_values - min_log_pred_singular_values
        )
        # TODO: For some reason all the condition numbers are 0?
        if np.all(np.isfinite(log_pred_singular_values)):
            summaries.extend(
                [
                    _summarize_histogram(
                        tag + "/min_log_pred_singular_values",
                        min_log_pred_singular_values,
                    ),
                    _summarize_histogram(
                        tag + "/max_log_pred_singular_values",
                        max_log_pred_singular_values,
                    ),
                    _summarize_histogram(
                        tag + "/log_pred_condition_numbers", log_pred_condition_numbers
                    ),
                ]
            )
        _summarize_and_log(
            tag + "/max_log_pred_condition_number", np.max(log_pred_condition_numbers)
        ),

    if log_probabilities is not None:
        summaries.extend(
            [
                _summarize_histogram(tag + "/log_probabilities", log_probabilities),
                _summarize_and_log(
                    tag + "/mean_log_probability", np.mean(log_probabilities)
                ),
            ]
        )

    return summaries


def _gp_parameter_summaries(tag, gp):
    """Generate summaries of a GP model parameters."""
    gp_params = gp.get_params()
    summaries = [
        _summarize_histogram(tag + "/signal_variance", gp_params.signal_variance),
        _summarize_and_log(
            tag + "/mean_signal_variance", np.mean(gp_params.signal_variance)
        ),
        _summarize_histogram(tag + "/noise_variance", gp_params.noise_variance),
        _summarize_and_log(
            tag + "/mean_noise_variance", np.mean(gp_params.noise_variance)
        ),
        _summarize_histogram(tag + "/length_scale", gp_params.length_scale),
        _summarize_and_log(tag + "/mean_length_scale", np.mean(gp_params.length_scale)),
    ]
    log_singular_values = np.log(np.diagonal(gp_params.gram_L)) / 2
    log_min_singular_value = np.min(log_singular_values)
    log_max_singular_value = np.max(log_singular_values)
    log_condition_number = log_max_singular_value - log_min_singular_value
    summaries.extend(
        [
            _summarize_and_log(tag + "/log_min_singular_value", log_min_singular_value),
            _summarize_and_log(tag + "/log_max_singular_value", log_max_singular_value),
            _summarize_and_log(tag + "/log_condition_number", log_condition_number),
        ]
    )
    return summaries
