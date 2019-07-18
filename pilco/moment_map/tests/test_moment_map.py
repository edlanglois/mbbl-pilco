"""Moment map unit tests."""
import functools

import numpy as np
import pytest
import tensorflow as tf

from pilco import moment_map


def composed_multiply_2_sub_1_moment_map(**kwargs):
    return moment_map.ComposedMomentMap(
        inner=moment_map.LinearMomentMap(scale=2, **kwargs),
        outer=moment_map.LinearMomentMap(offset=-1, **kwargs),
        **kwargs,
    )


# Try composition with changing dimensions
def composed_sin_index_moment_map(**kwargs):
    return moment_map.ComposedMomentMap(
        inner=moment_map.IndexMomentMap(0, **kwargs),
        outer=moment_map.SinMomentMap(**kwargs),
        **kwargs,
    )


def add_x1_2x2_moment_map(**kwargs):
    return moment_map.AddUncorrelatedMomentMap(
        [
            moment_map.IndexMomentMap(0, **kwargs),
            moment_map.LinearMomentMap(scale=2, **kwargs).compose(
                moment_map.IndexMomentMap(1, **kwargs)
            ),
        ],
        **kwargs,
    )


# Moment maps that work for any input size
# (moment_map_cls, the function they implement)
DETERMINISTIC_UNSIZED_MOMENT_MAPS = [
    (moment_map.SinMomentMap, np.sin),
    (functools.partial(moment_map.LinearMomentMap, scale=0.5), lambda x: x * 0.5),
    (functools.partial(moment_map.LinearMomentMap, offset=3), lambda x: x + 3),
    (
        functools.partial(moment_map.LinearMomentMap, scale=2, offset=3),
        lambda x: x * 2 + 3,
    ),
    (composed_multiply_2_sub_1_moment_map, lambda x: 2 * x - 1),
    (functools.partial(moment_map.IndexMomentMap, index=0), lambda x: x[..., 0:1]),
    (composed_sin_index_moment_map, lambda x: np.sin(x[..., 0:1])),
    (
        moment_map.SumSquaredMomentMap,
        lambda x: np.sum(np.square(x), axis=-1, keepdims=True),
    ),
]

DETERMINISTIC_SIZED_MOMENT_MAPS = [
    (moment_map.AbsMomentMap, 1, np.abs),
    (
        functools.partial(moment_map.LinearMomentMap, scale=[1.5], offset=[2.0]),
        1,
        lambda x: x * np.array([1.5]) + np.array([2.0]),
    ),
    (
        functools.partial(moment_map.LinearMomentMap, scale=[1.5, 2], offset=[-2, -3]),
        2,
        lambda x: x * np.array([1.5, 2]) + np.array([-2.0, -3.0]),
    ),
    (
        functools.partial(moment_map.LinearMomentMap, scale=[1.5, 2]),
        2,
        lambda x: x * np.array([1.5, 2]),
    ),
    (
        functools.partial(moment_map.LinearMomentMap, offset=[-2, -3]),
        2,
        lambda x: x + np.array([-2.0, -3.0]),
    ),
    (add_x1_2x2_moment_map, 2, lambda x: x[..., 0:1] + 2 * x[..., 1:2]),
    (
        functools.partial(moment_map.ElementProductMomentMap, i=0, j=1),
        2,
        lambda x: x[..., 0:1] * x[..., 1:2],
    ),
    *[
        (mm, size, fn)
        for (mm, fn) in DETERMINISTIC_UNSIZED_MOMENT_MAPS
        for size in (1, 2)
    ],
]


@pytest.fixture(params=DETERMINISTIC_SIZED_MOMENT_MAPS)
def deterministic_sized_moment_map(request):
    mm_cls, in_size, fn = request.param
    return mm_cls(assert_valid=True), in_size, fn


@pytest.fixture(params=DETERMINISTIC_SIZED_MOMENT_MAPS)
def deterministic_sized_moment_map_cls(request):
    mm_cls, in_size, fn = request.param
    return mm_cls, in_size, fn


@pytest.fixture(params=[np.float32, np.float64])
def dtype(request):
    return request.param


def test_moment_map_deterministic(deterministic_sized_moment_map):
    mm, in_size, fn = deterministic_sized_moment_map
    x = np.ones(in_size)
    result = mm(
        x, return_cov=False, return_io_cov=False, return_io_cov_inv_in_cov=False
    )
    assert result.output_mean == pytest.approx(fn(x))
    assert result.output_covariance is None
    assert result.output_input_covariance is None
    assert result.io_cov_inv_in_cov is None


def test_moment_map_deterministic_covariances(deterministic_sized_moment_map):
    mm, in_size, fn = deterministic_sized_moment_map
    x = np.ones(in_size)
    result = mm(x, return_cov=True, return_io_cov=True, return_io_cov_inv_in_cov=True)
    out_size = len(result.output_mean)

    assert result.output_mean == pytest.approx(fn(x))
    assert result.output_covariance == pytest.approx(np.zeros((out_size, out_size)))
    assert result.output_input_covariance == pytest.approx(
        np.zeros((out_size, in_size))
    )
    assert result.io_cov_inv_in_cov.shape == (out_size, in_size)


def test_moment_map_numerical_0cov_covariances(deterministic_sized_moment_map):
    """Test using a covariance matrix of 0s instead of None"""
    mm, in_size, fn = deterministic_sized_moment_map
    x = np.ones(in_size)
    cov = np.zeros((in_size, in_size))
    result = mm(
        x, cov, return_cov=True, return_io_cov=True, return_io_cov_inv_in_cov=True
    )
    out_size = len(result.output_mean)

    assert result.output_mean == pytest.approx(fn(x))
    assert result.output_covariance == pytest.approx(np.zeros((out_size, out_size)))
    assert result.output_input_covariance == pytest.approx(
        np.zeros((out_size, in_size))
    )
    assert result.io_cov_inv_in_cov.shape == (out_size, in_size)


def array_in_range(x, lower_bound, upper_bound):
    return np.all(lower_bound <= x) and np.all(x <= upper_bound)


def test_moment_map_empirical(deterministic_sized_moment_map):
    mm, in_size, fn = deterministic_sized_moment_map

    rand = np.random.RandomState(seed=1)
    in_mean = rand.uniform(-1, 1, size=in_size)
    A = rand.normal(size=(in_size, in_size), scale=np.sqrt(in_size))
    in_covariance = A @ A.T

    result = mm(in_mean, covariance=in_covariance, return_io_cov_inv_in_cov=True)

    # Empirical statistics
    num_samples = 200_000
    X = rand.multivariate_normal(in_mean, in_covariance, size=num_samples)
    Y = fn(X)
    # Use bootstrapping to get intervals
    num_bootstraps = 100
    quantiles = (0.01, 0.99)
    bootstrap_indices = np.random.choice(
        num_samples, size=(num_bootstraps, num_samples)
    )
    Y_boot = Y[bootstrap_indices, :]

    Y_means = np.mean(Y_boot, axis=1)
    Y_mean_lower_bound, Y_mean_upper_bound = np.quantile(Y_means, quantiles, axis=0)

    assert array_in_range(result.output_mean, Y_mean_lower_bound, Y_mean_upper_bound)

    Y_boot_centered = Y_boot - Y_means[:, None, :]
    Y_boot_centered_t = np.transpose(Y_boot_centered, (0, 2, 1))
    Y_covariances = Y_boot_centered_t @ Y_boot_centered / num_samples
    Y_cov_lower_bound, Y_cov_upper_bound = np.quantile(Y_covariances, quantiles, axis=0)

    if not isinstance(mm, moment_map.AddUncorrelatedMomentMap):
        # TODO: Better handling of this exception
        # Skip AddUncorrelatedMomentMap because it is adding correlated values in this
        # case and is therefore incorrect.
        assert array_in_range(
            result.output_covariance, Y_cov_lower_bound, Y_cov_upper_bound
        )

    X_boot = X[bootstrap_indices, :]
    X_boot_centered = X_boot - np.mean(X_boot, axis=1, keepdims=True)
    YX_covariances = Y_boot_centered_t @ X_boot_centered / num_samples
    YX_cov_lower_bound, YX_cov_upper_bound = np.quantile(
        YX_covariances, quantiles, axis=0
    )

    assert array_in_range(
        result.output_input_covariance, YX_cov_lower_bound, YX_cov_upper_bound
    )
    assert array_in_range(
        result.io_cov_inv_in_cov @ in_covariance, YX_cov_lower_bound, YX_cov_upper_bound
    )


def test_moment_map_tensorflow_construct(deterministic_sized_moment_map_cls, dtype):
    mm_cls, in_size, _ = deterministic_sized_moment_map_cls
    mm = mm_cls(backend="tensorflow", assert_valid=True, dtype=dtype)

    in_mean = tf.zeros(in_size, dtype=dtype)
    in_covariance = tf.eye(in_size, dtype=dtype)
    result1 = mm(in_mean)
    assert result1.output_mean.dtype == dtype
    assert result1.output_covariance.dtype == dtype
    assert result1.output_input_covariance.dtype == dtype

    result2 = mm(in_mean, in_covariance)
    assert result2.output_mean.dtype == dtype
    assert result2.output_covariance.dtype == dtype
    assert result2.output_input_covariance.dtype == dtype
