"""Unit tests for moment_map/gp.py"""

import numpy as np
import pytest
import scipy
import sklearn.gaussian_process as gp

import pilco.gp.gpflow
import pilco.gp.sklearn
import pilco.moment_map.gp as mmgp
from pilco import moment_map


def _gp_rbf_mean0(in_dim=1, out_dim=1, n=1, var=1):
    inducing_points = np.zeros([n, in_dim])
    # Broadcast along out_dim dimension
    if var > 0:
        gram_L = np.expand_dims(
            np.linalg.cholesky(np.ones([n, n]) + var * np.eye(n)), 0
        )
    else:
        gram_L = np.zeros([n, n])
        gram_L[:, 0] = 1
    coefficients = np.zeros([out_dim, n])
    signal_variance = np.ones([out_dim])
    length_scale = np.ones([out_dim, in_dim])
    return (inducing_points, gram_L, coefficients, signal_variance, length_scale)


@pytest.mark.parametrize(
    "in_dim,out_dim,n", [(1, 1, 1), (3, 1, 1), (1, 4, 1), (1, 1, 5), (3, 4, 5)]
)
def test_predict_from_gaussian_unit_to_const0var0_nocov(in_dim, out_dim, n):
    (
        inducing_points,
        gram_L,
        coefficients,
        signal_variance,
        length_scale,
    ) = _gp_rbf_mean0(in_dim=in_dim, out_dim=out_dim, n=n, var=0)
    gpm = mmgp.GaussianProcessMomentMap(
        inducing_points=inducing_points,
        coefficients=coefficients,
        gram_L=gram_L,
        signal_variance=signal_variance,
        length_scale=length_scale,
    )

    results = gpm(
        mean=np.zeros([in_dim]),
        covariance=np.ones([in_dim, in_dim]),
        return_cov=False,
        return_io_cov=False,
    )

    assert results.output_mean == pytest.approx(np.zeros([out_dim]))
    assert results.output_covariance is None
    assert results.output_input_covariance is None


@pytest.mark.parametrize(
    "in_dim,out_dim,n", [(1, 1, 1), (3, 1, 1), (1, 4, 1), (1, 1, 5), (3, 4, 5)]
)
def test_predict_from_gaussian_delta_to_const0var0(in_dim, out_dim, n):
    var = 1e-10
    inducing_points, gram_L, coefficients, signal_variance, length_scale = _gp_rbf_mean0(
        in_dim=in_dim, out_dim=out_dim, n=n, var=var
    )
    gpm = mmgp.GaussianProcessMomentMap(
        inducing_points=inducing_points,
        coefficients=coefficients,
        gram_L=gram_L,
        signal_variance=signal_variance,
        length_scale=length_scale,
    )

    results = gpm(
        mean=np.zeros([in_dim]),
        covariance=np.zeros([in_dim, in_dim]),
        return_cov=True,
        return_io_cov=True,
    )

    assert results.output_mean == pytest.approx(np.zeros([out_dim]))
    assert results.output_covariance == pytest.approx(var / n * np.eye(out_dim))
    assert results.output_input_covariance == pytest.approx(np.zeros([out_dim, in_dim]))


def test_predict_from_gaussian_delta_to_const0var0_large_lengtscale():
    in_dim = 2
    out_dim = 2
    n = 5
    var = 1e-10
    inducing_points, gram_L, coefficients, signal_variance, length_scale = _gp_rbf_mean0(
        in_dim=in_dim, out_dim=out_dim, n=n, var=var
    )
    length_scale[:, 1] = 1e10
    gpm = mmgp.GaussianProcessMomentMap(
        inducing_points=inducing_points,
        coefficients=coefficients,
        gram_L=gram_L,
        signal_variance=signal_variance,
        length_scale=length_scale,
    )

    results = gpm(
        mean=np.zeros([in_dim]),
        covariance=np.zeros([in_dim, in_dim]),
        return_cov=True,
        return_io_cov=True,
    )

    assert results.output_mean == pytest.approx(np.zeros([out_dim]))
    assert results.output_covariance == pytest.approx(var / n * np.eye(out_dim))
    assert results.output_input_covariance == pytest.approx(np.zeros([out_dim, in_dim]))


@pytest.mark.parametrize(
    "in_dim,out_dim,n,batch_shape",
    [(3, 4, 5, (1,)), (1, 1, 1, (5,)), (3, 4, 5, (2, 2))],
)
def test_predict_from_gaussian_batch_delta_to_const0var0(
    in_dim, out_dim, n, batch_shape
):
    var = 1e-10
    inducing_points, gram_L, coefficients, signal_variance, length_scale = _gp_rbf_mean0(
        in_dim=in_dim, out_dim=out_dim, n=n, var=var
    )
    gpm = mmgp.GaussianProcessMomentMap(
        inducing_points=inducing_points,
        coefficients=coefficients,
        gram_L=gram_L,
        signal_variance=signal_variance,
        length_scale=length_scale,
    )
    results = gpm(
        mean=np.zeros(batch_shape + (in_dim,)),
        covariance=np.zeros(batch_shape + (in_dim, in_dim)),
        return_cov=True,
        return_io_cov=True,
    )

    assert results.output_mean == pytest.approx(np.zeros(batch_shape + (out_dim,)))
    assert results.output_covariance == pytest.approx(
        np.tile(var / n * np.eye(out_dim), batch_shape + (1, 1))
    )
    assert results.output_input_covariance == pytest.approx(
        np.zeros(batch_shape + (out_dim, in_dim))
    )


@pytest.mark.parametrize(
    "in_dim,out_dim,n,batch_shape",
    [
        (1, 1, 1, ()),
        (3, 1, 1, ()),
        (1, 4, 1, ()),
        (1, 1, 5, ()),
        (1, 1, 1, (2,)),
        (1, 1, 1, (2, 3)),
        (3, 4, 5, (2,)),
    ],
)
def test_predict_to_const0var0(in_dim, out_dim, n, batch_shape):
    var = 1e-10
    inducing_points, gram_L, coefficients, signal_variance, length_scale = _gp_rbf_mean0(
        in_dim=in_dim, out_dim=out_dim, n=n, var=var
    )
    gpm = mmgp.GaussianProcessMomentMap(
        inducing_points=inducing_points,
        coefficients=coefficients,
        gram_L=gram_L,
        signal_variance=signal_variance,
        length_scale=length_scale,
    )
    results = gpm(
        mean=np.zeros(batch_shape + (in_dim,)), return_cov=True, return_io_cov=True
    )

    assert results.output_mean == pytest.approx(np.zeros(batch_shape + (out_dim,)))
    assert results.output_covariance == pytest.approx(
        np.tile(var / n * np.eye(out_dim), batch_shape + (1, 1))
    )
    assert np.allclose(results.output_input_covariance, 0)


@pytest.mark.parametrize(
    "in_dim,out_dim,n,var,covariance_none,shared_kernel",
    [
        (1, 1, 1, 3, False, False),
        (1, 1, 1, 0.5, False, False),
        (3, 1, 1, 3, False, False),
        (1, 4, 1, 3, False, False),
        (1, 1, 5, 3, False, False),
        (1, 1, 1, 3, True, False),
        (3, 4, 5, 3, False, False),
        (3, 4, 5, 3, True, False),
        (1, 1, 1, 3, False, True),
        (3, 4, 5, 3, True, True),
    ],
)
def test_predict_from_gaussian_delta_vs_sklearn(
    in_dim, out_dim, n, var, covariance_none, shared_kernel
):
    """Check agreement with GaussianProcessRegressor for deterministic input.
    """
    rand = np.random.RandomState(seed=2)

    input_mean = rand.normal(size=[in_dim])
    if covariance_none:
        input_covariance = np.zeros([in_dim, in_dim])
    else:
        input_covariance = None

    X = rand.normal(size=[n, in_dim])
    Y = X @ rand.normal(size=[in_dim, out_dim]) + 0.1 * rand.normal(size=[n, 1])

    gp_regressor = pilco.gp.sklearn.SklearnRBFGaussianProcessRegressor(
        shared_kernel=shared_kernel
    )
    gp_regressor.fit(X, Y)

    gp_params = gp_regressor.get_params()
    gpm = mmgp.GaussianProcessMomentMap.from_params(gp_params)
    results = gpm(
        mean=input_mean,
        covariance=input_covariance,
        return_cov=True,
        return_io_cov=True,
    )

    pred_y_mean, pred_y_var = gp_regressor.predict(input_mean[None, :], return_var=True)
    pred_y_mean = np.squeeze(pred_y_mean, 0)
    pred_y_var = np.squeeze(pred_y_var, 0)
    pred_y_cov = np.diag(pred_y_var)

    assert results.output_mean == pytest.approx(pred_y_mean)
    assert results.output_covariance == pytest.approx(pred_y_cov, abs=1e-8)
    assert results.output_input_covariance == pytest.approx(np.zeros([out_dim, in_dim]))


@pytest.mark.slow
@pytest.mark.parametrize(
    "in_dim,out_dim,n,var,covariance_none,shared_kernel",
    [
        (1, 1, 1, 3, False, False),
        (1, 1, 1, 3, True, False),
        (3, 4, 5, 3, False, True),
        (3, 4, 5, 3, True, True),
    ],
)
def test_predict_from_gaussian_delta_vs_gpflow_SGPR(
    in_dim, out_dim, n, var, covariance_none, shared_kernel
):
    """Check agreement with gpflow.SGPR for deterministic input.
    """
    rand = np.random.RandomState(seed=2)

    input_mean = rand.normal(size=[in_dim])
    if covariance_none:
        input_covariance = np.zeros([in_dim, in_dim])
    else:
        input_covariance = None

    X = rand.normal(size=[n, in_dim])
    Y = X @ rand.normal(size=[in_dim, out_dim]) + 0.1 * rand.normal(size=[n, 1])

    gp_regressor = pilco.gp.gpflow.GpflowRBFSparseVariationalGaussianProcessRegressor(
        num_inducing_points=2, shared_kernel=shared_kernel
    )
    gp_regressor.fit(X, Y)

    gp_params = gp_regressor.get_params()
    gpm = mmgp.GaussianProcessMomentMap.from_params(gp_params)
    results = gpm(
        mean=input_mean,
        covariance=input_covariance,
        return_cov=True,
        return_io_cov=True,
    )

    pred_y_mean, pred_y_var = gp_regressor.predict(input_mean[None, :], return_var=True)
    pred_y_mean = np.squeeze(pred_y_mean, 0)
    pred_y_var = np.squeeze(pred_y_var, 0)
    pred_y_cov = np.diag(pred_y_var)

    assert results.output_mean == pytest.approx(pred_y_mean)
    assert results.output_covariance == pytest.approx(pred_y_cov, abs=1e-8)
    assert results.output_input_covariance == pytest.approx(np.zeros([out_dim, in_dim]))


@pytest.mark.parametrize(
    "in_dim,out_dim,n,var,shared_kernel",
    [
        (1, 1, 1, 3, False),
        (1, 1, 1, 0.5, False),
        (3, 1, 1, 3, False),
        (1, 4, 1, 3, False),
        (1, 1, 5, 3, False),
        (3, 4, 5, 3, False),
        (1, 1, 1, 3, True),
        (3, 4, 5, 3, True),
    ],
)
def test_predict_vs_sklearn(in_dim, out_dim, n, var, shared_kernel):
    """Check agreement with GaussianProcessRegressor for deterministic input.
    """
    rand = np.random.RandomState(seed=3)

    input_ = rand.normal(size=[in_dim])

    X = rand.normal(size=[n, in_dim])
    Y = X @ rand.normal(size=[in_dim, out_dim]) + 0.1 * rand.normal(size=[n, 1])

    gp_regressor = pilco.gp.sklearn.SklearnRBFGaussianProcessRegressor(
        shared_kernel=shared_kernel
    )
    gp_regressor.fit(X, Y)

    gp_params = gp_regressor.get_params()
    gpm = mmgp.GaussianProcessMomentMap.from_params(gp_params)
    results = gpm(mean=input_, return_cov=True, return_io_cov=False)

    pred_y_mean, pred_y_var = gp_regressor.predict(input_[None, :], return_var=True)
    pred_y_mean = np.squeeze(pred_y_mean, 0)
    pred_y_var = np.squeeze(pred_y_var, 0)
    pred_y_cov = np.diag(pred_y_var)

    assert results.output_mean == pytest.approx(pred_y_mean)
    assert results.output_covariance == pytest.approx(pred_y_cov, abs=1e-8)
    assert results.output_input_covariance is None


@pytest.mark.slow
@pytest.mark.parametrize(
    "in_dim,out_dim,n,var,shared_kernel", [(1, 1, 1, 3, False), (3, 4, 5, 3, True)]
)
def test_predict_vs_gpflow_SGPR(in_dim, out_dim, n, var, shared_kernel):
    """Check agreement with GaussianProcessRegressor for deterministic input.
    """
    rand = np.random.RandomState(seed=3)

    input_ = rand.normal(size=[in_dim])

    X = rand.normal(size=[n, in_dim])
    Y = X @ rand.normal(size=[in_dim, out_dim]) + 0.1 * rand.normal(size=[n, 1])

    gp_regressor = pilco.gp.gpflow.GpflowRBFSparseVariationalGaussianProcessRegressor(
        num_inducing_points=2, shared_kernel=shared_kernel
    )
    gp_regressor.fit(X, Y)

    gp_params = gp_regressor.get_params()
    gpm = mmgp.GaussianProcessMomentMap.from_params(gp_params)
    results = gpm(mean=input_, return_cov=True, return_io_cov=False)

    pred_y_mean, pred_y_var = gp_regressor.predict(input_[None, :], return_var=True)
    pred_y_mean = np.squeeze(pred_y_mean, 0)
    pred_y_var = np.squeeze(pred_y_var, 0)
    pred_y_cov = np.diag(pred_y_var)

    assert results.output_mean == pytest.approx(pred_y_mean)
    assert results.output_covariance == pytest.approx(pred_y_cov, abs=1e-8)
    assert results.output_input_covariance is None


def random_positive_definite(rand, n):
    a = rand.normal(size=[n, n])
    return a @ a.T


def test_predict_from_gaussian_normal_vs_numerical_1d():
    """Compare predictions against numerical integration in 1D."""
    x_data = np.array([-0.8, -0.5, -0.25, 0.25, 0.7])
    y_data = np.array([0.4, 0, 0.3, -0.1, 0])

    kernel = pilco.gp.sklearn.anisotropic_rbf_kernel(
        1,
        length_scale=1.0,
        length_scale_bounds=(0.25, 1e1),
        noise_variance=1e-5,
        noise_variance_bounds=(1e-15, 1e-5),
    )
    gpr = gp.GaussianProcessRegressor(kernel)
    gpr.fit(x_data[:, None], y_data)

    x_grid, dx = np.linspace(-1, 1.5, 1000, retstep=True)
    # y conditional on x
    ycx_mean, ycx_std = gpr.predict(x_grid[:, None], return_std=True)
    # Remove noise variance from std - not included by
    # GaussianProcessMomentMap
    ycx_std = np.sqrt(np.square(ycx_std) - gpr.kernel_.k2.noise_level)

    input_mean = 0.2
    input_std = 0.3
    x_pdf = scipy.stats.norm.pdf(x_grid, loc=input_mean, scale=input_std)

    # Ensure that most of the probability mass is captured
    x_sum = sum(x_pdf) * dx
    if not 0.999 <= x_sum <= 1.0:
        raise ValueError(f"Bad domain; invalid distribution support {x_sum}")

    y_grid, dy = np.linspace(-1, 1, 500, retstep=True)

    # joint probability. Order: [Y, X]
    yx_pdf = (
        np.exp(-np.square((y_grid[:, None] - ycx_mean) / ycx_std) / 2)
        / (np.sqrt(2 * np.pi) * ycx_std)
        * x_pdf
    )
    y_pdf = np.sum(yx_pdf * dx, axis=-1)

    y_sum = sum(y_pdf) * dy
    if not 0.999 <= y_sum <= 1.0:
        raise ValueError(f"Bad domain; invalid distribution support {y_sum}")

    y_mean = np.sum(y_pdf * y_grid) * dy
    y_var = np.sum(y_pdf * np.square(y_grid - y_mean)) * dy
    xy_cov = (
        np.sum((y_grid - y_mean)[:, None] * (x_grid - input_mean) * yx_pdf) * dx * dy
    )

    gp_params = pilco.gp.sklearn.rbf_gaussian_process_regressor_parameters(gpr)
    gpm = mmgp.GaussianProcessMomentMap.from_params(gp_params)
    y_pred = gpm(
        mean=input_mean, covariance=input_std ** 2, return_cov=True, return_io_cov=True
    )

    assert y_pred.output_mean == pytest.approx(np.reshape(y_mean, [1]), rel=0.001)
    assert y_pred.output_covariance == pytest.approx(
        np.reshape(y_var, [1, 1]), rel=0.001
    )
    assert y_pred.output_input_covariance == pytest.approx(
        np.reshape(xy_cov, [1, 1]), rel=0.001
    )


def _discard_to_threshold(x, delta):
    """Indices into `x` such that `sum(x) - sum(x[result]) < delta`.

    Args:
        x: A 1d array-like.
        delta: Maximum sum of elements in x[~result].
    """
    sort_idx = np.argsort(x)
    unsort_idx = np.argsort(sort_idx)
    sorted_keep = np.cumsum(x[sort_idx]) >= delta
    return sorted_keep[unsort_idx]


def test_predict_from_gaussian_normal_vs_numerical_2d():
    """Compare predictions against numerical integration in 2D."""
    rand = np.random.RandomState(seed=1)

    in_dim = 2
    out_dim = 2
    x_data = rand.normal(scale=0.2, size=[6, in_dim])
    y_data = rand.normal(scale=0.2, size=[6, out_dim])

    gp_regressor = pilco.gp.sklearn.SklearnRBFGaussianProcessRegressor(
        signal_variance_bounds=(0.1, 10), shared_kernel=False
    )
    gp_regressor.fit(x_data, y_data)

    gp_params = gp_regressor.get_params()

    x_grid_1d, dx = np.linspace(-1, 1, 100, retstep=True)
    x_grid_2d = np.stack(
        np.meshgrid(*[x_grid_1d] * in_dim, indexing="ij"), axis=-1
    ).reshape([-1, in_dim])
    dxx = dx * dx
    del x_grid_1d

    assert in_dim == 2
    input_mean = np.array([0.1, -0.2])
    # std: (0.2, 0.1); correlation: 0.8
    input_cov = np.array([[0.04, 0.016], [0.016, 0.01]])
    x_pdf = scipy.stats.multivariate_normal.pdf(
        x_grid_2d, mean=input_mean, cov=input_cov
    )

    # Exclude low-probability samples for efficiency
    # Throw away the set of samples with smallest density and
    # whose total density is <= 0.001
    x_keep = _discard_to_threshold(x_pdf, 0.001)
    x_grid_2d = x_grid_2d[x_keep, :]
    x_pdf = x_pdf[x_keep]

    # Ensure that most of the probability mass is captured
    x_sum = np.sum(x_pdf) * dxx
    if not 0.998 <= x_sum <= 1.002:
        raise ValueError(f"Bad domain; invalid distribution support {x_sum}")

    # y conditional on x
    ycx_mean, ycx_var = gp_regressor.predict(x_grid_2d, return_var=True)
    ycx_std = np.sqrt(ycx_var)

    # Sampling grid for distribution over y
    y_grid_1d, dy = np.linspace(-1, 1, 100, retstep=True)
    y_grid_2d = np.stack(
        np.meshgrid(*[y_grid_1d] * out_dim, indexing="ij"), axis=-1
    ).reshape([-1, out_dim])
    dyy = dy * dy
    del y_grid_1d

    # joint probability. Order: [Y, X]
    yx_pdf = (
        np.exp(
            -0.5
            * np.sum(np.square((y_grid_2d[:, None, :] - ycx_mean) / ycx_std), axis=-1)
        )
        / ((2 * np.pi) ** (out_dim / 2) * np.prod(ycx_std, axis=-1))
        * x_pdf
    )
    y_pdf = np.sum(yx_pdf * dxx, axis=-1)

    # Exclude low-probability samples for efficiency
    y_keep = _discard_to_threshold(y_pdf, 0.001)
    y_grid_2d = y_grid_2d[y_keep, :]
    yx_pdf = yx_pdf[y_keep, :]
    y_pdf = y_pdf[y_keep]

    y_sum = np.sum(y_pdf) * dyy
    if not 0.997 <= y_sum <= 1.003:
        raise ValueError(f"Bad domain; invalid distribution support {y_sum}")

    y_mean = np.sum(y_pdf[:, None] * y_grid_2d, axis=0) * dyy
    y_cov = np.cov(m=y_grid_2d, rowvar=False, ddof=0, aweights=y_pdf)

    xy_cov = (
        np.sum(
            (y_grid_2d - y_mean)[:, None, :, None]
            * (x_grid_2d - input_mean)[None, :, None, :]
            * yx_pdf[:, :, None, None],
            axis=(0, 1),
        )
        * dxx
        * dyy
    )

    gpm = mmgp.GaussianProcessMomentMap.from_params(gp_params)
    y_pred = gpm(
        mean=input_mean, covariance=input_cov, return_cov=True, return_io_cov=True
    )

    assert y_pred.output_mean == pytest.approx(y_mean, rel=0.001, abs=0.001)
    assert y_pred.output_covariance == pytest.approx(y_cov, rel=0.001, abs=0.001)
    assert y_pred.output_input_covariance == pytest.approx(xy_cov, rel=0.001, abs=0.001)


def test_index_gp_vs_gp():
    """Test a GP(index) moment map vs a GP moment map with high length-scale."""

    gp_mean = np.array([2.0, 2.0])
    high_length_scale_gp = mmgp.DeterministicGaussianProcessMomentMap(
        inducing_points=gp_mean,
        coefficients=np.ones((1, 1)),
        signal_variance=1,
        length_scale=np.array([1e6, 1.0]),
    )

    index_gp = mmgp.DeterministicGaussianProcessMomentMap(
        inducing_points=gp_mean[..., -1:],
        coefficients=np.ones((1, 1)),
        signal_variance=1,
        length_scale=np.ones((1,)),
    ).compose(moment_map.core.IndexMomentMap(1))

    rand = np.random.RandomState(1)
    mean = gp_mean + rand.randn(2)
    covariance = np.eye(2)

    length_scale_pred = high_length_scale_gp(
        mean=mean, covariance=covariance, return_cov=True, return_io_cov_inv_in_cov=True
    )
    index_pred = index_gp(
        mean=mean, covariance=covariance, return_cov=True, return_io_cov_inv_in_cov=True
    )

    assert index_pred.output_mean == pytest.approx(length_scale_pred.output_mean)
    assert index_pred.output_covariance == pytest.approx(
        length_scale_pred.output_covariance
    )
    assert index_pred.output_input_covariance == pytest.approx(
        length_scale_pred.output_input_covariance, abs=1e-8
    )
