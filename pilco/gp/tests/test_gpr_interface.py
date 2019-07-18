"""Test Gaussian Process regressor implementations against the interface."""
import functools

import numpy as np
import pytest

import pilco.gp
import pilco.gp.gpflow
import pilco.gp.sklearn
from pilco import utils

gp_regressor_classes = {
    "SklearnRBFGaussianProcessRegressor": pilco.gp.sklearn.SklearnRBFGaussianProcessRegressor,
    "GpflowRBFGaussianProcessRegressor": pilco.gp.gpflow.GpflowRBFGaussianProcessRegressor,
    "GpflowRBFSparseVariationalGaussianProcessRegressor": functools.partial(
        pilco.gp.gpflow.GpflowRBFSparseVariationalGaussianProcessRegressor,
        num_inducing_points=10,
    ),
}


@pytest.fixture(params=gp_regressor_classes.values(), ids=gp_regressor_classes.keys())
def gpr_class(request):
    return request.param


def _get_dataset(num_input_dimensions, num_output_dimensions, num_points):
    rand = np.random.RandomState(1)
    X = rand.standard_normal([num_points, num_input_dimensions])
    M = rand.standard_normal([num_input_dimensions, num_output_dimensions])
    Y = X @ M + 0.01 * rand.standard_normal([num_points, num_output_dimensions])
    return X, Y


@pytest.mark.parametrize("in_dim,out_dim", [(1, 1), (3, 1), (1, 4), (3, 4)])
def test_gp_regressor_predict_trainset_close(gpr_class, in_dim, out_dim):
    n = 10
    X, Y = _get_dataset(in_dim, out_dim, n)
    gpr = gpr_class()
    gpr.fit(X, Y)
    Y_pred = gpr.predict(X)
    assert Y_pred.shape == (n, out_dim)
    assert gpr.predict(X) == pytest.approx(Y, abs=0.1)


@pytest.mark.slow
@pytest.mark.parametrize("in_dim,out_dim", [(1, 1), (3, 1), (1, 4), (3, 4)])
@pytest.mark.parametrize("shared_kernel", [(False,), (True,)])
def test_gp_regressor_predict_var(gpr_class, in_dim, out_dim, shared_kernel):
    n = 10
    X, Y = _get_dataset(in_dim, out_dim, n)
    gpr = gpr_class(shared_kernel=shared_kernel)
    gpr.fit(X, Y)
    Y_pred, Y_var = gpr.predict(X, return_var=True)
    assert Y_pred.shape == (n, out_dim)
    assert Y_var.shape == (n, out_dim)
    assert np.all(Y_var >= 0)
    if shared_kernel and out_dim > 1:
        assert np.diff(Y_var, axis=-1) == pytest.approx(np.zeros([n, out_dim - 1]))


@pytest.mark.slow
@pytest.mark.parametrize("in_dim,out_dim", [(1, 1), (3, 1), (1, 4), (3, 4)])
@pytest.mark.parametrize("shared_kernel", [(False,), (True,)])
def test_gp_regressor_predict_cov(gpr_class, in_dim, out_dim, shared_kernel):
    n = 10
    X, Y = _get_dataset(in_dim, out_dim, n)
    gpr = gpr_class(shared_kernel=shared_kernel)
    gpr.fit(X, Y)
    Y_pred, Y_cov = gpr.predict(X, return_cov=True)
    assert Y_pred.shape == (n, out_dim)
    assert Y_cov.shape == (out_dim, n, n)
    assert np.swapaxes(Y_cov, -2, -1) == pytest.approx(Y_cov, abs=1e-10)
    assert np.all(np.linalg.eigvalsh(Y_cov) > -1e-10)
    if shared_kernel and out_dim > 1:
        assert np.diff(Y_cov, axis=0) == pytest.approx(np.zeros([out_dim - 1, n, n]))


@pytest.mark.slow
@pytest.mark.parametrize("in_dim,out_dim", [(1, 1), (3, 1), (1, 4), (3, 4)])
@pytest.mark.parametrize("shared_kernel", [(False,), (True,)])
def test_gp_regressor_get_params(gpr_class, in_dim, out_dim, shared_kernel):
    n = 10
    X, Y = _get_dataset(in_dim, out_dim, n)
    gpr = gpr_class(shared_kernel=shared_kernel)
    gpr.fit(X, Y)

    params = gpr.get_params()
    assert params.inducing_points.shape == (n, in_dim)
    assert params.coefficients.shape == (out_dim, n)
    kernel_out_dim = 1 if shared_kernel else out_dim
    assert params.signal_variance.shape == (kernel_out_dim,)
    assert params.length_scale.shape == (kernel_out_dim, in_dim)
    assert params.noise_variance.shape == (kernel_out_dim,)
    if params.target_values is not None:
        assert params.target_values.shape == (n, out_dim)
    if params.gram_L is not None:
        assert params.gram_L.shape == (kernel_out_dim, n, n)
    if params.B_L is not None:
        assert params.B_L.shape == (kernel_out_dim, n, n)


@pytest.mark.slow
@pytest.mark.parametrize("in_dim,out_dim", [(1, 1), (3, 1), (1, 4), (3, 4)])
@pytest.mark.parametrize("shared_kernel", [(False,), (True,)])
def test_gp_regressor_get_params_verify_predict(
    gpr_class, in_dim, out_dim, shared_kernel
):
    n = 10
    X, Y = _get_dataset(in_dim, out_dim, n)
    gpr = gpr_class(shared_kernel=shared_kernel)
    gpr.fit(X, Y)

    params = gpr.get_params()

    rand = np.random.RandomState(2)
    m = 6
    X2 = rand.standard_normal([m, in_dim])

    # [M, OUT_DIM, N]
    Ksu = params.signal_variance[None, :, None] * np.exp(
        -0.5
        * np.sum(
            np.square(
                (X2[:, None, None, :] - params.inducing_points[None, :, :])
                / params.length_scale[:, None, :]
            ),
            axis=-1,
        )
    )
    Y2_param_mean = np.einsum("...i,...i->...", Ksu, params.coefficients)

    Y2_pred_mean, Y2_pred_cov = gpr.predict(X2, return_cov=True)
    assert Y2_param_mean == pytest.approx(Y2_pred_mean)

    # [OUT_DIM, M, M]
    Kss = params.signal_variance[:, None, None] * np.exp(
        -0.5
        * np.sum(
            np.square(
                (X2[:, None, :] - X2[None, :, :])
                / params.length_scale[:, None, None, :]
            ),
            axis=-1,
        )
    )

    # [OUT_DIM, N, M]
    Kus = np.transpose(Ksu, [1, 2, 0])
    # [OUT_DIM, N, M]
    v = utils.numpy.batch_solve(params.gram_L, Kus, triangular=True, lower=True)

    # [OUT_DIM, M, M]
    Y2_param_cov = Kss - np.transpose(v, [0, 2, 1]) @ v
    if params.B_L is not None:
        w = utils.numpy.batch_solve(params.B_L, v, triangular=True, lower=True)
        Y2_param_cov += np.transpose(w, [0, 2, 1]) @ w

    if shared_kernel:
        Y2_param_cov = np.broadcast_to(Y2_param_cov, [out_dim, m, m])

    assert Y2_param_cov == pytest.approx(Y2_pred_cov, abs=1e-10)

    Y2_pred_mean_2, Y2_pred_cov_noise = gpr.predict(
        X2, return_cov=True, predictive_noise=True
    )
    assert Y2_param_mean == pytest.approx(Y2_pred_mean_2)
    Y2_param_cov_noise = Y2_param_cov + np.eye(m) * params.noise_variance[:, None, None]
    assert Y2_param_cov_noise == pytest.approx(Y2_pred_cov_noise, abs=1e-10)
