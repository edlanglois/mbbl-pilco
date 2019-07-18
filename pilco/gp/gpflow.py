"""Gpflow Gaussian processes."""
import logging

import gpflow
import numpy as np
import scipy.cluster.vq
import tensorflow as tf

from pilco.third_party.gpflow_derived import sgpr

from . import base


class GpflowRBFGaussianProcessRegressor(base.DecoupledKernelGaussianProcessWrapper):
    """Radial Basis Gaussian Process regressor using gpflow."""

    def __init__(
        self,
        min_noise_variance=None,
        shared_kernel=True,
        max_iterations=1000,
        persist_hyperparameters=True,
    ):
        super().__init__(
            model_class=_GpflowRBFGaussianProcessRegressor_SharedKernel,
            shared_kernel=shared_kernel,
            max_iterations=max_iterations,
            persist_hyperparameters=persist_hyperparameters,
            min_noise_variance=min_noise_variance,
        )


class GpflowRBFSparseVariationalGaussianProcessRegressor(
    base.DecoupledKernelGaussianProcessWrapper
):
    """Sparse Radial Basis Gaussian Process regressor using gpflow."""

    def __init__(
        self,
        num_inducing_points,
        min_noise_variance=None,
        shared_kernel=True,
        max_iterations=1000,
        persist_hyperparameters=True,
    ):
        super().__init__(
            model_class=_GpflowRBFSparseVariationalGaussianProcessRegressor_SharedKernel,  # noqa
            shared_kernel=shared_kernel,
            num_inducing_points=num_inducing_points,
            max_iterations=max_iterations,
            min_noise_variance=min_noise_variance,
        )


class _BaseGpflowRBFGaussianProcessRegressor(base.BaseRBFGaussianProcessRegressor):
    """GPflow regression model base class."""

    def __init__(
        self, shared_kernel, max_iterations=1000, persist_hyperparameters=True
    ):
        """Initialize

        Args:
            persist_hyperparameters: If `True`, calls to fit() use the current set of
                hyper-parameters as initial values. Otherwise, starts from scratch every
                time.
            max_iterations: Maximum number of model fitting iterations.
        """
        super().__init__(shared_kernel=True)
        self._model = None
        self.persist_hyperparameters = persist_hyperparameters
        self.max_iterations = max_iterations

    def fit(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)
        if self._model is None or not self.persist_hyperparameters:
            self._model = self._make_model(X, Y)
        else:
            self._model.X = X
            self._model.Y = Y

        # TODO: this is apparently bad practice because it creates more graph objects
        # each time.
        # https://nbviewer.jupyter.org/github/GPflow/GPflow/blob/develop/doc/source/notebooks/tips_and_tricks.ipynb
        gpflow.train.ScipyOptimizer().minimize(self._model, maxiter=self.max_iterations)

    def _make_model(self, X, Y):
        """Create a new GP regression model initialized to the given data."""
        raise NotImplementedError


class _GpflowRBFGaussianProcessRegressor_SharedKernel(
    _BaseGpflowRBFGaussianProcessRegressor
):
    """Gpflow RBF Gaussian Process regressor; all outputs share a kernel."""

    def __init__(self, min_noise_variance=None, **kwargs):
        super().__init__(shared_kernel=True, **kwargs)
        self.min_noise_variance = min_noise_variance

    def _make_model(self, X, Y):
        _, num_input_dimensions = X.shape
        with gpflow.defer_build():
            kernel = gpflow.kernels.RBF(input_dim=num_input_dimensions, ARD=True)
            model = gpflow.models.GPR(X, Y, kern=kernel)
            if self.min_noise_variance is not None:
                model.likelihood.variance.transform = gpflow.transforms.Log1pe(
                    lower=self.min_noise_variance
                )
        model.build()
        return model

    def predict(
        self,
        X,
        return_var=False,
        return_cov=False,
        predictive_noise=False,
        broadcastable=False,
    ):
        del broadcastable  # Unused
        if self._model is None:
            raise ValueError("Cannot call predict() before fit()")

        if return_cov:
            y_mean, y_cov = self._model.predict_f_full_cov(X)
            if predictive_noise:
                diag_idx = np.arange(y_cov.shape[-1])
                y_cov[..., diag_idx, diag_idx] += self._model.likelihood.variance.value
            return y_mean, y_cov

        y_mean, y_var = self._model.predict_f(X)
        if return_var:
            if predictive_noise:
                y_var += self._model.likelihood.variance.value
            return y_mean, y_var
        return y_mean

    def get_params(self):
        return base.RBFGaussianProcessParameters(**_get_gpr_params(self._model))


@gpflow.decors.autoflow()
@gpflow.decors.params_as_tensors
def _get_gpr_params(gpr):
    """Get parameters from a Gpflow Gaussian Process Regressor."""
    K = (
        gpr.kern.K(gpr.X)
        + tf.eye(tf.shape(gpr.X)[0], dtype=gpflow.settings.float_type)
        * gpr.likelihood.variance
    )
    L = tf.cholesky(K)
    alpha = tf.matrix_transpose(tf.cholesky_solve(L, gpr.Y))

    return {
        "inducing_points": gpr.X,
        "coefficients": alpha,
        "target_values": gpr.Y,
        "signal_variance": gpr.kern.variance[None],
        "length_scale": gpr.kern.lengthscales[None, :],
        "noise_variance": gpr.likelihood.variance[None],
        "gram_L": L[None, :, :],
    }


class _GpflowRBFSparseVariationalGaussianProcessRegressor_SharedKernel(
    _BaseGpflowRBFGaussianProcessRegressor
):
    """Gpflow RBF Sparse Variational GP regressor; shared kernel."""

    def __init__(self, num_inducing_points, min_noise_variance=None, **kwargs):
        super().__init__(shared_kernel=True, **kwargs)
        self.num_inducing_points = num_inducing_points
        self.min_noise_variance = min_noise_variance

    def _make_model(self, X, Y):
        _, num_input_dimensions = X.shape

        if len(X) <= self.num_inducing_points:
            initial_inducing_points = np.random.normal(
                size=[self.num_inducing_points, X.shape[-1]]
            )
            initial_inducing_points[: len(X), :] = X
        else:
            initial_inducing_points, _ = scipy.cluster.vq.kmeans(
                X, self.num_inducing_points
            )
        with gpflow.defer_build():
            kernel = gpflow.kernels.RBF(input_dim=num_input_dimensions, ARD=True)
            model = gpflow.models.SGPR(X, Y, kern=kernel, Z=initial_inducing_points)
            if self.min_noise_variance is not None:
                model.likelihood.variance.transform = gpflow.transforms.Log1pe(
                    lower=self.min_noise_variance
                )
        model.build()
        return model

    def predict(
        self,
        X,
        return_var=False,
        return_cov=False,
        predictive_noise=False,
        broadcastable=False,
    ):
        del broadcastable  # Unused

        if return_cov:
            y_mean, y_cov = self._model.predict_f_full_cov(X)
            if predictive_noise:
                diag_idx = np.arange(y_cov.shape[-1])
                y_cov[..., diag_idx, diag_idx] += self._model.likelihood.variance.value
            return y_mean, y_cov

        y_mean, y_var = self._model.predict_f(X)
        if return_var:
            if predictive_noise:
                y_var += self._model.likelihood.variance.value
            return y_mean, y_var
        return y_mean

    # TODO: retrieved values might be stale?
    # https://nbviewer.jupyter.org/github/GPflow/GPflow/blob/develop/doc/source/notebooks/tips_and_tricks.ipynb
    def get_params(self):
        return base.RBFGaussianProcessParameters(
            **sgpr.get_sgpr_parameters(self._model)
        )


def _filter_logdensities_warning(record):
    """Filter out specific useless warnings from gpflow.logdensities."""
    if (
        record.levelname == "WARNING"
        and record.msg == "Shape of x must be 2D at computation."
    ):
        # This warning unavoidable, triggered by a tensor with unknown shape.
        # The event is harmless.
        return 0
    return 1


logging.getLogger("gpflow.logdensities").addFilter(_filter_logdensities_warning)
