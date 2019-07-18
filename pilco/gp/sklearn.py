"""Scikit-learn Gaussian processes."""
import numpy as np
import sklearn.gaussian_process as gp
import sklearn.gaussian_process.kernels as gp_kernels

from . import base


class SklearnRBFGaussianProcessRegressor(base.DecoupledKernelGaussianProcessWrapper):
    """Radial Basis Gaussian Process regressor using Scikit learn."""

    def __init__(
        self,
        signal_variance_bounds=(1e-5, 1e5),
        length_scale_bounds=(1e-5, 1e5),
        noise_variance_bounds=(1e-5, 1e5),
        shared_kernel=True,
    ):
        super().__init__(
            model_class=_SklearnRBFGaussianProcessRegressor_SharedKernel,
            signal_variance_bounds=signal_variance_bounds,
            length_scale_bounds=length_scale_bounds,
            noise_variance_bounds=noise_variance_bounds,
            shared_kernel=shared_kernel,
        )


class _SklearnRBFGaussianProcessRegressor_SharedKernel(
    base.BaseRBFGaussianProcessRegressor
):
    """Sklearn RBF Gaussian Process regressor; all outputs share a kernel."""

    def __init__(
        self,
        signal_variance_bounds=(1e-5, 1e5),
        length_scale_bounds=(1e-5, 1e5),
        noise_variance_bounds=(1e-5, 1e5),
    ):
        super().__init__(shared_kernel=True)
        if length_scale_bounds is None:
            length_scale_bounds = (1, 1)
        if noise_variance_bounds is None:
            noise_variance_bounds = (1, 1)
        self.signal_variance_bounds = signal_variance_bounds
        self.length_scale_bounds = length_scale_bounds
        self.noise_variance_bounds = noise_variance_bounds

        self._model = None
        self._model_input_dimensions = None
        self.num_output_dimensions = None

    def fit(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)
        _, num_input_dimensions = X.shape
        _, self.num_output_dimensions = Y.shape
        self._set_model_size(num_input_dimensions)
        self._model.fit(X, Y)

    def predict(
        self,
        X,
        return_var=False,
        return_cov=False,
        predictive_noise=False,
        broadcastable=False,
    ):
        X = np.asarray(X)
        if predictive_noise:
            return self._predict(
                X,
                return_var=return_var,
                return_cov=return_cov,
                broadcastable=broadcastable,
            )
        # Somewhat hacky.
        # The predictive noise is part of the kernel so it is included in the
        # predictive distribution by default.
        # To avoid numerical issues when subracting the noise afterwards,
        # it is better to temporarily set the noise to 0 in the kernel.
        # The gram matrix has already been computed so it is not altered.
        noise_level = self._model.kernel_.k2.noise_level
        try:
            self._model.kernel_.k2.noise_level = 0
            return self._predict(
                X,
                return_var=return_var,
                return_cov=return_cov,
                broadcastable=broadcastable,
            )
        finally:
            self._model.kernel_.k2.noise_level = noise_level

    def _predict(self, X, return_var, return_cov, broadcastable):
        """Predictive distribution. See predict()"""
        num_points, _ = X.shape
        result = self._model.predict(X, return_std=return_var, return_cov=return_cov)

        if return_var:
            mean, std = result
            var = np.square(std)[:, None]
            if not broadcastable:
                var = np.broadcast_to(var, [num_points, self.num_output_dimensions])
            return mean, var

        if return_cov:
            mean, cov = result
            if not broadcastable:
                cov = np.broadcast_to(
                    cov, [self.num_output_dimensions, num_points, num_points]
                )

            return mean, cov
        return result

    def get_params(self):
        return rbf_gaussian_process_regressor_parameters(self._model)

    def _set_model_size(self, num_input_dimensions):
        """Set the number of input dimensions of the model."""
        if self._model_input_dimensions == num_input_dimensions:
            return
        kernel = anisotropic_rbf_kernel(
            num_dimensions=num_input_dimensions,
            signal_variance_bounds=self.signal_variance_bounds,
            length_scale_bounds=self.length_scale_bounds,
            noise_variance_bounds=self.noise_variance_bounds,
        )
        self._model = gp.GaussianProcessRegressor(kernel=kernel)


def rbf_gaussian_process_regressor_parameters(regressor):
    """Extract parameters from a GP regressor with RBF kernel."""
    # Ensure all parameters have a dimension corresponding to
    # num_output_dimensions, even if it has size 1.
    target_values = regressor.y_train_
    if len(target_values.shape) == 1:
        target_values = target_values[:, None]

    coefficients = regressor.alpha_
    if len(coefficients.shape) == 1:
        coefficients = coefficients[None, :]
    else:
        coefficients = coefficients.T

    kernel = regressor.kernel_

    return base.RBFGaussianProcessParameters(
        inducing_points=regressor.X_train_,
        coefficients=coefficients,
        target_values=target_values,
        signal_variance=np.atleast_1d(kernel.k1.k1.constant_value),
        length_scale=np.atleast_2d(kernel.k1.k2.length_scale),
        noise_variance=np.atleast_1d(kernel.k2.noise_level),
        # atleast_3d appends instead of prepends
        gram_L=regressor.L_[None, :, :],
    )


def _logspace_mean(x, axis=None):
    """Mean in log space."""
    return np.exp(np.mean(np.log(x), axis=axis))


def anisotropic_rbf_kernel(
    num_dimensions,
    length_scale_bounds=(1e-5, 1e5),
    signal_variance_bounds=(1e-5, 1e5),
    noise_variance_bounds=(1e-5, 1e5),
    signal_variance=None,
    length_scale=None,
    noise_variance=None,
):
    """An anisotropic Gaussian Process RBF kernel with scale and noise terms.

    Args:
        num_dimensions: Number of input dimensions to kernel.

        length_scale_bounds: Bounds on the kernel length scale.
            An array-like broadcastable to shape `(num_dimensions, 2)`.

        signal_variance_bounds: Optional bounds on the signal variance
            (the kernel scale factor). If None, the kernel has no scale factor.
            An array-like broadcastable to shape `(2,)`

        signal_variance_bounds: Optional bounds on the noise variance (the
            kernel additive term). If None, the kernel has no additive noise.
            An array-like broadcastable to shape `(2,)`.

        length_scale: Optional initial length scales.
            A value broadcastable to shape `(num_dimensions,)`.
            Defaults to log-space midpoint of `length_scale_bounds`.

        signal_variance: Optional initial signal variance. A scalar.
            Defaults to log-space midpoint of `signal_variance_bounds`.

        noise_variance: Optional initial noise variance. A scalar.
            Defaults to log-space midpoint of `noise_variance_bounds`.
    """
    length_scale_bounds = np.broadcast_to(length_scale_bounds, (num_dimensions, 2))
    if length_scale is None:
        length_scale = _logspace_mean(length_scale_bounds, axis=-1)
    else:
        length_scale = np.broadcast_to(length_scale, (num_dimensions,))
    kernel = gp_kernels.RBF(
        length_scale=length_scale, length_scale_bounds=length_scale_bounds
    )

    if signal_variance_bounds is not None:
        signal_variance_bounds = np.broadcast_to(signal_variance_bounds, (2,))
        if signal_variance is None:
            signal_variance = _logspace_mean(signal_variance_bounds, axis=-1)
        kernel = (
            gp_kernels.ConstantKernel(
                constant_value=signal_variance,
                constant_value_bounds=signal_variance_bounds,
            )
            * kernel
        )

    if noise_variance_bounds is not None:
        noise_variance_bounds = np.broadcast_to(noise_variance_bounds, (2,))
        if noise_variance is None:
            noise_variance = _logspace_mean(noise_variance_bounds, axis=-1)
        kernel = kernel + gp_kernels.WhiteKernel(
            noise_level=noise_variance, noise_level_bounds=noise_variance_bounds
        )

    return kernel
