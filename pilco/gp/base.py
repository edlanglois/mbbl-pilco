"""Gaussian process base classes."""
import typing

import numpy as np


class RBFGaussianProcessParameters(typing.NamedTuple):
    """Parameters to a Gaussian process with a radial basis kernel.

    Attributes:
        inducing_points: array-like, shape = (n_points, n_input_dims)
            Gaussian process inducing points in input feature space.
        coefficients: array-like, shape = (n_output_dims, n_points)
            Dual coefficients of inducing_points in kernel space.
        target_values: array-like, shape = (n_points, n_output_dims)
            Output target value at each inducing point.
        signal_variance: array-like, shape = (n_kernels,)
            Kernel magnitude scaling factors.
        length_scale: array-like, shape = (n_kernels, n_input_dims)
            Kernel length scales.
        noise_variance: array-like, shape = (n_kernels,)
            Kernel additive noise levels.
        gram_L: array-like, shape = (n_kernels, n_points, n_points)
            Lower triangular Cholesky decomposition of the kernel Gram matrix
                over `inducing_points`.
        B_L: array-like, shape = (n_kernels, n_points, n_points)
            Optional lower triangular Cholesky decomposition of B used for
            sparse Gaussian processes.
            If included, instead of using the inverse gram matrix
                inv(gram) = L^{-T} L^{-1}  where L = gram_L,
            use
                inv(gram) - L^{-T} B^{-1} L^{-1}

    n_kernels is either 1 or n_output_dims
    """

    inducing_points: typing.Any
    coefficients: typing.Any
    target_values: typing.Any = None
    signal_variance: typing.Any = 1
    length_scale: typing.Any = 1
    noise_variance: typing.Any = 0
    gram_L: typing.Any = None
    B_L: typing.Any = None


class BaseRBFGaussianProcessRegressor:
    """Interface of radial basis kernel Gaussian Process regressor.

    Implements Gaussian Process regression with a radial basis kernel and
    additive Gaussian noise on the observations.

    Supports multiple output dimensions, each modelled by an independent
    Gaussian Process.
    For a single input point, the outputs dimensions are independently
    distributed Gaussians.

    Attributes:
        shared_kernel: Whether all output dimensions share a single kernel.
    """

    def __init__(self, shared_kernel):
        self.shared_kernel = shared_kernel

    def fit(self, X, Y):
        """Fit the Gaussian process to data.

        Args:
            X: Matrix of input feature vectors.
                An array of shape `(num_points, num_input_dimensions)`.
            Y: Matrix of regression target vectors.
                An array of shape `(num_points, num_output_dimensions)`.
        """
        raise NotImplementedError

    def predict(
        self,
        X,
        return_var=False,
        return_cov=False,
        predictive_noise=False,
        broadcastable=False,
    ):
        """Predict using the fitted Gaussian processes.

        Args:
            X: Matrix of query point feature vectors.
                An array of shape `(num_points, num_input_dimensions)`.
            return_var: If True, return the predictive variance at the query
                points.
            return_cov: If True, return the covariance of the joint predictive
                distribution at the query points.
                At most one of `return_var` and `return_cov` may be True.
            predictive_noise: If True, include the additive Gaussian noise in
                the predictive distribution.
            broadcastable: If True, outputs will take advantage of broadcasting
                semantics and may have dimensions with size 1 rather than full
                size.

        Returns:
            Y_mean: A matrix of predicted output means.
                An array of shape `(num_points, num_output_dimensions)`.
            Y_var: The predictive variance at each of the query points.
                An array of shape `(num_points, num_output_dimensions)` or
                Returned only if `return_var` is True.
            Y_cov: The covariance of the joint predictive distribution at the
                query points. An array of shape
                `(num_output_dimensions, num_points, num_points)`.
                Returned only if `return_cov` is True.
        """
        raise NotImplementedError

    def get_params(self):
        """Get the Gaussian processes parameters.

        Returns:
            An instance of `RBFGaussianProcessParameters`.
        """
        raise NotImplementedError


class DecoupledKernelGaussianProcessWrapper(BaseRBFGaussianProcessRegressor):
    """Wraps a GP regressor, giving output dims optionally independent kernels.
    """

    def __init__(self, model_class, shared_kernel=False, **model_kwargs):
        """Initialize a DecoupledKernelGaussianProcessWrapper.

        Args:
            model_class: Class to wrap. Uses a single kernel for all outputs.
                Must have interface of BaseRBFGaussianProcessRegressor.
            shared_kernel: If False, output dimensions have independent
                kernels. Otherwise, all share the same kernel parameters.
            **model_kwargs: Keyword arguments to pass to model_class.__init__
        """
        super().__init__(shared_kernel=shared_kernel)
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        if self.shared_kernel:
            self._models = [self.model_class(**self.model_kwargs)]
        else:
            self._models = None

    def fit(self, X, Y):
        Y = np.asarray(Y)
        _, num_output_dimensions = Y.shape

        if not self.shared_kernel:
            if self._models is None or len(self._models) != num_output_dimensions:
                self._models = [
                    self.model_class(**self.model_kwargs)
                    for _ in range(num_output_dimensions)
                ]
            for model, y in zip(self._models, Y.T):
                model.fit(X, y[:, None])
        else:
            model, = self._models
            model.fit(X, Y)

    def predict(
        self,
        X,
        return_var=False,
        return_cov=False,
        predictive_noise=False,
        broadcastable=False,
    ):
        mean_list = []
        var_list = []

        try:
            models = iter(self._models)
        except TypeError:
            raise RuntimeError("Must call fit() before predict().")
        for model in models:
            result = model.predict(
                X,
                return_var=return_var,
                return_cov=return_cov,
                predictive_noise=predictive_noise,
                broadcastable=broadcastable,
            )
            if return_var or return_cov:
                mean, var = result
                mean_list.append(mean)
                var_list.append(var)
            else:
                mean_list.append(result)

        mean = np.concatenate(mean_list, axis=-1)
        if return_var:
            var = np.concatenate(var_list, axis=-1)
            return mean, var
        if return_cov:
            cov = np.concatenate(var_list, axis=0)
            return mean, cov
        return mean

    def get_params(self):
        try:
            models = iter(self._models)
        except TypeError:
            raise RuntimeError("Must call fit() before get_params().")

        param_lists = {name: [] for name in RBFGaussianProcessParameters._fields}
        for model in models:
            model_params = model.get_params()
            for name, value_list in param_lists.items():
                value_list.append(getattr(model_params, name))

        inducing_points_list = param_lists["inducing_points"]
        inducing_points = inducing_points_list.pop()
        if any(
            not (np.array_equal(inducing_points, other_inducing_points))
            for other_inducing_points in inducing_points_list
        ):
            raise ValueError(
                "Cannot concatenate kernels with different inducing points."
            )

        param_dict = {
            "inducing_points": inducing_points,
            "coefficients": np.concatenate(param_lists["coefficients"], axis=0),
            "target_values": concatenate_if_not_none(
                param_lists["target_values"], axis=-1
            ),
            "signal_variance": np.concatenate(param_lists["signal_variance"]),
            "length_scale": np.concatenate(param_lists["length_scale"], axis=0),
            "noise_variance": np.concatenate(param_lists["noise_variance"]),
            "gram_L": concatenate_if_not_none(param_lists["gram_L"], axis=0),
            "B_L": concatenate_if_not_none(param_lists["B_L"], axis=0),
        }

        return RBFGaussianProcessParameters(**param_dict)


def concatenate_if_not_none(arrays, axis=0):
    """Concatenate a series of arrays along an axis if none are None."""
    # Numpy equality breaks `in`
    if any(array is None for array in arrays):
        return None
    return np.concatenate(arrays, axis=axis)
