"""Simple mathematical moment map functions."""
import logging

import pilco.third_party.pilco.moments as pilco_moments

from . import core

logger = logging.getLogger(__name__)


class WhiteNoiseMomentMap(core.MomentMap):
    """Additive Gaussian white noise moment map.

    f(x) = x + eps
    where eps ~ N(0, diag(noise_variance))

    Attributes:
        noise_variance: 1D array of noise variances. The diagonal of the noise
            covariance matrix.
        dtype: Data type of the GP input and output.
    """

    def __init__(self, noise_variance, input_dim=None, **kwargs):
        super().__init__(input_dim=input_dim, output_dim=input_dim, **kwargs)
        self._repr_arguments["noise_variance"] = noise_variance
        self.noise_variance = self._ops.atleast_1d(
            self._ops.asarray(noise_variance, dtype=self.dtype)
        )

    def _call(self, mean, covariance, return_cov, return_io_cov_inv_in_cov):
        output_mean = mean

        if return_cov:
            output_covariance = self._ops.diagonal_matrix(self.noise_variance)
            if covariance is not None:
                output_covariance = output_covariance + covariance
        else:
            output_covariance = None

        if return_io_cov_inv_in_cov:
            io_cov_inv_in_cov = self._identity
        else:
            io_cov_inv_in_cov = None
        return output_mean, output_covariance, io_cov_inv_in_cov


class LinearMomentMap(core.MomentMap):
    """Moment map that applies an elementwise linear transformation.

    y = scale * x + offset
    """

    def __init__(self, scale=None, offset=None, **kwargs):
        """Initialize LinearMomentMap:

        Args:
            scale: The scaling term. Either a scalar or a 1d array-like with the same
                length as the target input. Supports broadcasting.
        """
        super().__init__(input_dim=None, output_dim=None, **kwargs)
        if scale is not None:
            scale = self._ops.atleast_1d(self._ops.asarray(scale, dtype=self.dtype))
        self.scale = scale

        if offset is not None:
            offset = self._ops.atleast_1d(self._ops.asarray(offset, dtype=self.dtype))
        self.offset = offset
        self._repr_arguments["scale"] = scale
        self._repr_arguments["offset"] = offset

    def _call(self, mean, covariance, return_cov, return_io_cov_inv_in_cov):
        output_mean = mean
        if self.scale is not None:
            output_mean = output_mean * self.scale
        if self.offset is not None:
            output_mean = output_mean + self.offset

        if return_cov:
            if covariance is None:
                # No input covariance => no output covariance
                output_covariance = self._zero_covariance_for(output_mean)
            elif self.scale is None:
                # No scaling => output covariance is same as input
                output_covariance = covariance
            else:
                # Scaling => elementwise scaled output covariance
                # This works correctly for scalar scale via broadcasting.
                output_covariance = covariance * (
                    self.scale[:, None] * self.scale[None, :]
                )
        else:
            output_covariance = None

        if return_io_cov_inv_in_cov:
            if self.scale is None:
                io_cov_inv_in_cov = self._identity
            else:
                input_dim = self.input_dim_value(mean)
                io_cov_inv_in_cov = self._ops.diagonal_matrix(
                    self.scale, size=input_dim
                )
        else:
            io_cov_inv_in_cov = None

        return output_mean, output_covariance, io_cov_inv_in_cov


class AbsMomentMap(core.MomentMap):
    """Moment map for the elementwise absolute value function |x|.

    For 1D inputs this is the Folded Normal distribution.
    https://en.wikipedia.org/wiki/Folded_normal_distribution
    """

    def __init__(self, eps=1e-16, **kwargs):
        """Initialize AbsMomentMap

        Args:
            eps: Small value added to the covariance matrix for stability.
        """
        # Currently only supports 1 => 1
        # Could be extended to elementwise with arbitrary dimension.
        super().__init__(input_dim=1, output_dim=1, **kwargs)
        self.eps = eps

    def _call(self, mean, covariance, return_cov, return_io_cov_inv_in_cov):
        if covariance is None:
            output_mean = abs(mean)
            if return_cov:
                output_covariance = self._zero_covariance_for(output_mean)
            else:
                output_covariance = None
            if return_io_cov_inv_in_cov:
                # Effectively a step function from -1 to 1 w.r.t. mean
                io_cov_inv_in_cov = (1 - 2 * self._ops.normal_cdf(0, mean, self.eps))[
                    ..., None
                ]
            else:
                io_cov_inv_in_cov = None
            return output_mean, output_covariance, io_cov_inv_in_cov

        variance = self._ops.matrix_diag_part(covariance)
        variance_eps = variance + self.eps
        stddev_eps = self._ops.sqrt(variance_eps)

        # Density at zero
        f0 = self._ops.normal_pdf(0, mean, stddev_eps)
        # CDF at zero
        F0 = self._ops.normal_cdf(0, mean, stddev_eps)

        output_mean = 2 * variance * f0 + mean * (1 - 2 * F0)

        input_dim = self.input_dim_value(mean)
        if not return_cov:
            output_covariance = None
        else:
            if input_dim > 1:
                raise NotImplementedError(
                    "Covariance not implemented for multidimensional inputs."
                )
            output_variance = mean ** 2 + variance - output_mean ** 2
            output_covariance = output_variance[..., None]

        if return_io_cov_inv_in_cov:
            io_cov_inv_in_cov = (1 - 2 * F0)[..., None]
        else:
            io_cov_inv_in_cov = None

        return output_mean, output_covariance, io_cov_inv_in_cov


class SinMomentMap(core.MomentMap):
    """Moment map for the function a * sin(x).

    Attributes:
        output_scale: Scaling factor applied to sin(x). A scalar.
            The output at `x` is `output_scale * sin(x)`.
    """

    def __init__(self, output_scale=1, **kwargs):
        super().__init__(input_dim=None, output_dim=None, **kwargs)
        self._repr_arguments["output_scale"] = output_scale
        self.output_scale = self._ops.asarray(output_scale, dtype=self.dtype)

    def _call(self, mean, covariance, return_cov, return_io_cov_inv_in_cov):
        input_dim = self.input_dim_value(mean)

        if covariance is None:
            variance = self._ops.zeros(input_dim, dtype=mean.dtype)
        else:
            variance = self._ops.matrix_diag_part(covariance)

        sqrt_exp_neg_var = self._ops.exp(-variance / 2)
        output_mean = self.output_scale * sqrt_exp_neg_var * self._ops.sin(mean)

        if not return_cov:
            output_covariance = None
        elif covariance is None:
            output_covariance = self._zero_covariance_for(output_mean)
        else:
            output_covariance = pilco_moments.multivarate_normal_sin_covariance(
                mean=mean,
                variance=variance,
                covariance=covariance,
                output_scale=self.output_scale,
                ops=self._ops,
            )

        if not return_io_cov_inv_in_cov:
            io_cov_inv_in_cov = None
        else:
            io_cov_inv_in_cov = self._ops.diagonal_matrix(
                self.output_scale * sqrt_exp_neg_var * self._ops.cos(mean)
            )
        return output_mean, output_covariance, io_cov_inv_in_cov


class SumSquaredMomentMap(core.MomentMap):
    """Sum of squared elements of the input vector."""

    def __init__(self, eps=1e-16, **kwargs):
        """Initialize SumSquaredMomentMap

        Args:
            eps: Small value added to the covariance matrix for stability.
        """
        super().__init__(input_dim=None, output_dim=1, **kwargs)
        self.eps = eps

    def _call(self, mean, covariance, return_cov, return_io_cov_inv_in_cov):
        if covariance is None:
            output_mean = self._ops.vector_dot(mean, mean)[..., None]
            if return_cov:
                output_covariance = self._zero_covariance_for(output_mean)
            else:
                output_covariance = None
            if return_io_cov_inv_in_cov:
                # See below for derivation
                io_cov_inv_in_cov = 2 * mean[..., None, :]
            else:
                io_cov_inv_in_cov = None
            return output_mean, output_covariance, io_cov_inv_in_cov

        # Eigenvalue decomposition of the covariance matrix Σ := Cov[X, X]
        # Σ = U Λ U'
        # where
        #   Λ = diag(eig_values)
        #   U = eig_vects
        #   U' = U transpose
        #
        # Add a small diagonal for stability when the covariance has near-0 eigenvalues.
        input_dim = self.input_dim_value(mean)
        eig_values, eig_vects = self._ops.self_adjoint_eig(
            covariance + self.eps * self._ops.eye(input_dim, dtype=covariance.dtype)
        )

        # Mean vectors for normalized distribution
        # X ~ 𝒩(μ, Σ)
        # X = U √Λ Z + μ = U √Λ (Z + b)
        # for
        #   Z ~ 𝒩(0, I)
        #   b = inv(√Λ) U' μ
        b = self._ops.squeeze(
            self._ops.matmul(mean[..., None, :], eig_vects), axis=-2
        ) / self._ops.sqrt(eig_values)
        b_squared = self._ops.square(b)

        # X'X = (Z + b)' √Λ U' U √Λ (Z + b)
        #       = (Z + b)' Λ (Z + b)
        #       = sum_i λ_i (Z_i + b_i)²
        #
        # where (Z_i + b_i)² ~ χ_i² is a noncentral chi-squared distribution
        # with mean b_i and 1 DOF.

        # χ_i² has mean 1 + b_i^2
        # E[X'X] = sum_i λ_i (1 + b_i^2)
        output_mean = self._ops.vector_dot(eig_values, b_squared + 1)[..., None]

        # χ_i² has variance 2 + 4 b_i^2
        # Var[X'X] = sum_i λ_i² (2 + 4 b_i^2)
        # since Z_i are independent
        if return_cov:
            output_covariance = self._ops.vector_dot(
                self._ops.square(eig_values), 4 * b_squared + 2
            )[..., None, None]
        else:
            output_covariance = None

        # Want Cov[X'X, X] @ inv(Cov[X, X])
        #
        # Note: X^{⋅2} represents elementwise squaring
        #
        # Cov[X'X, X]
        #   = Cov[(Z + b)' Λ (Z + b), U √Λ (Z + b)]
        #   = Cov[λ' (Z + b)^{⋅2}, Z] √Λ U'
        #   = λ' Cov[(Z + b)^{⋅2}, Z] √Λ U'
        #
        # Cov[(Z + b)^{⋅2}, Z]
        #   = Cov[Z^{⋅2} + 2 diag(b) Z + b^{⋅2}, Z]
        #   = Cov[Z^{⋅2}, Z] + 2 diag(b) Cov[Z, Z] + Cov[b^{⋅2}, Z]
        #   = O + 2 diag(b) I + 0 = 2 diag(b)
        #
        # Cov[X'X, X]
        #   = 2 λ' diag(b) √Λ U'
        #   = 2 b' Λ^{3/2} U'
        #   = 2 μ' U Λ^{-1/2} Λ^{3/2} U'
        #   = 2 μ' U Λ U'
        #   = 2 μ' Cov[X, X]
        #
        # So Cov[X'X, X] @ inv(Cov[X, X]) = 2μ'
        if return_io_cov_inv_in_cov:
            io_cov_inv_in_cov = 2 * mean[..., None, :]
        else:
            io_cov_inv_in_cov = None

        return output_mean, output_covariance, io_cov_inv_in_cov


class ElementProductMomentMap(core.MomentMap):
    """Product of two elements of the input vector."""

    def __init__(self, i, j, eps=1e-16, **kwargs):
        """Initialize ElementProductMomentMap

        Args:
            i: Index of the first element in the product. Must be a scalar index.
            j: Index of the second element in the product. Must be a scalar index != i.
            eps: Small value added to division denominator for stability.
        """
        super().__init__(input_dim=None, output_dim=1, **kwargs)
        if i == j:
            raise ValueError("Require i != j")
        self.i = i
        self.j = j
        self.eps = eps
        self._repr_arguments["i"] = i
        self._repr_arguments["j"] = j

    def _call(self, mean, covariance, return_cov, return_io_cov_inv_in_cov):
        # See scalar_normal_product.mac for derivations
        mean_x = mean[..., self.i]
        mean_y = mean[..., self.j]

        if covariance is None:
            var_x = 0
            var_y = 0
            cov_xy = 0
        else:
            var_x = covariance[..., self.i, self.i]
            var_y = covariance[..., self.j, self.j]
            cov_xy = covariance[..., self.i, self.j]

        mean_xy = mean_x * mean_y + cov_xy
        output_mean = mean_xy[..., None]

        if return_cov:
            # E[X^2] = E[X]^2 + Var[X]
            mean_x2 = mean_x ** 2 + var_x
            mean_y2 = mean_y ** 2 + var_y
            output_covariance = (
                mean_x2 * mean_y2 + mean_xy ** 2 - 2 * mean_x ** 2 * mean_y ** 2
            )[..., None, None]
        else:
            output_covariance = None

        if return_io_cov_inv_in_cov:
            # Let a = Cov[X,Y] / Var[X]
            # Then Z = Y - a * X is independent of X
            #
            # Cov[XY, X] = Cov[X(Z + aX), X] = Cov[XZ, X] + a * Cov[X^2, X]
            # Cov[XZ, X] = E[Z] * Var[X] = (E[Y] - a * E[X]) * Var[X]
            # Cov[X^2, X] = E[X^3] - E[X^2]E[X] = 2 * E[X] * Var[X]
            #
            # So
            # Cov[XY, X]
            #   = E[Y] * Var[X] - a * E[X] * Var[X] + 2a * E[X] * Var[X]
            #   = E[Y] * Var[X] + a * E[X] * Var[X]
            #   = E[Y] * Var[X] + E[X] * Cov[X, Y]
            #
            # By symmetry,
            # Cov[XY, Y]
            #   = E[X] * Var[Y] + E[Y] * Cov[X, Y]
            #
            # Combined,
            # Cov[XY, (X, Y)]
            #   = [E[Y], E[X]] @ [  Var[X]   Cov[X, Y] ]
            #                    [ Cov[X, Y]  Var[Y]   ]
            #
            # More generally, for Cov[XY, W] we can write
            # X = Z_X + a_X * W,  a_X = Cov[X, W] / Var[W] => Z_X, W independent
            # Y = Z_Y + a_Y * W,  a_Y = Cov[Y, W] / Var[W] => Z_Y, W independent
            #
            # Cov[XY, W]
            #   = Cov[(Z_X + a_X * W)(Z_Y + a_Y * W), W]
            #   = Cov[Z_X * Z_Y, W] + a_Y Cov[Z_X * W, W] + a_X Cov[Z_Y * W, W]
            #       + a_X * a_Y * Cov[W^2, W]
            #
            #   = 0 + a_y E[Z_X]Var[W] + a_X E[Z_Y]Var[W] + a_X a_Y Cov[W^2, W]
            #   = ...
            #   = Cov[Y, W] * E[X] + Cov[X, W] * E[Y]
            #   = [0, ..., 0, E[Y], ..., E[Y], 0, ..., 0] @ Cov[input]
            #                  i          j
            input_dim = self.input_dim_value(mean)
            eye = self._ops.eye(input_dim, dtype=mean.dtype)
            io_cov_inv_in_cov = (mean_y * eye[self.i, :] + mean_x * eye[self.j, :])[
                ..., None, :
            ]
        else:
            io_cov_inv_in_cov = None

        return output_mean, output_covariance, io_cov_inv_in_cov
