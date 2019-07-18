"""Gaussian Process moment mapping."""
import logging

import numpy as np

from pilco import utils

from . import core

logger = logging.getLogger(__name__)


class _BaseGaussianProcessMomentMap(core.MomentMap):
    """Gaussian process base moment map.

    The Gaussian process must have an RBF kernel of the form:

    `k(x, y) = signal_variance * exp(-1/2 (x-y)^T Lambda^{-1} (x-y)) + noise`

    where Lambda is `length_scale**2`.
    Different output dimensions of the GP may use different kernel parameters.

    Attributes:
        inducing_points: Gaussian process inducing points.
            An array of shape `[N, IN_DIM]`.
        coefficients: Gaussian process regression coefficients in the dual
            space. An array of shape `[OUT_DIM, N]`.
        signal_variance: Kernel magnitude scaling factor.
            An array of shape `[OUT_DIM]`. Supports broadcasting.
        length_scale: Kernel input length scales.
            An array of shape `[OUT_DIM, IN_DIM]`. Supports broadcasting.
        gram_L: Lower triangular Cholesky decomposition of the Gram matrix of
            the kernel on `inducing_points`, including noise on the diagonal.
            An array of shape `[OUT_DIM, N, N]`.
            May be `None` if `deterministic` is true or if input_covariance is
            None.
        deterministic: If True, represent a deterministic GP that consists of
            only the mean function.
        B_L: Cholesky decomposition of the matrix B, an optional additional
            term on the inverse Gram matrix. If present, instead of using
            If included, instead of using the inverse gram matrix
                inv(gram) = L^{-T} L^{-1}  where L = `gram_L`,
            use
                inv(gram) - L^{-T} B^{-1} L^{-1}
            An array of shape `[OUT_DIM, N, N]`.
    """

    def __init__(
        self,
        inducing_points,
        coefficients,
        signal_variance,
        length_scale,
        gram_L,
        deterministic,
        *,
        B_L=None,
        debug_check_reference=False,
        **kwargs,
    ):
        try:
            input_dim = int(inducing_points.shape[-1])
        except IndexError:
            input_dim = 1
        try:
            output_dim = int(coefficients.shape[-2])
        except IndexError:
            output_dim = 1

        super().__init__(input_dim=input_dim, output_dim=output_dim, **kwargs)

        self.inducing_points = self._ops.atleast_2d(
            self._ops.asarray(inducing_points, dtype=self.dtype)
        )
        self.coefficients = self._ops.atleast_2d(
            self._ops.asarray(coefficients, dtype=self.dtype)
        )
        self.signal_variance = self._ops.atleast_1d(
            self._ops.asarray(signal_variance, dtype=self.dtype)
        )
        self.length_scale = self._ops.atleast_2d(
            self._ops.asarray(length_scale, dtype=self.dtype)
        )
        if gram_L is not None:
            gram_L = self._ops.atleast_3d(self._ops.asarray(gram_L, dtype=self.dtype))
        self.gram_L = gram_L

        if B_L is not None:
            B_L = self._ops.atleast_3d(self._ops.asarray(B_L, dtype=self.dtype))
        self.B_L = B_L

        self.deterministic = deterministic

        # Check results against reference implementation
        self.debug_check_reference = debug_check_reference

    def _call(self, mean, covariance, return_cov, return_io_cov_inv_in_cov):
        if covariance is None:
            with self._ops.name_scope("GPMomements_DeterministicInput"):
                (
                    output_mean,
                    output_covariance,
                    io_cov_inv_in_cov,
                ) = self._call_determistic_input(
                    input_mean=mean,
                    return_cov=return_cov,
                    return_io_cov_inv_in_cov=return_io_cov_inv_in_cov,
                )
        else:
            with self._ops.name_scope("GPMoments_NormalInput"):
                (
                    output_mean,
                    output_covariance,
                    io_cov_inv_in_cov,
                ) = self._call_normal_input(
                    input_mean=mean,
                    input_covariance=covariance,
                    return_cov=return_cov,
                    return_io_cov_inv_in_cov=return_io_cov_inv_in_cov,
                )

        # Check shapes
        assert output_mean.shape[-1] == self.output_dim
        if return_cov:
            assert output_covariance.shape[-2:] == (self.output_dim, self.output_dim)
        if return_io_cov_inv_in_cov:
            assert io_cov_inv_in_cov.shape[-2:] == (self.output_dim, self.input_dim)

        if self.debug_check_reference:
            if self.backend != "numpy":
                logger.warning("Can only check reference with numpy backend.")
            elif self.B_L is not None:
                logger.warning("Cannot check reference when using B_L")
            else:
                (
                    output_mean_ref,
                    output_covariance_ref,
                    io_covariance_ref,
                ) = _reference_gp_predict_moments(
                    input_mean=mean,
                    input_covariance=covariance,
                    inducing_points=self.inducing_points,
                    gram_L=self.gram_L,
                    coefficients=self.coefficients,
                    signal_variance=self.signal_variance,
                    length_scale=self.length_scale,
                    return_cov=return_cov,
                    return_io_cov=return_io_cov_inv_in_cov,
                )
                np.testing.assert_allclose(output_mean, output_mean_ref, rtol=1e-3)
                if return_cov:
                    np.testing.assert_allclose(output_covariance, output_covariance_ref)
                if return_io_cov_inv_in_cov and covariance is not None:
                    np.testing.assert_allclose(
                        io_cov_inv_in_cov @ covariance, io_covariance_ref
                    )

        return output_mean, output_covariance, io_cov_inv_in_cov

    def _call_determistic_input(self, input_mean, return_cov, return_io_cov_inv_in_cov):
        """Call for determistic input.

        Args:
            input_mean: Input distribution mean.
                An array of shape [..., IN_DIM].
            return_cov: Return the output covariance matrix.
                If false, `output_covariance` is None.
            return_io_cov_inv_in_cov: Return the input-output covariance matrix
                times inverse input covariance matrix.
                If false, `io_cov_inv_in_cov` is None.

        Returns:
            output_mean: Output mean vector. An array of shape [..., OUT_DIM].
            output_covariance: Output covariance matrix.
                An array of shape [..., OUT_DIM, OUT_DIM].
            io_cov_inv_in_cov:
                Input-output covariance matrix times inverse input covariance.
                May be `self._identity` if equal to the identity matrix.
                An array of shape [..., OUT_DIM, IN_DIM].
        """
        # Implementation based on:
        # Deisenroth MP.
        # Efficient reinforcement learning using Gaussian processes.
        # KIT Scientific Publishing; 2010.
        #
        # The notation generally follows this reference.
        # Capitalized variables represent matrices and lower case represent
        # vectors (ignoring batch and output dimensions).

        # beta:       [OUT_DIM, N]
        beta = self.coefficients

        # Equation 2.36
        # q = signal_variance * sqrt(|Sigma @ Lambda^{-1} - I|) * exp(...)
        # Sigma = 0 so |Sigma * Lamba^{-1} - I| = 1

        # q: [..., OUT_DIM, N]
        q = self._ops.squeeze(
            self._rbf_kernel(
                x=self.inducing_points,
                y=input_mean[..., None, :],
                signal_variance=self.signal_variance,
                length_scale=self.length_scale,
            ),
            -1,
        )

        # Equation 2.43
        # output_mean: [..., OUT_DIM]
        output_mean = self._ops.vector_dot(q, beta)

        if return_io_cov_inv_in_cov:
            io_cov_inv_in_cov = self._ops.zeros(
                [self.output_dim, self.input_dim], dtype=output_mean.dtype
            )
        else:
            io_cov_inv_in_cov = None

        if not return_cov:
            output_covariance = None
            return output_mean, output_covariance, io_cov_inv_in_cov

        if self.deterministic:
            output_covariance = self._ops.zeros(
                [self.output_dim, self.output_dim], dtype=output_mean.dtype
            )
            return output_mean, output_covariance, io_cov_inv_in_cov

        if self.gram_L is None:
            raise ValueError(
                "gram_L is required when return_cov is true " "and not deterministic"
            )

        # TODO: avoid expand / squeeze
        # InvL_q: [..., OUT_DIM, N, 1]
        InvL_q = self._ops.matrix_triangular_solve(self.gram_L, q[..., :, None])

        # q_InvG_q: [..., OUT_DIM]
        q_InvG_q = self._ops.sum(
            self._ops.square(self._ops.squeeze(InvL_q, axis=-1)), axis=-1
        )

        if self.B_L is not None:
            # InvBL_InvL_q: [..., OUT_DIM, N]
            InvBL_InvL_q = self._ops.squeeze(
                self._ops.matrix_triangular_solve(self.B_L, InvL_q), axis=-1
            )

            # q_InvG_q: [..., OUT_DIM]
            q_InvG_q = q_InvG_q - self._ops.sum(self._ops.square(InvBL_InvL_q), axis=-1)

        # output_covariance: [..., OUT_DIM, OUT_DIM]
        output_covariance = self._ops.diagonal_matrix(
            self.signal_variance - q_InvG_q, size=self.output_dim
        )

        return output_mean, output_covariance, io_cov_inv_in_cov

    def _call_normal_input(
        self, input_mean, input_covariance, return_cov, return_io_cov_inv_in_cov
    ):
        """Call for multivariate normal distributed input.

        Args:
            input_mean: Input distribution mean.
                An array of shape [..., IN_DIM].
            input_covariance: Input distribution covariance matrix.
                An array of shape [..., IN_DIM, IN_DIM].
            return_cov: Return the output covariance matrix.
                If false, `output_covariance` is None.
            return_io_cov_inv_in_cov: Return the input-output covariance matrix
                times inverse input covariance matrix.
                If false, `io_cov_inv_in_cov` is None.

        Returns:
            output_mean: Output mean vector. An array of shape [..., OUT_DIM].
            output_covariance: Output covariance matrix.
                An array of shape [..., OUT_DIM, OUT_DIM].
            io_cov_inv_in_cov:
                Input-output covariance matrix times inverse input covariance.
                May be `self._identity` if equal to the identity matrix.
                An array of shape [..., OUT_DIM, IN_DIM].
        """
        # Implementation based on:
        # Deisenroth MP.
        # Efficient reinforcement learning using Gaussian processes.
        # KIT Scientific Publishing; 2010.
        #
        # The notation generally follows this reference.
        # Capitalized variables represent matrices and lower case represent
        # vectors (ignoring batch and output dimensions).
        beta = self.coefficients

        # length_scale needs to have fully broadcasted IN_DIM due to
        # self._ops.sum(length_scale) & diagonal_matrix(lambda_)
        # length_scale: [OUT_DIM, IN_DIM]
        length_scale = self._ops.broadcast_axis_to(
            self.length_scale, axis=-1, size=self.input_dim
        )

        # lambda: [OUT_DIM, IN_DIM]
        lambda_ = self._ops.square(length_scale)

        # Equation 2.36
        # q = signal_variance * sqrt(|Sigma * Lambda^{-1} + I|) * exp(...)

        # expanded_Sigma: [..., OUT_DIM, IN_DIM, IN_DIM]
        expanded_Sigma = input_covariance[..., None, :, :]
        # Sigma_p_Lambda: [..., OUT_DIM, IN_DIM, IN_DIM]
        Sigma_p_Lambda = expanded_Sigma + self._ops.diagonal_matrix(
            lambda_, size=self.input_dim
        )

        # zeta: [..., N, IN_DIM]
        with self._ops.name_scope("zeta"):
            zeta = self.inducing_points - input_mean[..., None, :]

        # Sigma_p_Lambda_cholesky: [..., OUT_DIM, IN_DIM, IN_DIM]
        with self._ops.name_scope("Sigma_p_Lambda_cholesky"):
            Sigma_p_Lambda_cholesky = self._ops.cholesky(Sigma_p_Lambda)

        identity_input = self._ops.eye(
            self.input_dim, dtype=Sigma_p_Lambda_cholesky.dtype
        )

        # Inv_Sigma_p_Lambda: [..., OUT_DIM, IN_DIM, IN_DIM]
        with self._ops.name_scope("Inv_Sigma_p_Lambda"):
            Inv_Sigma_p_Lambda = self._ops.cholesky_solve(
                Sigma_p_Lambda_cholesky, identity_input
            )

        # Sigma_p_Lambda_cholesky_inv: [..., OUT_DIM, IN_DIM, IN_DIM]
        with self._ops.name_scope("Sigma_p_Lambda_cholesky_inv"):
            Sigma_p_Lambda_cholesky_inv = self._ops.matrix_triangular_solve(
                Sigma_p_Lambda_cholesky, identity_input, lower=True, adjoint=True
            )

        # Sigma_p_Lambda_halflogdet: [..., OUT_DIM]
        with self._ops.name_scope("Sigma_p_Lambda_halflogdet"):
            Sigma_p_Lambda_halflogdet = self._ops.sum(
                self._ops.log(self._ops.matrix_diag_part(Sigma_p_Lambda_cholesky)),
                axis=-1,
            )

        # Equation 2.36
        # q: [..., OUT_DIM, N]
        with self._ops.name_scope("q"):
            q = self._ops.exp(
                -0.5
                * self._ops.sum(
                    self._ops.square(
                        self._ops.matmul(
                            zeta[..., None, :, :], Sigma_p_Lambda_cholesky_inv
                        )
                    ),
                    axis=-1,
                )
            )

            # q *= signal_variance / sqrt(|Sigma*Lambda^{-1} + I|)
            #    = signal_variance * exp(
            #       sum(log(length_scale)) - logdet(Sigma+Lambda)/2)
            # q_scale: [OUT_DIM]
            q_scale = self.signal_variance * self._ops.exp(
                self._ops.sum(self._ops.log(length_scale), axis=-1)
                - Sigma_p_Lambda_halflogdet
            )
            q *= q_scale[..., None]

        # Equation 2.43
        # output_mean: [..., OUT_DIM]
        with self._ops.name_scope("output_mean"):
            output_mean = self._ops.vector_dot(q, beta)

        # if return_cov:
        if return_cov:
            with self._ops.name_scope("output_covariance"):
                output_covariance = self._output_covariance(
                    input_covariance=input_covariance,
                    output_mean=output_mean,
                    lambda_=lambda_,
                    zeta=zeta,
                )
        else:
            output_covariance = None

        if return_io_cov_inv_in_cov:
            with self._ops.name_scope("io_cov_inv_in_cov"):
                io_cov_inv_in_cov = self._io_cov_inv_in_cov(
                    q=q, zeta=zeta, Inv_Sigma_p_Lambda=Inv_Sigma_p_Lambda
                )
        else:
            io_cov_inv_in_cov = None

        return output_mean, output_covariance, io_cov_inv_in_cov

    def _output_covariance(self, input_covariance, output_mean, lambda_, zeta):
        """Calculate the output covariance matrix.

        Args:
            input_covariance: Input distribution covariance matrix.
                An array of shape [..., IN_DIM, IN_DIM].
            output_mean: Output mean vector. An array of shape [..., OUT_DIM].
            lambda_: Array of squared kernel length scales.
                An array of shape [OUT_DIM, IN_DIM]
            zeta: An array of shape [..., N, IN_DIM]

        Returns:
            output_covariance: Output covariance matrix.
                An array of shape [..., OUT_DIM, OUT_DIM].
        """
        beta = self.coefficients

        R_halflogdet, InvR_Sigma = self._output_covariance_r(
            input_covariance=input_covariance, lambda_=lambda_
        )

        # Equation 2.54
        # Efficient calculation of the z' R^{-1} Sigma z term in n^2.
        #
        # v_ai := Lambda^{-1}_a @ zeta_i
        # z_abij = v_ai + v_bj
        # M_ab := R^{-1}_ab @ Sigma
        #       = (Lambda^{-1}_a + Lambda^{-1}_b + Sigma)^{-1}
        # M_ab is symmetric: M_ab = M_ab'
        # Also, M_ab = M_ba
        #
        # o_abij := z_abij' @ M_ab @ z_abij
        # = (v_ai + v_bj)' @ M_ab @ (v_ai + b_bj)
        # = v_ai' @ M_ab @ v_ai + v_bj' @ M_ba @ v_bj + 2 @ v_bj' @ M_ab @ v_ai
        #
        # Let
        #   w_abi := M_ab @ v_ai = (v_ai' @ M_ab)'
        #   u_abi := v_ai' @ w_abi = v_ai' @ M_ab @ v_ai
        #
        # Then
        # o_abij = u_abi + u_baj + 2 * v_bj' @ w_abi
        #

        # v: [..., OUT_DIM, N, IN_DIM]
        v = zeta[..., None, :, :] / lambda_[..., :, None, :]
        # M: [..., OUT_DIM, OUT_DIM, IN_DIM, IN_DIM]
        M = InvR_Sigma

        # w: [..., OUT_DIM, OUT_DIM, N, IN_DIM]
        w = self._ops.matmul(v[..., :, None, :, :], M)
        # u: [..., OUT_DIM, OUT_DIM, N]
        u = self._ops.vector_dot(v[..., :, None, :, :], w)

        # n2: [..., OUT_DIM, OUT_DIM, N, N]
        n2 = (
            u[..., :, :, :, None]
            + self._ops.swapaxes(u, -3, -2)[..., :, :, None, :]
            + 2
            * self._ops.vector_dot(
                v[..., None, :, None, :, :], w[..., :, :, :, None, :]
            )
        )

        # zeta_InvLambda_zeta: [..., OUT_DIM, N]
        zeta_InvLambda_zeta = self._ops.vector_dot(zeta[..., None, :, :], v)
        if self._ops.data_is_mutable:  # n2 is huge so avoid copies if possible
            n2 -= zeta_InvLambda_zeta[..., :, None, :, None]
            n2 -= zeta_InvLambda_zeta[..., None, :, None, :]
            n2 /= 2
        else:
            n2 = (
                n2
                - zeta_InvLambda_zeta[..., :, None, :, None]
                - zeta_InvLambda_zeta[..., None, :, None, :]
            ) / 2

        # [OUT_DIM]
        log_signal_variance = self._ops.log(self.signal_variance)
        if self._ops.data_is_mutable:
            n2 += log_signal_variance[..., :, None, None, None]
            n2 += log_signal_variance[..., None, :, None, None]
        else:
            n2 = (
                n2
                + log_signal_variance[..., :, None, None, None]
                + log_signal_variance[..., None, :, None, None]
            )

        # Equation 2.53
        # Q: [..., OUT_DIM, OUT_DIM, N, N]
        with self._ops.name_scope("Q"):
            Q = self._ops.exp(n2 - R_halflogdet[..., None, None])

        # Equation 2.50
        # E_hh: [..., OUT_DIM, OUT_DIM]
        E_hh = self._ops.squeeze(
            self._ops.matmul(
                beta[..., :, None, None, :],
                self._ops.matmul(Q, beta[..., None, :, :, None]),
            ),
            axis=(-2, -1),
        )

        # Equation 2.45 & 2.55
        output_covariance = E_hh - output_mean[..., :, None] * output_mean[..., None, :]

        if not self.deterministic:
            if self.gram_L is None:
                raise ValueError(
                    "gram_L is required when return_cov is true "
                    "and not deterministic"
                )
            # Diagonal along output dimensions
            # DiagQ: [..., OUT_DIM, N, N]
            DiagQ = self._ops.diag_part_43(Q)

            # InvL_Q: [..., OUT_DIM, N, N]
            InvL_Q = self._ops.matrix_triangular_solve(self.gram_L, DiagQ)
            InvG_Q = self._ops.matrix_triangular_solve(
                self.gram_L, InvL_Q, adjoint=True
            )
            InvG_Q_traces = self._ops.trace(InvG_Q)

            if self.B_L is not None:
                InvB_InvL_Q = self._ops.cholesky_solve(self.B_L, InvL_Q)
                InvL_InvB_InvL_Q = self._ops.matrix_triangular_solve(
                    self.gram_L, InvB_InvL_Q, adjoint=True
                )
                InvG_Q_traces = InvG_Q_traces - self._ops.trace(InvL_InvB_InvL_Q)

            # G is the effective Gram matrix,
            # either G = LL' or G = inv(inv(L)'inv(L) + inv(L)'inv(B)inv(L))
            #
            # Note: We could instead pre-compute InvG and then we have
            # InvG_Q_traces = self._ops.sum(self.InvG * DiagQ, axis=(-2, -1))
            # This would be faster but less numerically stable.

            # output_extra_variance: [OUT_DIM]
            output_extra_variance = self._ops.maximum(
                self.signal_variance - InvG_Q_traces, 0
            )

            output_covariance = output_covariance + self._ops.diagonal_matrix(
                output_extra_variance, size=self.output_dim
            )

        # Try to mitigate potential asymmetry from numerical errors
        output_covariance = (
            output_covariance + self._ops.matrix_transpose(output_covariance)
        ) / 2
        return output_covariance

    def _output_covariance_r(self, input_covariance, lambda_):
        """Calculate log(sqrt(det(R))) and R⁻¹Σ for R = Σ(Λₐ⁻¹+Λₑ⁻¹)+I

        Helper function for _output_covariance

        Args:
            input_covariance: Covariance matrix of the input distribution.
                An array of shape `[..., IN_DIM, IN_DIM]`.
            lambda_: Squared length scales.
                The diagonal of he length scale matrix Lambda (
                An array of shape `[..., OUT_DIM, IN_DIM].

        Returns:
            R_halflogdet: log(sqrt(det(R)))
                An array of shape `[..., OUT_DIM, OUT_DIM]`.
            InvR_Sigma: R⁻¹Σ
                An array of shape `[..., OUT_DIM, OUT_DIM, IN_DIM, IN_DIM]`.
        """
        # We want to use the Cholesky decomposition for stability but R is not
        # symmetric.
        #
        # Let G = (Λₐ⁻¹+Λₑ⁻¹)
        # S = GR = GΣG + G is symmetric so use that instead.
        # Let L be the Cholesky decomposition of S. LL* = S
        #
        # |R| = |S| / |G|
        #     = |L|^2 / prod(g)
        # where g = diag_part(G)
        #
        # L is triangular so |L| = prod(diag_part(L))
        #
        # R⁻¹Σ = S⁻¹GΣ

        with self._ops.name_scope("inv_lambda"):
            inv_lambda = self._ops.reciprocal(lambda_)

        # g = inv_lambda_a + inv_lambda_b
        # g: [..., OUT_DIM, OUT_DIM, IN_DIM]
        with self._ops.name_scope("g"):
            g = inv_lambda[..., :, None, :] + inv_lambda[..., None, :, :]

        # InvLambda_Sigma: [..., OUT_DIM, OUT_DIM, IN_DIM, IN_DIM]
        with self._ops.name_scope("G_Sigma"):
            G_Sigma = g[..., :, None] * input_covariance[..., None, None, :, :]

        # S: [..., OUT_DIM, OUT_DIM, IN_DIM, IN_DIM]
        with self._ops.name_scope("S"):
            S = G_Sigma * g[..., None, :] + self._ops.diagonal_matrix(g)

        # S_cholesky: [..., OUT_DIM, OUT_DIM, IN_DIM, IN_DIM]
        with self._ops.name_scope("S_cholesky"):
            S_cholesky = self._ops.cholesky(S)

        # R_halflogdet: [..., OUT_DIM, OUT_DIM]
        with self._ops.name_scope("R_halflogdet"):
            S_halflogdet = self._ops.sum(
                self._ops.log(self._ops.matrix_diag_part(S_cholesky)), axis=-1
            )
            R_halflogdet = S_halflogdet - self._ops.sum(self._ops.log(g), axis=-1) / 2

        with self._ops.name_scope("InvR_Sigma"):
            InvR_Sigma = self._ops.cholesky_solve(S_cholesky, G_Sigma)

        return R_halflogdet, InvR_Sigma

    def _io_cov_inv_in_cov(self, q, zeta, Inv_Sigma_p_Lambda):
        """Calculate Cov(Output, Input) * Inv(Cov(Input, Input))."""
        # input-output covariance
        # Equation 2.70
        beta = self.coefficients

        # beta:       [OUT_DIM, N]
        # q:     [..., OUT_DIM, N]
        # zeta:   [..., N, IN_DIM]
        # io_cov_inv_in_cov: [OUT_DIM, IN_DIM]
        io_cov_inv_in_cov = self._ops.sum(
            (beta * q)[..., None]
            * self._ops.matmul(zeta[..., None, :, :], Inv_Sigma_p_Lambda),
            axis=-2,
        )
        return io_cov_inv_in_cov

    def _rbf_kernel(self, x=None, y=None, z=None, signal_variance=1, length_scale=1):
        """Evaluate a radial basis function kernel.

        signal_variance * exp((x - y)' @ diag(1 / length_scales) @ (x - y))

        Args:
            x: Array of shape [..., M, D].
            y: Array of shape [..., N, D].
            z: Array of shape [..., M, N, D] equal to x - y.
                May be provided instead of x and y.
            signal_variance: Array of shape [OUT_DIM].
                Multiplier on the kernel value.
            length_scale: Array of shape [OUT_DIM, D].
                Input length scale.

        Returns:
            An array of shape [..., OUT_DIM, M, N] containing the kernel
            evaluated at each pair (i, j) of x[..., i, :] - y[..., j, :]
        """
        with self._ops.name_scope("rbf_kernel"):
            if z is None:
                x = self._ops.atleast_2d(x)
                y = self._ops.atleast_2d(y)
                z = x[..., :, None, :] - y[..., None, :, :]
            else:
                z = self._ops.atleast_3d(z)

            signal_variance = self._ops.asarray(
                signal_variance, preferred_dtype=z.dtype
            )
            signal_variance = self._ops.atleast_1d(signal_variance)
            length_scale = self._ops.asarray(length_scale, preferred_dtype=z.dtype)
            length_scale = self._ops.atleast_2d(length_scale)

            return signal_variance[..., :, None, None] * self._ops.exp(
                -0.5
                * self._ops.sum(
                    self._ops.square(
                        z[..., None, :, :, :] / length_scale[..., :, None, None, :]
                    ),
                    axis=-1,
                )
            )


# Currently unused. Could be used to speed up repeated GP moment map evaluation
# at the expense of numerical stability. If gram_L or B_L are poorly
# conditioned then it is better to solve rather than invert.
def _inverse_effective_gram(ops, gram_L, B_L):
    """Inverse of the effective Gram matrix."""
    # gram_L: [OUT_DIM, N, N]
    # B_L: [OUT_DIM, N, N]
    eye = ops.eye(ops.shape(gram_L)[-1], dtype=gram_L.dtype)
    InvL = ops.matrix_triangular_solve(gram_L, eye)
    # InvG: [OUT_DIM, N, N]
    InvG = ops.matrix_triangular_solve(gram_L, InvL, adjoint=True)
    if B_L is not None:
        InvB_InvL = ops.cholesky_solve(B_L, InvL)
        InvL_InvB_InvL = ops.matrix_triangular_solve(gram_L, InvB_InvL, adjoint=True)
        InvG = InvG - InvL_InvB_InvL
    return InvG


class GaussianProcessMomentMap(_BaseGaussianProcessMomentMap):
    """A Gaussian process moment map."""

    def __init__(
        self,
        inducing_points,
        coefficients,
        signal_variance,
        length_scale,
        gram_L,
        B_L=None,
        **kwargs,
    ):
        super().__init__(
            inducing_points=inducing_points,
            coefficients=coefficients,
            signal_variance=signal_variance,
            length_scale=length_scale,
            gram_L=gram_L,
            B_L=B_L,
            deterministic=False,
            **kwargs,
        )

    @classmethod
    def from_params(cls, params, **kwargs):
        """Create an instance from `GaussianProcessParameters`.

        Args:
            params: A `GaussianProcessParameters` object.
            **kwargs: Additional keyword arguments passed to __init__.

        Returns:
            A `GaussianProcessMomentMap`.
        """
        return cls(
            inducing_points=params.inducing_points,
            coefficients=params.coefficients,
            signal_variance=params.signal_variance,
            length_scale=params.length_scale,
            gram_L=params.gram_L,
            B_L=params.B_L,
            **kwargs,
        )


class DeterministicGaussianProcessMomentMap(_BaseGaussianProcessMomentMap):
    """A deterministic Gaussian process moment map.

    This is a GP that has no variance; it represents a delta distribution on
    the mean.
    """

    def __init__(
        self, inducing_points, coefficients, signal_variance, length_scale, **kwargs
    ):
        super().__init__(
            inducing_points=inducing_points,
            coefficients=coefficients,
            signal_variance=signal_variance,
            length_scale=length_scale,
            gram_L=None,
            B_L=None,
            deterministic=True,
            **kwargs,
        )

    @classmethod
    def from_params(cls, params, **kwargs):
        """Create an instance from `GaussianProcessParameters`.

        Args:
            params: A `GaussianProcessParameters` object.
            **kwargs: Additional keyword arguments passed to __init__.

        Returns:
            A `DeterministicGaussianProcessMomentMap`.
        """
        return cls(
            inducing_points=params.inducing_points,
            coefficients=params.coefficients,
            signal_variance=params.signal_variance,
            length_scale=params.length_scale,
            **kwargs,
        )


def _reference_gp_predict_moments(
    input_mean,
    input_covariance,
    inducing_points,
    gram_L,
    coefficients,
    signal_variance,
    length_scale,
    return_cov=True,
    return_io_cov=True,
    asserts=True,
):
    """Reference implementation for GP moment prediction.

    This is a translation of the code in appendex E.1 of
    Deisenroth MP. Efficient reinforcement learning using Gaussian processes.
    KIT Scientific Publishing; 2010.

    Some code was removed in order to support the same set of arguments as
    _BaseGaussianProcessMomentMap.
    """
    # [N, IN_DIM]
    input_ = np.asarray(inducing_points)

    # [OUT_DIM, N]
    beta = np.atleast_2d(coefficients)

    n, D = input_.shape
    E, _ = beta.shape

    # [..., IN_DIM]
    m = np.atleast_1d(input_mean)
    # [..., IN_DIM, IN_DIM]
    s = np.atleast_2d(input_covariance)

    # logh[1:D] - [OUT_DIM, IN_DIM]
    length_scales = np.atleast_2d(length_scale)
    squared_length_scales = np.square(length_scales)
    log_length_scales = np.log(length_scales)

    # logh[D+1] - [OUT_DIM]
    signal_var = np.atleast_1d(signal_variance)
    signal_std = np.sqrt(signal_var)
    log_signal_std = np.log(signal_std)

    # [..., N, IN_DIM]
    inp = inducing_points - m[..., None, :]

    # 1) Compute predicted mean and covariance between input and prediction

    # [OUT_DIM, IN_DIM, IN_DIM]
    iLambda = utils.numpy.batch_diag(1 / squared_length_scales)
    # [..., OUT_DIM, IN_DIM, IN_DIM]
    R_1 = s[..., None, :, :] + utils.numpy.batch_diag(squared_length_scales)

    # [..., OUT_DIM, IN_DIM, IN_DIM]
    siLambda = s[..., None, :, :] / squared_length_scales[:, None, :]
    # [..., OUT_DIM, IN_DIM, IN_DIM]
    iR_1 = iLambda @ (np.eye(D) - np.linalg.solve(np.eye(D) + siLambda, siLambda))

    if asserts:
        assert np.allclose(iR_1 @ R_1, np.eye(D)) or np.allclose(
            iR_1, np.linalg.inv(R_1)
        )
        assert np.allclose(R_1 @ iR_1, np.eye(D)) or np.allclose(
            iR_1, np.linalg.inv(R_1)
        )

    # [..., OUT_DIM, N, IN_DIM]
    T = inp[..., None, :, :] @ iR_1

    # [..., OUT_DIM]
    c = (
        signal_var
        / np.sqrt(np.linalg.det(R_1))
        * np.exp(np.sum(log_length_scales, axis=-1))
    )

    # [..., OUT_DIM, N]
    # pylint: disable=invalid-unary-operand-type
    q = c[..., :, None] * np.exp(-utils.numpy.batch_inner(T, inp[..., None, :, :]) / 2)
    # pylint: enable=invalid-unary-operand-type

    # [..., OUT_DIM, N]
    qb = q * beta

    # [..., OUT_DIM]
    M = np.sum(qb, axis=-1)

    if return_io_cov:
        # [..., OUT_DIM, IN_DIM]
        V = np.squeeze(qb[..., None, :] @ T @ s[..., None, :, :], axis=-2)
    else:
        V = None

    if not return_cov:
        return M, None, V

    # [..., OUT_DIM, N, IN_DIM]
    v = inp[..., None, :, :] / length_scales[:, None, :]

    # [..., OUT_DIM, N]
    log_k = 2 * log_signal_std[:, None] - np.sum(np.square(v), axis=-1) / 2

    # 2) predictive covariance matrix (symmetric)
    # 2a) non-central moments

    # [..., OUT_DIM, N, IN_DIM]
    zeta = inp[..., None, :, :] / squared_length_scales[:, None, :]

    # [..., OUT_DIM, OUT_DIM, IN_DIM, IN_DIM]
    R_2 = s[..., None, None, :, :] * (
        1 / squared_length_scales[:, None, None, :]
        + 1 / squared_length_scales[None, :, None, :]
    ) + np.eye(D)
    # [..., OUT_DIM, OUT_DIM]
    t = 1 / np.sqrt(np.linalg.det(R_2))

    # [..., OUT_DIM, OUT_DIM, N, N]
    Q = t[..., None, None] * np.exp(
        log_k[..., :, None, :, None]
        + log_k[..., None, :, None, :]
        + _maha(
            a=zeta[..., :, None, :, :],
            b=-zeta[..., None, :, :, :],
            Q=np.linalg.solve(R_2, s[..., None, None, :, :]) / 2,
        )
    )

    # [OUT_DIM, N, N]
    L = np.asarray(gram_L)

    # [OUT_DIM, N, N]
    iK = utils.numpy.batch_cho_solve(L, np.eye(n))

    # [OUT_DIM, OUT_DIM, N, N]
    A = beta[:, None, :, None] * beta[None, :, None, :]
    A[np.arange(E), np.arange(E), :, :] -= iK

    # [..., OUT_DIM, OUT_DIM, N, N]
    A = A * Q
    # [..., OUT_DIM, OUT_DIM]
    S = np.sum(A, axis=(-2, -1))

    S[..., np.arange(E), np.arange(E)] += signal_var

    # 2b) centralize moments
    S -= M[..., :, None] * M[..., None, :]

    return M, S, V


def _maha(a, b, Q=None):
    """Squared Mahalanobis distance (a - b) @ Q @ (a - b).

    Args:
        a: Shape [..., M, K]
        b: Shape [..., N, K]
        Q: Shape [..., K, K] or None (defaults to identity)

    Returns:
        Array of shape [..., M, N]
    """
    if Q is None:
        return (
            np.sum(np.square(a), axis=-1)[..., :, None]
            + np.sum(np.square(b), axis=-1)[..., None, :]
            - 2 * a @ np.swapaxes(b, -2, -1)
        )

    aQ = a @ Q
    return (
        utils.numpy.batch_inner(aQ, a)[..., :, None]
        + utils.numpy.batch_inner(b @ Q, b)[..., None, :]
        - 2 * aQ @ np.swapaxes(b, -2, -1)
    )
