"""Core classes and methods for mapping moments through functions."""
import typing

from . import backends


class MomentMap:
    """Base class for moment map functions.

    Moment maps are callables that take in multivariate-normal distributed
    inputs and produce the mean, covariance of the output distribution, as well
    as the input-output covariance.

    Attributes:
        input_dim: Size of input dimension. May be None for indeterminant.
        output_dim: Size of output dimension.
            An integer, a function of the input dimension,
            or None to represent the identity function of the input dimension.
        backend: The backend array / tensor library.
        dtype: Data type of the input and output arrays.
            If None, no data type is imposed.
        assert_valid: Add assertions verifying the validity of the covariance
            matrices.
        valid_tolerance: Numerical tolerance on validity assertions.
    """

    # Subclasses can return this as the io_cov_inv_in_cov matrix to avoid
    # needless matrix multiplies with the identity matrix.
    _identity = object()

    def __init__(
        self,
        input_dim,
        output_dim,
        backend="numpy",
        dtype=None,
        assert_valid=False,
        valid_tolerance=1e-8,
    ):
        super().__init__()
        if output_dim is None:
            # Set to the identity function
            def output_dim(in_dim):
                return in_dim

        if input_dim is not None and callable(output_dim):
            # Immediately evaluate output_dim if possible
            output_dim = output_dim(input_dim)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.backend = backend
        self.dtype = dtype
        self.assert_valid = assert_valid
        self.valid_tolerance = valid_tolerance

        self._ops = backends.BACKENDS[self.backend]()
        self._repr_arguments = {}

    def __repr__(self):
        class_name = self.__class__.__name__
        arglist = ", ".join(
            f"{key}={value!r}" for key, value in self._repr_arguments.items()
        )
        return f"{class_name}({arglist})"

    def __call__(
        self,
        mean,
        covariance=None,
        return_cov=True,
        return_io_cov=True,
        return_io_cov_inv_in_cov=False,
    ):
        """Distribution of function values for Normal distribution inputs.

        Args:
            mean: Input mean vector. An array of shape [..., IN_DIM].
                May alternatively be a `moment_map` in which case this is instead
                an alias for `compose`.
            covariance: Input covariance matrix. Defaults to zero.
                An array of shape [..., IN_DIM, IN_DIM].
            return_cov: Return the output covariance matrix.
                If false, `output_covariance` is None.
            return_io_cov: Return the output-input covariance matrix.
                If false, `output_input_covariance` is None.
            return_io_cov_inv_in_cov: Return the output-input covariance matrix
                times inverse input covariance matrix.
                If false, `io_cov_inv_in_cov` is None.

        Supports broadcasting of the inputs.

        Returns:
                A MomentMapDistribution NamedTuple, containing
            output_mean:
            output_covariance:
            output_input_covariance:
            io_cov_inv_in_cov:

        Overload: This is an alias for `compose` if `mean` is a moment_map.
        """
        if isinstance(mean, MomentMap):
            return self.compose(mean)

        with self._ops.name_scope(type(self).__name__):
            mean = self._ops.asarray(mean, dtype=self.dtype, name="mean")
            mean = self._ops.atleast_1d(mean)

            input_dim = self.input_dim_value(mean)
            output_dim = self.output_dim_value(input_dim)

            if covariance is not None:
                covariance = self._ops.asarray(
                    covariance, dtype=self.dtype, name="covariance"
                )
                covariance = self._ops.atleast_2d(covariance)
                covariance = self._with_assert_valid_covariance(covariance)

            return_any_io_cov = return_io_cov or return_io_cov_inv_in_cov
            output_mean, output_covariance, io_cov_inv_in_cov = self._call(
                mean=mean,
                covariance=covariance,
                return_cov=return_cov,
                return_io_cov_inv_in_cov=return_any_io_cov,
            )

            assert output_mean.shape[-1:] == (output_dim,)
            if return_cov:
                assert output_covariance.shape[-2:] == (output_dim, output_dim)
                output_covariance = self._with_assert_valid_covariance(
                    output_covariance
                )

            if return_any_io_cov:
                assert (io_cov_inv_in_cov is self._identity) or (
                    io_cov_inv_in_cov.shape[-2:] == (output_dim, input_dim)
                )

            if return_io_cov:
                if covariance is None:
                    output_input_covariance = self._ops.zeros(
                        [output_dim, input_dim], dtype=output_mean.dtype
                    )
                elif io_cov_inv_in_cov is self._identity:
                    assert input_dim == output_dim
                    output_input_covariance = covariance
                else:
                    output_input_covariance = self._ops.matmul(
                        io_cov_inv_in_cov, covariance
                    )
            else:
                output_input_covariance = None

            if not return_io_cov_inv_in_cov:
                io_cov_inv_in_cov = None
            elif io_cov_inv_in_cov is self._identity:
                io_cov_inv_in_cov = self._ops.eye(output_dim, dtype=output_mean.dtype)

            return MomentMapDistribution(
                output_mean=output_mean,
                output_covariance=output_covariance,
                output_input_covariance=output_input_covariance,
                io_cov_inv_in_cov=io_cov_inv_in_cov,
            )

    def __getitem__(self, index):
        return IndexMomentMap(
            index,
            backend=self.backend,
            dtype=self.dtype,
            assert_valid=self.assert_valid,
            valid_tolerance=self.valid_tolerance,
        )(self)

    def _call(self, mean, covariance, return_cov, return_io_cov_inv_in_cov):
        """Distribution of function values for Normal distribution inputs.

        Name scope has been created and inputs have the correct type and inner
        shape.

        Args:
            mean: Input distribution mean.
                An array of shape [..., IN_DIM].
            covariance: Input distribution covariance matrix.
                May be `None` to represent the all-zeros covariance matrix.
                An array of shape [..., IN_DIM, IN_DIM].
            return_cov: Return the output covariance matrix.
                If false, `output_covariance` is None.
            return_io_cov_inv_in_cov: Return the input-output covariance matrix
                times inverse input covariance matrix.
                If false, `io_cov_inv_in_cov` is None.

        Supports broadcasting of the inputs.

        Returns:
            output_mean: Output mean vector. An array of shape [..., OUT_DIM].
            output_covariance: Output covariance matrix.
                An array of shape [..., OUT_DIM, OUT_DIM].
            io_cov_inv_in_cov:
                Output-input covariance matrix times inverse input covariance.
                May be `self._identity` if equal to the identity matrix.
                An array of shape [..., OUT_DIM, IN_DIM].
        """
        raise NotImplementedError

    def compose(self, other):
        """Create a new MomentMap representing `self(other(x))`.

        Args:
            A `MomentMap` instance.

        Returns:
            A `ComposedMomentMap` instance representing self(other(x))
        """
        # Inherit configuration parameters from self
        return ComposedMomentMap(
            inner=other,
            outer=self,
            backend=self.backend,
            dtype=self.dtype,
            assert_valid=self.assert_valid,
            valid_tolerance=self.valid_tolerance,
        )

    def _with_assert_valid_covariance(self, covariance):
        if not self.assert_valid:
            return covariance
        with self._ops.name_scope("assert_valid_covariance"):
            # Do in steps for easier debugging
            covariance = self._ops.with_assert(
                covariance,
                self._ops.allclose(
                    covariance,
                    self._ops.matrix_transpose(covariance),
                    atol=self.valid_tolerance,
                ),
                [covariance],
                name="assert_symmetric",
            )
            eigenvalues = self._ops.self_adjoint_eigvals(covariance)
            covariance = self._ops.with_assert(
                covariance,
                self._ops.all(eigenvalues >= -self.valid_tolerance),
                [covariance, eigenvalues],
                name="assert_nonneg_eigenvalues",
            )
            return covariance

    def input_dim_value(self, input_vector):
        """The value of the input dimension given a specific input vector."""
        if self.input_dim is not None:
            return self.input_dim
        return int(input_vector.shape[-1])

    def output_dim_value(self, input_dim):
        """The output dimension for a specific input dimension."""
        if self.input_dim is not None and self.input_dim != input_dim:
            raise ValueError(
                "Given input_dim ({input_dim}) does not match "
                "fixed self.input_dim ({self.input_dim})."
            )
        if callable(self.output_dim):
            return self.output_dim(input_dim)
        return self.output_dim

    def _zero_covariance_for(self, mean):
        """Create an all-zero covariance matrix for the given means vector."""
        dim = int(mean.shape[-1])
        return self._ops.zeros((dim, dim), dtype=mean.dtype)


class MomentMapDistribution(typing.NamedTuple):
    """Distribution produced by a MomentMap.

    Args:
        output_mean: Mean of the output distribution.
            An array of shape [..., OUT_DIM]
        output_covariance: Covariance of the output distribution.
            An array of shape [..., OUT_DIM, OUT_DIM]
        output_input_covariance: Output-input covariance matrix.
            An array of shape [..., OUT_DIM, IN_DIM]
        io_cov_inv_in_cov: Output-input covariance matrix times inverse
            input covariance matrix.
            An array of shape [..., OUT_DIM, IN_DIM]
    """

    output_mean: typing.Any
    output_covariance: typing.Any = None
    output_input_covariance: typing.Any = None
    io_cov_inv_in_cov: typing.Any = None


class ComposedMomentMap(MomentMap):
    """Create a new moment map f(g(x)) out of moment maps f and g.

    This proceeds by:
    1. Calculate the moments mu_g, sigma_g of g(x) for x ~ N(mu_x, sigma_x)
    2. Calculate the moments mu_f, sigma_f of g(y) for y ~ N(mu_g, sigma_g)
    3. Calculate the cross-covariance of x and g(y) by marginalizing out y.

    Since moment matching is performed after g and before f, the resulting
    distribution is an inexact approximation to the true moments of f(g(x)).

    Attributes:
        inner: An instance of `MomentMap` applied to the input.
        outer: An instance of `MomentMap` applied to the output of `inner`.

    Inner is `g` and outer is `f`.
    """

    def __init__(self, inner, outer, backend=None, **kwargs):
        if backend is None:
            backend = inner.backend
            if backend != outer.backend:
                raise ValueError(
                    "Inconsistent backends:"
                    f" outer.backend={outer.backend!r},"
                    f" inner.backend={inner.backend!r}"
                )
        elif (backend != outer.backend) or (backend != inner.backend):
            raise ValueError(
                "Inconsistent backends:"
                f" backend={backend!r},"
                f" outer.backend={outer.backend!r},"
                f" inner.backend={inner.backend!r}"
            )

        if (
            isinstance(inner.output_dim, int)
            and isinstance(outer.input_dim, int)
            and inner.output_dim != outer.input_dim
        ):
            raise ValueError(
                f"Output dimension of inner ({inner.output_dim}) "
                f"does not match input dimension of outer ({outer.input_dim})."
            )

        self.inner = inner
        self.outer = outer

        outer_output_dim = self.outer.output_dim
        inner_output_dim = self.inner.output_dim
        if callable(outer_output_dim):
            try:
                output_dim = outer_output_dim(inner_output_dim)
            except TypeError:

                def output_dim(in_dim):
                    return outer_output_dim(inner_output_dim(in_dim))

        else:
            output_dim = outer_output_dim

        super().__init__(
            input_dim=inner.input_dim, output_dim=output_dim, backend=backend, **kwargs
        )
        self._repr_arguments["outer"] = self.outer
        self._repr_arguments["inner"] = self.inner

    def _call(self, mean, covariance, return_cov, return_io_cov_inv_in_cov):
        intermediate = self.inner(
            mean=mean,
            covariance=covariance,
            return_cov=True,
            return_io_cov=False,
            return_io_cov_inv_in_cov=return_io_cov_inv_in_cov,
        )

        final = self.outer(
            mean=intermediate.output_mean,
            covariance=intermediate.output_covariance,
            return_cov=return_cov,
            return_io_cov=False,
            return_io_cov_inv_in_cov=return_io_cov_inv_in_cov,
        )

        if return_io_cov_inv_in_cov:
            # y = inner(x); z = outer(y)
            # x is conditionally independent of z given y
            # so
            #   cov_zx = cov_zy @ inv(cov_yy) @ cov_yx
            #          = cov_zy_icov_yy @ cov_yx
            #   cov_zx_icov_xx = cov_zy_icov_yy @ cov_yx_icov_xx

            io_cov_inv_in_cov = self._ops.matmul(
                final.io_cov_inv_in_cov, intermediate.io_cov_inv_in_cov
            )
        else:
            io_cov_inv_in_cov = None

        return final.output_mean, final.output_covariance, io_cov_inv_in_cov


class JointInputOutputMomentMap(MomentMap):
    """Create a MomentMap x -> (x, f(x)) out of moment map f.

    Augments the output space of a moment map by prepending the input space.

    Attributes:
        moment_map: The wrapped moment map.
    """

    def __init__(self, moment_map, backend=None, **kwargs):
        if backend is None:
            backend = moment_map.backend
        elif backend != moment_map.backend:
            raise ValueError(
                f"Given backend {backend!r} does not match "
                f"wrapped backend {moment_map.backend!r}."
            )

        if moment_map.input_dim is None:

            def output_dim(in_dim):
                return in_dim + moment_map.output_dim_value(in_dim)

        else:
            output_dim = moment_map.input_dim + moment_map.output_dim_value(
                moment_map.input_dim
            )

        super().__init__(
            input_dim=moment_map.input_dim,
            output_dim=output_dim,
            backend=backend,
            **kwargs,
        )
        self.moment_map = moment_map
        self._repr_arguments["moment_map"] = self.moment_map

    def _call(self, mean, covariance, return_cov, return_io_cov_inv_in_cov):
        output = self.moment_map(
            mean=mean,
            covariance=covariance,
            return_cov=return_cov,
            return_io_cov=return_cov,
            return_io_cov_inv_in_cov=return_io_cov_inv_in_cov,
        )
        with self._ops.name_scope("mean"):
            joint_mean = self._ops.concatenate([mean, output.output_mean], axis=-1)

        input_dim = self.input_dim_value(mean)

        if return_cov:
            with self._ops.name_scope("covariance"):
                if covariance is None:
                    covariance = self._zero_covariance_for(mean)

                # short names for clearer concatenation
                c_xx = covariance
                c_yx = output.output_input_covariance
                c_xy = self._ops.matrix_transpose(c_yx)
                c_yy = output.output_covariance
                joint_covariance = self._ops.concatenate(
                    [
                        self._ops.concatenate([c_xx, c_xy], axis=-1),
                        self._ops.concatenate([c_yx, c_yy], axis=-1),
                    ],
                    axis=-2,
                )
        else:
            joint_covariance = None

        if return_io_cov_inv_in_cov:
            with self._ops.name_scope("io_cov_inv_in_cov"):
                x_cov_inv_x_cov = self._ops.eye(
                    input_dim, dtype=output.io_cov_inv_in_cov.dtype
                )
                joint_io_cov_inv_in_cov = self._ops.concatenate(
                    [x_cov_inv_x_cov, output.io_cov_inv_in_cov], axis=-2
                )
        else:
            joint_io_cov_inv_in_cov = None
        return joint_mean, joint_covariance, joint_io_cov_inv_in_cov


class IndexMomentMap(MomentMap):
    """Returns a subset of the input dimensions by indexing."""

    def __init__(self, index, **kwargs):
        """Initialize an IndexMomentMap.

        Args:
            index: The index into the input dimension.
                Can be any index type supported by both numpy and the backend.
                Integers and slice objects are recommended.
        """
        # Output must be a vector not a scalar
        # so convert integer indices to size-1 lists.
        if isinstance(index, int):
            if index == -1:
                index = slice(-1, None)
            else:
                index = slice(index, index + 1)

        def output_dim(in_dim):
            # Try range() first because it is O(1)
            try:
                indexed_range = range(in_dim)[index]
            except TypeError:  # Not vanilla index (slice)
                pass
            else:
                return len(indexed_range)

            # Try numpy empty to get advanced indices but is O(in_dim) (I think)
            import numpy as np

            return len(np.empty(in_dim)[index])

        super().__init__(input_dim=None, output_dim=output_dim, **kwargs)
        self.index = index
        self._repr_arguments["index"] = index

    def _call(self, mean, covariance, return_cov, return_io_cov_inv_in_cov):
        output_mean = mean[..., self.index]

        if return_cov:
            if covariance is None:
                output_covariance = self._zero_covariance_for(output_mean)
            else:
                output_covariance = covariance[..., self.index, self.index]
        else:
            output_covariance = None

        if return_io_cov_inv_in_cov:
            input_dim = self.input_dim_value(mean)
            io_cov_inv_in_cov = self._ops.eye(input_dim, dtype=mean.dtype)[
                self.index, :
            ]
        else:
            io_cov_inv_in_cov = None

        return output_mean, output_covariance, io_cov_inv_in_cov


class AddUncorrelatedMomentMap(MomentMap):
    """Sum the outputs of a series of moment maps on the same inputs.

    It is assumed that the moment maps are uncorrelated from each other.
    The output covariance will be incorrect if this is not the case.
    """

    def __init__(self, moment_maps, **kwargs):
        """Initialize AddMomentMap.

        Args:
            moment_maps: A list of moment maps whose outputs to add together.
        """
        if not moment_maps:
            raise ValueError("Must specify at least one moment map.")

        input_dims = {mm.input_dim for mm in moment_maps}
        try:
            input_dims.remove(None)
        except KeyError:
            pass
        try:
            input_dim, = input_dims
        except ValueError:
            if not input_dims:
                # No elements other than None
                input_dim = None
            else:
                raise ValueError(
                    f"Inconsistent moment map input dimensions: {input_dims}"
                )

        if input_dim is None:
            # Cannot reasonably compare output_dims unless input_dim is set.
            # We'll let any inconsistencies arise in _call.
            output_dim = moment_maps[0].output_dim
        else:
            output_dims = {
                mm.output_dim(input_dim) if callable(mm.output_dim) else mm.output_dim
                for mm in moment_maps
            }
            try:
                output_dim, = output_dims
            except ValueError:
                raise ValueError(
                    f"Inconsistent moment map output dimensions: {output_dims}"
                )
        super().__init__(input_dim=input_dim, output_dim=output_dim, **kwargs)
        self.moment_maps = list(moment_maps)
        self._repr_arguments["moment_maps"] = self.moment_maps

    def _call(self, mean, covariance, return_cov, return_io_cov_inv_in_cov):
        results = [
            mm(
                mean,
                covariance,
                return_cov=return_cov,
                return_io_cov_inv_in_cov=return_io_cov_inv_in_cov,
            )
            for mm in self.moment_maps
        ]

        output_mean = sum(result.output_mean for result in results)

        if return_cov:
            # TODO: Maybe change to AddMomentMap and raise an exception here by default?
            # Could add an "uncorrelated" argument to __init__?
            # Not sure if that's actually OK because whether the outputs are correlated
            # could depend on the input covariance.
            output_covariance = sum(result.output_covariance for result in results)
        else:
            output_covariance = None

        if return_io_cov_inv_in_cov:
            io_cov_inv_in_cov = sum(result.io_cov_inv_in_cov for result in results)
        else:
            io_cov_inv_in_cov = None

        return output_mean, output_covariance, io_cov_inv_in_cov
