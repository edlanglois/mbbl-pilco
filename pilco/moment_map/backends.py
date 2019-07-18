"""Backends for moment maps."""
import contextlib
import functools

import numpy as np
import scipy.stats

from pilco import utils


def _numpy_reduce_wrapper(f):
    @functools.wraps(f)
    def wrapper(*args, keep_dims=False, **kwargs):
        return f(*args, keepdims=keep_dims, **kwargs)

    return wrapper


class _NumpyOps:
    """Numpy operations required by moment maps."""

    data_is_mutable = True
    has_gradients = False

    @staticmethod
    @contextlib.contextmanager
    def name_scope(name, default_name=None, values=None):
        """No-op name scope."""
        del name, default_name, values  # Unused
        yield

    shape = np.shape

    @staticmethod
    def asarray(value, dtype=None, name=None, preferred_dtype=None):
        """Convert the input to an array."""
        del name, preferred_dtype
        return np.asarray(value, dtype=dtype)

    atleast_1d = np.atleast_1d
    atleast_2d = np.atleast_2d

    @staticmethod
    def atleast_3d(array):
        # Different than numpy.atleast_3d. Prepends instead of appends.
        if array.ndim >= 3:
            return array
        return np.reshape(array, (1,) * (3 - array.ndim) + array.shape)

    squeeze = np.squeeze
    expand_dims = np.expand_dims
    swapaxes = np.swapaxes
    broadcast_to = np.broadcast_to

    @staticmethod
    def broadcast_axis_to(a, axis, size):
        """Broadcast a single axis to a target size."""
        shape = list(a.shape)
        shape[axis] = size
        return np.broadcast_to(a, shape)

    eye = np.eye
    zeros = np.zeros
    concatenate = np.concatenate

    minimum = np.minimum
    maximum = np.maximum
    clip = np.clip

    sum = _numpy_reduce_wrapper(np.sum)
    all = _numpy_reduce_wrapper(np.all)

    allclose = np.allclose

    square = np.square
    sqrt = np.sqrt
    exp = np.exp
    log = np.log
    sin = np.sin
    cos = np.cos
    reciprocal = np.reciprocal

    normal_pdf = scipy.stats.norm.pdf
    normal_cdf = scipy.stats.norm.cdf

    matmul = np.matmul
    trace = functools.partial(np.trace, axis1=-2, axis2=-1)

    diagonal_matrix = utils.numpy.batch_diag
    matrix_diag_part = functools.partial(np.diagonal, axis1=-2, axis2=-1)

    @staticmethod
    def diag_part_43(a, size=None):
        """Diagonal along 4th last and 3rd last dimensions of a.

        Args:
            a: Take diagonal of this array.
            size: Take the first `size` diagonal elements.
        """
        if size is None:
            size = a.shape[-4]
            if size != a.shape[-3]:
                raise ValueError(
                    f"Array with shape {a.shape} is not square along "
                    "4th & 3rd last dimensions."
                )
        range_ = np.arange(size)
        return a[..., range_, range_, :, :]

    matrix_transpose = functools.partial(np.swapaxes, axis1=-2, axis2=-1)
    matrix_inverse = np.linalg.inv
    matrix_solve = np.linalg.solve

    @staticmethod
    def matrix_triangular_solve(matrix, rhs, lower=True, adjoint=False):
        return utils.numpy.batch_solve(
            a=matrix, b=rhs, triangular=True, lower=lower, adjoint=adjoint
        )

    cholesky = np.linalg.cholesky
    cholesky_solve = utils.numpy.batch_cho_solve
    self_adjoint_eig = np.linalg.eigh
    self_adjoint_eigvals = np.linalg.eigvalsh

    @staticmethod
    def vector_dot(a, b):
        """Broadcasting vector dot product along last dimension."""
        return np.einsum("...i,...i->...", a, b)

    @staticmethod
    def logdet(a):
        """Log determiniant of positive definite matrices."""
        sign, logdet = np.linalg.slogdet(a)
        if np.any(sign <= 0):
            raise ValueError("Matrix is not positive definite.")
        return logdet

    @staticmethod
    @contextlib.contextmanager
    def assert_context(condition, data=None, name=None):
        """Create a context in which `condition` is asserted to hold.

        Args:
            condition: A boolean scalar.
            data: Optional list of data to print when the condition is false.
            name: A name for this operation (optional).
        """
        del name
        if data is not None:
            assert condition, data
        else:
            assert condition
        yield

    @staticmethod
    def with_assert(a, condition, data=None, name=None):
        """Attach an assertion as a dependency of an array.

        Returns:
            The array with assertion dependency.
        """
        del name
        if data is not None:
            assert condition, data
        else:
            assert condition
        return a

    @staticmethod
    def with_print(a, message, data=None):
        """Print contents of array."""
        if data is None:
            data = [a]
        print(message, *data)
        return a

    @staticmethod
    def stop_gradient(input):  # pylint: disable=redefined-builtin
        return input


_TensorFlowOpsCache = None


def get_tensorflow_ops():
    """Get or create a class of TensorFlow ops for moment maps."""
    # The point of this is to avoid importing tensorflow if not using
    # TensorFlowOps

    global _TensorFlowOpsCache
    if _TensorFlowOpsCache is not None:
        return _TensorFlowOpsCache

    # Lazy load of tensorflow
    import tensorflow as tf
    import tensorflow_probability as tfp
    import pilco.utils.tf as tf_utils

    class _TensorFlowOps:
        """TensorFlow operations required by moment maps."""

        data_is_mutable = False
        has_gradients = True

        name_scope = tf.name_scope

        shape = tf.shape

        asarray = tf.convert_to_tensor
        atleast_1d = functools.partial(tf_utils.atleast_nd, size=1)
        atleast_2d = functools.partial(tf_utils.atleast_nd, size=2)
        atleast_3d = functools.partial(tf_utils.atleast_nd, size=3)
        squeeze = tf.squeeze
        expand_dims = tf.expand_dims
        swapaxes = tf_utils.swapaxes
        broadcast_to = tf.broadcast_to
        broadcast_axis_to = tf_utils.broadcast_axis_to

        eye = tf.eye
        zeros = tf.zeros
        concatenate = tf.concat

        minimum = tf.minimum
        maximum = tf.maximum
        clip = tf.clip_by_value

        sum = tf.reduce_sum
        all = tf.reduce_all

        @staticmethod
        def allclose(a, b, rtol=1e-05, atol=1e-8):
            return tf.reduce_all(tf.abs(a - b) <= (atol + rtol * tf.abs(b)))

        square = tf.square
        sqrt = tf.sqrt
        exp = tf.exp
        log = tf.log
        sin = tf.sin
        cos = tf.cos
        reciprocal = tf.reciprocal

        @staticmethod
        def normal_pdf(x, loc, scale):
            return tfp.distributions.Normal(loc=loc, scale=scale).prob(x)

        @staticmethod
        def normal_cdf(x, loc, scale):
            return tfp.distributions.Normal(loc=loc, scale=scale).cdf(x)

        matmul = tf_utils.broadcast_matmul
        trace = tf.trace

        @staticmethod
        def diagonal_matrix(diagonal, size=None, name=None):
            with tf.name_scope(name, "diagonal_matrix", [diagonal]):
                diagonal = tf.convert_to_tensor(diagonal)
                if size is not None:
                    diagonal = _TensorFlowOps.broadcast_axis_to(
                        diagonal, axis=-1, size=size
                    )
                return tf.matrix_diag(diagonal)

        matrix_diag_part = tf.matrix_diag_part

        @staticmethod
        def diag_part_43(a, size=None):
            with tf.name_scope("diag_part_43"):
                a = tf.convert_to_tensor(a)
                if size is None:
                    try:
                        size_4 = int(a.shape[-4])
                        size_3 = int(a.shape[-3])
                    except TypeError:
                        shape = tf.shape(a)
                        with tf.control_dependencies(
                            [tf.assert_equal(shape[-4], shape[-3])]
                        ):
                            size = shape[-4]
                    else:
                        if size_4 != size_3:
                            raise ValueError(
                                f"Tensor {a} is not square along dimensions "
                                "-4 and -3."
                            )
                        size = size_4

                return tf_utils.gather_dims(
                    a, np.tile(np.arange(size)[:, None], (1, 2)), first_axis=-4
                )

        matrix_transpose = tf.matrix_transpose
        matrix_inverse = tf.matrix_inverse

        @staticmethod
        def matrix_solve(matrix, rhs, adjoint=False, name=None):
            """Broadcasting batch matrix solve."""
            try:
                return tf.matrix_solve(matrix, rhs, adjoint=adjoint, name=name)
            except ValueError:
                matrix, rhs = tf_utils.broadcast_outer_dims((matrix, 2), (rhs, 2))
                return tf.matrix_solve(matrix, rhs, adjoint=adjoint, name=name)

        @staticmethod
        def matrix_triangular_solve(matrix, rhs, lower=True, adjoint=False, name=None):
            try:
                return tf.matrix_triangular_solve(
                    matrix, rhs, lower=lower, adjoint=adjoint, name=name
                )
            except ValueError:
                matrix, rhs = tf_utils.broadcast_outer_dims((matrix, 2), (rhs, 2))
                return tf.matrix_triangular_solve(
                    matrix, rhs, lower=lower, adjoint=adjoint, name=name
                )

        cholesky = tf.cholesky

        @staticmethod
        def cholesky_solve(chol, rhs, name=None):
            """Broadcasting batch cholesky solve."""
            try:
                return tf.cholesky_solve(chol, rhs, name=name)
            except ValueError:
                chol, rhs = tf_utils.broadcast_outer_dims((chol, 2), (rhs, 2))
                return tf.cholesky_solve(chol, rhs, name=name)

        self_adjoint_eig = tf.self_adjoint_eig
        self_adjoint_eigvals = tf.self_adjoint_eigvals

        @staticmethod
        def vector_dot(a, b):
            """Broadcasting vector dot product along last dimension."""
            return tf.reduce_sum(a * b, axis=-1)

        logdet = tf.linalg.logdet

        @staticmethod
        @contextlib.contextmanager
        def assert_context(condition, data=None, name=None):
            """Create a context in which `condition` is asserted to hold.

            Args:
                condition: A boolean scalar tensor.
                data: Optional list of tensors to print when the condition is
                    false. Defaults to `condition`.
            """
            if data is None:
                data = (condition,)
            if condition in (True, False):
                # Allow immediate evaluation with with static conditions
                if data is None:
                    assert condition
                else:
                    assert condition, data
                yield
            else:
                assert_op = tf.Assert(condition, data, summarize=100, name=name)
                with tf.control_dependencies([assert_op]):
                    yield

        @staticmethod
        def with_assert(a, condition, data=None, name=None):
            """Attach an assertion as a dependency of a tensor.

            Returns:
                The tensor with assertion dependency.
            """
            with _TensorFlowOps.assert_context(condition, data=data, name=name):
                return tf.identity(a)

        @staticmethod
        def with_print(a, message, data=None):
            """Attach a print as a dependency of a tensor.

            Args:
                a: Attach dependency to this tensor.
                message: A message string to display before printing the tensor
                    contents.
                data: A list of tensors to print. Defaults to [a].

            Returns:
                The tensor with print dependency.
            """
            if data is None:
                data = [a]
            return tf.Print(a, data, message, summarize=100)

        stop_gradient = tf.stop_gradient

    _TensorFlowOpsCache = _TensorFlowOps
    return _TensorFlowOpsCache


BACKENDS = {"numpy": lambda: _NumpyOps, "tensorflow": get_tensorflow_ops}
