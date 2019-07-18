"""Numpy array manipulation utilities."""
# Copyright © 2019 Eric Langlois
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import functools
import itertools

import scipy.linalg

import numpy as np

#################
# Vectorization #
#################


def batch_apply(fn, x, in_dims=1, out=None):
    """Batch apply some function over an array-like.

    Args:
        fn: A function that maps arrays of shape [D_1, ..., D_N] to arrays with
            a consistent output shape [C_1, ..., C_K].
        x: An array of shape [B_1, ..., B_M, D_1, ..., D_N]
        in_dims: The number of input dimensions to `fn`.
            Equal to `N` in the array shape expressions.
        out: An optional output array of shape [B_1, ..., B_M, C_1, ..., C_K]
            into which the output is placed. This will be faster as it avoids
            concatenation across the batch dimensions.

    Returns:
        The batched output. An array of shape [B_1, ..., B_M, C_1, ..., C_K]
    """
    x = np.asarray(x)
    if in_dims == 0:
        batch_shape = x.shape
    else:
        batch_shape = x.shape[:-in_dims]

    if out is None:
        out = np.empty(batch_shape, dtype=object)
        merge_output = True
    else:
        merge_output = False

    for idx in np.ndindex(batch_shape):
        out[idx] = fn(x[idx])

    if merge_output:
        # Out consists of an array of size [B_1, ..., B_M] whose elements are arrays of
        # shape [C_1, ..., C_K]. Want to combine it into an array of shape
        # [B_1, ..., B_M, C_1, ..., C_K]
        out = np.array(out.tolist())
    return out


########
# Math #
########


def softmax(a, axis=None, out=None):
    """Return the softmax of an array along an axis.

    Args:
        a: Input array
        axis: Axis or axes along which to operate.
            By default, flattened input is used.
        out: Optional output array in which to place the result.
            Must have the same shape as the expected output.

    Returns:
        The softmax of `a`. The result is an array with the same shape as `a`.
    """
    intermediate = np.subtract(a, np.amax(a, axis=axis, keepdims=True), out=out)
    try:
        np.exp(intermediate, out=intermediate)
    except TypeError:
        if intermediate is not out:
            intermediate = np.exp(intermediate)
        else:
            raise
    out = intermediate
    np.divide(out, np.sum(out, axis=axis, keepdims=True), out=out)
    return out


##################
# Linear Algebra #
##################


def batch_inner(a, b, out=None):
    """Batch inner product along the last dimension of `a` and `b`.

    Supports broadcasting.

    Args:
        a: An array of shape [B1, ..., BM, N]
        b: An array of shape [B1, ..., BM, N]
        out: Optional output array in which to place the result.
            Must have the same shape as the expected output.

    Returns:
        An array of shape [B1, ..., BM]
        where result[idx] = sum(a[idx] * b[idx])
    """
    # Einsum doesn't accept out=None
    kwargs = {}
    if out is not None:
        kwargs["out"] = out

    return np.einsum("...i,...i->...", a, b, **kwargs)


def batch_solve(a, b, sym_pos=False, triangular=False, lower=False, adjoint=False):
    """Batch solve the linear equations Ax = B.

    Args:
        a: array-like, shape = (..., m, m)
        b: array-like, shape = (..., m, k)
        sym_pos: Assume `a` is symmetric and positive definite.
        triangular: Assume `a` is triangular.
        lower: Use only data contained in the lower triangle of `a`, if
            `sym_pos` or `triangular` is true.
            Default is to use upper triangle.
        adjoint: If True, solve the system adjoint(A)x = B instead.
            May be used only when `triangular` is True.

    Returns:
        x: array-like, shape = (..., m, k)
            Solution to Ax = B
    """
    a, b = outer_broadcast_arrays((a, 2), (b, 2))
    if triangular:
        solver = functools.partial(
            scipy.linalg.solve_triangular, trans="C" if adjoint else "N"
        )
    elif adjoint:
        raise ValueError("Adjoint may only be used when triangular=True")
    else:
        solver = functools.partial(scipy.linalg.solve, sym_pos=sym_pos)
    x = np.empty_like(b)
    for idx in np.ndindex(a.shape[:-2]):
        x[idx + (slice(None), slice(None))] = solver(
            a=a[idx + (slice(None), slice(None))],
            b=b[idx + (slice(None), slice(None))],
            lower=lower,
        )
    return x


def batch_cho_solve(c, b, lower=True):
    """Batch solve the linear equations Ax = B given Cholesky factorization.

    Args:
        c: array-like, shape = (..., m, m)
            The Cholesky factorization of A.
        b: array-like, shape = (..., m, k)
            The right-hand size matrix.
        lower: `True` if `c` is lower triangular and `False` if `c` is upper
            triangular.

    Returns:
        x: array-like, shape = (..., m, k)
            Solution to Ax = B
    """
    c, b = outer_broadcast_arrays((c, 2), (b, 2))
    x = np.empty_like(b)
    for idx in np.ndindex(c.shape[:-2]):
        # pylint: disable=no-member
        x[idx + (slice(None), slice(None))] = scipy.linalg.cho_solve(
            (c[idx + (slice(None), slice(None))], lower),
            b[idx + (slice(None), slice(None))],
        )
    return x


###############
# Array Shape #
###############


def batch_diag(x, size=None, out=None):
    """Batched diagonal matrices from the last dimension of `x`.

    Args:
        x: Array to diagonalize. An array of shape `[..., size]`
        size: Size of the resulting diagonal array if `out` is not provided.
            Defaults to the size of the last dimension of `x`.
        out: Optional output array in which to place the result.
            Must have the same shape as the expected output.

    Returns:
        An array of shape `[..., size, size]` with the diagonal along the last
        two dimensions set to `x` and `0` elsewhere.
    """
    x = np.asarray(x)
    if out is None:
        if size is None:
            try:
                size = x.shape[-1]
            except IndexError as e:
                raise ValueError("Cannot infer size from scalar") from e
        diag = np.zeros(x.shape[:-1] + (size, size))
    else:
        diag = out
        out.fill(0)
        size = diag.shape[-1]
        if diag.shape[-2] != size:
            raise ValueError("The last two dimensions of out are not equal.")
    diag[..., np.arange(size), np.arange(size)] = x
    return diag


################
# Broadcasting #
################


def broadcast_shapes(a, b):
    """Broadcast two shapes together.

    Returns:
        The resulting broadcasted shape as a tuple.

    Raises:
        ValueError: If `a` and `b` cannot be broadcasted together.
    """
    c = []
    for ai, bi in itertools.zip_longest(reversed(a), reversed(b), fillvalue=1):
        if ai != bi and ai != 1 and bi != 1:
            raise ValueError(f"Unable to broadcast shapes {a} and {b}")
        c.append(max(ai, bi))
    return tuple(reversed(c))


def outer_broadcast_arrays(*args):
    """Broadcast arrays ignoring inner dimensions.

    Args:
        *args: A list of `(array, suffix_size)` tuples.
            All but the last `suffix_size` dimensions are broadcasted.

    Returns:
        *arrays: A list of broadcasted arrays.
    """
    batch_shape = ()
    for array, suffix_size in args:
        batch_shape = broadcast_shapes(batch_shape, array.shape[:-suffix_size])

    return tuple(
        np.broadcast_to(array, batch_shape + array.shape[-suffix_size:])
        for array, suffix_size in args
    )


##################
# Joint Indexing #
##################


def joint_shuffle(*arrays, random_state=None):
    """Randomly shuffle the first dimension of multiple arrays with the same reordering.

    Does not modify the arrays in-place.

    Args:
        *arrays: A set of arrays to shuffle.
        axis: The axis along which to shuffle.
        random_state: An optional random state object used to generate the ordering.

    Returns:
        The same set of arrays but randomly shuffled along the given axis.
    """
    if not arrays:
        return arrays
    length = arrays[0].shape[0]
    if random_state is None:
        random_state = np.random
    indices = random_state.permutation(length)
    return tuple(array[indices] for array in arrays)
