"""TensorFlow Utilities."""
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

import numpy as np
import tensorflow as tf

__all__ = [
    "batch_gather",
    "gather_dims",
    "atleast_nd",
    "broadcast_axis_to",
    "broadcast_ranks",
    "broadcast",
    "flatten_dims",
    "swapaxes",
    "broadcast_matmul",
]


def batch_gather(params, indices, name=None):
    """Gather with batched indices.

    Args:
        params: The tensor or array from which to gather values.
            Shape `[B1, ..., BN, A, C1, ..., CM]`.
        indices: Batched index tensor or array. Elements are indices into
            the `N+1`th dimension of `params`. Shape `[B1, ..., BN, D]`.
            Must have known rank.
        name: Name for the operation (optional).

    Returns:
        A `Tensor` of shape `[B1, ..., BN, D, C1, ..., CM]` where
        `result[b1, ..., bN, i, c1, ..., cM] =
            params[b1, ..., bN, indices[b1, ..., bN, i], c1, ..., cM]`.


    Note: Does not yet support broadcasting
    """
    with tf.name_scope(name, "batch_gather", [params, indices]):
        params = tf.convert_to_tensor(params)
        indices = tf.convert_to_tensor(indices, preferred_dtype=tf.int32)

        axis = indices.shape.ndims - 1

        with tf.name_scope("nd_indices"):
            if axis > 0:
                indices_shape = tf.shape(indices)
                if indices_shape.dtype != indices.dtype:
                    indices_shape = tf.cast(indices_shape, indices.dtype)
                # Need each index array to have shape [B1, ..., BN, D]
                # nd_indices are constructed as
                # [b1, ..., bn, indices[b1, ..., bn, d]]
                # so don't need index array for D, discard it (was just
                # included for the shape of the other index arrays).
                batch_index_arrays = tf.meshgrid(
                    *[
                        tf.range(indices_shape[i], dtype=indices.dtype)
                        for i in range(axis + 1)
                    ],
                    indexing="ij",
                )[:-1]

                # Prepend the indices with indices into the batch dimensions.
                nd_indices = tf.stack(batch_index_arrays + [indices], axis=-1)
            else:
                nd_indices = tf.expand_dims(indices, -1)

        gathered = tf.gather_nd(params, nd_indices)
        gathered.set_shape(indices.shape.concatenate(params.shape[axis + 1 :]))
        return gathered


def gather_dims(params, indices, first_axis, name=None):
    """Gather slices along a contiguous range of dimensions.

    Equivalent to tf.gather_nd when first_axis=0
    Equivalent to tf.gather when `indices` has final dimension 1.

    Args:
        params: The tensor or array from which to gather values.
            Shape `[B1, ..., BN, A1, ..., AK, C1, ..., CM]`.
            where `N = first_axis - 1`.
        indices: Index tensor or array with shape `[L, K]`.
            Elements are multi-indices into the `K` dimensions of `params`
            at `[first_axis, first_axis + K)`.
            The size of the last dimension of `indices` must be statically
            known.
        first_axis: The start of the range of dimensions indexed by `indices`.
        name: Name for the operation (optional).

    Returns:
        A tensor `output` with shape `[B1, ..., BN, L, C1, ..., CM]` where
        `output[b1, ..., bN, l, c1, ... ,cM
            = params[b1, ..., bN, indices[l], c1, ..., cM]`.

    """
    with tf.name_scope(name, "gather_dims", [params, indices]):
        params = tf.convert_to_tensor(params)
        indices = tf.convert_to_tensor(indices, preferred_dtype=tf.int32)

        index_size = int(indices.shape[-1])
        end_axis = first_axis + index_size
        if first_axis < 0:
            if end_axis > 0:
                raise IndexError("Multi-index extends past the end of params.")
            elif end_axis == 0:
                end_axis = None

        with tf.name_scope("flat_index"):
            with tf.device("/cpu:0"):
                # GPU does not support cumsum on integers
                # Shape of dimensions being gathered.
                gather_shape = tf.shape(params)[first_axis:end_axis]
                # Sizes for multi-index ravel; equal to total size of all gather
                # dimensions after the current.
                ravel_sizes = tf.cumprod(gather_shape, exclusive=True, reverse=True)
                flat_index = tf.reduce_sum(indices * ravel_sizes, axis=-1)

        flat_params = flatten_dims(params, start=first_axis, end=end_axis)
        if first_axis < 0:
            gather_axis = first_axis + (index_size - 1)
        else:
            gather_axis = first_axis
        return tf.gather(flat_params, flat_index, axis=gather_axis)


def atleast_nd(tensor, size, name=None):
    """Prepend size-1 dimensions to a tensor so that it has rank at least size.

    Args:
        tensor: The tensor or array to reshape.
        size: Minimum number of dimensions in output tensor.
        name: Name for the operation (optional).

    Returns:
        A tensor object equal to `tensor` reshaped to have at least `size`
        dimensions.
    """
    with tf.name_scope(name, "atleast", [tensor]):
        tensor = tf.convert_to_tensor(tensor)
        if tensor.shape.ndims is not None and tensor.shape.ndims >= size:
            return tensor
        static_shape = tf.broadcast_static_shape(
            tensor.shape, tf.TensorShape([1] * size)
        )
        dynamic_shape = tf.broadcast_dynamic_shape(
            tf.shape(tensor), tf.ones(size, dtype=tf.int32)
        )
        tensor = tf.reshape(tensor, dynamic_shape)
        tensor.set_shape(static_shape)
        return tensor


# TODO: Remove once tf.broadcast_to supports gradients.
def broadcast_to(tensor, shape, static_shape=None, name=None):
    """Custom implementation of broadcast_to with gradients.

    Args:
        tensor: The tensor or array to broadcast.
        shape: The target shape. A tensor or array.
            Must itself have static shape.
        static_shape: An optional post-broadcast static shape.
            Used to validate the broadcasting and set the static shape of the
            result.
        name: Name for the operation (optional).

    Returns:
        The broadcasted tensor.
    """
    with tf.name_scope(name, "broadcast_to", [tensor, shape]):
        tensor = tf.convert_to_tensor(tensor)
        shape = tf.convert_to_tensor(shape, preferred_dtype=tf.int32)
        if static_shape is not None:
            bcast_static_shape = tf.broadcast_static_shape(tensor.shape, static_shape)
            if bcast_static_shape.as_list() != static_shape.as_list():
                raise ValueError(
                    f"{tensor} with shape {tensor.shape} cannot be "
                    f"broadcasted to shape {static_shape}."
                )

        target_rank, = shape.shape.as_list()
        tensor = atleast_nd(tensor, target_rank)
        tile_multiples = shape // tf.shape(tensor)
        broadcasted_tensor = tf.tile(tensor, tile_multiples)
        broadcasted_tensor.set_shape(static_shape)
        return broadcasted_tensor


def broadcast_axis_to(tensor, axis, size, name=None):
    """Broadcast a single axis to a target size.

    Args:
        tensor: The tensor or array to broadcast.
        axis: The axis of `tensor` to broadcast. An integer.
        size: The size of `axis` after broadcasting. An integer.
        name: Name for the operation (optional).

    Returns:
        The broadcasted tensor. If `tensor` has shape `[D1, ..., DN]` then
        the result has shape `[D1, ..., D[axis-1], size, D[axis+1], ..., DN]`.
    """
    with tf.name_scope(name, "broadcast_axis_to", [tensor]):
        tensor = tf.convert_to_tensor(tensor)
        in_static_shape = tensor.shape
        in_dynamic_shape = tf.shape(tensor)

        out_static_shape = in_static_shape[:axis].concatenate([size])
        out_dynamic_shape_list = [in_dynamic_shape[:axis], [size]]
        if axis != -1:
            out_static_shape = out_static_shape.concatenate(in_static_shape[axis + 1 :])
            out_dynamic_shape_list.append(in_dynamic_shape[axis + 1 :])

        out_dynamic_shape = tf.concat(out_dynamic_shape_list, axis=0)
        tensor_out = broadcast_to(tensor, out_dynamic_shape, out_static_shape)
        return tensor_out


def broadcast_ranks(*tensors, name=None):
    """Partially broadcast tensors to have the same rank.

    Does not perform axis expansion, just prepends size-1 dimensions so that
    the tensor ranks are equal.

    Args:
        *tensors: A list of tensors to broadcast
        name: Name for the operation (optional).

    Returns:
        The input tensors with size-1 dimensions prepended so that they all
        have the same rank.
    """
    with tf.name_scope(name, "broadcast_ranks", tensors):
        rank = max(len(t.shape) for t in tensors)
        return tuple(atleast_nd(t, rank) for t in tensors)


def broadcast(*tensors, name=None):
    """Explicit broadcast of tensors together.

    Args:
        *tensors: A list of tensors to broadcast.
        name: Name for the operation (optional).
    Returns:
        The broadcasted tensors.
    """
    with tf.name_scope(name, "broadcast", tensors):
        static_shape = functools.reduce(
            tf.broadcast_static_shape, (t.shape for t in tensors)
        )
        dynamic_shape = functools.reduce(
            tf.broadcast_dynamic_shape, (tf.shape(t) for t in tensors)
        )
        broadcasted_tensors = tuple(
            broadcast_to(t, dynamic_shape, static_shape) for t in tensors
        )
        return broadcasted_tensors


def broadcast_outer_dims(*args, name=None):
    """Broadcast tensors excluding inner dimensions.

    Args:
        *args: A list of `(tensor, suffix_size)` tuples.
            All but the last `suffix_size` dimensions are broadcasted.
        name: Name for the operation (optional).

    Returns:
        *tensors: A tuple of broadcasted arrays.
    """
    tensors, suffix_sizes = zip(*args)
    end_indices = tuple(-size if size > 0 else None for size in suffix_sizes)

    with tf.name_scope(name, "broadcast_outer_dims", tensors):
        tensors = tuple(tf.convert_to_tensor(t) for t in tensors)
        tensor_shapes = tuple(tf.shape(t) for t in tensors)

        outer_shape_static = functools.reduce(
            tf.broadcast_static_shape,
            (t.shape[:end] for t, end in zip(tensors, end_indices)),
        )
        outer_shape_dynamic = functools.reduce(
            tf.broadcast_dynamic_shape,
            (t_shape[:end] for t_shape, end in zip(tensor_shapes, end_indices)),
        )

        broadcasted_tensors = tuple(
            broadcast_to(
                t,
                shape=tf.concat([outer_shape_dynamic, t_shape[end:]], axis=0),
                static_shape=outer_shape_static.concatenate(t.shape[end:]),
            )
            if end is not None
            else broadcast_to(
                t, shape=outer_shape_dynamic, static_shape=outer_shape_static
            )
            for t, t_shape, end in zip(tensors, tensor_shapes, end_indices)
        )
        return broadcasted_tensors


def flatten_dims(tensor, start=0, end=None, name=None):
    """Flatten a contiguous range of dimensions in a tensor.

    Args:
        tensor: The tensor or array to reshape.
        start: First dimension to flatten. An integer.
        end: One past the last the last dimension to flatten.
            An integer or `None`. If `None`, flattens to the end.
        name: Name for the operation (optional).

    Returns:
        A tensor equal to `tensor` but with dimensions [start, end) flattened
        to a single dimension.
        If `tensor` has shape [D1, ..., DN] then the result has shape
        `[D1, ..., D[start-1], D[start] * ... * D[end - 1], D[end], ..., DN]`.
    """
    with tf.name_scope(name, "flatten_dims", [tensor]):
        tensor = tf.convert_to_tensor(tensor)
        in_static_shape = tensor.shape
        if in_static_shape[start:end].ndims == 1:
            # Flattening a single dimension, no change.
            return tensor

        in_dynamic_shape = tf.shape(tensor)

        out_static_shape = in_static_shape[:start].concatenate(
            [in_static_shape[start:end].num_elements()]
        )
        out_dynamic_shape_list = [
            in_dynamic_shape[:start],
            tf.reduce_prod(in_dynamic_shape[start:end])[None],
        ]
        if end is not None:
            out_static_shape = out_static_shape.concatenate(in_static_shape[end:])
            out_dynamic_shape_list.append(in_dynamic_shape[end:])
        out_dynamic_shape = tf.concat(out_dynamic_shape_list, axis=0)
        tensor_out = tf.reshape(tensor, out_dynamic_shape)
        tensor_out.set_shape(out_static_shape)
        return tensor_out


def swapaxes(tensor, axis1, axis2, name=None):
    """Swap two axes of a tensor.

    Args:
        tensor: The input tensor or array. Must have statically known rank.
        axis1: The first axis.
        axis2: The second axis.
        name: Name for the operation (optional).

    Returns:
        A copy of tensor with `axis1` and `axis2` swapped.
    """
    with tf.name_scope(name, "swapaxes", [tensor]):
        tensor = tf.convert_to_tensor(tensor)
        perm = list(range(len(tensor.shape)))
        perm[axis1], perm[axis2] = perm[axis2], perm[axis1]
        return tf.transpose(tensor, perm=perm)


def broadcast_matmul(a, b, transpose_a=False, transpose_b=False, name=None):
    """Broadcasting batch matrix multiplication.

    Like tf.matmul but also broadcasts tensors.
    """
    try:
        return tf.matmul(
            a, b, transpose_a=transpose_a, transpose_b=transpose_b, name=name
        )
    except ValueError:
        pass

    with tf.name_scope(name, "matmul", [a, b]):
        b_batch_size = b.shape[:-2].num_elements()
        if b_batch_size == 1 and not transpose_a and not transpose_b:
            # Batch is only present in a. (b may have batch dims but all 1)
            # Flatten into first dim, do matmul via tensordot, then reshape.
            flat_a = flatten_dims(a, start=0, end=-1)
            flat_b = flatten_dims(b, start=0, end=-1)
            flat_result = tf.tensordot(flat_a, flat_b, axes=1)
            result = tf.reshape(
                flat_result, tf.concat([tf.shape(a)[:-1], tf.shape(b)[-1:]], axis=0)
            )
            result.set_shape(a.shape[:-1].concatenate(b.shape[-1:]))

            rank_b = b.shape.ndims
            assert (
                rank_b is not None
            ), "Shouldn't be possible if b_batch_size is not None"
            if rank_b > 2:
                # Ensure result includes b batch dims if b has any.
                result = atleast_nd(result, rank_b)
            return result

        a_bcast, b_bcast = broadcast_outer_dims((a, 2), (b, 2))
        return tf.matmul(
            a_bcast, b_bcast, transpose_a=transpose_a, transpose_b=transpose_b
        )


def _np_zeros_for_shape(shape, dtype):
    """Numpy array of zeros consistent with the given shape."""
    try:
        dtype = dtype.as_numpy_dtype
    except AttributeError:
        pass
    if shape is None:
        shape = ()
    else:
        shape = [x if x is not None else 1 for x in shape]
    return np.zeros(shape, dtype=dtype)


class PersistentTensor:
    """A tensor that persists between calls to session.run() via handles.

    Unlike `tf.Variable`, this does not have a fixed shape.

    Attributes:
        session: The session in which the tensor persists.

        value: The op representing the tensor value.
            `handle_ph` must be set when evaluating graphs containing `value`.
        handle_ph: The handle placeholder. Must be given a string handle
            obtained by calling `assign` or by evaluating `assign_op`.

        assign_ph: Placeholder that is assigned into the persistent tensor.
        assign_op: Op that evaluates to a `TensorHandle` object.
            `TensorHandle.handle` is a string handle that should be fed to
            `handle_ph` in a feed_dict. `TensorHandle.delete` should be called
            when done to free the memory in the session.

        handles: A dictionary of all handle objects created by `assign()`.
            indexed by the handle string returned by `assign()`.
            Used internally to free the data associated with the handles when
            `free_assignments()` is called but may also be used externally to
            access the `TensorHandle` object associated with a handle string.

    Example:
    >>> sess = tf.Session()
    >>> a = PersistentTensor(sess, dtype=tf.float32)
    >>> b = a.value * 2

    >>> a_zero_handle = a.assign(0.0)
    >>> print(sess.run(b, feed_dict={a.handle_ph: a_zero_handle}))
    0.0

    >>> a_ones_handle = a.assign(np.ones(3))
    >>> print(list(sess.run(b, feed_dict={a.handle_ph: a_ones_handle})))
    [2.0, 2.0, 2.0]

    >>> print(sess.run(b, feed_dict={a.handle_ph: a_zero_handle}))
    0.0
    """

    def __init__(self, session, dtype, shape=None, name=None):
        """Create a PersistentTensor.

        Args:
            session: The session in which the tensor persists.
            dtype: The tensor data type.
            shape: Optional shape of the tensor.
            name: Optional name of the tensor.
        """
        with tf.name_scope(name, "PersistentTensor"):
            self.session = session
            self.assign_ph = tf.placeholder(dtype, shape=shape, name="assign_ph")
            self.assign_op = tf.get_session_handle(self.assign_ph, name="assign_op")

            # Need to create dummy handle in order to call
            # tf.get_session_tensor
            dummy_handle = session.run(
                self.assign_op,
                feed_dict={self.assign_ph: _np_zeros_for_shape(shape, dtype=dtype)},
            )
            try:
                self.handle_ph, self.value = tf.get_session_tensor(
                    dummy_handle.handle, dtype=dtype, name="value"
                )
            finally:
                dummy_handle.delete()
            self.value.set_shape(tf.TensorShape(shape))
            self.handles = {}

    def assign(self, value):
        """Create a handle representing a value assigned to this tensor.

        Args:
            value: A numpy array to be assigned to the tensor.

        Returns:
            A string handle representing the assignment. Set as the value for
            `handle_ph` in a feed_dict when running graphs containing
            `self.value`.
        """
        handle = self.session.run(self.assign_op, feed_dict={self.assign_ph: value})
        self.handles[handle.handle] = handle
        return handle.handle

    def free_assignments(self):
        """Free the memory created by all calls to assign().

        Invalidates all handles previously returned by assign.
        """
        for handle in self.handles.values():
            handle.delete()
        self.handles = {}

    def __str__(self):
        return f"{self.__class__.__name__}(value={self.value})"
