"""TensorFlow Utilities Tests"""
import numpy as np
import pytest
import tensorflow as tf

from pilco.utils import tf as tfu


@pytest.fixture
def graph():
    return tf.Graph()


def test_batch_gather_1d(graph):
    with graph.as_default():
        params = tf.range(5)
        indices = tf.convert_to_tensor([3, 1, 2])
        gathered = tfu.batch_gather(params, indices)
        assert gathered.shape == (3,)

        with tf.Session() as session:
            gathered_value = session.run(gathered)
        assert np.array_equal(gathered_value, np.array([3, 1, 2]))


def test_batch_gather_list_inputs(graph):
    with graph.as_default():
        params = list(range(5))
        indices = [3, 1, 2]
        gathered = tfu.batch_gather(params, indices)
        assert gathered.shape == (3,)

        with tf.Session() as session:
            gathered_value = session.run(gathered)
        assert np.array_equal(gathered_value, np.array([3, 1, 2]))


def test_batch_gather_2d(graph):
    with graph.as_default():
        params = tf.convert_to_tensor(np.arange(12).reshape(3, 4))
        indices = tf.convert_to_tensor([[0, 1], [3, 1], [2, 2]])
        gathered = tfu.batch_gather(params, indices)
        assert gathered.shape == (3, 2)

        with tf.Session() as session:
            gathered_value = session.run(gathered)
        # (0) [1]  2    3
        #  4  [5]  6   (7)
        #  8   9 [(10)] 11
        assert np.array_equal(gathered_value, np.array([[0, 1], [7, 5], [10, 10]]))


def test_batch_gather_2d_slice(graph):
    with graph.as_default():
        params = tf.convert_to_tensor(np.arange(24).reshape(3, 4, 2))
        indices = tf.convert_to_tensor([[0, 1], [3, 1], [2, 2]])
        gathered = tfu.batch_gather(params, indices)
        assert gathered.shape == (3, 2, 2)

        with tf.Session() as session:
            gathered_value = session.run(gathered)
        # (0)  [2]    4     6
        #  8   [10]   12   (14)
        #  16   18  [(20)]  22
        assert np.array_equal(
            gathered_value,
            np.array([[[0, 1], [2, 3]], [[14, 15], [10, 11]], [[20, 21], [20, 21]]]),
        )


def test_batch_gather_none_batch(graph):
    with graph.as_default():
        params = tf.placeholder(dtype=tf.int32, shape=(None, 4))
        indices = tf.placeholder(dtype=tf.int32, shape=(None, 2))
        gathered = tfu.batch_gather(params, indices)
        assert gathered.shape.as_list() == [None, 2]

        with tf.Session() as session:
            gathered_value = session.run(
                gathered,
                feed_dict={
                    params: np.arange(12).reshape(3, 4),
                    indices: np.array([[0, 1], [3, 1], [2, 2]]),
                },
            )

        assert np.array_equal(gathered_value, np.array([[0, 1], [7, 5], [10, 10]]))


def test_batch_gather_none_axis(graph):
    with graph.as_default():
        params = tf.placeholder(dtype=tf.int32, shape=(3, None))
        indices = tf.placeholder(dtype=tf.int32, shape=(3, None))
        gathered = tfu.batch_gather(params, indices)
        assert gathered.shape.as_list() == [3, None]

        with tf.Session() as session:
            gathered_value = session.run(
                gathered,
                feed_dict={
                    params: np.arange(12).reshape(3, 4),
                    indices: np.array([[0, 1], [3, 1], [2, 2]]),
                },
            )

        assert np.array_equal(gathered_value, np.array([[0, 1], [7, 5], [10, 10]]))


def test_batch_gather_1d_int64_index(graph):
    with graph.as_default():
        params = tf.range(5)
        indices = tf.convert_to_tensor([3, 1, 2], dtype=tf.int64)
        gathered = tfu.batch_gather(params, indices)
        assert gathered.shape == (3,)

        with tf.Session() as session:
            gathered_value = session.run(gathered)
        assert np.array_equal(gathered_value, np.array([3, 1, 2]))


def test_batch_gather_2d_int64_index(graph):
    with graph.as_default():
        params = tf.convert_to_tensor(np.arange(12).reshape(3, 4))
        indices = tf.convert_to_tensor([[0, 1], [3, 1], [2, 2]], dtype=tf.int64)
        gathered = tfu.batch_gather(params, indices)
        assert gathered.shape == (3, 2)

        with tf.Session() as session:
            gathered_value = session.run(gathered)
        # (0) [1]  2    3
        #  4  [5]  6   (7)
        #  8   9 [(10)] 11
        assert np.array_equal(gathered_value, np.array([[0, 1], [7, 5], [10, 10]]))


@pytest.mark.skip(reason="TF>=1.9 no longer raises InvalidArgumentError here")
def test_batch_gather_invalid_index(graph):
    with graph.as_default():
        params = tf.convert_to_tensor(np.arange(12).reshape(3, 4))
        indices = tf.convert_to_tensor([[0, 4], [3, 1], [2, 2]])
        gathered = tfu.batch_gather(params, indices)

        with pytest.raises(tf.errors.InvalidArgumentError):
            with tf.Session() as session:
                session.run(gathered)


def test_persistent_tensor_assign_no_shape(graph):
    with graph.as_default(), tf.Session() as sess:
        a = tfu.PersistentTensor(sess, dtype=tf.float32)
        b = a.value * 2

        a_zero_handle = a.assign(0.0)
        assert np.array_equal(
            sess.run(b, feed_dict={a.handle_ph: a_zero_handle}), np.zeros(())
        )

        a_ones_handle = a.assign(np.ones(3))
        assert np.array_equal(
            sess.run(b, feed_dict={a.handle_ph: a_ones_handle}), np.ones(3) * 2
        )

        assert np.array_equal(
            sess.run(b, feed_dict={a.handle_ph: a_zero_handle}), np.zeros(())
        )


def test_persistent_tensor_assign_op_no_shape(graph):
    with graph.as_default(), tf.Session() as sess:
        a = tfu.PersistentTensor(sess, dtype=tf.float32)
        b = a.value * 2

        a_zero_handle = sess.run(a.assign_op, feed_dict={a.assign_ph: 1.0})
        assert np.array_equal(
            sess.run(b, feed_dict={a.handle_ph: a_zero_handle.handle}), 2 * np.ones(())
        )
        a_zero_handle.delete()


# test free_assignments


def test_persistent_tensor_assign_shape(graph):
    with graph.as_default(), tf.Session() as sess:
        a = tfu.PersistentTensor(sess, dtype=tf.float32, shape=[1, None, 2])
        b = a.value * 2

        assert a.value.shape.as_list() == [1, None, 2]

        a_zero_handle = a.assign(np.zeros([1, 1, 2]))
        assert np.array_equal(
            sess.run(b, feed_dict={a.handle_ph: a_zero_handle}), np.zeros([1, 1, 2])
        )

        a_ones_handle = a.assign(np.ones([1, 3, 2]))
        assert np.array_equal(
            sess.run(b, feed_dict={a.handle_ph: a_ones_handle}), np.ones([1, 3, 2]) * 2
        )


def test_persistent_tensor_assign_shape_invalid(graph):
    with graph.as_default(), tf.Session() as sess:
        a = tfu.PersistentTensor(sess, dtype=tf.float32, shape=[1, None, 2])

        assert a.value.shape.as_list() == [1, None, 2]

        with pytest.raises(ValueError):
            a.assign(np.zeros([3]))


def test_persistent_tensor_free_assignments(graph):
    with graph.as_default(), tf.Session() as sess:
        a = tfu.PersistentTensor(sess, dtype=tf.float32, name="somename")

        handle = a.assign(np.ones([1, 2]))
        assert np.array_equal(
            sess.run(a.value, feed_dict={a.handle_ph: handle}), np.ones([1, 2])
        )

        a.free_assignments()

        handle = a.assign(np.zeros([1, 2]))
        assert np.array_equal(
            sess.run(a.value, feed_dict={a.handle_ph: handle}), np.zeros([1, 2])
        )


def test_persistent_tensor_use_after_free_assignments(graph):
    with graph.as_default(), tf.Session() as sess:
        a = tfu.PersistentTensor(sess, dtype=tf.float32, name="somename")

        handle = a.assign(np.ones([1, 2]))
        assert np.array_equal(
            sess.run(a.value, feed_dict={a.handle_ph: handle}), np.ones([1, 2])
        )

        a.free_assignments()

        with pytest.raises(tf.errors.InvalidArgumentError):
            sess.run(a.value, feed_dict={a.handle_ph: handle})
