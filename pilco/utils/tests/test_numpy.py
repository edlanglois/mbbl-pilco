"""Numpy utilities tests."""
import numpy as np
import pytest

from pilco.utils import numpy as np_utils


def test_batch_apply_sum_in_dims0():
    x = np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5)
    target = pytest.approx(x)
    assert np_utils.batch_apply(np.sum, x, in_dims=0) == target

    out = np.empty_like(x)
    assert np_utils.batch_apply(np.sum, x, in_dims=0, out=out) == target
    assert out == target


def test_batch_apply_sum_in_dims1():
    x = np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5)
    target = pytest.approx(np.sum(x, axis=-1))
    assert np_utils.batch_apply(np.sum, x, in_dims=1) == target

    out = np.empty(x.shape[:-1], dtype=x.dtype)
    assert np_utils.batch_apply(np.sum, x, in_dims=1, out=out) == target
    assert out == target


def test_batch_apply_sum_in_dims2():
    x = np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5)
    target = pytest.approx(np.sum(x, axis=(-2, -1)))
    assert np_utils.batch_apply(np.sum, x, in_dims=2) == target

    out = np.empty(x.shape[:-2], dtype=x.dtype)
    assert np_utils.batch_apply(np.sum, x, in_dims=2, out=out) == target
    assert out == target


def test_softmax_float():
    x = np.array([1, 0, 0], dtype=float)
    y = np.array([np.exp(1), 1, 1]) / (np.exp(1) + 1 + 1)
    assert np_utils.softmax(x) == pytest.approx(y)


def test_softmax_int():
    x = np.array([1, 0, 0], dtype=int)
    y = np.array([np.exp(1), 1, 1]) / (np.exp(1) + 1 + 1)
    assert np_utils.softmax(x) == pytest.approx(y)


def test_softmax_out_other():
    x = np.array([1, 0, 0], dtype=float)
    out = np.empty_like(x)
    y = np.array([np.exp(1), 1, 1]) / (np.exp(1) + 1 + 1)
    result = np_utils.softmax(x, out=out)
    assert result == pytest.approx(y)
    assert result is out


def test_softmax_out_self():
    x = np.array([1, 0, 0], dtype=float)
    y = np.array([np.exp(1), 1, 1]) / (np.exp(1) + 1 + 1)
    result = np_utils.softmax(x, out=x)
    assert result == pytest.approx(y)
    assert result is x


def test_softmax_out_int():
    x = np.array([1, 0, 0], dtype=int)
    with pytest.raises(TypeError):
        # Cannot put softmax into integer array
        np_utils.softmax(x, out=x)


def test_softmax_2d_noaxis():
    x = np.array([[1, 0, 0], [0, 0, 0]], dtype=float)
    y = np.array([[np.exp(1), 1, 1], [1, 1, 1]]) / (np.exp(1) + 5)
    assert np_utils.softmax(x) == pytest.approx(y)


def test_softmax_2d_axis():
    x = np.array([[1, 0, 0], [0, 0, 0]], dtype=float)
    y0 = np.array(
        [[np.exp(1) / (1 + np.exp(1)), 0.5, 0.5], [1 / (1 + np.exp(1)), 0.5, 0.5]]
    )
    y1 = np.array(
        [
            [np.exp(1) / (2 + np.exp(1)), 1 / (2 + np.exp(1)), 1 / (2 + np.exp(1))],
            [1 / 3, 1 / 3, 1 / 3],
        ]
    )
    assert np_utils.softmax(x, axis=0) == pytest.approx(y0)
    assert np_utils.softmax(x, axis=-2) == pytest.approx(y0)
    assert np_utils.softmax(x, axis=1) == pytest.approx(y1)
    assert np_utils.softmax(x, axis=-1) == pytest.approx(y1)


def test_batch_inner_1d_1d():
    x = np.array([1, 2, 3], dtype=float)
    y = np.array([2, 3, 4], dtype=float)
    assert np_utils.batch_inner(x, y) == pytest.approx(20)


def test_batch_inner_1d_1d_out():
    x = np.array([1, 2, 3], dtype=float)
    y = np.array([2, 3, 4], dtype=float)
    z = np.zeros([])
    assert np_utils.batch_inner(x, y, out=z) == pytest.approx(20)
    assert z == pytest.approx(20)


def test_batch_inner_1d_1d_int():
    x = np.array([1, 2, 3], dtype=int)
    y = np.array([2, 3, 4], dtype=int)
    assert np_utils.batch_inner(x, y) == 20


def test_batch_inner_1d_0d():
    x = np.array([1, 2, 3], dtype=float)
    y = np.array([2], dtype=float)
    assert np_utils.batch_inner(x, y) == pytest.approx(12)


def test_batch_inner_batch_nobroadcast():
    x = np.array([[1, 2, 3], [0.5, 3, -1]], dtype=float)
    y = np.array([[2, 3, 4], [5, 6, 2.1]], dtype=float)
    assert np_utils.batch_inner(x, y) == pytest.approx(np.array([20, 2.5 + 18 - 2.1]))


def test_batch_inner_batch_broadcast():
    x = np.array([[1, 2]])
    y = np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], dtype=float)
    assert np_utils.batch_inner(x, y) == pytest.approx(
        np.array([[5, 11, 17], [23, 29, 35]])
    )


def test_batch_cho_solve_identity():
    # a = np.stack([2 * np.eye(3), 1 / 3 * np.eye(3)])
    c = np.stack([np.sqrt(2) * np.eye(3), 1 / np.sqrt(3) * np.eye(3)])

    rand = np.random.RandomState(seed=1)
    b = rand.normal(size=[1, 2, 3, 4])

    x = np_utils.batch_cho_solve(c, b)
    assert x == pytest.approx(b / np.array([2, 1 / 3])[:, None, None])


@pytest.mark.parametrize(
    "asize,bsize",
    [
        [(4, 4), (4, 1)],
        [(1, 1), (1, 1)],
        [(2, 3, 2, 2), (2, 3, 2, 4)],
        [(1, 4, 2, 2), (3, 1, 2, 2)],
        [(4, 2, 2), (3, 1, 2, 2)],
    ],
)
def test_batch_cho_solve_random(asize, bsize):
    rand = np.random.RandomState(seed=1)
    a = rand.normal(size=asize)
    np.matmul(a, np.swapaxes(a, axis1=-2, axis2=-1), out=a)

    b = rand.normal(size=bsize)

    c = np.linalg.cholesky(a)
    x = np_utils.batch_cho_solve(c, b)
    b_bcast = np.broadcast_to(b, x.shape)
    assert a @ x == pytest.approx(b_bcast)


def test_batch_diag_shape_1():
    x = np.array([2])
    np.testing.assert_array_equal(np_utils.batch_diag(x), np.array([[2]]))


def test_batch_diag_shape_1_size():
    x = np.array([2])
    np.testing.assert_array_equal(np_utils.batch_diag(x, size=3), 2 * np.eye(3))


def test_batch_diag_0d():
    with pytest.raises(ValueError):
        # Cannot infer diagonal size
        assert np_utils.batch_diag(2)


def test_batch_diag_0d_size():
    np.testing.assert_array_equal(np_utils.batch_diag(2, size=3), 2 * np.eye(3))


def test_batch_diag_shape_3():
    x = np.array([1, 2, 3])
    diagx = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    np.testing.assert_array_equal(np_utils.batch_diag(x), diagx)
    np.testing.assert_array_equal(np_utils.batch_diag(x, size=3), diagx)


def test_batch_diag_shape_2_3():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    diagx = np.array(
        [[[1, 0, 0], [0, 2, 0], [0, 0, 3]], [[4, 0, 0], [0, 5, 0], [0, 0, 6]]]
    )
    np.testing.assert_array_equal(np_utils.batch_diag(x), diagx)
    np.testing.assert_array_equal(np_utils.batch_diag(x, size=3), diagx)


def test_batch_diag_shape_2_3_wrongsize():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        np_utils.batch_diag(x, size=2)


def test_batch_diag_shape_2_3_out():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    diagx = np.array(
        [[[1, 0, 0], [0, 2, 0], [0, 0, 3]], [[4, 0, 0], [0, 5, 0], [0, 0, 6]]]
    )
    out = np.empty([2, 3, 3])
    np.testing.assert_array_equal(np_utils.batch_diag(x, out=out), diagx)
    np.testing.assert_array_equal(out, diagx)
    out.fill(0)
    np.testing.assert_array_equal(np_utils.batch_diag(x, size=3, out=out), diagx)
    np.testing.assert_array_equal(out, diagx)


def test_batch_diag_0d_out():
    out = np.empty([2, 3, 3])
    diagx = np.array([2 * np.eye(3), 2 * np.eye(3)])
    np.testing.assert_array_equal(np_utils.batch_diag(2, out=out), diagx)
    np.testing.assert_array_equal(out, diagx)


def test_batch_diag_out_notsquare():
    x = np.array([3])
    out = np.empty([2, 3])
    with pytest.raises(ValueError):
        np_utils.batch_diag(x, out=out)


@pytest.mark.parametrize(
    "a,b,result",
    [
        [(), (), ()],
        [(2,), (2,), (2,)],
        [(3,), (1,), (3,)],
        [(1,), (3,), (3,)],
        [(1, 3), (2, 1), (2, 3)],
        [(2, 3), (3,), (2, 3)],
        [(2, 3), (), (2, 3)],
    ],
)
def test_broadcast_shapes(a, b, result):
    assert np_utils.broadcast_shapes(a, b) == result


@pytest.mark.parametrize("a,b", [[(2,), (3,)], [(2, 3), (2,)]])
def test_broadcast_shapes_error(a, b):
    with pytest.raises(ValueError):
        np_utils.broadcast_shapes(a, b)


def test_outer_broadcast_arrays_11():
    a = np.zeros([1, 3, 5])
    b = np.zeros([2, 2, 1, 4])
    bcast_a, bcast_b = np_utils.outer_broadcast_arrays((a, 1), (b, 1))

    np.testing.assert_array_equal(bcast_a, np.zeros([2, 2, 3, 5]))
    np.testing.assert_array_equal(bcast_b, np.zeros([2, 2, 3, 4]))


def test_outer_broadcast_arrays_12():
    a = np.zeros([1, 3, 5])
    b = np.zeros([2, 3, 1, 3])
    bcast_a, bcast_b = np_utils.outer_broadcast_arrays((a, 1), (b, 2))

    np.testing.assert_array_equal(bcast_a, np.zeros([2, 3, 5]))
    np.testing.assert_array_equal(bcast_b, np.zeros([2, 3, 1, 3]))
