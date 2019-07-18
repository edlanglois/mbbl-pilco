import numpy as np
import pytest

from pilco.utils import stats


@pytest.mark.parametrize("size", [2, 5, 8, 11])
def test_online_mean_variance_correct_mean(size):
    rand = np.random.RandomState(seed=size)
    mv = stats.OnlineMeanVariance()

    x = rand.normal(size=size)
    for xi in x:
        mv.add(xi)

    assert mv.mean() == pytest.approx(np.mean(x))


def test_online_mean_variance_undefined_mean_raises():
    mv = stats.OnlineMeanVariance()
    with pytest.raises(stats.UndefinedError):
        mv.mean()


@pytest.mark.parametrize("size", [2, 5, 8, 11])
def test_online_mean_variance_correct_variance(size):
    rand = np.random.RandomState(seed=size)
    mv = stats.OnlineMeanVariance()

    x = rand.normal(size=size)
    for xi in x:
        mv.add(xi)

    assert mv.variance() == pytest.approx(np.var(x))


def test_online_mean_variance_undefined_var_0_raises():
    mv = stats.OnlineMeanVariance()
    with pytest.raises(stats.UndefinedError):
        mv.variance()


def test_online_mean_variance_undefined_var_1_raises():
    mv = stats.OnlineMeanVariance()
    mv.add(1)
    with pytest.raises(stats.UndefinedError):
        mv.variance()
