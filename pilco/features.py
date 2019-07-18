"""Feature networks."""

import gym.spaces
import numpy as np
import tensorflow as tf


class FeatureNet:
    """Feature vector generation network for `gym.Space` samples."""

    def __init__(self, space, extension=None):
        """Initialize a feature network for a `gym.Space`.

        Args:
            space: A `gym.Space` instance. Generates feature for samples from
                this spaces.
            extension: An optional additional function applied to the feature
                vector created by `build` before it is returned.
        """
        self.space = space
        self.extension = extension

    def prepare(self, xs):
        """Prepare a batch of `self.space` samples.

        Args:
            xs: An iterable of samples from `self.space`.

        Returns:
            A numpy array representing `xs` suitable to be assigned to the
            `input_placeholder` returned by `build`.
            The first dimension is the batch dimension.
        """
        return np.asarray(xs)

    def build(self):
        """Build the feature network.

        Returns:
            input_placeholder: Placeholder for a batch of samples from `space`.
            feature: Tensorflow op containing a batch feature vectors for the
                input.
        """
        input_placeholder, feature = self._build()
        if self.extension:
            feature = self.extension(feature)
        return input_placeholder, feature

    def _build(self):
        """Build the feature network (without extension).

        Returns:
            input_placeholder: Placeholder for a batch of samples from `space`.
            feature: Tensorflow op containing a batch feature vectors for the
                input.
        """
        raise NotImplementedError


def make_flatten_space(space):
    """Make a function that flattens sampls from space.

    Args:
        space: The space to flatten. A `gym.Space` instance.

    Returns:
        n: The length of the flat feature vectors.
        flatten: A function that takes a list of elements from space and
            returns a 2d array where each row is a feature vector.
    """
    if isinstance(space, gym.spaces.Box):
        n = np.prod(space.shape)

        def flatten(xs):
            return np.reshape(xs, [len(xs), n])

        return n, flatten

    elif isinstance(space, gym.spaces.Discrete):
        n = space.n

        def flatten(xs):
            return np.eye(n)[np.asarray(xs)]

        return n, flatten

    elif isinstance(space, gym.spaces.Tuple):
        ns, flattens = zip(*[make_flatten_space(subspace) for subspace in space.spaces])
        n = sum(ns)

        def flatten(xs):
            return np.concatenate(
                [f(subxs) for f, subxs in zip(flattens, zip(*xs))], axis=-1
            )

        return n, flatten

    try:
        # Assume a box-like space if it has `.shape`
        shape = space.shape
    except AttributeError:
        pass
    else:
        n = np.prod(shape)

        def flatten(xs):
            return np.reshape(xs, [len(xs), n])

        return n, flatten

    raise TypeError("No support for space {}".format(space))


class FlatFeatures(FeatureNet):
    """Convert samples to flat real-value feature vectors."""

    def __init__(self, space, extension=None):
        super().__init__(space, extension=extension)
        self.n, self.prepare = make_flatten_space(space)

    def _build(self):
        input_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, self.n))
        return input_placeholder, input_placeholder


class FullyConnectedFeatures(FlatFeatures):
    """Features are the output of a deep fully-connected network."""

    def __init__(self, space, widths, extension=None, **fc_kwargs):
        """Initialize a `FullyConnectedFeatures` instance.

        Args:
            space: A `gym.Space` instance to generate features for.
            widths: A list of layer widths.
            extension: An optional additional function applied to the feature
                vector created by `build` before it is returned.
            fc_kwargs: Additional arguments passed to
                `tf.contrib.layers.fully_connected`.
        """

        def fc_extension(features):
            """Fully-connected network extension to features."""
            for width in widths:
                features = tf.contrib.layers.fully_connected(
                    features, num_outputs=width, **fc_kwargs
                )
            if extension is not None:
                features = extension(features)
            return features

        super().__init__(space, extension=fc_extension)
