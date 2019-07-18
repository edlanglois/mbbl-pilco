"""Code derived from gpflow sgpr.py"""
# Copyright 2018 Eric Langlois, edl@cs.toronto.edu
# Copyright 2016 James Hensman, alexggmatthews, Mark van der Wilk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# Modifications:
# The code is based on a function in Gpflow and modified by Eric Langlois to
# expose the internal parameters.
import gpflow
import tensorflow as tf


@gpflow.decors.autoflow()
@gpflow.decors.params_as_tensors
def get_sgpr_parameters(self):
    """Get parameters from a Gpflow Sparse Variational GP Regressor."""
    num_inducing_points = len(self.feature)

    # Reference
    # https://github.com/GPflow/GPflow/blob/develop/doc/source/notebooks/SGPR_notes.ipynb
    #
    # Predictive distribution
    # p(f*) = Normal(mean=K_{*u} L^{-T}L_B^{-T}c,
    #                cov=K_{**} - K_{*u} L^{-T} (1-B^{-1}) L^{-1} K_{u*})
    #
    # where
    # u: Inducing points
    # f: Data points
    # *: Prediction points
    #
    # Code based on SGPR._build_predict

    with tf.name_scope("Kuf"):
        # [NUM_INDUCING, NUM_DATA]
        Kuf = gpflow.features.Kuf(self.feature, self.kern, self.X)
    with tf.name_scope("Kuu"):
        # [NUM_INDUCING, NUM_INDUCING]
        Kuu = gpflow.features.Kuu(
            self.feature, self.kern, jitter=gpflow.settings.numerics.jitter_level
        )
    with tf.name_scope("sigma"):
        # []
        sigma = tf.sqrt(self.likelihood.variance)

    with tf.name_scope("eye"):
        # [NUM_INDUCING, NUM_INDUCING]
        eye = tf.eye(num_inducing_points, dtype=gpflow.settings.float_type)

    with tf.name_scope("L"):
        # [NUM_INDUCING, NUM_INDUCING]
        L = tf.cholesky(Kuu)
    with tf.name_scope("A"):
        # [NUM_INDUCING, NUM_DATA]
        A = tf.matrix_triangular_solve(L, Kuf, lower=True) / sigma
    with tf.name_scope("B"):
        # [NUM_INDUCING, NUM_INDUCING]
        B = tf.matmul(A, A, transpose_b=True) + eye

    with tf.name_scope("LB"):
        # [NUM_INDUCING, NUM_INDUCING]
        LB = tf.cholesky(B)

    with tf.name_scope("Ay"):
        # [NUM_INDUCING, OUT_DIM]
        Ay = tf.matmul(A, self.Y)
    with tf.name_scope("c"):
        # [NUM_INDUCING, OUT_DIM]
        c = tf.matrix_triangular_solve(LB, Ay, lower=True) / sigma
    with tf.name_scope("tmp1"):
        # [NUM_INDUCING, NUM_INDUCING]
        tmp1 = tf.matrix_triangular_solve(L, eye, lower=True)
    with tf.name_scope("tmp2"):
        # [NUM_INDUCING, NUM_INDUCING]
        tmp2 = tf.matrix_triangular_solve(LB, tmp1, lower=True)
    with tf.name_scope("alpha"):
        # [NUM_INDUCING, OUT_DIM]
        alpha = tf.matmul(tmp2, c, transpose_a=True)

    return {
        "inducing_points": self.feature.Z,
        "coefficients": tf.matrix_transpose(alpha),
        "signal_variance": self.kern.variance[None],
        "length_scale": self.kern.lengthscales[None, :],
        "noise_variance": self.likelihood.variance[None],
        "gram_L": L[None, :, :],
        "B_L": LB[None, :, :],
    }
