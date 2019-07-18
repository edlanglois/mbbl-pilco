"""Code derived from the PILCO source, adapted to Python by Eric Langlois."""
# Copyright (C) 2018 by Eric Langlois
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of
# conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list
# of conditions and the following disclaimer in the documentation and/or other materials
# provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY MARC DEISENROTH, ANDREW MCHUTCHON, JOE HALL, and CARL
# EDWARD RASMUSSEN ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL MARC DEISENROTH, ANDREW MCHUTCHON, JOE HALL,
# and CARL EDWARD RASMUSSEN OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.
#
# The views and conclusions contained in the software and documentation are those of the
# authors and should not be interpreted as representing official policies, either
# expressed or implied, of Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward
# Rasmussen.
#
# The code and associated documentation is available from
# http://mlg.eng.cam.ac.uk/pilco/


def multivarate_normal_sin_covariance(mean, variance, covariance, output_scale, ops):
    """The covariance of applying sin(x) to each dimension of a MVN distribution.

    Args:
        mean: The mean of the input distribution.
            An array of shape [..., N]
        variance: The variance of the input distribution.
            An array of shape [..., N]
        covariance: The covariance of the input distribution.
            An array of shape [..., N, N]
        output_mean: Scaling factor applied the output of sin(x).
            An array of shape [..., N]
        ops: A tensor library. e.g. numpy

    Supports broadcasting.
    """
    # Based off of the file util/gSin.m in the PILCO source.
    lq = -(variance[..., :, None] + variance[..., None, :]) / 2
    q = ops.exp(lq)

    V = (ops.exp(lq + covariance) - q) * ops.cos(
        mean[..., :, None] - mean[..., None, :]
    ) - (ops.exp(lq - covariance) - q) * ops.cos(
        mean[..., :, None] + mean[..., None, :]
    )
    output_scale = ops.atleast_1d(output_scale)
    return output_scale[:, None] * output_scale[None, :] * V / 2
