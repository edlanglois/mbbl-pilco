"""Statistics utilities."""
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

class UndefinedError(Exception):
    """The value in question is undefined."""

    pass


class OnlineMeanVariance:
    """Online mean & variance calculation with O(1) space and O(1) updates.

    Uses Welford's algorithm.
    """

    def __init__(self):
        self.count = 0
        self._mean = 0
        # squared_error_sum = sum_{i=1}^n (x_i - xbar_n)^2
        self.squared_error_sum = 0

    def __str__(self):
        mean = self._mean if self.count >= 1 else float("nan")
        variance = (
            self.squared_error_sum / self.count if self.count >= 2 else float("nan")
        )
        return f"[mean={mean}, var={variance}, n={self.count}]"

    def add(self, x):
        """Add a new sample to the collected statistics."""
        self.count += 1
        delta1 = x - self._mean
        self._mean += delta1 / self.count
        delta2 = x - self._mean
        self.squared_error_sum += delta1 * delta2

    def mean(self):
        if not self.count:
            raise UndefinedError("mean")
        return self._mean

    def variance(self):
        if self.count < 2:
            raise UndefinedError("variance")
        return self.squared_error_sum / self.count
