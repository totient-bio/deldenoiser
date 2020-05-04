##############################################################################
# Implementation of the Poisson Background Subtraction algorithm
# It minimizes the log-likelihood of the background intensity beta
#
# logL(beta) = c0 * beta + sum_{r: n_r > 0} logQ (n_r + 1,  a_r + b_r beta)
#
# where
#   c0 can be positive or negative,
#   n_r is positive,
#   a_r and b_r are non-negative
#   logQ() is np.log(scipy.special.gammaincc())
#
# Optimization is done on [0, inf) interval.
# If c0 <= 0, the optimal beta is 0.0, this is quick to return.
# If c0 > 0, the optimum exists, it is unique and
# is guaranteed to be between 0.0 and beta_1
# where beta_1 = (sum_{r: n_r > 0} n_r) / ((sum_{r: n_r > 0} b_r) - c0)
#
# If no starting beta (beta_start) is provided, the guess beta_0 is used
# beta_0 = (#[r: n_r > 0] * median_{r: n_r > 0}(n_r))
#          / (sum_{r: n_r > 0} b_r - c0)
#
# Steps of the algorithm:
# 1. Run Newton-Raphson algorithm, using the 1st and second
#    derivative of logL
#    f(beta) = d/(d beta) logL =
#              c0 - sum_{r: n_1 > 0} b_r H_r
#    f'(beta) = d/(d beta) f =
#               - sum_{r: n_1 > 0} (b_r)^2 H_r ((n_r / xi_r) + H_r - 1)
#    i.e. beta_new = beta_old - f(beta_old) / f'(beta_old)
# where
#    H_r = (xi_r)^(n_r) * exp(-xi_r) / Gamma(n_r + 1, xi_r)
#    xi_r = a_r + b_r beta
#    Gamma(s, x) is th upper incomplete gamma function,
#    i.e. scipy.special.gamma(s) * scipy.special.gammaincc(s,x)
#
# 2. If the Newton-Raphson update fails to decrease |f|, then fall back to
#    Anderson-Bjoerk algorithm on an interval determined by
#    the following rules:
#    - evaluate f at 0, beta_1 and beta_start
#    - if the Newton update (beta') falls between 0 and beta_1,
#      evaluate f(beta')
#    - use the 3 (or 4) points to find the sub-interval
#      on which f is guaranteed tp change sign
#
# 3. Run Anderson-Bjoerck algorithm,
#
#
# Copyright (C) 2020  Totient, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License v3.0
# along with this program.
# If not, see <https://www.gnu.org/licenses/gpl-3.0.en.html>.
#
# Developer:
# Peter Komar (peter.komar@totient.bio)
##############################################################################

import numpy as np
from scipy.special import gammaln, gammaincc

NUMERICAL_FLOOR = 1e-10


def gammainccln(s, x):
    """Computes numpy.log(scipy.special.gammaincc(s,x)), avoiding underflow.

    See: https://en.wikipedia.org/wiki/Incomplete_gamma_function#Asymptotic_behavior  # noqa

    Args:
        s (numpy.ndarray): first argument of incomplete gamma function
        x (numpy.ndarray): second argument of incomplete gamma function

    Returns:
        numpy.ndarray: ln(gammaincc(s,x)) for arrays s, x of the same shape
    """
    result = gammaincc(s, x)

    # For some inputs scipy.special.gammaincc returns 0.0.
    # This happens when x >> s.
    # Let's avoid taking the log of these entries,
    # and evaluate the first 3 terms of the asymptotic series instead.
    underflow = (result == 0.0)
    result[~underflow] = np.log(result[~underflow])
    x = x[underflow]
    s = s[underflow]
    result[underflow] = -x + (s - 1) * np.log(x) + np.log(1 + (s - 1) / x) \
                        - gammaln(s)
    return result


def anderson_bjoerk_step(f, start, end, last_yc):
    """Performs a single iteration of Anderson-Bjoerck root-finding algorithm

    Args:
        f: univariate function the root of which we are looking for
        start: x and y values on the left of the interval, tuple of (a, ya)
        end: x and y values on the right of the interval, tuple of (b, yb)
        last_yc: value of the previously discovered y value, yc

    Returns:
         tuple of tuples: new values of (a, ya), (b, yb) and (c, yc)
         where c and yc are the newly discovered x and y values
    """
    a, ya = start
    b, yb = end
    s = last_yc

    # compute middle point using classical regula falsi update
    c = (yb * a - ya * b) / (yb - ya)
    yc = f(c)

    if yc * ya > 0:  # c is on the same side of f as the left limit (a)
        if yc * s > 0:
            m = 1 - yc / ya
            m = m if m > 0 else 0.5
            yb *= m
        a = c
        ya = yc
    else:  # c is on the same side of f as the right limit (b)
        if yc * s > 0:
            m = 1 - yc / yb
            m = m if m > 0 else 0.5
            ya *= m
        b = c
        yb = yc

    return (a, ya), (b, yb), (c, yc)


def _logL(beta, n, a, b, c0):
    """Compute the log-likelihood of the Poisson Background subtraction model

    Args:
        beta (float): background intensity
        n (numpy.ndarray): array of positive numbers
        a (numpy.ndarray): array of non-negative numbers (same length as n)
        b (numpy.ndarray): array of non-negative numbers (same length as n)
        c0 (float):

    Returns:
        float: log-likelihood at beta value

    """
    xi = a + b * beta
    xi[xi <= 0.0] = NUMERICAL_FLOOR
    logQ = gammainccln(n + 1, xi)
    logL = beta * c0 + np.sum(logQ)
    return logL


def _f_fprime(beta, n, a, b, c0):
    """Compute the derivative (f) and the second derivative (fprime) of logL

    Args:
        beta (float): background intensity
        n (numpy.ndarray): array of positive numbers
        a (numpy.ndarray): array of non-negative numbers (same length as n)
        b (numpy.ndarray): array of non-negative numbers (same length as n)
        c0 (float):

    Returns:
        tuple: (f, fprime) at beta value

    """
    xi = a + b * beta
    xi[xi <= 0.0] = NUMERICAL_FLOOR
    logQ = gammainccln(n + 1, xi)
    logH = n * np.log(xi) - xi - gammaln(n + 1) - logQ
    H = np.exp(logH)
    f = c0 - np.sum(b * H)
    fprime = - np.sum(b ** 2 * H * (n / xi + H - 1))

    return f, fprime


def _f_null(n, a, b, c0):
    """Compute f for beta = 0.0

    Args:
        n (numpy.ndarray): array of positive numbers
        a (numpy.ndarray): array of non-negative numbers (same length as n)
        b (numpy.ndarray): array of non-negative numbers (same length as n)
        c0 (float):

    Returns:
        float: f(0.0)
    """
    logQ = gammainccln(n + 1, a)
    if (a > 0.0).all():
        logH = n * np.log(a) - a - gammaln(n + 1) - logQ
        H = np.exp(logH)
    else:
        H = np.zeros_like(a)
        zeros = (a == 0)
        H[zeros] = 0.0  # assuming  n > 0
        H[~zeros] = np.exp(n[~zeros] * np.log(a[~zeros])
                           - a[~zeros]
                           - gammaln(n[~zeros] + 1)
                           - logQ[~zeros])

    f = c0 - np.sum(b * H)
    return f


def pbs_optimize(n, a, b, c0, beta_start=None, itermax=10, b_beta_tol=0.5):
    """Performs maximization of the PBS log-likelihood

    Log likelihood:
        logL(beta) = c0 * beta
                     + sum_{r: n_r > 0} logQ (n_r + 1,  a_r + b_r beta)

    If beta_start is None, the guess
    beta_guess = len(n) * np.median(n) / float(np.sum(b) - c0)
    is used as starting point.

    In any case, it is assumed that beta_start is between
    0.0 and beta_max, i.e. beta_max = np.sum(n) / float(np.sum(b) - c0)

    First Newton-Raphson algorithm is run. If this fails to get closer to
    the root of f = d/(d beta) logL, then Anderson-Bjoerck algorithm is run.

    Optimization terminates early, if b * beta changes less than b_beta_tol
    between iterations.

    Args:
        n: array of positives (zeros are not allowed)
        a: array of non-negatives (same length as n)
        b: array of positives (same length as n)
        c0: float
        beta_start: float, starting point of iteration
        itermax: maximum number of Anderson-Bjoerck iterations
        b_beta_tol: minimal value of the change of b * beta between iterations
            that is considered significant. If b * beta changes less than this
            between two iterations, the algorithm terminates early.

    Returns:
        float: new beta value
    """

    # define function the root of which we are looking for
    def _f(_beta):
        f_val, fprime_val = _f_fprime(_beta, n, a, b, c0)
        return f_val

    # If the f curve is decreasing even at 0.0, that is the optimum
    f_null = _f_null(n, a, b, c0)
    if f_null < 0.0:
        return 0.0

    betas = [0]
    fs = [f_null]

    # Compute upper limit (beta_max)
    beta_max = np.sum(n) / float(np.sum(b) - c0)
    for _ in range(10):
        f_at_beta_max = _f(beta_max)
        if f_at_beta_max < 0:
            break
        beta_max *= 2

    # Compute f and f' at starting point
    if beta_start is None:
        beta_start = len(n) * np.median(n) / float(np.sum(b) - c0)

    # Run Newton-Raphson algorithm
    beta = beta_start
    new_beta = beta
    for it in range(itermax):
        betas.append(beta)
        f, fprime = _f_fprime(beta, n, a, b, c0)
        fs.append(f)
        if fprime == 0.0:
            new_beta = -np.inf
            break

        new_beta = beta - f / fprime

        if (new_beta <= 0) or (new_beta >= beta_max):
            break
        if (np.abs(new_beta - beta) * b < b_beta_tol).all():
            return new_beta
        beta = new_beta
    f_start = fs[1]

    # Examine the results of N-R algorithm
    # If the new beta improves |f|, we accept it
    # otherwise, we use it as just another potential starting point
    # for Anderson-Bjoerck algorithm below
    if (new_beta > 0) and (new_beta < beta_max):
        f_new, fprime_new = _f_fprime(new_beta, n, a, b, c0)
        if np.abs(f_new) < np.abs(f_start):
            return new_beta
        betas.append(new_beta)
        fs.append(f_new)

    # Find the interval where f changes sign
    betas.append(beta_max)
    fs.append(f_at_beta_max)

    betas = np.array(betas)
    fs = np.array(fs)
    sort_order = np.argsort(betas)
    betas = betas[sort_order]
    fs = fs[sort_order]
    start = (betas[0], fs[0])
    end = (betas[len(fs) - 1], fs[len(fs) - 1])
    for idx_start, idx_end in zip(range(0, len(fs) - 1), range(1, len(fs))):
        if np.sign(fs[idx_start]) != np.sign(fs[idx_end]):
            start = (betas[idx_start], fs[idx_start])
            end = (betas[idx_end], fs[idx_end])
            break
    if start[1] * end[1] > 0:
        raise ValueError(f'f has the same sign at start and end, '
                         f'{start}, {end}')

    # Run Anderson-Bjoerck algorithm
    updated = (0.0, 0.0)
    for it in range(itermax):
        if start[0] == end[0]:
            return start[0]
        if start[1] == 0.0:
            return start[0]
        if end[1] == 0.0:
            return end[0]
        if start[1] * end[1] > 0:
            raise ValueError(
                f'f has the same sign at start and end, {start}, {end}')
        new_start, new_end, new_updated = anderson_bjoerk_step(_f, start, end,
                                                               updated[1])

        if (np.abs(new_updated[0] - updated[0]) * b < b_beta_tol).all():
            return new_updated[0]
        start = new_start
        end = new_end
        updated = new_updated

    return updated[0]
