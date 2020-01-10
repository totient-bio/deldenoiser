##############################################################################
# deldenoiser is a python package supporting the deldenoiser command line tool
# which is used to denoise read counts of a DNA-encoded experiment.
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

import sys
import numpy as np
from itertools import product, repeat
import warnings
from scipy.special import gammaln, gammaincc, logsumexp
from scipy.optimize import minimize_scalar
from multiprocessing import Pool, cpu_count

##############################################################################
# Utility functions
##############################################################################


def one_hot_vector(indexes, max_index=None, one_based=False):
    """Generates "one-hot" vectors for a 1-d array of indexes.

    Args:
        indexes (numpy.ndarray): 1-d array of integers
        max_index (int): highest index to be represented
        one_based (bool): if True, the indexes are interpreted as 1-based

    Returns:
        numpy.ndarray: 2-d array of shape (len(indexes), max_index),
        where each row contains only one 1 and all other entries are 0s

    Raises:
        ValueError: if any of the indexes is outside the range defined by
            max_index and one_based
    """
    lowest_allowed_index = 1 if one_based else 0
    if np.min(indexes) < lowest_allowed_index:
        raise ValueError(f'indexes must be >= {lowest_allowed_index}')
    found_max = np.max(indexes)
    if max_index is None:
        max_index = found_max
    if max_index < found_max:
        raise ValueError(f'indexes must be <= max_index ({found_max})')
    if one_based:
        indexes = indexes - 1
        max_index = max_index - 1
    one_hot = np.zeros((len(indexes), max_index + 1), dtype=int)
    one_hot[np.arange(one_hot.shape[0]), indexes] = 1
    return one_hot


def estimate_scalar_background(N, floor, xb, xs, alpha_b, alpha_s, gamma):
    """Finds the optimal positive beta value that minimizes the PBS likelihood.

    log-likelihood(beta) = - alpha_b * beta + alpha_s * sum(xb/xs) * beta
    + sum( logQ( N/gamma + 1 , (alpha_s/xs + 1/gamma)*(x_b*beta + floor) ))
    where logQ(...) is numpy.log(scipy.special.gammaincc(...))

    The interpretation of this model is the following:

    Counts, N, come from a dispersed Poisson distribution,
    whose over-dispersion parameter is gamma,
    and its intensity consists of the following components

    1. An intensity floor, which we already know is present.

    2. Background intensity, with a known profile xLT,
    but unknown intensity

    3. Unknown and independent signals with typical size of 1/alpha,
    that affect the intensity with a known profile xs.

    In other words,
    intensity[i] = floor[i] + xLT[i]*background + xs[i]*signals[i]
    where signals[i] ~ Exponential(alpha) for all i

    Computation is made efficient by first checking the sign of the derivative
    of the negative log-likelihood at beta = 0. If it is positive,
    the minimum is at beta = 0, else it is at some beta > 0 value, which
    we find numerically.

    Args:
        N (numpy.ndarray): counts vector of length L (>= 0, integers)
        floor (numpy.ndarray): intensity floor vector of length L (>= 0)
        xb (numpy.ndarray): background profile vector of length L (> 0)
        xs (numpy.ndarray): signal profile vector of length L (> 0)
        alpha_b (float): strength of regularization of background (> 0)
        alpha_s (float): strength of regularization of signals (> 0)
        gamma (float): over-dispersion parameter (> 0, usually >= 1)

    Returns:
        float: optimal beta value, which minimizes the log-likelihood
    """
    c0 = alpha_b - alpha_s * np.sum(xb / xs)

    # Check if the c0 term alone is enough to make the
    # negative log-likelihood increasing at beta = 0
    if c0 >= 0.0:
        return 0.0

    c1 = (alpha_s / xs + 1.0 / gamma) * floor
    c2 = (alpha_s / xs + 1.0 / gamma) * xb
    n = N / float(gamma) + 1

    # Check if the negative log-likelihood is increasing at beta = 0,
    # by checking the sign of the derivative, i.e.
    # c0 + sum( c2 * c1^(n-1) * exp(-c1) / Gamma(n, c1) ) >? 0

    # if n == 1, the formula is much simpler
    # because Gamma(1, z) = exp(-z)
    terms_1 = np.log(c2[n == 1])

    # For n > 1, we only need to compute terms where c1 > 0,
    # because terms where c1 == 0 (if n == 1) contribute 0
    compute = ((n > 1) & (c1 > 0.0))
    terms_2 = (np.log(c2[compute])
               + (n[compute] - 1) * np.log(c1[compute])
               - c1[compute]
               - gammaln(n[compute])
               - gammainccln(n[compute], c1[compute])
               )

    terms = np.concatenate([terms_1, terms_2])
    if len(terms) == 0:
        logsumexp_of_terms = -np.inf
    else:
        logsumexp_of_terms = logsumexp(terms)
    if logsumexp_of_terms > np.log(-c0):
        return 0.0

    # Now, since the negative log-likelihood is decreasing at beta = 0,
    # we perform numerical minimization
    def mlog_likelihood(beta):
        return c0 * beta - np.sum(gammainccln(n, c1 + c2 * beta))

    upper_bound = max(0, np.max((N - floor) / xb))
    res = minimize_scalar(mlog_likelihood,
                          method='bounded',
                          bounds=(0, upper_bound))
    if not res.success:
        warnings.warn('Background estimation failed.')
    beta_opt = res.x
    return beta_opt


def gammainccln(s, x):
    """Computes ln(gammaincc(s,x)) without underflow. Fast if s has many 1s.

    See: https://en.wikipedia.org/wiki/Incomplete_gamma_function#Asymptotic_behavior  # noqa

    Args:
        s (numpy.ndarray): first argument of incomplete gamma function
        x (numpy.ndarray): second argument of incomplete gamma function

    Returns:
        numpy.ndarray: ln(gammaincc(s,x)) for arrays s, x of the same shape
    """
    arr = np.zeros_like(x, dtype=float)

    # For s = 1, use the the identity ln(gammaincc(1, x)) = -x
    # This speeds up evaluation for inputs where many s entries are 1
    is_s_one = (s == 1)
    arr[is_s_one] = -x[is_s_one]

    # For all other s values, call scipy.special.gammaincc,
    # and don't forget to take the log of these in the next steps.
    arr[~is_s_one] = gammaincc(s[~is_s_one], x[~is_s_one])

    # For some inputs scipy.special.gammaincc returns 0.0.
    # This happens when x >> s.
    # Let's avoid taking the log of these entries,
    # and evaluate the first 3 terms of the asymptotic series instead.
    underflow = (~is_s_one) & (arr == 0.0)
    take_log = (~is_s_one) & (arr > 0.0)
    arr[take_log] = np.log(arr[take_log])
    x = x[underflow]
    s = s[underflow]
    arr[underflow] = -x + (s - 1) * np.log(x) + np.log(1 + (s - 1) / x) \
                     - gammaln(s)
    return arr


##############################################################################
# Main functions
##############################################################################

def _estimate_one_fitness(T_and_beta, X, N, floor, design, lane_indexes,
                          alpha, gamma):
    """Estimate beta[T] for one truncate T.

    Args:
        T_and_beta (sequence): truncate index, (T[1], T[2], ... T[cmax]),
            T[c] = 0, 1, ... lmax[c], and initial beta value, which is
            already incorporated in floor
        X (numpy.ndarray): effect matrix connecting beta to intensities
        N (numpy.ndarray): 1-d array of read counts
            associated with each lane index combination, len: prod(design)
        floor (numpy.ndarray): intensity floor, same shape as N
        design (numpy.ndarray): vector of number of lanes in each cycle
        lane_indexes (numpy.ndarray): 2-d array of all lane index combinations
        alpha (float): regularization strength
            (i.e. inverse of the typical expected fitness coefficient,
             usually 1.0 is a good choice)
        gamma (float): dispersion parameter of the Poisson noise
            (the more PCR cycles are used that higher gamma should be)

    Returns:
        dict
            "T_idx": index of the T extended lane combination in
                the L_extended array
            "beta": estimated fitness for truncate T
            "floor_increment": amount by which this truncate
                raises the Poisson intensity floor
    """
    cmax = len(design)
    T, beta_in_floor = T_and_beta

    s_idx = np.sign(T).dot(2 ** np.arange(cmax - 1, -1, -1))
    T_idx = np.array(T).dot(np.concatenate([
        np.cumprod(np.array(design[::-1]) + 1)[::-1][1:], [1]
    ]))
    affected = np.ones(lane_indexes.shape[0], dtype=bool)
    for c, Tc in enumerate(T):
        if Tc == 0:  # 0 means this truncate affects all lanes of cycle c
            continue
        affected[lane_indexes[:, c] != Tc] = False

    XT = X[:, s_idx][affected]
    XLT = X[affected, -1]  # removing column of legitimate compounds
    NT = N[affected]
    floorT = floor[affected]
    floorT_without_beta_in_floor = np.max([np.zeros_like(floorT),
                                           floorT - beta_in_floor * XT],
                                          axis=0)
    beta_T = estimate_scalar_background(NT, floorT_without_beta_in_floor,
                                        XT, XLT,
                                        alpha, alpha,
                                        gamma)
    return {'T_idx': T_idx,
            'beta': beta_T}


def _estimate_many_fitness(T_and_beta_list, X, N, floor, design, lane_indexes,
                           alpha, gamma):
    results = []
    for T_and_beta in T_and_beta_list:
        results.append(
            _estimate_one_fitness(T_and_beta,
                                  X, N, floor, design, lane_indexes,
                                  alpha, gamma)
        )
    return results


class NullblockModel:
    """Realizes the DEL denoising model that uses 'null' blocks.

    Attributes:
        design (numpy.ndarray): 1-d array of number of lanes in each cycle
        lane_indexes (numpy.ndarray): 2-d array of all lane index combinations
        bb_indexes (numpy.ndarray): 2-d array of all possible
            building block combinations
        alpha (float): regularization strength
            (i.e. inverse of the typical expected fitness coefficient,
             usually 1.0 is a good choice)
        gamma (float): dispersion parameter of the Poisson noise
            (the more PCR cycles are used that higher gamma should be)
        yields (list): list of yield vectors, where each vector contains
            the yields of lanes in the corresponding cycle
        inventory_matrix (numpy.ndarray): 2-d array where each row
            contains the fractional composition corresponding to a
            lane combination


    """

    def __init__(self, design, alpha, gamma):
        NullblockModel._check_design(design)
        NullblockModel._check_positive_float(alpha, 'alpha')
        NullblockModel._check_positive_float(gamma, 'gamma')
        self.design = np.array(design)
        self.lane_indexes = self._create_lane_index(include_zero=False)
        self.bb_indexes = self._create_lane_index(include_zero=True)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.yields = None
        self.inventory_matrix = None

    @staticmethod
    def _check_design(design):
        error_message = 'design must be a sequence of positive integers, ' \
            f'value passed: {design}'
        try:
            d_arr = np.array(design)
            if len(d_arr.shape) != 1:
                raise ValueError(error_message)
            if len(d_arr) == 0:
                raise ValueError(error_message)
            if np.sum(d_arr <= 0) > 0:
                raise ValueError(error_message)
        except ValueError:
            raise ValueError(error_message)

    @staticmethod
    def _check_positive_float(arg, name):
        error_message = f'{name} must be a positive floating point number, ' \
            f'value passed: {arg}'
        try:
            f = float(arg)
            if f <= 0.0:
                raise ValueError(error_message)
        except ValueError:
            raise ValueError(error_message)

    def _create_lane_index(self, include_zero):
        """Generates the matrix of all lane index combinations.

        Args:
            include_zero (bool): if True, the lane indexes are allowed to be 0
                otherwise they start from 1

        Returns:
            numpy.ndarray: matrix of lane index combinations of shape
                [prod(design), len(design)]
        """
        lmin = 0 if include_zero else 1
        lanes = np.array(
            list(
                product(
                    *[list(range(lmin, lmax_c + 1, 1))
                      for lmax_c in self.design])
            )
        )
        return lanes

    def _check_yields(self, yields):
        error_message = 'yields must be a sequence of vectors, where' \
                        'len(y[c]) == design[c], and all elements are ' \
                        f'between 0 and 1; value passed: {yields}'
        try:
            if (np.array(list(map(len, yields))) != self.design).any():
                raise ValueError(error_message)
            for c, y in enumerate(yields):
                for l in range(self.design[c]):
                    if y[l] > 1.0 or y[l] < 0.0:
                        raise ValueError(error_message)
        except ValueError:
            raise ValueError(error_message)

    def _create_inventory_matrix(self, yields=None):
        """Combines the design and the yields list into the J matrix.

        J[L, s] = prod_{c} (y_[c][L[c]])*s[c] + (1 -y_[c][L[c]])*(1 - s[c])
        where
            L is the lane index combinations (vector of cmax indexes)
            s is the success vector (vector of cmax 0 or 1 values)
            y is a list of yield vectors (each vector for one cycle)

        Returns:
            numpy.ndarray: matrix J[L, s] of shape [prod(design), 2^cmax],
            where each value represents the fraction of compounds
            corresponding to success vector s and tag L.
        """
        if yields is None:
            yields = self.yields
        else:
            self._check_yields(yields)
        if yields is None:
            raise ValueError('Setting self.yields is required. It is None.')
        for c, lmaxc in enumerate(self.design):
            try:
                lmaxc = int(lmaxc)
            except ValueError:
                raise ValueError('design must contain positive integers '
                                 f'(error at index {c}, value: {lmaxc}')
            if lmaxc < 1:
                raise ValueError('design must contain positive integers '
                                 f'(error at index {c}, value: {lmaxc}')
        cmax = len(self.design)
        y = yields
        s_list = list(product(*[[0, 1] for _ in range(cmax)]))
        J = np.zeros([np.prod(self.design), 2**cmax])
        for s_idx, s in enumerate(s_list):
            ys = []
            for c in range(cmax):
                ys.append(s[c] * y[c] + (1 - s[c]) * (1 - y[c]))
            Y = np.array(
                list(product(*ys)))  # spread out the ys entries across L
            J[:, s_idx] = np.prod(Y, axis=1)

        return J

    def load_yields(self, yields):
        """Load and preprocess yields for NullblockModel.

        Args:
            yields (list): list of 1-d yield vectors,
                where each vector contains the yields of lanes in the
                corresponding cycle

        Raises:
            ValueError, if yields does not conform with self.design
        """
        self._check_yields(yields)
        if self.yields is not None:
            warnings.warn('yields was set already, '
                          'its value has been overwritten.')
        self.yields = yields
        self.inventory_matrix = self._create_inventory_matrix()

    def fit_sequencing_bias(self, pre_selection_readcounts):
        """Estimates sequencing bias factors from pre-selection read counts.

        It assumes that the cycles have independent and multiplicative effect.

        Args:
            pre_selection_readcounts (numpy.ndarray): 1-d array of read counts
            associated with each lane index combination, length: prod(design)

        Returns:
            numpy.ndarray, list: numpy.ndarray: estimated bias
            (normalized: sums to total read number) for each lane combination

            list: list of cycle- and lane- specific sequencing bias,
            this has the same shape as yields
        """
        L = self.lane_indexes
        N = pre_selection_readcounts
        Ntot = float(np.sum(N))

        cycle_probs = []
        for c, lmax in enumerate(self.design):
            p = np.zeros(lmax)
            for l in range(1, lmax + 1, 1):
                p[l - 1] = np.sum(N[L[:, c] == l]) / Ntot
            cycle_probs.append(p)

        p_on_L = np.array(list(product(*cycle_probs)))
        bias = np.prod(p_on_L, axis=1) * Ntot
        return bias, cycle_probs

    def _get_s_index(self, T):
        cmax = len(T)
        s = np.sign(T)
        place_value = 2 ** np.arange(cmax - 1, -1, -1)
        return s.dot(place_value)

    def _get_T_index(self, T):
        place_value = np.concatenate([
            np.cumprod(np.array(self.design[::-1]) + 1)[::-1][1:], [1]
        ])
        return np.array(T).dot(place_value)

    def _get_affected_lanes_T(self, T):
        L = self.lane_indexes
        affected = np.ones(L.shape[0], dtype=bool)
        for c, Tc in enumerate(T):
            if Tc == 0:  # 0 means this truncate affects all lanes of cycle c
                continue
            affected[L[:, c] != Tc] = False
        return affected

    def fit_truncates(self,
                      sequencing_bias,
                      post_selection_readcounts,
                      beta_init=None,
                      max_processes=None,
                      maxiter=2,
                      tol=0.1,
                      debug=False):
        """Estimates the fitness parameter, beta[T] for each of truncate T.

        Args:
            sequencing_bias (numpy.ndarray): 1-d array of length prod(design),
                containing the sequencing biases estimated from pre-selection
                read counts
            post_selection_readcounts (numpy.ndarray): 1-d array of read counts
                associated with each lane index combination,
                length: prod(design)
            beta_init (numpy.ndarray or None): 1-d array of initial beta values
            max_processes (int or None): maximum number of parallel processes
                to use, default is the number of system CPUs
            maxiter (int): maximum number of iterations of coordinate descent
            tol (float): if the intensity due to truncates changes less than
                this amount, the iterations are terminated early
            debug (bool): if True, progress of coordinate descend is printed
                to stderr

        Returns:
            numpy.ndarray, numpy.ndarray: numpy.ndarray: beta,
            fitted fitness vector of all truncates (length: prod(design+1))

            numpy.ndarray: floor, total intensity contribution of truncates
            according to the fitted fitness values (length: prod(design))
        """
        if max_processes is None:
            max_processes = cpu_count()
        if max_processes > 1000:
            warnings.warn('Max number of processes is reduced to 1000.')
            max_processes = 1000

        cmax = len(self.design)
        J = self.inventory_matrix
        bias = (np.sum(post_selection_readcounts)
                * sequencing_bias
                / np.sum(sequencing_bias)
                )
        X = np.einsum('L,Ls->Ls', bias, J)
        N = post_selection_readcounts
        s_list = list(product(*[[0, 1] for _ in range(cmax)]))
        s_list.sort(key=lambda s: sum(s))

        if beta_init is None:
            beta = np.zeros(np.prod(self.design + 1))
        else:
            beta = beta_init.copy()
        floor = np.sum(X * self._create_beta_matrix(beta), axis=1)

        if debug:
            loglike = self.loglikelihood_of_truncates(
                self._create_beta_matrix(beta),
                sequencing_bias,
                post_selection_readcounts)
            print(f'fitting truncates, '
                  f'initial, '
                  f'loglike: {loglike}',
                  file=sys.stderr
                  )

        for it in range(maxiter):

            floor_start = floor.copy()

            # Iterate through all s success vectors {0,1}^cmax,
            # omitting the s vector (1,1,... 1)
            # Collect the T indexes corresponding to it.
            # Estimate the fitness (beta) of each T, using the same floor.
            # After estimating all beta for a given s, update the floor.
            for s in s_list[:-1]:

                Tc_lists = []
                for c, lmax_c in enumerate(self.design):
                    if s[c] == 0:
                        Tc_lists.append([0])
                    else:
                        Tc_lists.append(list(range(1, lmax_c + 1, 1)))
                T_list = list(product(*Tc_lists))
                T_arr = np.array(T_list)
                beta_list = beta[self._get_T_index(T_arr)]
                T_and_beta_list = list(zip(T_list, beta_list))

                processes = min([len(T_and_beta_list), max_processes])
                T_and_beta_lists = [[] for _ in range(processes)]
                for idx, T_and_beta in enumerate(T_and_beta_list):
                    T_and_beta_lists[idx % processes].append(T_and_beta)
                with Pool(processes) as pool:
                    results = pool.starmap(_estimate_many_fitness,
                                           zip(T_and_beta_lists,
                                               repeat(X),
                                               repeat(N),
                                               repeat(floor),
                                               repeat(self.design),
                                               repeat(self.lane_indexes),
                                               repeat(self.alpha),
                                               repeat(self.gamma)
                                               )
                                           )

                for result in results:
                    for res in result:
                        beta[res['T_idx']] = res['beta']

                floor = np.sum(X * self._create_beta_matrix(beta), axis=1)

                if debug:
                    loglike = self.loglikelihood_of_truncates(
                        self._create_beta_matrix(beta),
                        sequencing_bias,
                        post_selection_readcounts)
                    print(f'fitting truncates, '
                          f'iteration {it+1}, '
                          f'class {"".join(map(str, s))}, '
                          f'log-likelihood: {loglike}',
                          file=sys.stderr
                          )

            if (np.abs(floor - floor_start) <= tol).all():
                break

        return beta, floor

    def fit_legitimates(self,
                        sequencing_bias,
                        post_selection_readcounts,
                        floor):
        """Estimates fitness parameter (beta) of legitimate products.

        Args:
            sequencing_bias (numpy.ndarray): 1-d array of length prod(design),
                containing the sequencing biases estimated from pre-selection
                read counts
            post_selection_readcounts (numpy.ndarray): 1-d array of read counts
                associated with each lane index combination,
                length: prod(design)
            floor (numpy.ndarray): 1-d array of intensity floor estimated by
                NullblockModel().fit_truncates

        Returns:
            dict:
                'mode': numpy.ndarray: maximum a posteriori estimated of
                each fitness parameter

                'mean': numpy.ndarray: posterior mean
                of each fitness parameter

                'std': numpy.ndarray: posterior standard deviation of
                each fitness parameter
        """
        J = self.inventory_matrix
        gamma = self.gamma
        alpha = self.alpha
        bias = (np.sum(post_selection_readcounts)
                * sequencing_bias
                / np.sum(sequencing_bias)
                )
        X = np.einsum('L,Ls->Ls', bias, J)
        XL = X[:, -1]
        N = post_selection_readcounts

        beta_mode = N / (XL + gamma * alpha) - floor / XL
        beta_mode[beta_mode < 0] = 0.0

        n = N / gamma + 1
        x = floor * (1.0 / gamma + alpha / XL)
        g1 = np.exp((gammaln(n + 1) + gammainccln(n + 1, x)) -
                    (gammaln(n) + gammainccln(n, x)))
        g2 = np.exp((gammaln(n + 2) + gammainccln(n + 2, x)) -
                    (gammaln(n) + gammainccln(n, x)))

        beta_mean = gamma / (XL + alpha * gamma) * g1 - floor / XL
        beta_var = gamma**2 / (XL + alpha * gamma)**2 * (g2 - g1**2)
        beta_std = np.sqrt(beta_var)

        return {
            'mode': beta_mode,
            'mean': beta_mean,
            'std': beta_std
        }

    def _create_beta_matrix(self, beta):
        """Transforms the beta[D] vector into beta[L,s] matrix.

        Args:
            beta (numpy.ndarray): 1-d array of fitness coefficients beta[D]

        Returns:
            numpy.ndarray: 2-d array of beta[L,s] = beta[D] if D = L * s

        """
        cmax = len(self.design)
        L = self.lane_indexes
        beta_matrix_columns = []
        for s in list(product(*[[0, 1] for _ in range(cmax)])):
            sL = np.array(s) * L
            beta_sL = beta[self._get_T_index(sL)]
            beta_matrix_columns.append(beta_sL)
        beta_matrix = np.concatenate(beta_matrix_columns) \
            .reshape([2 ** cmax, -1]).T
        return beta_matrix

    def compute_readcount_breakdown(self,
                                    sequencing_bias,
                                    beta_truncates,
                                    beta_legitimates,
                                    post_selection_readcounts):
        """Computes the estimated breakdown of observed read counts.

        Args:
            sequencing_bias (numpy.ndarray): 1-d array of length prod(design),
                containing the sequencing biases estimated from pre-selection
                read counts
            beta_truncates (numpy.ndarray): fitness of truncates,
                estimated by NullblockModel.fit_truncates()
            beta_legitimates (numpy.ndarray): fitness of legitimate compounds,
                estimated by NullblockModel.fit_legitimates()
            post_selection_readcounts (numpy.ndarray): 1-d array of read counts
                associated with each lane index combination,
                length: prod(design)

        Returns:
            numpy.ndarray: 2-d array of read counts, where each row sums to
            the observed read count, and the value are the mean posterior
            estimates of the counts broken down by success status, s.
        """
        J = self.inventory_matrix
        N = post_selection_readcounts
        bias = np.sum(N) * sequencing_bias / np.sum(sequencing_bias)
        X = np.einsum('L,Ls->Ls', bias, J)

        Lext = self.bb_indexes
        truncates = (np.sum(Lext == 0, axis=1) > 0)
        beta = beta_truncates.copy()
        beta[~truncates] = beta_legitimates
        beta_matrix = self._create_beta_matrix(beta)
        intensity_breakdown_matrix = X * beta_matrix
        denom = np.sum(intensity_breakdown_matrix, axis=1)
        denom[denom == 0] = 1e-6
        norm = N / denom
        count_breakdown_matrix = np.einsum('Ls,L->Ls',
                                           intensity_breakdown_matrix, norm)
        return count_breakdown_matrix

    def loglikelihood_of_truncates(self,
                                   beta_matrix,
                                   sequencing_bias,
                                   post_selection_readcounts,
                                   yields=None):
        """Computes the log likelihood of the truncates.

        Args:
            beta_matrix (numpy.ndarray): 2-d array of beta_matrix[L,s]
                of truncates (which is the same as beta_vector[D] for D = L*s)
            sequencing_bias (numpy.ndarray): 1-d array of length prod(design),
                containing the sequencing biases estimated from pre-selection
                read counts
            post_selection_readcounts (numpy.ndarray): 1-d array of read counts
                associated with each lane index combination,
                length: prod(design)
            yields (list or None): list of yield vectors,
                where each vector contains
            the yields of lanes in the corresponding cycle

        Returns:
            float: log-likelihood of truncate fitness parameters,
            where the fitness parameters of legitimate compounds are
            averaged over their prior

        """
        if yields is None:
            yields = self.yields
        else:
            self._check_yields(yields)
        if yields is None:
            raise ValueError('Setting self.yields is required. It is None.')
        alpha = self.alpha
        gamma = self.gamma

        N = post_selection_readcounts

        J = self._create_inventory_matrix(yields)
        bias = np.sum(N) * sequencing_bias / np.sum(sequencing_bias)
        X = np.einsum('L,Ls->Ls', bias, J)
        XL = X[:, -1]  # last column contains to the X_{LL} entries
        beta_truncates_matrix = beta_matrix[:, :-1]
        F = np.sum(X[:, :-1] * beta_truncates_matrix, axis=1)

        n = N / gamma + 1
        loglike_terms = (np.log(gamma)
                         - np.log(XL)
                         - n * np.log(gamma * alpha / XL + 1)
                         + alpha * F / XL
                         + gammainccln(n, (alpha / XL + 1.0 / gamma) * F)
                         )
        loglike = np.sum(loglike_terms)

        return loglike
