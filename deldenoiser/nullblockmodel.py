##############################################################################
# Implementation of the Null-block model
#
#
# Stored attributes:
#
#   cycles: number of synthesis cycles
#   design: list of lane numbers in each cycle, r_c^{max}
#   alpha: L1 regularization parameter
#   gamma: dispersion parameter
#   bhat: sequence imbalance estimates, bhat_{c, r_c}
#   yields: yields of synthesis reactions, Y_{c, r_c}
#   x: x_{c, q_c} = bhat_{c,q_c} Y_{c, q_c},  if q_c > 0
#      x_{c, 0} = 1.0 - sum_{r_c} bhat_{c, r_c} Y_{c, r_c}
#   r_arr: array of reaction index vectors, 1 <= r_c <= r_c^{max}
#       for which N_r > 0
#   n: vector of scaled post-selection readcounts, n_r = N_r / gamma
#   ntot: total scaled readcount
#   Z: vector of Z_{r} = alpha + X_{r,r}/gamma,
#      where X_{r,r} = N_tot * prod_{c} x_{c, r_c}
#   a: background intensity, a_r = sum_q b_r[q] * F_q
#      where b_r[q] = (alpha + X_{r,r} / gamma)
#                     * prod_{c: q_c==0} (1 - yields[c,r_c]) / yields[c,r_c]
#   q_arr: array of truncates indexes which have potentially F_q > 0
#   F: vector of fitness coefficients of the truncates, F_q
#   c0: c0[q] = - alpha + sum_{r} b_r[q]
#               - ntot * prod_{c} x_{c, q_c}
#       where b_r[q] = Z_{r}
#                      * prod_{c: q_c==0} (1 - yields[c,r_c]) / yields[c,r_c]
#
#
# Functions implementing the following steps:
#
#   1. fit_sequence_imbalance(): Read pre-selection read count data,
#       line-by-line, and compute bhat
#   2. load_yields(): Load yields
#   3. load_postselection_readcount(): Load post-selection read count data
#       entirely, and compute Z and a
#   4. guess_truncates(): Find truncates (q) that are non-zero
#       and compute initial guesses
#   5. fit_truncates(): Fit fitness of truncates
#   6. fit_fullcycles(): Fit full-cycle product fitness coefficients
#   7. compute_clean_readcounts(): Compute clean readcounts
#       for full-cycle products
#
#
# Multiprocessing considerations:
#
#   The most time-consuming steps are step 4 and 5.
#   They can be sped up by distributing the tasks for different q truncates
#   to different processes.
#   The processes share the following arrays: r_arr, n, Z, a
#   To save memory,
#   these arrays must be initialized using the following recipe:
#   >>> import multiprocessing as mp
#   >>> import ctypes
#   >>> import numpy as np
#   >>> base = mp.Array(ctypes.c_double, <size>)  # or with `ctypes.c_int`
#   >>> shared_array = np.ctypeslib.as_array(base.get_obj())
#   >>> shared_array = shared_array.reshape(<shape>)
#   Furthermore, mp.Array() returns a multiprocessing.RLock object,
#   which is used to block overlapping edits on array a
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

import sys
import gzip
import numpy as np
import pandas as pd
import multiprocessing as mp
from itertools import product, chain, cycle
import ctypes
from warnings import warn
from deldenoiser.pbs_algorithm import pbs_optimize, gammainccln

##############################################################################
# Globals ####################################################################
##############################################################################
# These arrays must be stored globally, because they are accessed by functions
# below that also must be defined globally for the multiprocessing starmap()
# to be able to distribute them on pool workers.
#
# Because of this design choice at most one instance of NullBlockModel class
# should be created. Constructing NullBlockModel object resets these globals.

global_r_arr = None
global_n = None
global_Z = None
global_a = None
global_a_lock = None

NUMERICAL_FLOOR = 1e-10

##############################################################################


def _read_gzip_or_nogzip(path):
    if path.endswith('.gz'):
        with gzip.open(path, mode='rt', encoding='utf-8') as f:
            for line in f:
                yield line
    else:
        with open(path, 'r') as f:
            for line in f:
                yield line


def _create_shared_array(shape, ctype, lock=True):
    """Create a numpy array that can be shared between parallel processes.

    Args:
        shape (iterable of int): shape of numpy array
        ctype (ctypes type): dtype of numpy array
        lock (bool): if True, the array has multiprocessing lock

    Returns:
        numpy.ndarray: numpy array referring to the data in shared memory
        multiprocessing.synchronize.RLock or None: lock object associated with
            the array
    """
    base = mp.Array(ctype, int(np.prod(shape)), lock=lock)
    if lock:
        arr_lock = base.get_lock()
        base = base.get_obj()
    else:
        arr_lock = None
    arr = np.ctypeslib.as_array(base)
    arr = arr.reshape(shape)
    return arr, arr_lock


def _guess_one_truncate(q, cycles, alpha, ntot, x, yields):
    c0, b, n_median = NullBlockModel._compute_c0_b_nmedian(
        q, cycles, alpha, ntot, x, yields)
    if c0 < 0.0:
        return q, 0.0, c0
    guess = len(b) * n_median / float(np.sum(b) - c0)
    return q, guess, c0


def _fit_one_truncate(q_idx, q, Fq, c0, yields, inner_itermax, tol):
    n, a, b, idxs = NullBlockModel._compute_n_a_b_idx(q, Fq, yields)
    new_Fq = pbs_optimize(n, a, b, c0,
                          beta_start=Fq,
                          itermax=inner_itermax,
                          b_beta_tol=tol)
    global_a_lock.acquire()
    if Fq is None:
        global_a[idxs] += b * new_Fq
    else:
        global_a[idxs] = np.max(np.stack([np.ones_like(b) * NUMERICAL_FLOOR,
                                          global_a[idxs] + b * (new_Fq - Fq)
                                          ]),
                                axis=0)
    global_a_lock.release()
    return q_idx, new_Fq


class NullBlockModel:
    """Implements the Null-block model for denoising DEL readcounts

    Attributes:
        cycles (int): number of synthsis cycles
        design (numpy.ndarray): 1-d array of positive integers,
            each indicating the number of synthesis lanes in the cycles
        alpha (float): positive regularization constant,
            equivalent to L1 (lasso) regularization constant
        gamma (float): positive dispersion constant,
            gamma == 1 means a Poisson noise is assumed
            gamma > 1 means an over-dispersed Poisson distribution is assumed
            gamma < 1 means an under-dispersed Poisson distribution is used
        bhat (list of numpy.ndarray): sequence imbalance factors,
            each bhat[c][r_c] stand for cycle c, reaction r_c
        yields (list of numpy.ndarray): reaction yields,
            each yields[c][r_c] stands for cycle c, reaction r_c
        x (list of numpy.ndarray): intermediary variable
            x = {x[c][q_c]: c = 1, 2, len(design), q_c = 0, 1, ... design[c]}
            x[c][q_c] = bhat[c][q_c] * yields[c][q_c], if q_c != 0
            x[c][q_c] = sum_{r_c} bhat[c][r_c]*(1 - yields[c][r_c]), if q_c==0
        r_arr (numpy.ndarray): 2-d integer array containing
            the reaction indexes
            of the non-zero counts in the post-selection read count data
        n (numpy.ndarray): 1-d integer array containing the non-zero
            scaled counts, n = N / gamma
        ntot (int): total scaled counts
        Z (numpy.ndarray): 1-d array, Z_{r} = alpha + X_{r,r}/gamma,
            where X_{r,r} = N_tot * prod_{c} x[c][r_c]
        a (numpy.ndarray): 1-d array, a_r = sum_q b_r[q] * F_q
            where
            where b_r[q] = Z_{r} *
                prod_{c: q_c==0} (1 - yields[c,r_c]) / yields[c,r_c]
            and F_q is the fitness coefficient of truncate q
        q_arr (numpy.ndarray): 2-d array of truncate indexes
            where only those are stored that have potentially non-zero F_q
        F (numpy.ndarray): 1-d array, F_q fitness coefficients,
            for only those q, that are stored in q_arr
        c0 (numpy.ndarray): 1-d array, sum_{r} X_{r,q}, intermediary array
            c0[q] = - alpha + sum_{r} b_r[q]
                    - ntot * prod_{c} prod_{c} x[c][q_c]
    """
    def __init__(self, design, alpha=1.0, gamma=1.0, default_yield=0.5):
        # check input arguments
        NullBlockModel._check_design(design)
        NullBlockModel._check_positive_float(alpha, 'alpha')
        NullBlockModel._check_positive_float(gamma, 'gamma')

        # load attributes directly from input arguments
        self.cycles = len(design)
        self.design = design
        self.alpha = float(alpha)
        self.gamma = float(gamma)

        # load default values for bhat, yields and x
        self.bhat = [np.ones(lanes, dtype=float) / lanes for lanes in design]
        self.yields = [default_yield * np.ones(lanes, dtype=float)
                       for lanes in design]
        self.x = self._compute_x()

        # tie r_arr, n, Z, a to the globals and reset them
        global global_r_arr
        global_r_arr = None
        self.r_arr = global_r_arr

        global global_n
        global_n = None
        self.n = global_n

        global global_Z
        global_Z = None
        self.Z = global_Z

        global global_a
        global_a = None
        self.a = global_a

        global global_a_lock
        global_a_lock = None

        # initialize ntot, q_arr, F, c0 to empty values
        self.ntot = 0.0
        self.q_arr = np.zeros([0, self.cycles], dtype=int)
        self.F = np.zeros(0, dtype=float)
        self.c0 = np.zeros(0, dtype=float)

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

    def _compute_x(self):
        cycles = self.cycles
        bhat = self.bhat
        yields = self.yields
        x = [np.concatenate((np.array([0]),
                             bhat[c] * yields[c])) for c in range(cycles)]
        for c in range(cycles):
            x[c][0] = 1.0 - np.sum(x[c])
        return x

    def _check_read_count_file(self, read_count_file):
        expected_header = [f'cycle_{c+1}_lane' for c in range(self.cycles)] \
            + ['readcount']
        line = _read_gzip_or_nogzip(read_count_file).__next__()
        header = line.strip().split('\t')
        if header != expected_header:
            error_message = f'The header {expected_header} is expected, '\
                f'but not found in {read_count_file}. ' \
                f'Header found: {header}'
            raise ValueError(error_message)

    def fit_sequence_imbalance(self, pre_selection_read_count_file):
        """Fit the sequence imbalance factors, bhat[c][r_c] from data file

        Using the formula:
            ```
            bhat[c][r_c] = (1 + sum_{r': r'_c == r_c} N[r']) /
                           (N0[c] +  sum_{r'} N[r'])
            ```
        where N[r'] is the number of pre-selection read count for index r',
        and N0[c] is the number of different r_c values available for cycle c.
        This assumes flat priors for all {bhat[c][r_c]: r_c = 1..r_c^{max}}

        The input file is not loaded to memory, instead it is streamed.
        This allows computing the bhat factors from very big DEL libraries.

        Does not return anything. It updates self.bhat and self.x.

        Args:
            pre_selection_read_count_file (str): file path

        """
        self._check_read_count_file(
            pre_selection_read_count_file)

        cycles = self.cycles
        design = self.design
        bhat = [np.ones(lanes, dtype=float) for lanes in design]
        Ntot = 0
        fin = _read_gzip_or_nogzip(pre_selection_read_count_file)
        fin.__next__()  # skip header

        for line_num, line in enumerate(fin):
            try:
                record = list(map(int, line.strip().split('\t')))
            except ValueError:
                warn(f'Line {line_num + 1} cannot be processed, '
                     f'it is skipped: {line}')
                continue
            if len(record) != cycles + 1:
                warn(f'Line {line_num + 1} contains wrong number of entries, '
                     f'it is skipped: {line}')
                continue

            r = record[:-1]
            Nr = record[-1]
            if Nr > 0:
                for c in range(cycles):
                    bhat[c][r[c] - 1] += Nr
                Ntot += Nr

        for c in range(cycles):
            bhat[c] /= float(design[c] + Ntot)

        self.bhat = bhat
        self.x = self._compute_x()

    def _check_yields_file(self, yields_file, minyield, maxyield):
        if yields_file.endswith('.gz'):
            compression = 'gzip'
        else:
            compression = None
        df_yields = pd.read_csv(yields_file, sep='\t',
                                compression=compression)
        if list(df_yields.columns) != ['cycle', 'lane', 'yield']:
            raise ValueError('yields file must have columns '
                             '"cycle", "lane", "yields')

        error_message = '"cycle" column of yields file ' \
                        'must contain positive integers'
        if ((df_yields['cycle'].values !=
             df_yields['cycle'].values.astype(int)).any()):
            raise ValueError(error_message)
        if ((df_yields['cycle'].values <= 0).any()):
            raise ValueError(error_message)

        error_message = '"lane" column of yields file ' \
                        'must contain positive integers'
        if ((df_yields['lane'].values !=
             df_yields['lane'].values.astype(int)).any()):
            raise ValueError(error_message)
        if (df_yields['lane'].values <= 0).any():
            raise ValueError(error_message)

        error_message = '"yield" column of yields file must contain ' \
                        'floating point values, 0.0 < yield < 1.0'
        df_yields.loc[df_yields['yield'] <= minyield, 'yield'] = minyield
        df_yields.loc[df_yields['yield'] >= maxyield, 'yield'] = maxyield
        if (df_yields['yield'] <= 0.0).any() or (
                df_yields['yield'] >= 1.0).any():
            raise ValueError(error_message)
        if np.isnan(df_yields['yield']).any():
            raise ValueError(error_message)

        df_yields.sort_values(by=['cycle', 'lane'], inplace=True)
        cycles = df_yields['cycle'].values
        lanes = df_yields['lane'].values
        yields = df_yields['yield'].values

        error_message = '"cycle" and "lane" columns in yields file ' \
                        'must list every cycle-lane ' \
                        'combination in the design in a sorted order.'
        cmax = self.cycles
        design = self.design
        expected_cycles = np.concatenate([(c + 1) * np.ones(design[c])
                                          for c in range(cmax)])
        expected_lanes = np.concatenate([np.arange(1, design[c] + 1, 1)
                                         for c in range(cmax)])
        if (cycles != expected_cycles).any() or (
                lanes != expected_lanes).any():
            raise ValueError(error_message)

        return yields

    def load_yields(self, yields_file, minyield, maxyield):
        """Check and load the reaction yields from file.

        Nothing is returned, self.yields and self.x gets updated.

        Args:
            yields_file (str): file path
            minyield (float): lowest allowed yield value, lower inputs are
                censored to this level
            maxyield (float): highest allowed yield value, higher inputs are
                censored to this level

        """
        cycles = self.cycles
        design = self.design
        yields = self._check_yields_file(yields_file, minyield, maxyield)
        yield_list = []
        idx_start = 0
        for c in range(cycles):
            yield_list.append(yields[idx_start:idx_start + design[c]])
            idx_start += design[c]

        self.yields = yield_list
        self.x = self._compute_x()

    def load_postselection_readcount(self,
                                     post_selection_read_count_file,
                                     index_dtype=ctypes.c_int16,
                                     count_dtype=ctypes.c_int32,
                                     float_dtype=ctypes.c_float):
        """Check and load post-selection read count data.

        Only records that contain positive read counts are kept in memory.
        Data is loaded into multiprocessing Arrays, so they
        can be concurrently accessed by parallel processes.

        Nothing is returned, self.r_arr, self.n, self.Z, self.a get updated.

        Args:
            post_selection_read_count_file (str): file path
            index_dtype (ctypes type): C type of r index
            count_dtype (ctypes type): C type of N readcounts
            float_dtype (ctypes type): C type of n = N/gamma

        """

        self._check_read_count_file(post_selection_read_count_file)

        cycles = self.cycles
        index_cols = [f'cycle_{c+1}_lane' for c in range(cycles)]
        dtypes = dict(zip(index_cols, [index_dtype] * cycles))
        dtypes['readcount'] = count_dtype
        if post_selection_read_count_file.endswith('gz'):
            compression = 'gzip'
        else:
            compression = None
        df_N = pd.read_csv(post_selection_read_count_file, sep='\t',
                           low_memory=True, dtype=dtypes,
                           error_bad_lines=False, warn_bad_lines=True,
                           compression=compression)
        if np.sum(df_N['readcount'] == 0) > 0:
            df_N = df_N[df_N['readcount'] > 0]

        records = len(df_N)

        global global_r_arr
        global_r_arr, _ = _create_shared_array([records, cycles],
                                               index_dtype, lock=False)
        self.r_arr = global_r_arr
        self.r_arr[:, :] = df_N[index_cols].values

        global global_n
        global_n, _ = _create_shared_array([records], float_dtype, lock=False)
        self.n = global_n
        self.n[:] = df_N['readcount'].values / float(self.gamma)
        self.ntot = np.sum(self.n)

        global global_Z
        global_Z, _ = _create_shared_array([records], float_dtype, lock=False)
        self.Z = global_Z
        self.Z[:] = self.ntot
        for c in range(cycles):
            self.Z *= self.x[c][self.r_arr[:, c]]
        self.Z += self.alpha

        global global_a
        global global_a_lock
        global_a, global_a_lock = _create_shared_array([records],
                                                       float_dtype, lock=True)
        self.a = global_a
        self.a += NUMERICAL_FLOOR

    @staticmethod
    def _compute_c0_b_nmedian(q, cycles, alpha, ntot, x, yields):
        """Computes the c0 and b parameters and the median n.

        c0 = -alpha + sum_r(b) - ntot * prod_{c} x[c][q_c]
        where b_r = Z_r * prod_{c: q_c==0} (1 - yields[c,r_c]) / yields[c,r_c]

        Args:
            q (iterable): truncate index vector
            guess (bool): if True, return a guess for

        Returns:
            (float, numpy.ndarray, float): c0, b, median(n)

        """
        q = np.array(q)
        qc_is_zero = (q == 0)
        truncated_c_idxs = np.argwhere(qc_is_zero)[:, 0]
        non_truncated_c_idxs = np.argwhere(~qc_is_zero)[:, 0]

        r_arr_aff = global_r_arr
        Z_aff = global_Z
        n_aff = global_n
        for c in non_truncated_c_idxs:
            condition = (r_arr_aff[:, c] == q[c])
            r_arr_aff = r_arr_aff[condition, :]
            Z_aff = Z_aff[condition]
            n_aff = n_aff[condition]
        b = Z_aff.copy()
        for c in truncated_c_idxs:
            b *= (1.0 / yields[c][r_arr_aff[:, c] - 1] - 1)
        c0 = (- alpha
              + np.sum(b)
              - ntot * np.prod([x[c][q[c]] for c in range(cycles)])
              )
        if len(n_aff) == 0:
            n_median = 0
        else:
            n_median = np.median(n_aff)

        return c0, b, n_median

    def guess_truncates(self, processes=mp.cpu_count()):
        """Determine which q truncates have non-zero F_q, and provide guesses

        The sign of the c0 parameter is used to determine if a truncate has
        non-zero optimal fitness.

        For those that are selected, the following guess is computed
            F_q^{guess} = N_aff * median(N_aff) / float(np.sum(b) - c0)
            where
                c0 = -alpha + sum_r(b) - (1/gamma) * Ntot * prod_c(x[c][q_c])
                b_r = (alpha + X_{r,r} / gamma)
                      * prod_{c: q_c==0} (1 - yields[c,r_c]) / yields[c,r_c]

        Nothing is returned. The arrays self.q_arr, self.F and self.c0
        get updated with values corresponding to q truncates that are deemed
        to have non-zero F_q fitness.

        Args:
            processes (int): number of parallel processes to start

        """
        cycles = self.cycles
        design = self.design
        alpha = self.alpha
        ntot = self.ntot
        x = self.x
        yields = self.yields

        # generate qs, the iterator for all q truncate indexes
        q_iters = []
        for mask in product(*[(0, 1)] * cycles):
            if sum(mask) == cycles:
                continue
            qc_values = []
            for m, d in zip(mask, design):
                if m == 0:
                    qc_values.append((0,))
                else:
                    qc_values.append(tuple(range(1, d + 1)))
            q_iter = product(*qc_values)
            q_iters.append(q_iter)
        qs = chain(*q_iters)

        pool = mp.Pool(processes=processes)
        # syntax: _guess_one_truncate(q, cycles, alpha, ntot, x, yields)
        results = pool.starmap(_guess_one_truncate,
                               zip(qs, cycle([cycles]), cycle([alpha]),
                                   cycle([ntot]), cycle([x]),
                                   cycle([yields])))
        pool.close()

        results_nonzero = []
        for res in results:
            q, Fq, c0 = res
            if Fq == 0.0:
                continue
            results_nonzero.append(res)
        results_nonzero.sort(key=lambda t: t[0])
        self.q_arr = np.array([res[0] for res in results_nonzero])
        self.F = np.array([res[1] for res in results_nonzero])
        self.c0 = np.array([res[2] for res in results_nonzero])

    @staticmethod
    def _compute_n_a_b_idx(q, Fq, yields):
        if Fq is None:
            Fq = 0.0
        q = np.array(q)
        qc_is_zero = (q == 0)
        truncated_c_idxs = np.argwhere(qc_is_zero)[:, 0]
        non_truncated_c_idxs = np.argwhere(~qc_is_zero)[:, 0]

        r_arr_aff = global_r_arr
        idx_aff = np.arange(r_arr_aff.shape[0])
        for c in non_truncated_c_idxs:
            condition = (r_arr_aff[:, c] == q[c])
            r_arr_aff = r_arr_aff[condition, :]
            idx_aff = idx_aff[condition]

        Z_aff = global_Z[idx_aff]
        n_aff = global_n[idx_aff]
        a_aff = global_a[idx_aff].copy()

        # Compute b_r = (alpha_s + X_{r,r}/gamma)
        #               * prod_{c: q_c = 0} (1-Y_{c,q_c}) / Y_{c, q_c}
        b = Z_aff.copy()
        for c in truncated_c_idxs:
            b *= (1.0 / yields[c][r_arr_aff[:, c] - 1] - 1)

        # Compute a_r = (alpha_s / X_{r,r,}  +  1 / gamma) * B_r - b_r F_q
        # but also make sure it's never negative
        a = np.max(np.stack((np.ones_like(a_aff) * NUMERICAL_FLOOR,
                             a_aff - b * Fq)), axis=0)

        return n_aff, a, b, idx_aff

    def fit_truncates(self, outer_itermax=100, inner_itermax=10, tol=0.5,
                      processes=mp.cpu_count(), debug=False, max_downsteps=5,
                      F_init=None):
        """Optimize the fitness of truncates iteratively.

        Args:
            outer_itermax (int): max number of iteration loops over truncates
            inner_itermax (int): max number of N-R or A-B iteration
            tol (float): iterations terminate if self.a changes less than this
                between iterations
            processes (int): number of parallel processes to start
            debug (bool): if True, print messages to stderr
            max_downsteps (int): if log-likelihood decreases
                this number of times throughout the optimization,
                the optimization is terminated
            F_init (None or float): staring guess for fitness values,
                if None the default guess of pbs_optimize() is used

        Nothing is returned. Arrays self.F and self.a are updated.

        """

        # setting initial F values to None makes pbs_algorithm compute the
        # same initial staring guess as self.guess_truncates()
        #
        if F_init is not None:
            F_init = float(F_init)
        Fqs = np.array([F_init for _ in range(self.q_arr.shape[0])])

        pool = mp.Pool(processes=processes)
        q_idxs = np.arange(self.q_arr.shape[0])

        a_curr = self.a.copy()
        logL_prev = -np.inf
        downward_steps = 0
        break_next = False
        for it in range(outer_itermax):
            results = pool.starmap(
                _fit_one_truncate,
                zip(q_idxs, self.q_arr, Fqs, self.c0, cycle([self.yields]),
                    cycle([inner_itermax]), cycle([tol / 2**self.cycles])))
            for res in results:
                q_idx, Fq = res
                Fqs[q_idx] = Fq
            max_change = np.max(np.abs(a_curr - self.a))
            a_curr[:] = self.a
            self.F = Fqs
            logL = self.log_likelihood()
            if debug:
                print(f'iteration {it+1}, logL: {logL}',
                      file=sys.stderr)
            if (max_change < tol) or break_next:
                break
            if logL < logL_prev:
                downward_steps += 1
            logL_prev = logL
            if downward_steps > max_downsteps:
                break_next = True
        pool.close()

    def fit_fullcycles(self):
        """Compute the maximum likelihood estimate of full-cycle fitnesses.

        hat F_r = max(0, N_r / (X_{r,r} + gamma * alpha) - hat B_r / X_{rr} )
        = max(0, (n_r - a_r) / Z_r )

        Returns:
            numpy.ndarray: estimated F_r fitness values for all r reactions
            that have non-zero counts

        """

        F = np.max(np.stack([
            np.zeros_like(self.n),
            (self.n - self.a) / self.Z
        ]), axis=0)
        return F

    def compute_clean_readcounts(self, F):
        """Compute the expectation value of N_{r,r}, given N_r and F.

        N_{r,r} = N_r * (X_{r,r} F_r) / sum_{q} X_{r,q} F_q
        = gamma * n_r * (Z_r F_r) / (Z_r F_r + a_r)

        Args:
            F (numpy.ndarray): fitness of full cycle products, computed by
                self.fit_fullcycles()

        Returns:
            numpy.ndarray: expectation value of N_{r,r}, i.e. the number of
            reads that are correctly attributed to the full cycle product r.

        """
        N_clean_ev = self.gamma * self.n * self.Z * F / (self.Z * F + self.a)
        return N_clean_ev

    def log_likelihood(self):
        logL = 0.0
        logL += np.sum(self.c0 * self.F)
        logL += np.sum(gammainccln(self.n + 1, self.a))
        return logL
