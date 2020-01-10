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

import pytest

import os

try:
    from deldenoiser import deldenoiser as deldenoiser
except ImportError:
    import sys

    sys.path.append(
        os.path.dirname(
            os.path.dirname(os.path.realpath(__file__))))
    from deldenoiser import deldenoiser as deldenoiser

import numpy as np


def is_approx(A, B, rel_error=1e-8, abs_error_floor=1e-10):
    A = np.array(A)
    B = np.array(B)
    if A.shape == B.shape and A.size == 0:
        return True
    if not (np.isinf(A) == np.isinf(B)).all():
        return False
    if not (A[np.isinf(A)] == B[np.isinf(B)]).all():
        return False
    A = A[~np.isinf(A)]
    B = B[~np.isinf(B)]
    diff = np.abs(A - B)
    size = np.max(np.abs([A, B]), axis=0)
    return (diff <= rel_error * size + abs_error_floor).all()


@pytest.mark.parametrize(
    "indexes, max_index, one_based, expected_results",
    [
        (np.array([0, 2, 4]), 5, False,
         np.array([
             [1, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0]
         ])),
        (np.array([1, 2, 4]), 4, True,
         np.array([
             [1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]
         ])),
        (np.array([1, 2, 4]), None, True,
         np.array([
             [1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]
         ])),
    ]
)
def test_one_hot_vector(indexes, max_index, one_based, expected_results):
    result = deldenoiser.one_hot_vector(indexes, max_index, one_based)
    assert (result == expected_results).all()


@pytest.mark.parametrize(
    "N, floor, xb, xs, alpha_b, alpha_s, gamma, expected_range",
    [
        (np.array([100, 1, 0, 1, 2, 1]),
         np.array([0, 0, 0, 0, 0, 0]),
         np.array([1, 1, 1, 1, 1, 1]),
         np.array([1, 1, 1, 1, 1, 1]),
         1.0,
         1.0,
         1.0,
         (0.1, 10.0)  # ideally approx. 1.0, but we can't expect that
         )
    ]
)
def test_estimate_scalar_background(N, floor, xb, xs, alpha_b, alpha_s, gamma,
                                    expected_range):
    res = deldenoiser.estimate_scalar_background(
        N, floor, xb, xs, alpha_b, alpha_s, gamma)
    assert (res >= expected_range[0]) and (res <= expected_range[1])


@pytest.mark.parametrize(
    "s, x, expected_result",
    [
        (np.array([1, 1, 2]), np.array([0.0, 0.5, 0.5]),
         np.array([0.0,
                   -0.5,
                   -0.09453489189183561802198688453565086342800957653750580238]
                  )
         )
    ]
)
def test_gammainccln(s, x, expected_result):
    assert is_approx(deldenoiser.gammainccln(s, x), expected_result)


model = deldenoiser.NullblockModel([10, 10, 10], 1.0, 1.0)
model.load_yields([np.array([0.8] * 10)] * 3)
path, file = os.path.split(os.path.realpath(__file__))
N_test = np.loadtxt(os.path.join(path, 'N.txt'), dtype=int)


@pytest.mark.parametrize(
    "T_and_beta, X, N, floor, design, lane_indexes, alpha, gamma, "
    "expected_range",
    [
        (
            ((0, 2, 3), 0.0),
            model.inventory_matrix,
            N_test,
            np.zeros_like(N_test),
            model.design,
            model.lane_indexes,
            model.alpha,
            model.gamma,
            (0.1, 10)
        )
    ]
)
def test_estimate_one_fitness(
        T_and_beta, X, N, floor, design, lane_indexes, alpha, gamma,
        expected_range):
    res = deldenoiser._estimate_one_fitness(
        T_and_beta, X, N, floor, design, lane_indexes, alpha, gamma)
    assert res['T_idx'] == 2 * 11 + 3
    assert (res['beta'] >= expected_range[0]) and \
           (res['beta'] <= expected_range[1])


@pytest.mark.parametrize(
    "design, alpha, gamma, yields, Npre, N, expected_truncate_beta_range",
    [
        (
            (10, 10, 10),
            0.01,
            1.0,
            [np.array([0.8] * 10)] * 3,
            np.ones(1000, dtype=int),
            np.loadtxt(os.path.join(path, 'N.txt'), dtype=int),
            (0.0, 10.0)
        )
    ]
)
def test_NullblockModel(design, alpha, gamma, yields, Npre, N,
                        expected_truncate_beta_range):

    model = deldenoiser.NullblockModel(design, alpha, gamma)
    model.load_yields(yields)

    beta_start = np.zeros(np.prod(np.array(design) + 1))
    bias, cycle_probs = model.fit_sequencing_bias(Npre)
    beta_trunc, floor = model.fit_truncates(bias, N)
    loglike_start = model.loglikelihood_of_truncates(
        model._create_beta_matrix(beta_start),
        bias, N, yields)
    loglike_end = model.loglikelihood_of_truncates(
        model._create_beta_matrix(beta_trunc),
        bias, N, yields)
    assert loglike_end > loglike_start

    beta_leg_dict = model.fit_legitimates(bias, N, floor)
    count_breakdown_matrix = model.compute_readcount_breakdown(
        bias, beta_trunc, beta_leg_dict['mode'], N
    )
    assert (np.round(np.sum(count_breakdown_matrix, axis=1)).astype(int)
            == N).all()

    assert (beta_trunc >= expected_truncate_beta_range[0]).all() and \
           (beta_trunc <= expected_truncate_beta_range[1]).all()
