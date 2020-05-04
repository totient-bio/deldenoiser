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
    from deldenoiser import nullblockmodel as nbm
    from deldenoiser import pbs_algorithm as pbs
except ImportError:
    import sys

    sys.path.append(
        os.path.dirname(
            os.path.dirname(os.path.realpath(__file__))))
    from deldenoiser import nullblockmodel as nbm
    from deldenoiser import pbs_algorithm as pbs

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
    assert is_approx(pbs.gammainccln(s, x), expected_result)


def test_NullblockModel():
    design = [10, 10, 10]
    alpha = 1.0
    gamma = 1.0
    yields = [np.array([0.8] * 10)] * 3
    path, file = os.path.split(os.path.realpath(__file__))
    Npost_file = os.path.join(path, 'Npost.tsv')

    model = nbm.NullBlockModel(design, alpha, gamma)

    model.yields = yields
    model.x = model._compute_x()
    model.load_postselection_readcount(Npost_file)

    model.guess_truncates(processes=2)
    model.fit_truncates()
    F = model.fit_fullcycles()
    model.compute_clean_readcounts(F)
