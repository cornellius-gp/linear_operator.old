#!/usr/bin/env python3

from __future__ import annotations

import math
import os
import random
import unittest

import torch

import linear_operator
from linear_operator import settings
from linear_operator.test.utils import approx_equal
from linear_operator.utils import pivoted_cholesky


def rbf_kernel(x1, x2=None):
    if x2 is None:
        x2 = x1
    if x1.dim() == 1:
        x1 = x1.unsqueeze(-1)
    if x2.dim() == 1:
        x2 = x2.unsqueeze(-1)

    dist = (x1.unsqueeze(-2) - x2.unsqueeze(-3)).norm(p=2, dim=-1).pow(2)
    return dist.div(-2.0).exp()


class TestPivotedCholesky(unittest.TestCase):
    def setUp(self):
        if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            random.seed(0)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_pivoted_cholesky(self):
        size = 100
        train_x = torch.linspace(0, 1, size)
        covar_matrix = rbf_kernel(train_x, train_x)
        piv_chol = pivoted_cholesky.pivoted_cholesky(covar_matrix, 10)
        covar_approx = piv_chol @ piv_chol.transpose(-1, -2)
        self.assertTrue(approx_equal(covar_approx, covar_matrix, 2e-4))

    def test_solve_qr(self, dtype=torch.float64, tol=1e-8):
        size = 50
        X = torch.rand((size, 2)).to(dtype=dtype)
        y = torch.sin(torch.sum(X, 1)).unsqueeze(-1).to(dtype=dtype)
        with settings.min_preconditioning_size(0):
            noise = torch.DoubleTensor(size).uniform_(math.log(1e-3), math.log(1e-1)).exp_().to(dtype=dtype)
            linear_op = linear_operator.to_linear_operator(rbf_kernel(X)).add_diag(noise)
            precondition_qr, _, logdet_qr = linear_op._preconditioner()

            F = linear_op._piv_chol_self
            M = noise.diag() + F.matmul(F.t())

        x_exact = torch.solve(y, M)[0]
        x_qr = precondition_qr(y)

        self.assertTrue(approx_equal(x_exact, x_qr, tol))

        logdet = 2 * torch.cholesky(M).diag().log().sum(-1)
        self.assertTrue(approx_equal(logdet, logdet_qr, tol))

    def test_solve_qr_constant_noise(self, dtype=torch.float64, tol=1e-8):
        size = 50
        X = torch.rand((size, 2)).to(dtype=dtype)
        y = torch.sin(torch.sum(X, 1)).unsqueeze(-1).to(dtype=dtype)

        with settings.min_preconditioning_size(0):
            noise = 1e-2 * torch.ones(size, dtype=dtype)
            linear_op = linear_operator.to_linear_operator(rbf_kernel(X)).add_diag(noise)
            precondition_qr, _, logdet_qr = linear_op._preconditioner()

            F = linear_op._piv_chol_self
        M = noise.diag() + F.matmul(F.t())

        x_exact = torch.solve(y, M)[0]
        x_qr = precondition_qr(y)

        self.assertTrue(approx_equal(x_exact, x_qr, tol))

        logdet = 2 * torch.cholesky(M).diag().log().sum(-1)
        self.assertTrue(approx_equal(logdet, logdet_qr, tol))

    def test_solve_qr_float32(self):
        self.test_solve_qr(dtype=torch.float32, tol=1e-2)

    def test_solve_qr_constant_noise_float32(self):
        self.test_solve_qr_constant_noise(dtype=torch.float32, tol=1e-3)


class TestPivotedCholeskyBatch(unittest.TestCase):
    def setUp(self):
        if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            random.seed(0)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_pivoted_cholesky(self):
        size = 100
        train_x = torch.cat(
            [torch.linspace(0, 1, size).unsqueeze(0), torch.linspace(0, 0.5, size).unsqueeze(0)], 0
        ).unsqueeze(-1)
        covar_matrix = rbf_kernel(train_x, train_x)
        piv_chol = pivoted_cholesky.pivoted_cholesky(covar_matrix, 10)
        covar_approx = piv_chol @ piv_chol.transpose(-1, -2)

        self.assertTrue(approx_equal(covar_approx, covar_matrix, 2e-4))


class TestPivotedCholeskyMultiBatch(unittest.TestCase):
    def setUp(self):
        if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            random.seed(0)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_pivoted_cholesky(self):
        size = 100
        train_x = torch.cat(
            [
                torch.linspace(0, 1, size).unsqueeze(0),
                torch.linspace(0, 0.5, size).unsqueeze(0),
                torch.linspace(0, 0.25, size).unsqueeze(0),
                torch.linspace(0, 1.25, size).unsqueeze(0),
                torch.linspace(0, 1.5, size).unsqueeze(0),
                torch.linspace(0, 1, size).unsqueeze(0),
                torch.linspace(0, 0.5, size).unsqueeze(0),
                torch.linspace(0, 0.25, size).unsqueeze(0),
                torch.linspace(0, 1.25, size).unsqueeze(0),
                torch.linspace(0, 1.25, size).unsqueeze(0),
                torch.linspace(0, 1.5, size).unsqueeze(0),
                torch.linspace(0, 1, size).unsqueeze(0),
            ],
            0,
        ).unsqueeze(-1)
        covar_matrix = rbf_kernel(train_x, train_x).view(2, 2, 3, size, size)
        piv_chol = pivoted_cholesky.pivoted_cholesky(covar_matrix, 10)
        covar_approx = piv_chol @ piv_chol.transpose(-1, -2)

        self.assertTrue(approx_equal(covar_approx, covar_matrix, 2e-4))


if __name__ == "__main__":
    unittest.main()
