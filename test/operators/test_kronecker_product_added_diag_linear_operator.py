#!/usr/bin/env python3

import unittest
from unittest import mock

import torch

from linear_operator import settings
from linear_operator.operators import (
    DiagLinearOperator,
    KroneckerProductAddedDiagLinearOperator,
    KroneckerProductLinearOperator,
    NonLinearOperator,
)
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase


class TestKroneckerProductAddedDiagLinearOperator(unittest.TestCase, LinearOperatorTestCase):
    # this lienar operator has an explicit inverse so we don't need to run these
    skip_slq_tests = True
    should_call_lanczos = False
    should_call_cg = False

    def create_linear_operator(self):
        a = torch.tensor([[4, 0, 2], [0, 3, -1], [2, -1, 3]], dtype=torch.float)
        b = torch.tensor([[2, 1], [1, 2]], dtype=torch.float)
        c = torch.tensor([[4, 0.5, 1, 0], [0.5, 4, -1, 0], [1, -1, 3, 0], [0, 0, 0, 4]], dtype=torch.float)
        a.requires_grad_(True)
        b.requires_grad_(True)
        c.requires_grad_(True)
        kp_linear_operator = KroneckerProductLinearOperator(
            NonLinearOperator(a), NonLinearOperator(b), NonLinearOperator(c)
        )

        return KroneckerProductAddedDiagLinearOperator(
            kp_linear_operator, DiagLinearOperator(0.1 * torch.ones(kp_linear_operator.shape[-1]))
        )

    def evaluate_linear_operator(self, linear_operator):
        tensor = linear_operator._linear_operator.evaluate()
        diag = linear_operator._diag_tensor._diag
        return tensor + diag.diag()

    def test_if_cholesky_used(self):
        linear_operator = self.create_linear_operator()
        rhs = torch.randn(linear_operator.size(-1))
        # Check that cholesky is not called
        with mock.patch.object(linear_operator, "cholesky") as chol_mock:
            self._test_inv_matmul(rhs, cholesky=False)
            chol_mock.assert_not_called()

    def test_root_inv_decomposition_no_cholesky(self):
        with settings.max_cholesky_size(0):
            linear_operator = self.create_linear_operator()
            test_mat = torch.randn(*linear_operator.batch_shape, linear_operator.size(-1), 5)
            # Check that cholesky is not called
            with mock.patch.object(linear_operator, "cholesky") as chol_mock:
                root_approx = linear_operator.root_inv_decomposition()
                res = root_approx.matmul(test_mat)
                actual = linear_operator.inv_matmul(test_mat)
                self.assertAllClose(res, actual, rtol=0.05, atol=0.02)
                chol_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
