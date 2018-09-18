from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import unittest
from gpytorch.lazy import CholLazyTensor
from gpytorch.utils import approx_equal


class TestCholLazyTensor(unittest.TestCase):
    def setUp(self):
        chol = torch.tensor(
            [[3, 0, 0, 0, 0], [-1, 2, 0, 0, 0], [1, 4, 1, 0, 0], [0, 2, 3, 2, 0], [-4, -2, 1, 3, 4]], dtype=torch.float
        )
        vecs = torch.randn(5, 2)

        self.chol_var = torch.tensor(chol, requires_grad=True)
        self.chol_var_copy = torch.tensor(chol, requires_grad=True)
        self.actual_mat = self.chol_var_copy.matmul(self.chol_var_copy.transpose(-1, -2))
        self.vecs = torch.tensor(vecs, requires_grad=True)
        self.vecs_copy = torch.tensor(vecs, requires_grad=True)

    def test_matmul(self):
        # Forward
        res = CholLazyTensor(self.chol_var).matmul(self.vecs)
        actual = self.actual_mat.matmul(self.vecs_copy)
        self.assertTrue(approx_equal(res, actual))

        # Backward
        grad_output = torch.randn(*self.vecs.size())
        res.backward(gradient=grad_output)
        actual.backward(gradient=grad_output)
        self.assertTrue(approx_equal(self.chol_var.grad, self.chol_var_copy.grad))
        self.assertTrue(approx_equal(self.vecs.grad, self.vecs_copy.grad))

    def test_inv_matmul(self):
        # Forward
        res = CholLazyTensor(self.chol_var).inv_matmul(self.vecs)
        actual = self.actual_mat.inverse().matmul(self.vecs_copy)
        self.assertLess(torch.max((res - actual).abs() / actual.norm()), 1e-2)

    def test_inv_quad_log_det(self):
        # Forward
        res_inv_quad, res_log_det = CholLazyTensor(self.chol_var).inv_quad_log_det(inv_quad_rhs=self.vecs, log_det=True)
        res = res_inv_quad + res_log_det
        actual_inv_quad = self.actual_mat.inverse().matmul(self.vecs_copy).mul(self.vecs_copy).sum()
        actual = actual_inv_quad + torch.log(torch.det(self.actual_mat))
        self.assertLess(((res - actual) / actual).abs().item(), 1e-2)

    def test_diag(self):
        res = CholLazyTensor(self.chol_var).diag()
        actual = self.actual_mat.diag()
        self.assertTrue(approx_equal(res, actual))

    def test_getitem(self):
        res = CholLazyTensor(self.chol_var)[2:4, -2]
        actual = self.actual_mat[2:4, -2]
        self.assertTrue(approx_equal(res, actual))

    def test_evaluate(self):
        res = CholLazyTensor(self.chol_var).evaluate()
        actual = self.actual_mat
        self.assertTrue(approx_equal(res, actual))


class TestCholLazyTensorBatch(unittest.TestCase):
    def setUp(self):
        chol = torch.tensor(
            [
                [[3, 0, 0, 0, 0], [-1, 2, 0, 0, 0], [1, 4, 1, 0, 0], [0, 2, 3, 2, 0], [-4, -2, 1, 3, 4]],
                [[2, 0, 0, 0, 0], [3, 1, 0, 0, 0], [-2, 3, 2, 0, 0], [-2, 1, -1, 3, 0], [-4, -4, 5, 2, 3]],
            ],
            dtype=torch.float,
        )
        vecs = torch.randn(2, 5, 3)

        self.chol_var = torch.tensor(chol, requires_grad=True)
        self.chol_var_copy = torch.tensor(chol, requires_grad=True)
        self.actual_mat = self.chol_var_copy.matmul(self.chol_var_copy.transpose(-1, -2))
        self.actual_mat_inv = torch.cat(
            [self.actual_mat[0].inverse().unsqueeze(0), self.actual_mat[1].inverse().unsqueeze(0)], 0
        )

        self.vecs = torch.tensor(vecs, requires_grad=True)
        self.vecs_copy = torch.tensor(vecs, requires_grad=True)

    def test_matmul(self):
        # Forward
        res = CholLazyTensor(self.chol_var).matmul(self.vecs)
        actual = self.actual_mat.matmul(self.vecs_copy)
        self.assertTrue(approx_equal(res, actual))

        # Backward
        grad_output = torch.randn(*self.vecs.size())
        res.backward(gradient=grad_output)
        actual.backward(gradient=grad_output)
        self.assertTrue(approx_equal(self.chol_var.grad, self.chol_var_copy.grad))
        self.assertTrue(approx_equal(self.vecs.grad, self.vecs_copy.grad))

    def test_inv_matmul(self):
        # Forward
        res = CholLazyTensor(self.chol_var).inv_matmul(self.vecs)
        actual = self.actual_mat_inv.matmul(self.vecs_copy)
        self.assertLess(torch.max((res - actual).abs() / actual.norm()), 1e-2)

    def test_inv_quad_log_det(self):
        # Forward
        res_inv_quad, res_log_det = CholLazyTensor(self.chol_var).inv_quad_log_det(inv_quad_rhs=self.vecs, log_det=True)
        res = res_inv_quad + res_log_det
        actual_inv_quad = self.actual_mat_inv.matmul(self.vecs_copy).mul(self.vecs_copy).sum(-1).sum(-1)
        actual_log_det = torch.tensor(
            [torch.log(torch.det(self.actual_mat[0])), torch.log(torch.det(self.actual_mat[1]))]
        )

        actual = actual_inv_quad + actual_log_det
        self.assertLess(torch.max((res - actual).abs() / actual.norm()), 1e-2)

    def test_diag(self):
        res = CholLazyTensor(self.chol_var).diag()
        actual = torch.cat([self.actual_mat[0].diag().unsqueeze(0), self.actual_mat[1].diag().unsqueeze(0)], 0)
        self.assertTrue(approx_equal(res, actual))

    def test_getitem(self):
        res = CholLazyTensor(self.chol_var)[1, 2:4, -2]
        actual = self.actual_mat[1, 2:4, -2]
        self.assertTrue(approx_equal(res, actual))

    def test_evaluate(self):
        res = CholLazyTensor(self.chol_var).evaluate()
        actual = self.actual_mat
        self.assertTrue(approx_equal(res, actual))


if __name__ == "__main__":
    unittest.main()
