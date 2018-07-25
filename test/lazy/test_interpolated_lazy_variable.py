from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import gpytorch
import torch
import unittest
import os
import random
from gpytorch.lazy import NonLazyVariable, InterpolatedLazyVariable
from gpytorch.utils import approx_equal


class TestInterpolatedLazyVariable(unittest.TestCase):
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

    def test_matmul(self):
        left_interp_indices = torch.LongTensor([[2, 3], [3, 4], [4, 5]]).repeat(3, 1)
        left_interp_values = torch.Tensor([[1, 2], [0.5, 1], [1, 3]]).repeat(3, 1)
        left_interp_values_copy = left_interp_values.clone()
        left_interp_values.requires_grad = True
        left_interp_values_copy.requires_grad = True
        right_interp_indices = torch.LongTensor([[0, 1], [1, 2], [2, 3]]).repeat(3, 1)
        right_interp_values = torch.Tensor([[1, 2], [2, 0.5], [1, 3]]).repeat(3, 1)
        right_interp_values_copy = right_interp_values.clone()
        right_interp_values.requires_grad = True
        right_interp_values_copy.requires_grad = True

        base_lazy_variable_mat = torch.randn(6, 6)
        base_lazy_variable_mat = base_lazy_variable_mat.t().matmul(base_lazy_variable_mat)
        base_variable = base_lazy_variable_mat
        base_variable.requires_grad = True
        base_variable_copy = base_lazy_variable_mat
        base_lazy_variable = NonLazyVariable(base_variable)

        test_matrix = torch.randn(9, 4)

        interp_lazy_var = InterpolatedLazyVariable(
            base_lazy_variable, left_interp_indices, left_interp_values, right_interp_indices, right_interp_values
        )
        res = interp_lazy_var.matmul(test_matrix)

        left_matrix = torch.zeros(9, 6)
        right_matrix = torch.zeros(9, 6)
        left_matrix.scatter_(1, left_interp_indices, left_interp_values_copy)
        right_matrix.scatter_(1, right_interp_indices, right_interp_values_copy)

        actual = left_matrix.matmul(base_variable_copy).matmul(right_matrix.t()).matmul(test_matrix)
        self.assertTrue(approx_equal(res.data, actual.data))

        res.sum().backward()
        actual.sum().backward()

        self.assertTrue(approx_equal(base_variable.grad.data, base_variable_copy.grad.data))
        self.assertTrue(approx_equal(left_interp_values.grad.data, left_interp_values_copy.grad.data))

    def test_batch_matmul(self):
        left_interp_indices = torch.LongTensor([[2, 3], [3, 4], [4, 5]]).repeat(5, 3, 1)
        left_interp_values = torch.Tensor([[1, 2], [0.5, 1], [1, 3]]).repeat(5, 3, 1)
        left_interp_values_copy = left_interp_values.clone()
        left_interp_values.requires_grad = True
        left_interp_values_copy.requires_grad = True
        right_interp_indices = torch.LongTensor([[0, 1], [1, 2], [2, 3]]).repeat(5, 3, 1)
        right_interp_values = torch.Tensor([[1, 2], [2, 0.5], [1, 3]]).repeat(5, 3, 1)
        right_interp_values_copy = right_interp_values.clone()
        right_interp_values.requires_grad = True
        right_interp_values_copy.requires_grad = True

        base_lazy_variable_mat = torch.randn(5, 6, 6)
        base_lazy_variable_mat = base_lazy_variable_mat.transpose(-1, -2).matmul(base_lazy_variable_mat)
        base_variable = base_lazy_variable_mat
        base_variable_copy = base_variable.clone()
        base_variable.requires_grad = True
        base_variable_copy.requires_grad = True
        base_lazy_variable = NonLazyVariable(base_variable)

        test_matrix = torch.randn(5, 9, 4)

        interp_lazy_var = InterpolatedLazyVariable(
            base_lazy_variable, left_interp_indices, left_interp_values, right_interp_indices, right_interp_values
        )
        res = interp_lazy_var.matmul(test_matrix)

        left_matrix_comps = []
        right_matrix_comps = []
        for i in range(5):
            left_matrix_comp = torch.zeros(9, 6)
            right_matrix_comp = torch.zeros(9, 6)
            left_matrix_comp.scatter_(1, left_interp_indices[i], left_interp_values_copy[i])
            right_matrix_comp.scatter_(1, right_interp_indices[i], right_interp_values_copy[i])
            left_matrix_comps.append(left_matrix_comp.unsqueeze(0))
            right_matrix_comps.append(right_matrix_comp.unsqueeze(0))
        left_matrix = torch.cat(left_matrix_comps)
        right_matrix = torch.cat(right_matrix_comps)

        actual = left_matrix.matmul(base_variable_copy).matmul(right_matrix.transpose(-1, -2))
        actual = actual.matmul(test_matrix)
        self.assertTrue(approx_equal(res.data, actual.data))

        res.sum().backward()
        actual.sum().backward()

        self.assertTrue(approx_equal(base_variable.grad.data, base_variable_copy.grad.data))
        self.assertTrue(approx_equal(left_interp_values.grad.data, left_interp_values_copy.grad.data))

    def test_inv_matmul(self):
        base_lazy_variable_mat = torch.randn(6, 6)
        base_lazy_variable_mat = base_lazy_variable_mat.t().matmul(base_lazy_variable_mat)
        test_matrix = torch.randn(3, 4)

        left_interp_indices = torch.LongTensor([[2, 3], [3, 4], [4, 5]])
        left_interp_values = torch.Tensor([[1, 2], [0.5, 1], [1, 3]])
        left_interp_values_copy = left_interp_values.clone()
        left_interp_values.requires_grad = True
        left_interp_values_copy.requires_grad = True

        right_interp_indices = torch.LongTensor([[2, 3], [3, 4], [4, 5]])
        right_interp_values = torch.Tensor([[1, 2], [0.5, 1], [1, 3]])
        right_interp_values_copy = right_interp_values.clone()
        right_interp_values.requires_grad = True
        right_interp_values_copy.requires_grad = True

        base_lazy_variable = base_lazy_variable_mat
        base_lazy_variable.requires_grad = True
        base_lazy_variable_copy = base_lazy_variable_mat
        test_matrix_var = test_matrix
        test_matrix_var.requires_grad = True
        test_matrix_var_copy = test_matrix

        interp_lazy_var = InterpolatedLazyVariable(
            NonLazyVariable(base_lazy_variable),
            left_interp_indices,
            left_interp_values,
            right_interp_indices,
            right_interp_values,
        )
        res = interp_lazy_var.inv_matmul(test_matrix_var)

        left_matrix = torch.zeros(3, 6)
        right_matrix = torch.zeros(3, 6)
        left_matrix.scatter_(1, left_interp_indices, left_interp_values_copy)
        right_matrix.scatter_(1, right_interp_indices, right_interp_values_copy)
        actual_mat = left_matrix.matmul(base_lazy_variable_copy).matmul(right_matrix.transpose(-1, -2))
        actual = gpytorch.inv_matmul(actual_mat, test_matrix_var_copy)

        self.assertTrue(approx_equal(res.data, actual.data))

        # Backward pass
        res.sum().backward()
        actual.sum().backward()

        self.assertTrue(approx_equal(base_lazy_variable.grad.data, base_lazy_variable_copy.grad.data))
        self.assertTrue(approx_equal(left_interp_values.grad.data, left_interp_values_copy.grad.data))

    def test_inv_matmul_batch(self):
        base_lazy_variable = torch.randn(6, 6)
        base_lazy_variable = (base_lazy_variable.t().matmul(base_lazy_variable)).unsqueeze(0).repeat(5, 1, 1)
        base_lazy_variable_copy = base_lazy_variable.clone()
        base_lazy_variable.requires_grad = True
        base_lazy_variable_copy.requires_grad = True

        test_matrix_var = torch.randn(5, 3, 4)
        test_matrix_var_copy = test_matrix_var.clone()
        test_matrix_var.requires_grad = True
        test_matrix_var_copy.requires_grad = True

        left_interp_indices = torch.LongTensor([[2, 3], [3, 4], [4, 5]]).unsqueeze(0).repeat(5, 1, 1)
        left_interp_values = torch.Tensor([[1, 2], [0.5, 1], [1, 3]]).unsqueeze(0).repeat(5, 1, 1)
        left_interp_values_copy = left_interp_values.clone()
        left_interp_values.requires_grad = True
        left_interp_values_copy.requires_grad = True

        right_interp_indices = torch.LongTensor([[2, 3], [3, 4], [4, 5]]).unsqueeze(0).repeat(5, 1, 1)
        right_interp_values = torch.Tensor([[1, 2], [0.5, 1], [1, 3]]).unsqueeze(0).repeat(5, 1, 1)
        right_interp_values_copy = right_interp_values.clone()
        right_interp_values.requires_grad = True
        right_interp_values_copy.requires_grad = True

        interp_lazy_var = InterpolatedLazyVariable(
            NonLazyVariable(base_lazy_variable),
            left_interp_indices,
            left_interp_values,
            right_interp_indices,
            right_interp_values,
        )
        res = interp_lazy_var.inv_matmul(test_matrix_var)

        left_matrix_comps = []
        right_matrix_comps = []
        for i in range(5):
            left_matrix_comp = torch.zeros(3, 6)
            right_matrix_comp = torch.zeros(3, 6)
            left_matrix_comp.scatter_(1, left_interp_indices[i], left_interp_values_copy[i])
            right_matrix_comp.scatter_(1, right_interp_indices[i], right_interp_values_copy[i])
            left_matrix_comps.append(left_matrix_comp.unsqueeze(0))
            right_matrix_comps.append(right_matrix_comp.unsqueeze(0))
        left_matrix = torch.cat(left_matrix_comps)
        right_matrix = torch.cat(right_matrix_comps)
        actual_mat = left_matrix.matmul(base_lazy_variable_copy).matmul(right_matrix.transpose(-1, -2))
        actual = gpytorch.inv_matmul(actual_mat, test_matrix_var_copy)

        self.assertTrue(approx_equal(res.data, actual.data))

        # Backward pass
        res.sum().backward()
        actual.sum().backward()

        self.assertTrue(approx_equal(base_lazy_variable.grad.data, base_lazy_variable_copy.grad.data))
        self.assertTrue(approx_equal(left_interp_values.grad.data, left_interp_values_copy.grad.data))

    def test_matmul_batch(self):
        left_interp_indices = torch.LongTensor([[2, 3], [3, 4], [4, 5]]).repeat(5, 3, 1)
        left_interp_values = torch.Tensor([[1, 2], [0.5, 1], [1, 3]]).repeat(5, 3, 1)
        right_interp_indices = torch.LongTensor([[0, 1], [1, 2], [2, 3]]).repeat(5, 3, 1)
        right_interp_values = torch.Tensor([[1, 2], [2, 0.5], [1, 3]]).repeat(5, 3, 1)

        base_lazy_variable_mat = torch.randn(5, 6, 6)
        base_lazy_variable_mat = base_lazy_variable_mat.transpose(1, 2).matmul(base_lazy_variable_mat)
        base_lazy_variable_mat.requires_grad = True
        test_matrix = torch.randn(1, 9, 4)

        base_lazy_variable = NonLazyVariable(base_lazy_variable_mat)
        interp_lazy_var = InterpolatedLazyVariable(
            base_lazy_variable, left_interp_indices, left_interp_values, right_interp_indices, right_interp_values
        )
        res = interp_lazy_var.matmul(test_matrix)

        left_matrix = torch.Tensor(
            [
                [0, 0, 1, 2, 0, 0],
                [0, 0, 0, 0.5, 1, 0],
                [0, 0, 0, 0, 1, 3],
                [0, 0, 1, 2, 0, 0],
                [0, 0, 0, 0.5, 1, 0],
                [0, 0, 0, 0, 1, 3],
                [0, 0, 1, 2, 0, 0],
                [0, 0, 0, 0.5, 1, 0],
                [0, 0, 0, 0, 1, 3],
            ]
        ).repeat(5, 1, 1)

        right_matrix = torch.Tensor(
            [
                [1, 2, 0, 0, 0, 0],
                [0, 2, 0.5, 0, 0, 0],
                [0, 0, 1, 3, 0, 0],
                [1, 2, 0, 0, 0, 0],
                [0, 2, 0.5, 0, 0, 0],
                [0, 0, 1, 3, 0, 0],
                [1, 2, 0, 0, 0, 0],
                [0, 2, 0.5, 0, 0, 0],
                [0, 0, 1, 3, 0, 0],
            ]
        ).repeat(5, 1, 1)
        actual = (
            left_matrix.matmul(base_lazy_variable_mat).matmul(right_matrix.transpose(-1, -2)).matmul(test_matrix.data)
        )

        self.assertTrue(approx_equal(res.data, actual))

    def test_getitem_batch(self):
        left_interp_indices = torch.LongTensor([[2, 3], [3, 4], [4, 5]]).repeat(5, 1, 1)
        left_interp_values = torch.Tensor([[1, 1], [1, 1], [1, 1]]).repeat(5, 1, 1)
        right_interp_indices = torch.LongTensor([[0, 1], [1, 2], [2, 3]]).repeat(5, 1, 1)
        right_interp_values = torch.Tensor([[1, 1], [1, 1], [1, 1]]).repeat(5, 1, 1)

        base_lazy_variable_mat = torch.randn(5, 6, 6)
        base_lazy_variable_mat = base_lazy_variable_mat.transpose(1, 2).matmul(base_lazy_variable_mat)
        base_lazy_variable_mat.requires_grad = True

        base_lazy_variable = NonLazyVariable(base_lazy_variable_mat)
        interp_lazy_var = InterpolatedLazyVariable(
            base_lazy_variable, left_interp_indices, left_interp_values, right_interp_indices, right_interp_values
        )

        actual = (
            base_lazy_variable[:, 2:5, 0:3]
            + base_lazy_variable[:, 2:5, 1:4]
            + base_lazy_variable[:, 3:6, 0:3]
            + base_lazy_variable[:, 3:6, 1:4]
        ).evaluate()

        self.assertTrue(approx_equal(interp_lazy_var[2].evaluate().data, actual[2].data))
        self.assertTrue(approx_equal(interp_lazy_var[0:2].evaluate().data, actual[0:2].data))
        self.assertTrue(approx_equal(interp_lazy_var[:, 2:3].evaluate().data, actual[:, 2:3].data))
        self.assertTrue(approx_equal(interp_lazy_var[:, 0:2].evaluate().data, actual[:, 0:2].data))
        self.assertTrue(approx_equal(interp_lazy_var[1, :1, :2].evaluate().data, actual[1, :1, :2].data))
        self.assertTrue(approx_equal(interp_lazy_var[1, 1, :2].data, actual[1, 1, :2].data))
        self.assertTrue(approx_equal(interp_lazy_var[1, :1, 2].data, actual[1, :1, 2].data))

    def test_diag(self):
        left_interp_indices = torch.LongTensor([[2, 3], [3, 4], [4, 5]])
        left_interp_values = torch.Tensor([[1, 1], [1, 1], [1, 1]])
        right_interp_indices = torch.LongTensor([[0, 1], [1, 2], [2, 3]])
        right_interp_values = torch.Tensor([[1, 1], [1, 1], [1, 1]])

        base_lazy_variable_mat = torch.randn(6, 6)
        base_lazy_variable_mat = base_lazy_variable_mat.t().matmul(base_lazy_variable_mat)
        base_lazy_variable_mat.requires_grad = True

        base_lazy_variable = NonLazyVariable(base_lazy_variable_mat)
        interp_lazy_var = InterpolatedLazyVariable(
            base_lazy_variable, left_interp_indices, left_interp_values, right_interp_indices, right_interp_values
        )

        actual = interp_lazy_var.evaluate()
        self.assertTrue(approx_equal(actual.diag().data, interp_lazy_var.diag().data))

    def test_batch_diag(self):
        left_interp_indices = torch.LongTensor([[2, 3], [3, 4], [4, 5]]).repeat(5, 1, 1)
        left_interp_values = torch.Tensor([[1, 1], [1, 1], [1, 1]]).repeat(5, 1, 1)
        right_interp_indices = torch.LongTensor([[0, 1], [1, 2], [2, 3]]).repeat(5, 1, 1)
        right_interp_values = torch.Tensor([[1, 1], [1, 1], [1, 1]]).repeat(5, 1, 1)

        base_lazy_variable_mat = torch.randn(5, 6, 6)
        base_lazy_variable_mat = base_lazy_variable_mat.transpose(1, 2).matmul(base_lazy_variable_mat)
        base_lazy_variable_mat.requires_grad = True

        base_lazy_variable = NonLazyVariable(base_lazy_variable_mat)
        interp_lazy_var = InterpolatedLazyVariable(
            base_lazy_variable, left_interp_indices, left_interp_values, right_interp_indices, right_interp_values
        )

        actual = interp_lazy_var.evaluate()
        actual_diag = torch.stack(
            [actual[0].diag(), actual[1].diag(), actual[2].diag(), actual[3].diag(), actual[4].diag()]
        )

        self.assertTrue(approx_equal(actual_diag.data, interp_lazy_var.diag().data))


if __name__ == "__main__":
    unittest.main()
