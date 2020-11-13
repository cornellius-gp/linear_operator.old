#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import LinearOperator, RootLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase


def make_random_mat(size, rank, batch_shape=torch.Size(())):
    res = torch.randn(*batch_shape, size, rank)
    return res


class TestMulLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 10

    def create_linear_operator(self):
        mat1 = make_random_mat(6, 6)
        mat2 = make_random_mat(6, 6)
        res = RootLinearOperator(mat1) * RootLinearOperator(mat2)
        return res.add_diag(torch.tensor(2.0))

    def evaluate_linear_operator(self, linear_operator):
        diag_tensor = linear_operator._diag_tensor.evaluate()
        res = torch.mul(
            linear_operator._linear_operator.left_linear_operator.evaluate(), linear_operator._linear_operator.right_linear_operator.evaluate()
        )
        res = res + diag_tensor
        return res

    def test_quad_form_derivative(self):
        linear_operator = self.create_linear_operator().requires_grad_(True)
        linear_operator._diag_tensor.requires_grad_(False)
        linear_operator_clone = linear_operator.clone().detach_().requires_grad_(True)
        linear_operator_clone._diag_tensor.requires_grad_(False)
        left_vecs = torch.randn(*linear_operator.batch_shape, linear_operator.size(-2), 2)
        right_vecs = torch.randn(*linear_operator.batch_shape, linear_operator.size(-1), 2)
        deriv_custom = linear_operator._quad_form_derivative(left_vecs, right_vecs)
        deriv_auto = LinearOperator._quad_form_derivative(linear_operator_clone, left_vecs, right_vecs)

        for dc, da in zip(deriv_custom, deriv_auto):
            if dc is not None or da is not None:
                self.assertAllClose(dc, da)


class TestMulLinearOperatorBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 2

    def create_linear_operator(self):
        mat1 = make_random_mat(6, rank=6, batch_shape=torch.Size((2,)))
        mat2 = make_random_mat(6, rank=6, batch_shape=torch.Size((2,)))
        res = RootLinearOperator(mat1) * RootLinearOperator(mat2)
        return res.add_diag(torch.tensor(2.0))

    def evaluate_linear_operator(self, linear_operator):
        diag_tensor = linear_operator._diag_tensor.evaluate()
        res = torch.mul(
            linear_operator._linear_operator.left_linear_operator.evaluate(), linear_operator._linear_operator.right_linear_operator.evaluate()
        )
        res = res + diag_tensor
        return res

    def test_quad_form_derivative(self):
        linear_operator = self.create_linear_operator().requires_grad_(True)
        linear_operator._diag_tensor.requires_grad_(False)
        linear_operator_clone = linear_operator.clone().detach_().requires_grad_(True)
        linear_operator_clone._diag_tensor.requires_grad_(False)
        left_vecs = torch.randn(*linear_operator.batch_shape, linear_operator.size(-2), 2)
        right_vecs = torch.randn(*linear_operator.batch_shape, linear_operator.size(-1), 2)
        deriv_custom = linear_operator._quad_form_derivative(left_vecs, right_vecs)
        deriv_auto = LinearOperator._quad_form_derivative(linear_operator_clone, left_vecs, right_vecs)

        for dc, da in zip(deriv_custom, deriv_auto):
            if dc is not None or da is not None:
                self.assertAllClose(dc, da)


class TestMulLinearOperatorMultiBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 1
    skip_slq_tests = True

    def create_linear_operator(self):
        mat1 = make_random_mat(6, rank=6, batch_shape=torch.Size((2, 3)))
        mat2 = make_random_mat(6, rank=6, batch_shape=torch.Size((2, 3)))
        res = RootLinearOperator(mat1) * RootLinearOperator(mat2)
        return res.add_diag(torch.tensor(0.5))

    def evaluate_linear_operator(self, linear_operator):
        diag_tensor = linear_operator._diag_tensor.evaluate()
        res = torch.mul(
            linear_operator._linear_operator.left_linear_operator.evaluate(), linear_operator._linear_operator.right_linear_operator.evaluate()
        )
        res = res + diag_tensor
        return res

    def test_inv_quad_logdet(self):
        pass

    def test_quad_form_derivative(self):
        linear_operator = self.create_linear_operator().requires_grad_(True)
        linear_operator._diag_tensor.requires_grad_(False)
        linear_operator_clone = linear_operator.clone().detach_().requires_grad_(True)
        linear_operator_clone._diag_tensor.requires_grad_(False)
        left_vecs = torch.randn(*linear_operator.batch_shape, linear_operator.size(-2), 2)
        right_vecs = torch.randn(*linear_operator.batch_shape, linear_operator.size(-1), 2)
        deriv_custom = linear_operator._quad_form_derivative(left_vecs, right_vecs)
        deriv_auto = LinearOperator._quad_form_derivative(linear_operator_clone, left_vecs, right_vecs)

        for dc, da in zip(deriv_custom, deriv_auto):
            if dc is not None or da is not None:
                self.assertAllClose(dc, da)


if __name__ == "__main__":
    unittest.main()
