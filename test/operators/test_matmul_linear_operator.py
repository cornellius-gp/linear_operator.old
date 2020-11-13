#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import MatmulLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase, RectangularLinearOperatorTestCase


class TestMatmulLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 1

    def create_linear_operator(self):
        lhs = torch.randn(5, 6, requires_grad=True)
        rhs = lhs.clone().detach().transpose(-1, -2)
        covar = MatmulLinearOperator(lhs, rhs)
        return covar

    def evaluate_linear_operator(self, linear_operator):
        return linear_operator.left_linear_operator.tensor.matmul(linear_operator.right_linear_operator.tensor)


class TestMatmulLinearOperatorBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 3

    def create_linear_operator(self):
        lhs = torch.randn(5, 5, 6, requires_grad=True)
        rhs = lhs.clone().detach().transpose(-1, -2)
        covar = MatmulLinearOperator(lhs, rhs)
        return covar

    def evaluate_linear_operator(self, linear_operator):
        return linear_operator.left_linear_operator.tensor.matmul(linear_operator.right_linear_operator.tensor)


class TestMatmulLinearOperatorRectangular(RectangularLinearOperatorTestCase, unittest.TestCase):
    def create_linear_operator(self):
        lhs = torch.randn(5, 3, requires_grad=True)
        rhs = torch.randn(3, 6, requires_grad=True)
        covar = MatmulLinearOperator(lhs, rhs)
        return covar

    def evaluate_linear_operator(self, linear_operator):
        return linear_operator.left_linear_operator.tensor.matmul(linear_operator.right_linear_operator.tensor)


class TestMatmulLinearOperatorRectangularMultiBatch(RectangularLinearOperatorTestCase, unittest.TestCase):
    def create_linear_operator(self):
        lhs = torch.randn(2, 3, 5, 3, requires_grad=True)
        rhs = torch.randn(2, 3, 3, 6, requires_grad=True)
        covar = MatmulLinearOperator(lhs, rhs)
        return covar

    def evaluate_linear_operator(self, linear_operator):
        return linear_operator.left_linear_operator.tensor.matmul(linear_operator.right_linear_operator.tensor)


if __name__ == "__main__":
    unittest.main()
