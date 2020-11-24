#!/usr/bin/env python3

from __future__ import annotations

import unittest

import torch

from linear_operator.operators import DenseLinearOperator, KroneckerProductLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase, RectangularLinearOperatorTestCase


def kron(a, b):
    res = []
    for i in range(a.size(-2)):
        row_res = []
        for j in range(a.size(-1)):
            row_res.append(b * a[..., i, j].unsqueeze(-1).unsqueeze(-2))
        res.append(torch.cat(row_res, -1))
    return torch.cat(res, -2)


class TestKroneckerProductLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_operator(self):
        a = torch.tensor([[4, 0, 2], [0, 3, -1], [2, -1, 3]], dtype=torch.float)
        b = torch.tensor([[2, 1], [1, 2]], dtype=torch.float)
        c = torch.tensor([[4, 0.5, 1, 0], [0.5, 4, -1, 0], [1, -1, 3, 0], [0, 0, 0, 4]], dtype=torch.float)
        a.requires_grad_(True)
        b.requires_grad_(True)
        c.requires_grad_(True)
        kp_linear_operator = KroneckerProductLinearOperator(
            DenseLinearOperator(a), DenseLinearOperator(b), DenseLinearOperator(c)
        )
        return kp_linear_operator

    def evaluate_linear_operator(self, linear_operator):
        res = kron(linear_operator.linear_operators[0].tensor, linear_operator.linear_operators[1].tensor)
        res = kron(res, linear_operator.linear_operators[2].tensor)
        return res


class TestKroneckerProductLinearOperatorBatch(TestKroneckerProductLinearOperator):
    seed = 0

    def create_linear_operator(self):
        a = torch.tensor([[4, 0, 2], [0, 3, -1], [2, -1, 3]], dtype=torch.float).repeat(3, 1, 1)
        b = torch.tensor([[2, 1], [1, 2]], dtype=torch.float).repeat(3, 1, 1)
        c = torch.tensor([[4, 0, 1, 0], [0, 4, -1, 0], [1, -1, 3, 0], [0, 0, 0, 4]], dtype=torch.float).repeat(3, 1, 1)
        a.requires_grad_(True)
        b.requires_grad_(True)
        c.requires_grad_(True)
        kp_linear_operator = KroneckerProductLinearOperator(
            DenseLinearOperator(a), DenseLinearOperator(b), DenseLinearOperator(c)
        )
        return kp_linear_operator


class TestKroneckerProductLinearOperatorRectangular(RectangularLinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_operator(self):
        a = torch.randn(2, 3, requires_grad=True)
        b = torch.randn(5, 2, requires_grad=True)
        c = torch.randn(6, 4, requires_grad=True)
        kp_linear_operator = KroneckerProductLinearOperator(
            DenseLinearOperator(a), DenseLinearOperator(b), DenseLinearOperator(c)
        )
        return kp_linear_operator

    def evaluate_linear_operator(self, linear_operator):
        res = kron(linear_operator.linear_operators[0].tensor, linear_operator.linear_operators[1].tensor)
        res = kron(res, linear_operator.linear_operators[2].tensor)
        return res


class TestKroneckerProductLinearOperatorRectangularMultiBatch(TestKroneckerProductLinearOperatorRectangular):
    seed = 0

    def create_linear_operator(self):
        a = torch.randn(3, 4, 2, 3, requires_grad=True)
        b = torch.randn(3, 4, 5, 2, requires_grad=True)
        c = torch.randn(3, 4, 6, 4, requires_grad=True)
        kp_linear_operator = KroneckerProductLinearOperator(
            DenseLinearOperator(a), DenseLinearOperator(b), DenseLinearOperator(c)
        )
        return kp_linear_operator


if __name__ == "__main__":
    unittest.main()
