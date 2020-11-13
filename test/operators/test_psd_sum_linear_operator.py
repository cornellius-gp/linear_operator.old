#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import NonLinearOperator, PsdSumLinearOperator, ToeplitzLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase


class TestPsdSumLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True

    def create_linear_operator(self):
        c1 = torch.tensor([5, 1, 2, 0], dtype=torch.float, requires_grad=True)
        t1 = ToeplitzLinearOperator(c1)
        c2 = torch.tensor([6, 0, 1, -1], dtype=torch.float, requires_grad=True)
        t2 = ToeplitzLinearOperator(c2)
        return PsdSumLinearOperator(t1, t2)

    def evaluate_linear_operator(self, linear_operator):
        tensors = [lt.evaluate() for lt in linear_operator.linear_operators]
        return sum(tensors)


class TestPsdSumLinearOperatorBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True

    def create_linear_operator(self):
        c1 = torch.tensor([[2, 0.5, 0, 0], [5, 1, 2, 0]], dtype=torch.float, requires_grad=True)
        t1 = ToeplitzLinearOperator(c1)
        c2 = torch.tensor([[2, 0.5, 0, 0], [6, 0, 1, -1]], dtype=torch.float, requires_grad=True)
        t2 = ToeplitzLinearOperator(c2)
        return PsdSumLinearOperator(t1, t2)

    def evaluate_linear_operator(self, linear_operator):
        tensors = [lt.evaluate() for lt in linear_operator.linear_operators]
        return sum(tensors)


class TestPsdSumLinearOperatorMultiBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    # Because these LTs are large, we'll skil the big tests
    should_test_sample = False
    skip_slq_tests = True

    def create_linear_operator(self):
        mat1 = torch.randn(2, 3, 4, 4)
        lt1 = NonLinearOperator(mat1 @ mat1.transpose(-1, -2))
        mat2 = torch.randn(2, 3, 4, 4)
        lt2 = NonLinearOperator(mat2 @ mat2.transpose(-1, -2))
        return PsdSumLinearOperator(lt1, lt2)

    def evaluate_linear_operator(self, linear_operator):
        tensors = [lt.evaluate() for lt in linear_operator.linear_operators]
        return sum(tensors)


if __name__ == "__main__":
    unittest.main()