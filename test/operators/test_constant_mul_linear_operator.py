#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import ToeplitzLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase
from linear_operator.utils.toeplitz import sym_toeplitz


class TestConstantMulLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_operator(self):
        column = torch.tensor([5, 1, 2, 0], dtype=torch.float, requires_grad=True)
        constant = 2.5
        return ToeplitzLinearOperator(column) * constant

    def evaluate_linear_operator(self, linear_operator):
        constant = linear_operator.expanded_constant
        column = linear_operator.base_linear_operator.column
        return sym_toeplitz(column) * constant


class TestConstantMulLinearOperatorBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_operator(self):
        column = torch.tensor([[5, 1, 2, 0]], dtype=torch.float).repeat(2, 1)
        column.requires_grad_(True)
        constant = torch.tensor([2.5, 1.0]).view(2, 1, 1)
        return ToeplitzLinearOperator(column) * constant

    def evaluate_linear_operator(self, linear_operator):
        constant = linear_operator.expanded_constant
        column = linear_operator.base_linear_operator.column
        return torch.cat([sym_toeplitz(column[0]).unsqueeze(0), sym_toeplitz(column[1]).unsqueeze(0)]) * constant.view(
            2, 1, 1
        )


class TestConstantMulLinearOperatorMultiBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    # Because these LTs are large, we'll skil the big tests
    should_test_sample = False
    skip_slq_tests = True

    def create_linear_operator(self):
        column = torch.tensor([[5, 1, 2, 0]], dtype=torch.float).repeat(3, 2, 1)
        column.requires_grad_(True)
        constant = torch.randn(3, 2, 1, 1).abs()
        return ToeplitzLinearOperator(column) * constant

    def evaluate_linear_operator(self, linear_operator):
        constant = linear_operator.expanded_constant
        toeplitz = linear_operator.base_linear_operator
        return toeplitz.evaluate() * constant


class TestConstantMulLinearOperatorMultiBatchBroadcastConstant(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    # Because these LTs are large, we'll skil the big tests
    should_test_sample = False
    skip_slq_tests = True

    def create_linear_operator(self):
        column = torch.tensor([[5, 1, 2, 0]], dtype=torch.float).repeat(3, 2, 1)
        column.requires_grad_(True)
        constant = torch.randn(2, 1, 1).abs()
        return ToeplitzLinearOperator(column) * constant

    def evaluate_linear_operator(self, linear_operator):
        constant = linear_operator.expanded_constant
        toeplitz = linear_operator.base_linear_operator
        return toeplitz.evaluate() * constant


if __name__ == "__main__":
    unittest.main()
