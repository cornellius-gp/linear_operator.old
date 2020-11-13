#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import RootLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase


class TestRootLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True
    should_call_lanczos = False

    def create_linear_operator(self):
        root = torch.randn(3, 5, requires_grad=True)
        return RootLinearOperator(root)

    def evaluate_linear_operator(self, linear_operator):
        root = linear_operator.root.tensor
        res = root.matmul(root.transpose(-1, -2))
        return res


class TestRootLinearOperatorBatch(TestRootLinearOperator):
    seed = 1

    def create_linear_operator(self):
        root = torch.randn(3, 5, 5)
        root.add_(torch.eye(5).unsqueeze(0))
        root.requires_grad_(True)
        return RootLinearOperator(root)


class TestRootLinearOperatorMultiBatch(TestRootLinearOperator):
    seed = 1
    # Because these LTs are large, we'll skil the big tests
    should_test_sample = False
    skip_slq_tests = True

    def create_linear_operator(self):
        root = torch.randn(4, 3, 5, 5)
        root.requires_grad_(True)
        return RootLinearOperator(root)


if __name__ == "__main__":
    unittest.main()
