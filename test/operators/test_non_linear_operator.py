#!/usr/bin/env python3

import unittest

import torch

from linear_operator import settings
from linear_operator.operators import NonLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase


class TestNonLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_operator(self):
        mat = torch.randn(5, 6)
        mat = mat.matmul(mat.transpose(-1, -2))
        mat.requires_grad_(True)
        return NonLinearOperator(mat)

    def evaluate_linear_operator(self, linear_operator):
        return linear_operator.tensor

    def test_root_decomposition_exact(self):
        linear_operator = self.create_linear_operator()
        test_mat = torch.randn(*linear_operator.batch_shape, linear_operator.size(-1), 5)
        with settings.fast_computations(covar_root_decomposition=False):
            root_approx = linear_operator.root_decomposition()
            res = root_approx.matmul(test_mat)
            actual = linear_operator.matmul(test_mat)
            self.assertLess(torch.norm(res - actual) / actual.norm(), 0.1)


class TestNonLinearOperatorBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_operator(self):
        mat = torch.randn(3, 5, 6)
        mat = mat.matmul(mat.transpose(-1, -2))
        mat.requires_grad_(True)
        return NonLinearOperator(mat)

    def evaluate_linear_operator(self, linear_operator):
        return linear_operator.tensor

    def test_root_decomposition_exact(self):
        linear_operator = self.create_linear_operator()
        test_mat = torch.randn(*linear_operator.batch_shape, linear_operator.size(-1), 5)
        with settings.fast_computations(covar_root_decomposition=False):
            root_approx = linear_operator.root_decomposition()
            res = root_approx.matmul(test_mat)
            actual = linear_operator.matmul(test_mat)
            self.assertLess(torch.norm(res - actual) / actual.norm(), 0.1)


class TestNonLinearOperatorMultiBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    # Because these LTs are large, we'll skil the big tests
    should_test_sample = False
    skip_slq_tests = True

    def create_linear_operator(self):
        mat = torch.randn(2, 3, 5, 6)
        mat = mat.matmul(mat.transpose(-1, -2))
        mat.requires_grad_(True)
        return NonLinearOperator(mat)

    def evaluate_linear_operator(self, linear_operator):
        return linear_operator.tensor
