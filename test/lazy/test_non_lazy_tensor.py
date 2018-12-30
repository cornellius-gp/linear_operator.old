#!/usr/bin/env python3

import torch
import unittest
import gpytorch
from gpytorch.lazy import NonLazyTensor
from test.lazy._lazy_tensor_test_case import LazyTensorTestCase


class TestNonLazyTensor(LazyTensorTestCase, unittest.TestCase):
    seed = 0

    def create_lazy_tensor(self):
        mat = torch.randn(5, 6)
        mat = mat.matmul(mat.transpose(-1, -2))
        mat.requires_grad_(True)
        return NonLazyTensor(mat)

    def evaluate_lazy_tensor(self, lazy_tensor):
        return lazy_tensor.tensor

    def test_root_decomposition_exact(self):
        lazy_tensor = self.create_lazy_tensor()
        test_mat = torch.randn(*lazy_tensor.batch_shape, lazy_tensor.size(-1), 5)
        with gpytorch.settings.fast_computations(covar_root_decomposition=False):
            root_approx = lazy_tensor.root_decomposition()
            res = root_approx.matmul(test_mat)
            actual = lazy_tensor.matmul(test_mat)
            self.assertLess(torch.norm(res - actual) / actual.norm(), 0.1)


class TestNonLazyTensorBatch(LazyTensorTestCase, unittest.TestCase):
    seed = 0

    def create_lazy_tensor(self):
        mat = torch.randn(3, 5, 6)
        mat = mat.matmul(mat.transpose(-1, -2))
        mat.requires_grad_(True)
        return NonLazyTensor(mat)

    def evaluate_lazy_tensor(self, lazy_tensor):
        return lazy_tensor.tensor

    def test_root_decomposition_exact(self):
        lazy_tensor = self.create_lazy_tensor()
        test_mat = torch.randn(*lazy_tensor.batch_shape, lazy_tensor.size(-1), 5)
        with gpytorch.settings.fast_computations(covar_root_decomposition=False):
            root_approx = lazy_tensor.root_decomposition()
            res = root_approx.matmul(test_mat)
            actual = lazy_tensor.matmul(test_mat)
            self.assertLess(torch.norm(res - actual) / actual.norm(), 0.1)
