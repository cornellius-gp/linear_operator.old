#!/usr/bin/env python3

from __future__ import annotations

import unittest

import torch

from linear_operator.operators import ToeplitzLinearOperator, to_linear_operator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase


class TestSumLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_operator(self):
        c1 = torch.tensor([5, 1, 2, 0], dtype=torch.float, requires_grad=True)
        t1 = ToeplitzLinearOperator(c1)
        c2 = torch.tensor([6, 0, 1, -1], dtype=torch.float, requires_grad=True)
        t2 = ToeplitzLinearOperator(c2)
        return t1 + t2

    def evaluate_linear_operator(self, linear_operator):
        tensors = [lt.evaluate() for lt in linear_operator.linear_operators]
        return sum(tensors)


class TestSumLinearOperatorBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_operator(self):
        c1 = torch.tensor([[2, 0.5, 0, 0], [5, 1, 2, 0]], dtype=torch.float, requires_grad=True)
        t1 = ToeplitzLinearOperator(c1)
        c2 = torch.tensor([[2, 0.5, 0, 0], [6, 0, 1, -1]], dtype=torch.float, requires_grad=True)
        t2 = ToeplitzLinearOperator(c2)
        return t1 + t2

    def evaluate_linear_operator(self, linear_operator):
        tensors = [lt.evaluate() for lt in linear_operator.linear_operators]
        return sum(tensors)


class TestSumLinearOperatorMultiBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    # Because these LTs are large, we'll skil the big tests
    skip_slq_tests = True

    def create_linear_operator(self):
        c1 = torch.tensor(
            [[[2, 0.5, 0, 0], [5, 1, 2, 0]], [[2, 0.5, 0, 0], [5, 1, 2, 0]]], dtype=torch.float, requires_grad=True,
        )
        t1 = ToeplitzLinearOperator(c1)
        c2 = torch.tensor(
            [[[2, 0.5, 0, 0], [5, 1, 2, 0]], [[2, 0.5, 0, 0], [6, 0, 1, -1]]], dtype=torch.float, requires_grad=True,
        )
        t2 = ToeplitzLinearOperator(c2)
        return t1 + t2

    def evaluate_linear_operator(self, linear_operator):
        tensors = [lt.evaluate() for lt in linear_operator.linear_operators]
        return sum(tensors)


class TestSumLinearOperatorBroadcasting(unittest.TestCase):
    def test_broadcast_same_shape(self):
        test1 = to_linear_operator(torch.randn(30, 30))

        test2 = torch.randn(30, 30)
        res = test1 + test2
        final_res = res + test2

        torch_res = res.evaluate() + test2

        self.assertEqual(final_res.shape, torch_res.shape)
        self.assertEqual((final_res.evaluate() - torch_res).sum(), 0.0)

    def test_broadcast_tensor_shape(self):
        test1 = to_linear_operator(torch.randn(30, 30))

        test2 = torch.randn(30, 1)
        res = test1 + test2
        final_res = res + test2

        torch_res = res.evaluate() + test2

        self.assertEqual(final_res.shape, torch_res.shape)
        self.assertEqual((final_res.evaluate() - torch_res).sum(), 0.0)

    def test_broadcast_lo_shape(self):
        test1 = to_linear_operator(torch.randn(30, 1))

        test2 = torch.randn(30, 30)
        res = test1 + test2
        final_res = res + test2

        torch_res = res.evaluate() + test2

        self.assertEqual(final_res.shape, torch_res.shape)
        self.assertEqual((final_res.evaluate() - torch_res).sum(), 0.0)


if __name__ == "__main__":
    unittest.main()
