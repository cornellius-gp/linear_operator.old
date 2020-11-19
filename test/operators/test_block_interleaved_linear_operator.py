#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import BlockInterleavedLinearOperator, DenseLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase


class TestBlockInterleavedLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True

    def create_linear_operator(self):
        blocks = torch.randn(8, 4, 4)
        blocks = blocks.matmul(blocks.transpose(-1, -2))
        blocks.add_(torch.eye(4, 4).unsqueeze_(0))
        return BlockInterleavedLinearOperator(DenseLinearOperator(blocks))

    def evaluate_linear_operator(self, linear_operator):
        blocks = linear_operator.base_linear_operator.tensor
        actual = torch.zeros(32, 32)
        for i in range(8):
            for j in range(4):
                for k in range(4):
                    actual[j * 8 + i, k * 8 + i] = blocks[i, j, k]
        return actual


class TestBlockInterleavedLinearOperatorBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True

    def create_linear_operator(self):
        blocks = torch.randn(2, 6, 4, 4)
        blocks = blocks.matmul(blocks.transpose(-1, -2))
        blocks.add_(torch.eye(4, 4))
        return BlockInterleavedLinearOperator(DenseLinearOperator(blocks), block_dim=2)

    def evaluate_linear_operator(self, linear_operator):
        blocks = linear_operator.base_linear_operator.tensor
        actual = torch.zeros(2, 24, 24)
        for i in range(2):
            for j in range(6):
                for k in range(4):
                    for m in range(4):
                        actual[i, k * 6 + j, m * 6 + j] = blocks[i, j, k, m]
        return actual


class TestBlockInterleavedLinearOperatorMultiBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    # Because these LTs are large, we'll skil the big tests
    should_test_sample = False
    skip_slq_tests = True

    def create_linear_operator(self):
        blocks = torch.randn(2, 6, 5, 4, 4)
        blocks = blocks.matmul(blocks.transpose(-1, -2))
        blocks.add_(torch.eye(4, 4))
        blocks.detach_()
        return BlockInterleavedLinearOperator(DenseLinearOperator(blocks), block_dim=1)

    def evaluate_linear_operator(self, linear_operator):
        blocks = linear_operator.base_linear_operator.tensor
        actual = torch.zeros(2, 5, 24, 24)
        for i in range(2):
            for j in range(6):
                for k in range(5):
                    for m in range(4):
                        for n in range(4):
                            actual[i, k, m * 6 + j, n * 6 + j] = blocks[i, k, j, m, n]
        return actual


if __name__ == "__main__":
    unittest.main()
