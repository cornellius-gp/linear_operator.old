#!/usr/bin/env python3

import math
import unittest
import warnings
from unittest.mock import MagicMock, patch

import torch

from linear_operator import settings, utils
from linear_operator.operators import CachedCGLinearOperator, NonLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase, _ensure_symmetric_grad
from linear_operator.utils.warnings import ExtraComputationWarning


class TestCachedCGLinearOperatorNoLogdet(LinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_operator(self, with_solves=False, with_logdet=False):
        mat = torch.randn(5, 6)
        mat = mat.matmul(mat.transpose(-1, -2))
        mat.requires_grad_(True)

        linear_operator = NonLinearOperator(mat)
        eager_rhs = torch.randn(5, 10).detach()
        if with_solves:
            with settings.num_trace_samples(1000 if with_logdet else 1):  # For inv_quad_logdet tests
                solve, probe_vecs, probe_vec_norms, probe_vec_solves, tmats = CachedCGLinearOperator.precompute_terms(
                    linear_operator, eager_rhs.detach(), logdet_terms=with_logdet
                )
                eager_rhss = [eager_rhs.detach(), eager_rhs[..., -2:-1].detach()]
                solves = [solve.detach(), solve[..., -2:-1].detach()]
        else:
            eager_rhss = [eager_rhs]
            solves = []
            probe_vecs = torch.tensor([], dtype=mat.dtype, device=mat.device)
            probe_vec_norms = torch.tensor([], dtype=mat.dtype, device=mat.device)
            probe_vec_solves = torch.tensor([], dtype=mat.dtype, device=mat.device)
            tmats = torch.tensor([], dtype=mat.dtype, device=mat.device)

        return CachedCGLinearOperator(
            linear_operator, eager_rhss, solves, probe_vecs, probe_vec_norms, probe_vec_solves, tmats
        )

    def evaluate_linear_operator(self, linear_operator):
        return linear_operator.base_linear_operator.tensor

    def _test_inv_matmul(self, rhs, lhs=None, cholesky=False):
        if cholesky:  # These tests don't make sense for CachedCGLinearOperator
            return

        linear_operator = self.create_linear_operator(with_solves=True).requires_grad_(True)
        linear_operator_copy = linear_operator.clone().detach_().requires_grad_(True)
        evaluated = self.evaluate_linear_operator(linear_operator_copy)
        evaluated.register_hook(_ensure_symmetric_grad)

        # Rather than the supplied rhs and lhs,
        # we'll replace them with ones that we've precomputed solves for
        rhs_orig = rhs
        if rhs_orig.dim() == 1:
            rhs = linear_operator.eager_rhss[0][..., -1].squeeze(-1).clone().detach().requires_grad_(True)
            # Make sure we're setting this test up correctly
            self.assertEqual(rhs_orig.shape, rhs.shape)
        else:
            if lhs is not None:
                rhs = linear_operator.eager_rhss[0][..., 2:].clone().detach().requires_grad_(True)
            else:
                rhs = linear_operator.eager_rhss[0].clone().detach().requires_grad_(True)
            # Make sure we're setting this test up correctly
            self.assertEqual(rhs_orig.shape[:-1], rhs.shape[:-1])

        lhs = lhs
        if lhs is not None:
            lhs_orig = lhs
            if rhs_orig.dim() == 1:
                lhs = linear_operator.eager_rhss[0][..., :-1].transpose(-1, -2).clone().detach().requires_grad_(True)
            else:
                lhs = linear_operator.eager_rhss[0][..., :2].transpose(-1, -2).clone().detach().requires_grad_(True)

            # Make sure we're setting this test up correctly
            self.assertEqual(lhs_orig.shape[:-2], lhs.shape[:-2])

        # Create a test right hand side and left hand side
        rhs.requires_grad_(True)
        rhs_copy = rhs.clone().detach().requires_grad_(True)
        if lhs is not None:
            lhs.requires_grad_(True)
            lhs_copy = lhs.clone().detach().requires_grad_(True)

        _wrapped_cg = MagicMock(wraps=utils.linear_cg)
        with patch("linear_operator.utils.linear_cg", new=_wrapped_cg) as linear_cg_mock:
            with settings.max_cholesky_size(math.inf if cholesky else 0), settings.cg_tolerance(1e-4):
                with warnings.catch_warnings(record=True) as ws:
                    # Makes sure warnings we catch don't cause `-w error` to fail
                    warnings.simplefilter("always", ExtraComputationWarning)

                    # Perform the inv_matmul
                    if lhs is not None:
                        res = linear_operator.inv_matmul(rhs, lhs)
                        actual = lhs_copy @ evaluated.inverse() @ rhs_copy
                    else:
                        res = linear_operator.inv_matmul(rhs)
                        actual = evaluated.inverse().matmul(rhs_copy)
                    self.assertAllClose(res, actual, rtol=0.02, atol=1e-5)
                    self.assertFalse(any(issubclass(w.category, ExtraComputationWarning) for w in ws))

                with warnings.catch_warnings(record=True) as ws:
                    # Makes sure warnings we catch don't cause `-w error` to fail
                    warnings.simplefilter("always", ExtraComputationWarning)

                    # Perform backward pass
                    grad = torch.randn_like(res)
                    res.backward(gradient=grad)
                    actual.backward(gradient=grad)
                    for arg, arg_copy in zip(linear_operator.representation(), linear_operator_copy.representation()):
                        if arg_copy.grad is not None:
                            self.assertAllClose(arg.grad, arg_copy.grad, rtol=0.03, atol=1e-5)
                    self.assertAllClose(rhs.grad, rhs_copy.grad, rtol=0.03, atol=1e-5)
                    if lhs is not None:
                        self.assertAllClose(lhs.grad, lhs_copy.grad, rtol=0.03, atol=1e-5)

                    # Determine if we've called CG or not
                    # We shouldn't if we supplied a lhs
                    if lhs is None:
                        self.assertEqual(len([w for w in ws if issubclass(w.category, ExtraComputationWarning)]), 1)
                        if not cholesky and self.__class__.should_call_cg:
                            self.assertTrue(linear_cg_mock.called)
                    else:
                        self.assertFalse(any(issubclass(w.category, ExtraComputationWarning) for w in ws))
                        self.assertFalse(linear_cg_mock.called)

    def _test_inv_quad_logdet(self, reduce_inv_quad=True, cholesky=False):
        with warnings.catch_warnings():
            # Makes sure warnings we catch don't cause `-w error` to fail
            warnings.simplefilter("ignore", ExtraComputationWarning)
            super()._test_inv_quad_logdet(reduce_inv_quad=reduce_inv_quad, cholesky=cholesky)

    def test_inv_matmul_vector(self):
        # Skipping this test because it's not really necessary for CachedCGLinearOperator
        # We'll only ever be performing inv_matmul against matrices ,r owhen a left hand side is supplied
        pass

    def test_inv_matmul_matrix_broadcast(self):
        pass

    def test_inv_quad_logdet(self):
        pass

    def test_inv_quad_logdet_no_reduce(self):
        pass

    def test_root_inv_decomposition(self):
        linear_operator = self.create_linear_operator()
        root_approx = linear_operator.root_inv_decomposition()

        test_mat = linear_operator.eager_rhss[0].clone().detach()

        res = root_approx.matmul(test_mat)
        with warnings.catch_warnings():
            # Makes sure warnings we catch don't cause `-w error` to fail
            warnings.simplefilter("ignore", ExtraComputationWarning)
            actual = linear_operator.inv_matmul(test_mat)
        self.assertLess(torch.norm(res - actual) / actual.norm(), 0.1)


class TestCachedCGLinearOperator(TestCachedCGLinearOperatorNoLogdet):
    seed = 0

    def test_inv_quad_logdet(self):
        # Forward
        linear_operator = self.create_linear_operator(with_solves=True, with_logdet=True)
        evaluated = self.evaluate_linear_operator(linear_operator)
        flattened_evaluated = evaluated.view(-1, *linear_operator.matrix_shape)

        vecs = linear_operator.eager_rhss[0].clone().detach().requires_grad_(True)
        vecs_copy = linear_operator.eager_rhss[0].clone().detach().requires_grad_(True)

        with settings.num_trace_samples(128), warnings.catch_warnings(record=True) as ws:
            res_inv_quad, res_logdet = linear_operator.inv_quad_logdet(inv_quad_rhs=vecs, logdet=True)
            self.assertFalse(any(issubclass(w.category, ExtraComputationWarning) for w in ws))
        res = res_inv_quad + res_logdet

        actual_inv_quad = evaluated.inverse().matmul(vecs_copy).mul(vecs_copy).sum(-2).sum(-1)
        actual_logdet = torch.cat(
            [torch.logdet(flattened_evaluated[i]).unsqueeze(0) for i in range(linear_operator.batch_shape.numel())]
        ).view(linear_operator.batch_shape)
        actual = actual_inv_quad + actual_logdet

        diff = (res - actual).abs() / actual.abs().clamp(1, math.inf)
        self.assertLess(diff.max().item(), 15e-2)

    def test_inv_quad_logdet_no_reduce(self):
        # Forward
        linear_operator = self.create_linear_operator(with_solves=True, with_logdet=True)
        evaluated = self.evaluate_linear_operator(linear_operator)
        flattened_evaluated = evaluated.view(-1, *linear_operator.matrix_shape)

        vecs = linear_operator.eager_rhss[0].clone().detach().requires_grad_(True)
        vecs_copy = linear_operator.eager_rhss[0].clone().detach().requires_grad_(True)

        with settings.num_trace_samples(128), warnings.catch_warnings(record=True) as ws:
            res_inv_quad, res_logdet = linear_operator.inv_quad_logdet(
                inv_quad_rhs=vecs, logdet=True, reduce_inv_quad=False
            )
            self.assertFalse(any(issubclass(w.category, ExtraComputationWarning) for w in ws))
        res = res_inv_quad.sum(-1) + res_logdet

        actual_inv_quad = evaluated.inverse().matmul(vecs_copy).mul(vecs_copy).sum(-2).sum(-1)
        actual_logdet = torch.cat(
            [torch.logdet(flattened_evaluated[i]).unsqueeze(0) for i in range(linear_operator.batch_shape.numel())]
        ).view(linear_operator.batch_shape)
        actual = actual_inv_quad + actual_logdet

        diff = (res - actual).abs() / actual.abs().clamp(1, math.inf)
        self.assertLess(diff.max().item(), 15e-2)


class TestCachedCGLinearOperatorNoLogdetBatch(TestCachedCGLinearOperatorNoLogdet):
    seed = 0

    def create_linear_operator(self, with_solves=False, with_logdet=False):
        mat = torch.randn(3, 5, 6)
        mat = mat.matmul(mat.transpose(-1, -2))
        mat.requires_grad_(True)

        linear_operator = NonLinearOperator(mat)
        eager_rhs = torch.randn(3, 5, 10).detach()
        if with_solves:
            with settings.num_trace_samples(1000 if with_logdet else 1):  # For inv_quad_logdet tests
                solve, probe_vecs, probe_vec_norms, probe_vec_solves, tmats = CachedCGLinearOperator.precompute_terms(
                    linear_operator, eager_rhs.detach(), logdet_terms=with_logdet
                )
                eager_rhss = [eager_rhs.detach(), eager_rhs[..., -2:-1].detach()]
                solves = [solve.detach(), solve[..., -2:-1].detach()]
        else:
            eager_rhss = [eager_rhs]
            solves = []
            probe_vecs = torch.tensor([], dtype=mat.dtype, device=mat.device)
            probe_vec_norms = torch.tensor([], dtype=mat.dtype, device=mat.device)
            probe_vec_solves = torch.tensor([], dtype=mat.dtype, device=mat.device)
            tmats = torch.tensor([], dtype=mat.dtype, device=mat.device)

        return CachedCGLinearOperator(
            linear_operator, eager_rhss, solves, probe_vecs, probe_vec_norms, probe_vec_solves, tmats
        )


class TestCachedCGLinearOperatorBatch(TestCachedCGLinearOperator):
    seed = 0

    def create_linear_operator(self, with_solves=False, with_logdet=False):
        mat = torch.randn(3, 5, 6)
        mat = mat.matmul(mat.transpose(-1, -2))
        mat.requires_grad_(True)

        linear_operator = NonLinearOperator(mat)
        eager_rhs = torch.randn(3, 5, 10).detach()
        if with_solves:
            with settings.num_trace_samples(1000 if with_logdet else 1):  # For inv_quad_logdet tests
                solve, probe_vecs, probe_vec_norms, probe_vec_solves, tmats = CachedCGLinearOperator.precompute_terms(
                    linear_operator, eager_rhs.detach(), logdet_terms=with_logdet
                )
                eager_rhss = [eager_rhs.detach(), eager_rhs[..., -2:-1].detach()]
                solves = [solve.detach(), solve[..., -2:-1].detach()]
        else:
            eager_rhss = [eager_rhs]
            solves = []
            probe_vecs = torch.tensor([], dtype=mat.dtype, device=mat.device)
            probe_vec_norms = torch.tensor([], dtype=mat.dtype, device=mat.device)
            probe_vec_solves = torch.tensor([], dtype=mat.dtype, device=mat.device)
            tmats = torch.tensor([], dtype=mat.dtype, device=mat.device)

        return CachedCGLinearOperator(
            linear_operator, eager_rhss, solves, probe_vecs, probe_vec_norms, probe_vec_solves, tmats
        )


class TestCachedCGLinearOperatorMultiBatch(TestCachedCGLinearOperator):
    seed = 0
    # Because these LTs are large, we'll skil the big tests
    should_test_sample = False
    skip_slq_tests = True

    def create_linear_operator(self, with_solves=False, with_logdet=False):
        mat = torch.randn(2, 3, 5, 6)
        mat = mat.matmul(mat.transpose(-1, -2))
        mat.requires_grad_(True)

        linear_operator = NonLinearOperator(mat)
        eager_rhs = torch.randn(2, 3, 5, 10).detach()
        if with_solves:
            with settings.num_trace_samples(1000 if with_logdet else 1):  # For inv_quad_logdet tests
                solve, probe_vecs, probe_vec_norms, probe_vec_solves, tmats = CachedCGLinearOperator.precompute_terms(
                    linear_operator, eager_rhs.detach(), logdet_terms=with_logdet
                )
                eager_rhss = [eager_rhs.detach(), eager_rhs[..., -2:-1].detach()]
                solves = [solve.detach(), solve[..., -2:-1].detach()]
        else:
            eager_rhss = [eager_rhs]
            solves = []
            probe_vecs = torch.tensor([], dtype=mat.dtype, device=mat.device)
            probe_vec_norms = torch.tensor([], dtype=mat.dtype, device=mat.device)
            probe_vec_solves = torch.tensor([], dtype=mat.dtype, device=mat.device)
            tmats = torch.tensor([], dtype=mat.dtype, device=mat.device)

        return CachedCGLinearOperator(
            linear_operator, eager_rhss, solves, probe_vecs, probe_vec_norms, probe_vec_solves, tmats
        )
