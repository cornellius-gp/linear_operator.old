#!/usr/bin/env python3

import itertools
import math
from abc import abstractmethod
from itertools import combinations, product
from unittest.mock import MagicMock, patch

import torch

from .base_test_case import BaseTestCase
from ..operators import to_dense, LinearOperator
from .. import settings
from ..utils import linear_cg
from ..utils.lanczos import lanczos_tridiag


def _ensure_symmetric_grad(grad):
    """
    A gradient-hook hack to ensure that symmetric matrix gradients are symmetric
    """
    res = torch.add(grad, grad.transpose(-1, -2)).mul(0.5)
    return res


class RectangularLinearOperatorTestCase(BaseTestCase):
    @abstractmethod
    def create_linear_operator(self):
        raise NotImplementedError()

    @abstractmethod
    def evaluate_linear_operator(self):
        raise NotImplementedError()

    def _test_matmul(self, rhs):
        linear_operator = self.create_linear_operator().requires_grad_(True)
        linear_operator_copy = linear_operator.clone().detach_().requires_grad_(True)
        evaluated = self.evaluate_linear_operator(linear_operator_copy)

        res = linear_operator.matmul(rhs)
        actual = evaluated.matmul(rhs)
        self.assertAllClose(res, actual)

        grad = torch.randn_like(res)
        res.backward(gradient=grad)
        actual.backward(gradient=grad)
        for arg, arg_copy in zip(linear_operator.representation(), linear_operator_copy.representation()):
            if arg_copy.requires_grad and arg_copy.is_leaf and arg_copy.grad is not None:
                self.assertAllClose(arg.grad, arg_copy.grad, rtol=1e-3)

    def test_add(self):
        linear_operator = self.create_linear_operator()
        evaluated = self.evaluate_linear_operator(linear_operator)

        rhs = torch.randn(linear_operator.shape)
        self.assertAllClose((linear_operator + rhs).evaluate(), evaluated + rhs)

        rhs = torch.randn(linear_operator.matrix_shape)
        self.assertAllClose((linear_operator + rhs).evaluate(), evaluated + rhs)

        rhs = torch.randn(2, *linear_operator.shape)
        self.assertAllClose((linear_operator + rhs).evaluate(), evaluated + rhs)

    def test_matmul_vec(self):
        linear_operator = self.create_linear_operator()
        rhs = torch.randn(linear_operator.size(-1))

        # We skip this test if we're dealing with batch LinearOperators
        # They shouldn't multiply by a vec
        if linear_operator.ndimension() > 2:
            return
        else:
            return self._test_matmul(rhs)

    def test_matmul_matrix(self):
        linear_operator = self.create_linear_operator()
        rhs = torch.randn(*linear_operator.batch_shape, linear_operator.size(-1), 4)
        return self._test_matmul(rhs)

    def test_matmul_matrix_broadcast(self):
        linear_operator = self.create_linear_operator()

        # Right hand size has one more batch dimension
        batch_shape = torch.Size((3, *linear_operator.batch_shape))
        rhs = torch.randn(*batch_shape, linear_operator.size(-1), 4)
        self._test_matmul(rhs)

        if linear_operator.ndimension() > 2:
            # Right hand size has one fewer batch dimension
            batch_shape = torch.Size(linear_operator.batch_shape[1:])
            rhs = torch.randn(*batch_shape, linear_operator.size(-1), 4)
            self._test_matmul(rhs)

            # Right hand size has a singleton dimension
            batch_shape = torch.Size((*linear_operator.batch_shape[:-1], 1))
            rhs = torch.randn(*batch_shape, linear_operator.size(-1), 4)
            self._test_matmul(rhs)

    def test_constant_mul(self):
        linear_operator = self.create_linear_operator()
        evaluated = self.evaluate_linear_operator(linear_operator)
        self.assertAllClose((linear_operator * 5).evaluate(), evaluated * 5)

    def test_evaluate(self):
        linear_operator = self.create_linear_operator()
        evaluated = self.evaluate_linear_operator(linear_operator)
        self.assertAllClose(linear_operator.evaluate(), evaluated)

    def test_getitem(self):
        linear_operator = self.create_linear_operator()
        evaluated = self.evaluate_linear_operator(linear_operator)

        # Non-batch case
        if linear_operator.ndimension() == 2:
            res = linear_operator[1]
            actual = evaluated[1]
            self.assertAllClose(res, actual)
            res = linear_operator[0:2].evaluate()
            actual = evaluated[0:2]
            self.assertAllClose(res, actual)
            res = linear_operator[:, 0:2].evaluate()
            actual = evaluated[:, 0:2]
            self.assertAllClose(res, actual)
            res = linear_operator[0:2, :].evaluate()
            actual = evaluated[0:2, :]
            self.assertAllClose(res, actual)
            res = linear_operator[..., 0:2].evaluate()
            actual = evaluated[..., 0:2]
            self.assertAllClose(res, actual)
            res = linear_operator[0:2, ...].evaluate()
            actual = evaluated[0:2, ...]
            self.assertAllClose(res, actual)
            res = linear_operator[..., 0:2, 2]
            actual = evaluated[..., 0:2, 2]
            self.assertAllClose(res, actual)
            res = linear_operator[0:2, ..., 2]
            actual = evaluated[0:2, ..., 2]
            self.assertAllClose(res, actual)

        # Batch case
        else:
            res = linear_operator[1].evaluate()
            actual = evaluated[1]
            self.assertAllClose(res, actual)
            res = linear_operator[0:2].evaluate()
            actual = evaluated[0:2]
            self.assertAllClose(res, actual)
            res = linear_operator[:, 0:2].evaluate()
            actual = evaluated[:, 0:2]
            self.assertAllClose(res, actual)

            for batch_index in product([1, slice(0, 2, None)], repeat=(linear_operator.dim() - 2)):
                res = linear_operator.__getitem__((*batch_index, slice(0, 1, None), slice(0, 2, None))).evaluate()
                actual = evaluated.__getitem__((*batch_index, slice(0, 1, None), slice(0, 2, None)))
                self.assertAllClose(res, actual)
                res = linear_operator.__getitem__((*batch_index, 1, slice(0, 2, None)))
                actual = evaluated.__getitem__((*batch_index, 1, slice(0, 2, None)))
                self.assertAllClose(res, actual)
                res = linear_operator.__getitem__((*batch_index, slice(1, None, None), 2))
                actual = evaluated.__getitem__((*batch_index, slice(1, None, None), 2))
                self.assertAllClose(res, actual)

            # Ellipsis
            res = linear_operator.__getitem__((Ellipsis, slice(1, None, None), 2))
            actual = evaluated.__getitem__((Ellipsis, slice(1, None, None), 2))
            self.assertAllClose(res, actual)
            res = linear_operator.__getitem__((slice(1, None, None), Ellipsis, 2))
            actual = evaluated.__getitem__((slice(1, None, None), Ellipsis, 2))
            self.assertAllClose(res, actual)

    def test_getitem_tensor_index(self):
        linear_operator = self.create_linear_operator()
        evaluated = self.evaluate_linear_operator(linear_operator)

        # Non-batch case
        if linear_operator.ndimension() == 2:
            index = (torch.tensor([0, 0, 1, 2]), torch.tensor([0, 1, 0, 2]))
            res, actual = linear_operator[index], evaluated[index]
            self.assertAllClose(res, actual)
            index = (torch.tensor([0, 0, 1, 2]), slice(None, None, None))
            res, actual = to_dense(linear_operator[index]), evaluated[index]
            self.assertAllClose(res, actual)
            index = (slice(None, None, None), torch.tensor([0, 0, 1, 2]))
            res, actual = to_dense(linear_operator[index]), evaluated[index]
            self.assertAllClose(res, actual)
            index = (torch.tensor([0, 0, 1, 2]), Ellipsis)
            res, actual = to_dense(linear_operator[index]), evaluated[index]
            self.assertAllClose(res, actual)
            index = (Ellipsis, torch.tensor([0, 0, 1, 2]))
            res, actual = to_dense(linear_operator[index]), evaluated[index]
            self.assertAllClose(res, actual)
            index = (Ellipsis, torch.tensor([0, 0, 1, 2]), torch.tensor([0, 1, 0, 2]))
            res, actual = linear_operator[index], evaluated[index]
            self.assertAllClose(res, actual)

        # Batch case
        else:
            for batch_index in product(
                [torch.tensor([0, 1, 1, 0]), slice(None, None, None)], repeat=(linear_operator.dim() - 2)
            ):
                index = (*batch_index, torch.tensor([0, 1, 0, 2]), torch.tensor([1, 2, 0, 1]))
                res, actual = linear_operator[index], evaluated[index]
                self.assertAllClose(res, actual)
                index = (*batch_index, torch.tensor([0, 1, 0, 2]), slice(None, None, None))
                res, actual = to_dense(linear_operator[index]), evaluated[index]
                self.assertAllClose(res, actual)
                index = (*batch_index, slice(None, None, None), torch.tensor([0, 1, 2, 1]))
                res, actual = to_dense(linear_operator[index]), evaluated[index]
                self.assertAllClose(res, actual)
                index = (*batch_index, slice(None, None, None), slice(None, None, None))
                res, actual = linear_operator[index].evaluate(), evaluated[index]
                self.assertAllClose(res, actual)

            # Ellipsis
            res = linear_operator.__getitem__((Ellipsis, torch.tensor([0, 1, 0, 2]), torch.tensor([1, 2, 0, 1])))
            actual = evaluated.__getitem__((Ellipsis, torch.tensor([0, 1, 0, 2]), torch.tensor([1, 2, 0, 1])))
            self.assertAllClose(res, actual)
            res = to_dense(
                linear_operator.__getitem__((torch.tensor([0, 1, 0, 1]), Ellipsis, torch.tensor([1, 2, 0, 1])))
            )
            actual = evaluated.__getitem__((torch.tensor([0, 1, 0, 1]), Ellipsis, torch.tensor([1, 2, 0, 1])))
            self.assertAllClose(res, actual)

    def test_permute(self):
        linear_operator = self.create_linear_operator()
        if linear_operator.dim() >= 4:
            evaluated = self.evaluate_linear_operator(linear_operator)
            dims = torch.randperm(linear_operator.dim() - 2).tolist()
            res = linear_operator.permute(*dims, -2, -1).evaluate()
            actual = evaluated.permute(*dims, -2, -1)
            self.assertAllClose(res, actual)

    def test_quad_form_derivative(self):
        linear_operator = self.create_linear_operator().requires_grad_(True)
        linear_operator_clone = linear_operator.clone().detach_().requires_grad_(True)
        left_vecs = torch.randn(*linear_operator.batch_shape, linear_operator.size(-2), 2)
        right_vecs = torch.randn(*linear_operator.batch_shape, linear_operator.size(-1), 2)

        deriv_custom = linear_operator._quad_form_derivative(left_vecs, right_vecs)
        deriv_auto = LinearOperator._quad_form_derivative(linear_operator_clone, left_vecs, right_vecs)

        for dc, da in zip(deriv_custom, deriv_auto):
            self.assertAllClose(dc, da)

    def test_sum(self):
        linear_operator = self.create_linear_operator()
        evaluated = self.evaluate_linear_operator(linear_operator)

        self.assertAllClose(linear_operator.sum(-1), evaluated.sum(-1))
        self.assertAllClose(linear_operator.sum(-2), evaluated.sum(-2))
        if linear_operator.ndimension() > 2:
            self.assertAllClose(linear_operator.sum(-3).evaluate(), evaluated.sum(-3))
        if linear_operator.ndimension() > 3:
            self.assertAllClose(linear_operator.sum(-4).evaluate(), evaluated.sum(-4))

    def test_transpose_batch(self):
        linear_operator = self.create_linear_operator()
        evaluated = self.evaluate_linear_operator(linear_operator)

        if linear_operator.dim() >= 4:
            for i, j in combinations(range(linear_operator.dim() - 2), 2):
                res = linear_operator.transpose(i, j).evaluate()
                actual = evaluated.transpose(i, j)
                self.assertAllClose(res, actual, rtol=1e-4, atol=1e-5)


class LinearOperatorTestCase(RectangularLinearOperatorTestCase):
    should_test_sample = False
    skip_slq_tests = False
    should_call_cg = True
    should_call_lanczos = True

    def _test_inv_matmul(self, rhs, lhs=None, cholesky=False):
        linear_operator = self.create_linear_operator().requires_grad_(True)
        linear_operator_copy = linear_operator.clone().detach_().requires_grad_(True)
        evaluated = self.evaluate_linear_operator(linear_operator_copy)
        evaluated.register_hook(_ensure_symmetric_grad)

        # Create a test right hand side and left hand side
        rhs.requires_grad_(True)
        rhs_copy = rhs.clone().detach().requires_grad_(True)
        if lhs is not None:
            lhs.requires_grad_(True)
            lhs_copy = lhs.clone().detach().requires_grad_(True)

        _wrapped_cg = MagicMock(wraps=linear_cg)
        with patch("linear_operator.utils.linear_cg", new=_wrapped_cg) as linear_cg_mock:
            with settings.max_cholesky_size(math.inf if cholesky else 0), settings.cg_tolerance(1e-4):
                # Perform the inv_matmul
                if lhs is not None:
                    res = linear_operator.inv_matmul(rhs, lhs)
                    actual = lhs_copy @ evaluated.inverse() @ rhs_copy
                else:
                    res = linear_operator.inv_matmul(rhs)
                    actual = evaluated.inverse().matmul(rhs_copy)
                self.assertAllClose(res, actual, rtol=0.02, atol=1e-5)

                # Perform backward pass
                grad = torch.randn_like(res)
                res.backward(gradient=grad)
                actual.backward(gradient=grad)
                for arg, arg_copy in zip(linear_operator.representation(), linear_operator_copy.representation()):
                    if arg_copy.requires_grad and arg_copy.is_leaf and arg_copy.grad is not None:
                        self.assertAllClose(arg.grad, arg_copy.grad, rtol=0.03, atol=1e-5)
                self.assertAllClose(rhs.grad, rhs_copy.grad, rtol=0.03, atol=1e-5)
                if lhs is not None:
                    self.assertAllClose(lhs.grad, lhs_copy.grad, rtol=0.03, atol=1e-5)

            # Determine if we've called CG or not
            if not cholesky and self.__class__.should_call_cg:
                self.assertTrue(linear_cg_mock.called)
            else:
                self.assertFalse(linear_cg_mock.called)

    def _test_inv_quad_logdet(self, reduce_inv_quad=True, cholesky=False):
        if not self.__class__.skip_slq_tests:
            # Forward
            linear_operator = self.create_linear_operator()
            evaluated = self.evaluate_linear_operator(linear_operator)
            flattened_evaluated = evaluated.view(-1, *linear_operator.matrix_shape)

            vecs = torch.randn(*linear_operator.batch_shape, linear_operator.size(-1), 3, requires_grad=True)
            vecs_copy = vecs.clone().detach_().requires_grad_(True)

            _wrapped_cg = MagicMock(wraps=linear_cg)
            with patch("linear_operator.utils.linear_cg", new=_wrapped_cg) as linear_cg_mock:
                with settings.num_trace_samples(256), settings.max_cholesky_size(
                    math.inf if cholesky else 0
                ), settings.cg_tolerance(1e-5):

                    res_inv_quad, res_logdet = linear_operator.inv_quad_logdet(
                        inv_quad_rhs=vecs, logdet=True, reduce_inv_quad=reduce_inv_quad
                    )

            actual_inv_quad = evaluated.inverse().matmul(vecs_copy).mul(vecs_copy).sum(-2)
            if reduce_inv_quad:
                actual_inv_quad = actual_inv_quad.sum(-1)
            actual_logdet = torch.cat(
                [torch.logdet(flattened_evaluated[i]).unsqueeze(0) for i in range(linear_operator.batch_shape.numel())]
            ).view(linear_operator.batch_shape)

            self.assertAllClose(res_inv_quad, actual_inv_quad, rtol=0.01, atol=0.01)
            self.assertAllClose(res_logdet, actual_logdet, rtol=0.2, atol=0.03)

            if not cholesky and self.__class__.should_call_cg:
                self.assertTrue(linear_cg_mock.called)
            else:
                self.assertFalse(linear_cg_mock.called)

    def test_add_diag(self):
        linear_operator = self.create_linear_operator()
        evaluated = self.evaluate_linear_operator(linear_operator)

        other_diag = torch.tensor(1.5)
        res = linear_operator.add_diag(other_diag).evaluate()
        actual = evaluated + torch.eye(evaluated.size(-1)).view(
            *[1 for _ in range(linear_operator.dim() - 2)], evaluated.size(-1), evaluated.size(-1)
        ).repeat(*linear_operator.batch_shape, 1, 1).mul(1.5)
        self.assertAllClose(res, actual)

        other_diag = torch.tensor([1.5])
        res = linear_operator.add_diag(other_diag).evaluate()
        actual = evaluated + torch.eye(evaluated.size(-1)).view(
            *[1 for _ in range(linear_operator.dim() - 2)], evaluated.size(-1), evaluated.size(-1)
        ).repeat(*linear_operator.batch_shape, 1, 1).mul(1.5)
        self.assertAllClose(res, actual)

        other_diag = torch.randn(linear_operator.size(-1)).pow(2)
        res = linear_operator.add_diag(other_diag).evaluate()
        actual = evaluated + other_diag.diag().repeat(*linear_operator.batch_shape, 1, 1)
        self.assertAllClose(res, actual)

        for sizes in product([1, None], repeat=(linear_operator.dim() - 2)):
            batch_shape = [linear_operator.batch_shape[i] if size is None else size for i, size in enumerate(sizes)]
            other_diag = torch.randn(*batch_shape, linear_operator.size(-1)).pow(2)
            res = linear_operator.add_diag(other_diag).evaluate()
            actual = evaluated.clone().detach()
            for i in range(other_diag.size(-1)):
                actual[..., i, i] = actual[..., i, i] + other_diag[..., i]
            self.assertAllClose(res, actual, rtol=1e-2, atol=1e-5)

    def test_cholesky(self):
        linear_operator = self.create_linear_operator()
        evaluated = self.evaluate_linear_operator(linear_operator)
        for upper in (False, True):
            res = linear_operator.cholesky(upper=upper).evaluate()
            actual = torch.cholesky(evaluated, upper=upper)
            self.assertAllClose(res, actual, rtol=1e-3, atol=1e-5)
            # TODO: Check gradients

    def test_diag(self):
        linear_operator = self.create_linear_operator()
        evaluated = self.evaluate_linear_operator(linear_operator)

        res = linear_operator.diag()
        actual = evaluated.diagonal(dim1=-2, dim2=-1)
        actual = actual.view(*linear_operator.batch_shape, -1)
        self.assertAllClose(res, actual, rtol=1e-2, atol=1e-5)

    def test_inv_matmul_vector(self, cholesky=False):
        linear_operator = self.create_linear_operator()
        rhs = torch.randn(linear_operator.size(-1))

        # We skip this test if we're dealing with batch LinearOperators
        # They shouldn't multiply by a vec
        if linear_operator.ndimension() > 2:
            return
        else:
            return self._test_inv_matmul(rhs)

    def test_inv_matmul_vector_with_left(self, cholesky=False):
        linear_operator = self.create_linear_operator()
        rhs = torch.randn(linear_operator.size(-1))
        lhs = torch.randn(6, linear_operator.size(-1))

        # We skip this test if we're dealing with batch LinearOperators
        # They shouldn't multiply by a vec
        if linear_operator.ndimension() > 2:
            return
        else:
            return self._test_inv_matmul(rhs, lhs=lhs)

    def test_inv_matmul_vector_with_left_cholesky(self):
        linear_operator = self.create_linear_operator()
        rhs = torch.randn(*linear_operator.batch_shape, linear_operator.size(-1), 5)
        lhs = torch.randn(*linear_operator.batch_shape, 6, linear_operator.size(-1))
        return self._test_inv_matmul(rhs, lhs=lhs, cholesky=True)

    def test_inv_matmul_matrix(self, cholesky=False):
        linear_operator = self.create_linear_operator()
        rhs = torch.randn(*linear_operator.batch_shape, linear_operator.size(-1), 5)
        return self._test_inv_matmul(rhs, cholesky=cholesky)

    def test_inv_matmul_matrix_cholesky(self):
        return self.test_inv_matmul_matrix(cholesky=True)

    def test_inv_matmul_matrix_with_left(self):
        linear_operator = self.create_linear_operator()
        rhs = torch.randn(*linear_operator.batch_shape, linear_operator.size(-1), 5)
        lhs = torch.randn(*linear_operator.batch_shape, 3, linear_operator.size(-1))
        return self._test_inv_matmul(rhs, lhs=lhs)

    def test_inv_matmul_matrix_broadcast(self):
        linear_operator = self.create_linear_operator()

        # Right hand size has one more batch dimension
        batch_shape = torch.Size((3, *linear_operator.batch_shape))
        rhs = torch.randn(*batch_shape, linear_operator.size(-1), 5)
        self._test_inv_matmul(rhs)

        if linear_operator.ndimension() > 2:
            # Right hand size has one fewer batch dimension
            batch_shape = torch.Size(linear_operator.batch_shape[1:])
            rhs = torch.randn(*batch_shape, linear_operator.size(-1), 5)
            self._test_inv_matmul(rhs)

            # Right hand size has a singleton dimension
            batch_shape = torch.Size((*linear_operator.batch_shape[:-1], 1))
            rhs = torch.randn(*batch_shape, linear_operator.size(-1), 5)
            self._test_inv_matmul(rhs)

    def test_inv_quad_logdet(self):
        return self._test_inv_quad_logdet(reduce_inv_quad=False, cholesky=False)

    def test_inv_quad_logdet_no_reduce(self):
        return self._test_inv_quad_logdet(reduce_inv_quad=True, cholesky=False)

    def test_inv_quad_logdet_no_reduce_cholesky(self):
        return self._test_inv_quad_logdet(reduce_inv_quad=True, cholesky=True)

    def test_prod(self):
        with settings.fast_computations(covar_root_decomposition=False):
            linear_operator = self.create_linear_operator()
            evaluated = self.evaluate_linear_operator(linear_operator)

            if linear_operator.ndimension() > 2:
                self.assertAllClose(linear_operator.prod(-3).evaluate(), evaluated.prod(-3), atol=1e-2, rtol=1e-2)
            if linear_operator.ndimension() > 3:
                self.assertAllClose(linear_operator.prod(-4).evaluate(), evaluated.prod(-4), atol=1e-2, rtol=1e-2)

    def test_root_decomposition(self, cholesky=False):
        _wrapped_lanczos = MagicMock(wraps=lanczos_tridiag)
        with patch("linear_operator.utils.lanczos.lanczos_tridiag", new=_wrapped_lanczos) as lanczos_mock:
            linear_operator = self.create_linear_operator()
            test_mat = torch.randn(*linear_operator.batch_shape, linear_operator.size(-1), 5)
            with settings.max_cholesky_size(math.inf if cholesky else 0):
                root_approx = linear_operator.root_decomposition()
                res = root_approx.matmul(test_mat)
                actual = linear_operator.matmul(test_mat)
                self.assertAllClose(res, actual, rtol=0.05)

            # Make sure that we're calling the correct function
            if not cholesky and self.__class__.should_call_lanczos:
                self.assertTrue(lanczos_mock.called)
            else:
                self.assertFalse(lanczos_mock.called)

    def test_root_decomposition_cholesky(self):
        return self.test_root_decomposition(cholesky=True)

    def test_root_inv_decomposition(self):
        linear_operator = self.create_linear_operator()
        root_approx = linear_operator.root_inv_decomposition()

        test_mat = torch.randn(*linear_operator.batch_shape, linear_operator.size(-1), 5)

        res = root_approx.matmul(test_mat)
        actual = linear_operator.inv_matmul(test_mat)
        self.assertAllClose(res, actual, rtol=0.05, atol=0.02)

    def test_sample(self):
        if self.__class__.should_test_sample:
            linear_operator = self.create_linear_operator()
            evaluated = self.evaluate_linear_operator(linear_operator)

            samples = linear_operator.zero_mean_mvn_samples(50000)
            sample_covar = samples.unsqueeze(-1).matmul(samples.unsqueeze(-2)).mean(0)
            self.assertAllClose(sample_covar, evaluated, rtol=0.3, atol=0.3)

    def test_sqrt_inv_matmul(self):
        linear_operator = self.create_linear_operator().requires_grad_(True)
        if len(linear_operator.batch_shape):
            return

        linear_operator_copy = linear_operator.clone().detach_().requires_grad_(True)
        evaluated = self.evaluate_linear_operator(linear_operator_copy)
        evaluated.register_hook(_ensure_symmetric_grad)

        # Create a test right hand side and left hand side
        rhs = torch.randn(*linear_operator.shape[:-1], 3).requires_grad_(True)
        lhs = torch.randn(*linear_operator.shape[:-2], 2, linear_operator.size(-1)).requires_grad_(True)
        rhs_copy = rhs.clone().detach().requires_grad_(True)
        lhs_copy = lhs.clone().detach().requires_grad_(True)

        # Perform forward pass
        with settings.max_cg_iterations(200):
            sqrt_inv_matmul_res, inv_quad_res = linear_operator.sqrt_inv_matmul(rhs, lhs)
        evals, evecs = evaluated.symeig(eigenvectors=True)
        matrix_inv_root = evecs @ (evals.sqrt().reciprocal().unsqueeze(-1) * evecs.transpose(-1, -2))
        sqrt_inv_matmul_actual = lhs_copy @ matrix_inv_root @ rhs_copy
        inv_quad_actual = (lhs_copy @ matrix_inv_root).pow(2).sum(dim=-1)

        # Check forward pass
        self.assertAllClose(sqrt_inv_matmul_res, sqrt_inv_matmul_actual, rtol=1e-4, atol=1e-3)
        self.assertAllClose(inv_quad_res, inv_quad_actual, rtol=1e-4, atol=1e-3)

        # Perform backward pass
        sqrt_inv_matmul_grad = torch.randn_like(sqrt_inv_matmul_res)
        inv_quad_grad = torch.randn_like(inv_quad_res)
        ((sqrt_inv_matmul_res * sqrt_inv_matmul_grad).sum() + (inv_quad_res * inv_quad_grad).sum()).backward()
        ((sqrt_inv_matmul_actual * sqrt_inv_matmul_grad).sum() + (inv_quad_actual * inv_quad_grad).sum()).backward()

        # Check grads
        self.assertAllClose(rhs.grad, rhs_copy.grad, rtol=1e-4, atol=1e-3)
        self.assertAllClose(lhs.grad, lhs_copy.grad, rtol=1e-4, atol=1e-3)
        for arg, arg_copy in zip(linear_operator.representation(), linear_operator_copy.representation()):
            if arg_copy.requires_grad and arg_copy.is_leaf and arg_copy.grad is not None:
                self.assertAllClose(arg.grad, arg_copy.grad, rtol=1e-4, atol=1e-3)

    def test_sqrt_inv_matmul_no_lhs(self):
        linear_operator = self.create_linear_operator().requires_grad_(True)
        if len(linear_operator.batch_shape):
            return

        linear_operator_copy = linear_operator.clone().detach_().requires_grad_(True)
        evaluated = self.evaluate_linear_operator(linear_operator_copy)
        evaluated.register_hook(_ensure_symmetric_grad)

        # Create a test right hand side and left hand side
        rhs = torch.randn(*linear_operator.shape[:-1], 3).requires_grad_(True)
        rhs_copy = rhs.clone().detach().requires_grad_(True)

        # Perform forward pass
        with settings.max_cg_iterations(200):
            sqrt_inv_matmul_res = linear_operator.sqrt_inv_matmul(rhs)
        evals, evecs = evaluated.symeig(eigenvectors=True)
        matrix_inv_root = evecs @ (evals.sqrt().reciprocal().unsqueeze(-1) * evecs.transpose(-1, -2))
        sqrt_inv_matmul_actual = matrix_inv_root @ rhs_copy

        # Check forward pass
        self.assertAllClose(sqrt_inv_matmul_res, sqrt_inv_matmul_actual, rtol=1e-4, atol=1e-3)

        # Perform backward pass
        sqrt_inv_matmul_grad = torch.randn_like(sqrt_inv_matmul_res)
        ((sqrt_inv_matmul_res * sqrt_inv_matmul_grad).sum()).backward()
        ((sqrt_inv_matmul_actual * sqrt_inv_matmul_grad).sum()).backward()

        # Check grads
        self.assertAllClose(rhs.grad, rhs_copy.grad, rtol=1e-4, atol=1e-3)
        for arg, arg_copy in zip(linear_operator.representation(), linear_operator_copy.representation()):
            if arg_copy.requires_grad and arg_copy.is_leaf and arg_copy.grad is not None:
                self.assertAllClose(arg.grad, arg_copy.grad, rtol=1e-4, atol=1e-3)

    def test_symeig(self):
        linear_operator = self.create_linear_operator().requires_grad_(True)
        linear_operator_copy = linear_operator.clone().detach_().requires_grad_(True)
        evaluated = self.evaluate_linear_operator(linear_operator_copy)

        # Perform forward pass
        evals_unsorted, evecs_unsorted = linear_operator.symeig(eigenvectors=True)
        evecs_unsorted = evecs_unsorted.evaluate()

        # since LinearOperator.symeig does not sort evals, we do this here for the check
        evals, idxr = torch.sort(evals_unsorted, dim=-1, descending=False)
        evecs = torch.gather(evecs_unsorted, dim=-1, index=idxr.unsqueeze(-2).expand(evecs_unsorted.shape))

        evals_actual, evecs_actual = torch.symeig(evaluated.double(), eigenvectors=True)
        evals_actual = evals_actual.to(dtype=evaluated.dtype)
        evecs_actual = evecs_actual.to(dtype=evaluated.dtype)

        # Check forward pass
        self.assertAllClose(evals, evals_actual, rtol=1e-4, atol=1e-3)
        lt_from_eigendecomp = evecs @ torch.diag_embed(evals) @ evecs.transpose(-1, -2)
        self.assertAllClose(lt_from_eigendecomp, evaluated, rtol=1e-4, atol=1e-3)

        # if there are repeated evals, we'll skip checking the eigenvectors for those
        any_evals_repeated = False
        evecs_abs, evecs_actual_abs = evecs.abs(), evecs_actual.abs()
        for idx in itertools.product(*[range(b) for b in evals_actual.shape[:-1]]):
            eval_i = evals_actual[idx]
            if torch.unique(eval_i.detach()).shape[-1] == eval_i.shape[-1]:  # detach to avoid pytorch/pytorch#41389
                self.assertAllClose(evecs_abs[idx], evecs_actual_abs[idx], rtol=1e-4, atol=1e-3)
            else:
                any_evals_repeated = True

        # Perform backward pass
        symeig_grad = torch.randn_like(evals)
        ((evals * symeig_grad).sum()).backward()
        ((evals_actual * symeig_grad).sum()).backward()

        # Check grads if there were no repeated evals
        if not any_evals_repeated:
            for arg, arg_copy in zip(linear_operator.representation(), linear_operator_copy.representation()):
                if arg_copy.requires_grad and arg_copy.is_leaf and arg_copy.grad is not None:
                    self.assertAllClose(arg.grad, arg_copy.grad, rtol=1e-4, atol=1e-3)

        # Test with eigenvectors=False
        _, evecs = linear_operator.symeig(eigenvectors=False)
        self.assertIsNone(evecs)

    def test_svd(self):
        linear_operator = self.create_linear_operator().requires_grad_(True)
        linear_operator_copy = linear_operator.clone().detach_().requires_grad_(True)
        evaluated = self.evaluate_linear_operator(linear_operator_copy)

        # Perform forward pass
        U_unsorted, S_unsorted, V_unsorted = linear_operator.svd()
        U_unsorted, V_unsorted = U_unsorted.evaluate(), V_unsorted.evaluate()

        # since LinearOperator.svd does not sort the singular values, we do this here for the check
        S, idxr = torch.sort(S_unsorted, dim=-1, descending=True)
        idxr = idxr.unsqueeze(-2).expand(U_unsorted.shape)
        U = torch.gather(U_unsorted, dim=-1, index=idxr)
        V = torch.gather(V_unsorted, dim=-1, index=idxr)

        # compute expected result from full tensor
        U_actual, S_actual, V_actual = torch.svd(evaluated.double())
        U_actual = U_actual.to(dtype=evaluated.dtype)
        S_actual = S_actual.to(dtype=evaluated.dtype)
        V_actual = V_actual.to(dtype=evaluated.dtype)

        # Check forward pass
        self.assertAllClose(S, S_actual, rtol=1e-4, atol=1e-3)
        lt_from_svd = U @ torch.diag_embed(S) @ V.transpose(-1, -2)
        self.assertAllClose(lt_from_svd, evaluated, rtol=1e-4, atol=1e-3)

        # if there are repeated singular values, we'll skip checking the singular vectors
        U_abs, U_actual_abs = U.abs(), U_actual.abs()
        V_abs, V_actual_abs = V.abs(), V_actual.abs()
        any_svals_repeated = False
        for idx in itertools.product(*[range(b) for b in S_actual.shape[:-1]]):
            Si = S_actual[idx]
            if torch.unique(Si.detach()).shape[-1] == Si.shape[-1]:  # detach to avoid pytorch/pytorch#41389
                self.assertAllClose(U_abs[idx], U_actual_abs[idx], rtol=1e-4, atol=1e-3)
                self.assertAllClose(V_abs[idx], V_actual_abs[idx], rtol=1e-4, atol=1e-3)
            else:
                any_svals_repeated = True

        # Perform backward pass
        svd_grad = torch.randn_like(S)
        ((S * svd_grad).sum()).backward()
        ((S_actual * svd_grad).sum()).backward()

        # Check grads if there were no repeated singular values
        if not any_svals_repeated:
            for arg, arg_copy in zip(linear_operator.representation(), linear_operator_copy.representation()):
                if arg_copy.requires_grad and arg_copy.is_leaf and arg_copy.grad is not None:
                    self.assertAllClose(arg.grad, arg_copy.grad, rtol=1e-4, atol=1e-3)
