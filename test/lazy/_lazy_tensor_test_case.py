from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import gpytorch
import torch
import os
import random
from abc import abstractmethod
from test._utils import approx_equal


class RectangularLazyTensorTestCase(object):
    @abstractmethod
    def create_lazy_tensor(self):
        raise NotImplementedError()

    @abstractmethod
    def evaluate_lazy_tensor(self):
        raise NotImplementedError()

    def setUp(self):
        if hasattr(self.__class__, "seed"):
            seed = self.__class__.seed
            if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
                self.rng_state = torch.get_rng_state()
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                random.seed(seed)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_matmul_vec(self):
        lazy_tensor = self.create_lazy_tensor()
        lazy_tensor_copy = lazy_tensor.clone()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)

        test_vector = torch.randn(lazy_tensor.size(-1))
        res = lazy_tensor.matmul(test_vector)
        actual = evaluated.matmul(test_vector)
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 3e-1)

        grad = torch.randn_like(res)
        res.backward(gradient=grad)
        actual.backward(gradient=grad)
        for arg, arg_copy in zip(lazy_tensor.representation(), lazy_tensor_copy.representation()):
            if arg_copy.grad is not None:
                self.assertLess(
                    ((arg.grad - arg_copy.grad).abs() / arg_copy.grad.abs().clamp(1, 1e5)).max().item(), 3e-1
                )

    def test_matmul_matrix(self):
        lazy_tensor = self.create_lazy_tensor()
        lazy_tensor_copy = lazy_tensor.clone()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)

        test_vector = torch.randn(lazy_tensor.size(-1), 5)
        res = lazy_tensor.matmul(test_vector)
        actual = evaluated.matmul(test_vector)
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 3e-1)

        grad = torch.randn_like(res)
        res.backward(gradient=grad)
        actual.backward(gradient=grad)
        for arg, arg_copy in zip(lazy_tensor.representation(), lazy_tensor_copy.representation()):
            if arg_copy.grad is not None:
                self.assertLess(
                    ((arg.grad - arg_copy.grad).abs() / arg_copy.grad.abs().clamp(1, 1e5)).max().item(), 3e-1
                )

    def test_evaluate(self):
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)
        self.assertTrue(approx_equal(lazy_tensor.evaluate(), evaluated))

    def test_getitem(self):
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)

        res = lazy_tensor[1]
        actual = evaluated[1]
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
        res = lazy_tensor[0:2].evaluate()
        actual = evaluated[0:2]
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
        res = lazy_tensor[:, 0:2].evaluate()
        actual = evaluated[:, 0:2]
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
        res = lazy_tensor[0:2, :].evaluate()
        actual = evaluated[0:2, :]
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)

    def test_getitem_tensor_index(self):
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)

        index = (torch.tensor([0, 0, 1, 2]), torch.tensor([0, 1, 0, 2]))
        res, actual = lazy_tensor[index], evaluated[index]
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
        index = (torch.tensor([0, 0, 1, 2]), slice(None, None, None))
        res, actual = lazy_tensor[index], evaluated[index]
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
        index = (slice(None, None, None), torch.tensor([0, 0, 1, 2]))
        res, actual = lazy_tensor[index], evaluated[index]
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)


class LazyTensorTestCase(RectangularLazyTensorTestCase):
    should_test_sample = False

    def test_add_diag(self):
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)

        other_diag = torch.tensor(1.5)
        res = lazy_tensor.add_diag(other_diag).evaluate()
        actual = evaluated + torch.eye(evaluated.size(-1)).mul(1.5)
        self.assertTrue(approx_equal(res, actual))

        other_diag = torch.tensor([1.5])
        res = lazy_tensor.add_diag(other_diag).evaluate()
        actual = evaluated + torch.eye(evaluated.size(-1)).mul(1.5)
        self.assertTrue(approx_equal(res, actual))

        other_diag = torch.randn(lazy_tensor.size(-1)).pow(2)
        res = lazy_tensor.add_diag(other_diag).evaluate()
        actual = evaluated + other_diag.diag()
        self.assertTrue(approx_equal(res, actual))

    def test_inv_matmul_vec(self):
        lazy_tensor = self.create_lazy_tensor()
        lazy_tensor_copy = lazy_tensor.clone()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)

        test_vector = torch.randn(lazy_tensor.size(-1))
        with gpytorch.settings.max_cg_iterations(200):
            res = lazy_tensor.inv_matmul(test_vector)
        actual = evaluated.inverse().matmul(test_vector)
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 3e-1)

        grad = torch.randn_like(res)
        res.backward(gradient=grad)
        actual.backward(gradient=grad)
        for arg, arg_copy in zip(lazy_tensor.representation(), lazy_tensor_copy.representation()):
            if arg_copy.grad is not None:
                self.assertLess(
                    ((arg.grad - arg_copy.grad).abs() / arg_copy.grad.abs().clamp(1, 1e5)).max().item(), 3e-1
                )

    def test_inv_matmul_matrix(self):
        lazy_tensor = self.create_lazy_tensor()
        lazy_tensor_copy = lazy_tensor.clone()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)

        test_vector = torch.randn(lazy_tensor.size(-1), 5)
        with gpytorch.settings.max_cg_iterations(100):
            res = lazy_tensor.inv_matmul(test_vector)
        actual = evaluated.inverse().matmul(test_vector)
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 3e-1)

        grad = torch.randn_like(res)
        res.backward(gradient=grad)
        actual.backward(gradient=grad)
        for arg, arg_copy in zip(lazy_tensor.representation(), lazy_tensor_copy.representation()):
            if arg_copy.grad is not None:
                self.assertLess(
                    ((arg.grad - arg_copy.grad).abs() / arg_copy.grad.abs().clamp(1, 1e5)).max().item(), 3e-1
                )

    def test_diag(self):
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)

        res = lazy_tensor.diag()
        actual = evaluated.diag()
        self.assertEqual(res.size(), lazy_tensor.size()[:-1])
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 3e-1)

    def test_inv_quad_log_det(self):
        # Forward
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)

        vecs = torch.randn(lazy_tensor.size(1), 3, requires_grad=True)
        vecs_copy = vecs.clone()

        with gpytorch.settings.num_trace_samples(128):
            res_inv_quad, res_log_det = lazy_tensor.inv_quad_log_det(inv_quad_rhs=vecs, log_det=True)
        res = res_inv_quad + res_log_det

        actual_inv_quad = evaluated.inverse().matmul(vecs_copy).mul(vecs_copy).sum()
        actual = actual_inv_quad + torch.logdet(evaluated)

        diff = (res - actual).abs() / actual.abs().clamp(1, 1e10)
        self.assertLess(diff.item(), 15e-2)

    def test_sample(self):
        if self.__class__.should_test_sample:
            lazy_tensor = self.create_lazy_tensor()
            evaluated = self.evaluate_lazy_tensor(lazy_tensor)

            samples = lazy_tensor.zero_mean_mvn_samples(10000)
            sample_covar = samples.unsqueeze(-1).matmul(samples.unsqueeze(-2)).mean(0)
            self.assertLess(((sample_covar - evaluated).abs() / evaluated.abs().clamp(1, 1e5)).max().item(), 3e-1)


class RectangularBatchLazyTensorTestCase(object):
    def test_matmul_matrix(self):
        lazy_tensor = self.create_lazy_tensor()
        lazy_tensor_copy = lazy_tensor.clone()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)

        test_vector = torch.randn(lazy_tensor.size(0), lazy_tensor.size(-1), 5)
        res = lazy_tensor.matmul(test_vector)
        actual = evaluated.matmul(test_vector)
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 3e-1)

        grad = torch.randn_like(res)
        res.backward(gradient=grad)
        actual.backward(gradient=grad)
        for arg, arg_copy in zip(lazy_tensor.representation(), lazy_tensor_copy.representation()):
            if arg_copy.grad is not None:
                self.assertLess(
                    ((arg.grad - arg_copy.grad).abs() / arg_copy.grad.abs().clamp(1, 1e5)).max().item(), 3e-1
                )

    def test_evaluate(self):
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)
        self.assertTrue(approx_equal(lazy_tensor.evaluate(), evaluated))

    def test_getitem(self):
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)

        res = lazy_tensor[1].evaluate()
        actual = evaluated[1]
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
        res = lazy_tensor[0:2].evaluate()
        actual = evaluated[0:2]
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
        res = lazy_tensor[:, 0:2].evaluate()
        actual = evaluated[:, 0:2]
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
        res = lazy_tensor[1, :1, :2].evaluate()
        actual = evaluated[1, :1, :2]
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
        res = lazy_tensor[1, 1, :2]
        actual = evaluated[1, 1, :2]
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
        res = lazy_tensor[1, :1, 2]
        actual = evaluated[1, :1, 2]
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)

    def test_getitem_tensor_index(self):
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)

        index = (torch.tensor([0, 1, 1, 0]), torch.tensor([0, 1, 0, 2]), torch.tensor([1, 2, 0, 1]))
        res, actual = lazy_tensor[index], evaluated[index]
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
        index = (torch.tensor([0, 1, 1, 0]), torch.tensor([0, 1, 0, 2]), slice(None, None, None))
        res, actual = lazy_tensor[index], evaluated[index]
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
        index = (torch.tensor([0, 1, 1]), slice(None, None, None), torch.tensor([0, 1, 2]))
        res, actual = lazy_tensor[index], evaluated[index]
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
        index = (slice(None, None, None), torch.tensor([0, 1, 1, 0]), torch.tensor([0, 1, 0, 2]))
        res, actual = lazy_tensor[index], evaluated[index]
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
        index = (torch.tensor([0, 0, 1, 1]), slice(None, None, None), slice(None, None, None))
        res, actual = lazy_tensor[index].evaluate(), evaluated[index]
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
        index = (slice(None, None, None), torch.tensor([0, 0, 1, 2]), torch.tensor([0, 0, 1, 1]))
        res, actual = lazy_tensor[index], evaluated[index]
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
        index = (torch.tensor([0, 1, 1, 0]), torch.tensor([0, 1, 0, 2]), slice(None, None, None))
        res, actual = lazy_tensor[index], evaluated[index]
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)
        index = (torch.tensor([0, 0, 1, 0]), slice(None, None, None), torch.tensor([0, 0, 1, 1]))
        res, actual = lazy_tensor[index], evaluated[index]
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 1e-1)


class BatchLazyTensorTestCase(RectangularBatchLazyTensorTestCase):
    should_test_sample = False

    @abstractmethod
    def create_lazy_tensor(self):
        raise NotImplementedError()

    @abstractmethod
    def evaluate_lazy_tensor(self):
        raise NotImplementedError()

    def setUp(self):
        if hasattr(self.__class__, "seed"):
            seed = self.__class__.seed
            if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
                self.rng_state = torch.get_rng_state()
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                random.seed(seed)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_add_diag(self):
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)

        other_diag = torch.tensor(1.5)
        res = lazy_tensor.add_diag(other_diag).evaluate()
        actual = evaluated + torch.eye(evaluated.size(-1)).unsqueeze(0).repeat(evaluated.size(0), 1, 1).mul(1.5)
        self.assertTrue(approx_equal(res, actual))

        other_diag = torch.tensor([1.5])
        res = lazy_tensor.add_diag(other_diag).evaluate()
        actual = evaluated + torch.eye(evaluated.size(-1)).unsqueeze(0).repeat(evaluated.size(0), 1, 1).mul(1.5)
        self.assertTrue(approx_equal(res, actual))

        other_diag = torch.randn(lazy_tensor.size(-1)).pow(2)
        res = lazy_tensor.add_diag(other_diag).evaluate()
        actual = evaluated + other_diag.diag().unsqueeze(0).repeat(evaluated.size(0), 1, 1)
        self.assertTrue(approx_equal(res, actual))

        other_diag = torch.randn(1, lazy_tensor.size(-1)).pow(2)
        res = lazy_tensor.add_diag(other_diag).evaluate()
        actual = evaluated + other_diag[0].diag().unsqueeze(0).repeat(evaluated.size(0), 1, 1)
        self.assertTrue(approx_equal(res, actual))

        other_diag = torch.randn(lazy_tensor.size(0), lazy_tensor.size(-1)).pow(2)
        res = lazy_tensor.add_diag(other_diag).evaluate()
        actual = evaluated + torch.cat([other_diag[i].diag().unsqueeze(0) for i in range(lazy_tensor.size(0))])
        self.assertTrue(approx_equal(res, actual))

        other_diag = torch.randn(lazy_tensor.size(0), 1).pow(2)
        res = lazy_tensor.add_diag(other_diag).evaluate()
        actual = evaluated + torch.cat(
            [torch.eye(lazy_tensor.size(-1)).mul(other_diag[i, 0]).unsqueeze(0) for i in range(lazy_tensor.size(0))]
        )
        self.assertTrue(approx_equal(res, actual))

    def test_inv_matmul_matrix(self):
        lazy_tensor = self.create_lazy_tensor()
        lazy_tensor_copy = lazy_tensor.clone()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)

        test_vector = torch.randn(lazy_tensor.size(0), lazy_tensor.size(-1), 5)
        with gpytorch.settings.max_cg_iterations(200):
            res = lazy_tensor.inv_matmul(test_vector)
        actual = torch.cat(
            [evaluated[i].inverse().matmul(test_vector[i]).unsqueeze(0) for i in range(lazy_tensor.size(0))]
        )
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 3e-1)

        grad = torch.randn_like(res)
        res.backward(gradient=grad)
        actual.backward(gradient=grad)
        for arg, arg_copy in zip(lazy_tensor.representation(), lazy_tensor_copy.representation()):
            if arg_copy.grad is not None:
                self.assertLess(
                    ((arg.grad - arg_copy.grad).abs() / arg_copy.grad.abs().clamp(1, 1e5)).max().item(), 3e-1
                )

    def test_diag(self):
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)

        res = lazy_tensor.diag()
        actual = torch.stack([evaluated[i].diag() for i in range(evaluated.size(0))])
        self.assertEqual(res.size(), lazy_tensor.size()[:-1])
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 3e-1)

    def test_inv_quad_log_det(self):
        # Forward
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)

        vecs = torch.randn(lazy_tensor.size(0), lazy_tensor.size(1), 3, requires_grad=True)
        vecs_copy = vecs.clone()

        with gpytorch.settings.num_trace_samples(128):
            res_inv_quad, res_log_det = lazy_tensor.inv_quad_log_det(inv_quad_rhs=vecs, log_det=True)
        res = res_inv_quad + res_log_det

        actual_inv_quad = torch.cat(
            [
                evaluated[i].inverse().matmul(vecs_copy[i]).mul(vecs_copy[i]).sum().unsqueeze(0)
                for i in range(lazy_tensor.size(0))
            ]
        )
        actual = actual_inv_quad + torch.cat(
            [torch.logdet(evaluated[i]).unsqueeze(0) for i in range(lazy_tensor.size(0))]
        )
        diffs = (res - actual).abs() / actual.abs().clamp(1, 1e10)
        for i in range(lazy_tensor.size(0)):
            self.assertLess(diffs[i].item(), 15e-2)

    def test_sample(self):
        if self.__class__.should_test_sample:
            lazy_tensor = self.create_lazy_tensor()
            evaluated = self.evaluate_lazy_tensor(lazy_tensor)

            samples = lazy_tensor.zero_mean_mvn_samples(10000)
            sample_covar = samples.unsqueeze(-1).matmul(samples.unsqueeze(-2)).mean(0)
            self.assertLess(((sample_covar - evaluated).abs() / evaluated.abs().clamp(1, 1e5)).max().item(), 3e-1)
