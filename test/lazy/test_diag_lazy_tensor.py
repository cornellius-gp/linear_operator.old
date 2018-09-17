from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import unittest
import gpytorch
from gpytorch.lazy import DiagLazyTensor


diag = torch.tensor([1, 2, 3], dtype=torch.float)


class TestDiagLazyTensor(unittest.TestCase):
    def test_evaluate(self):
        diag_lv = DiagLazyTensor(diag)
        self.assertTrue(torch.equal(diag_lv.evaluate().data, diag.diag()))

    def test_function_factory(self):
        # 1d
        diag_var1 = torch.tensor(diag.data, requires_grad=True)
        diag_var2 = torch.tensor(diag.data, requires_grad=True)
        test_mat = torch.Tensor([3, 4, 5])

        diag_lv = DiagLazyTensor(diag_var1)
        diag_ev = DiagLazyTensor(diag_var2).evaluate()

        # Forward
        res = diag_lv.matmul(test_mat)
        actual = torch.matmul(diag_ev, test_mat)
        self.assertLess(torch.norm(res.data - actual.data), 1e-4)

        # Backward
        res.sum().backward()
        actual.sum().backward()
        self.assertLess(torch.norm(diag_var1.grad.data - diag_var2.grad.data), 1e-3)

        # 2d
        diag_var1 = torch.tensor(diag.data, requires_grad=True)
        diag_var2 = torch.tensor(diag.data, requires_grad=True)
        test_mat = torch.eye(3)

        diag_lv = DiagLazyTensor(diag_var1)
        diag_ev = DiagLazyTensor(diag_var2).evaluate()

        # Forward
        res = diag_lv.matmul(test_mat)
        actual = torch.matmul(diag_ev, test_mat)
        self.assertLess(torch.norm(res.data - actual.data), 1e-4)

        # Backward
        res.sum().backward()
        actual.sum().backward()
        self.assertLess(torch.norm(diag_var1.grad.data - diag_var2.grad.data), 1e-3)

    def test_batch_function_factory(self):
        # 2d
        diag_var1 = torch.tensor(diag.data.repeat(5, 1), requires_grad=True)
        diag_var2 = torch.tensor(diag.data.repeat(5, 1), requires_grad=True)
        test_mat = torch.eye(3).repeat(5, 1, 1)

        diag_lv = DiagLazyTensor(diag_var1)
        diag_ev = DiagLazyTensor(diag_var2).evaluate()

        # Forward
        res = diag_lv.matmul(test_mat)
        actual = torch.matmul(diag_ev, test_mat)
        self.assertLess(torch.norm(res.data - actual.data), 1e-4)

        # Backward
        res.sum().backward()
        actual.sum().backward()
        self.assertLess(torch.norm(diag_var1.grad.data - diag_var2.grad.data), 1e-3)

    def test_getitem(self):
        diag_lv = DiagLazyTensor(diag)
        diag_ev = diag_lv.evaluate()
        self.assertTrue(torch.equal(diag_lv[0:2].evaluate().data, diag_ev[0:2].data))

    def test_batch_getitem(self):
        # 2d
        diag_lv = DiagLazyTensor(diag.repeat(5, 1))
        diag_ev = diag_lv.evaluate()

        self.assertTrue(torch.equal(diag_lv[0, 0:2].evaluate().data, diag_ev[0, 0:2].data))
        self.assertTrue(torch.equal(diag_lv[0, 0:2, :3].evaluate().data, diag_ev[0, 0:2, :3].data))
        self.assertTrue(torch.equal(diag_lv[:, 0:2, :3].evaluate().data, diag_ev[:, 0:2, :3].data))

    def test_sample(self):
        res = DiagLazyTensor(diag)
        actual = res.evaluate()

        with gpytorch.settings.max_root_decomposition_size(1000):
            samples = res.zero_mean_mvn_samples(10000)
            sample_covar = samples.unsqueeze(-1).matmul(samples.unsqueeze(-2)).mean(0)
        self.assertLess(((sample_covar - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 4e-1)

    def test_batch_sample(self):
        res = DiagLazyTensor(diag.repeat(5, 1))
        actual = res.evaluate()

        with gpytorch.settings.max_root_decomposition_size(1000):
            samples = res.zero_mean_mvn_samples(10000)
            sample_covar = samples.unsqueeze(-1).matmul(samples.unsqueeze(-2)).mean(0)
        self.assertLess(((sample_covar - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 4e-1)


if __name__ == "__main__":
    unittest.main()
