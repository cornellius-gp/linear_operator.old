from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import unittest
from gpytorch.lazy import ZeroLazyTensor


class TestZeroLazyTensor(unittest.TestCase):
    def test_evaluate(self):
        lv = ZeroLazyTensor(5, 4, 3)
        actual = torch.zeros(5, 4, 3)
        res = lv.evaluate()
        self.assertLess(torch.norm(res - actual), 1e-4)

    def test_getitem(self):
        lv = ZeroLazyTensor(5, 4, 3)

        res_one = lv[0].evaluate()
        res_two = lv[:, 1, :].evaluate()
        res_three = lv[:, :, 2].evaluate()

        self.assertLess(torch.norm(res_one - torch.zeros(4, 3)), 1e-4)
        self.assertLess(torch.norm(res_two - torch.zeros(5, 3)), 1e-4)
        self.assertLess(torch.norm(res_three - torch.zeros(5, 4)), 1e-4)

    def test_getitem_complex(self):
        lv = ZeroLazyTensor(5, 4, 3)

        res_one = lv[[0, 1]].evaluate()
        res_two = lv[:, [0, 1], :].evaluate()
        res_three = lv[:, :, [0, 2]].evaluate()

        self.assertLess(torch.norm(res_one - torch.zeros(2, 4, 3)), 1e-4)
        self.assertLess(torch.norm(res_two - torch.zeros(5, 2, 3)), 1e-4)
        self.assertLess(torch.norm(res_three - torch.zeros(5, 4, 2)), 1e-4)


if __name__ == "__main__":
    unittest.main()
