from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from .lazy_tensor import LazyTensor
from .non_lazy_tensor import NonLazyTensor


def _inner_repeat(tensor, amt):
    return tensor.unsqueeze(-1).repeat(amt, 1).squeeze(-1)


def _outer_repeat(tensor, amt):
    return tensor.unsqueeze(-1).repeat(1, amt).view(-1)


class MatmulLazyTensor(LazyTensor):
    def __init__(self, left_lazy_tensor, right_lazy_tensor):
        if not isinstance(left_lazy_tensor, LazyTensor):
            left_lazy_tensor = NonLazyTensor(left_lazy_tensor)
        if not isinstance(right_lazy_tensor, LazyTensor):
            right_lazy_tensor = NonLazyTensor(right_lazy_tensor)

        super(MatmulLazyTensor, self).__init__(left_lazy_tensor, right_lazy_tensor)
        self.left_lazy_tensor = left_lazy_tensor
        self.right_lazy_tensor = right_lazy_tensor

    @property
    def _evaluated_left_lazy_tensor(self):
        if not hasattr(self, "_evaluated_left_lazy_tensor_memo"):
            self._evaluated_left_lazy_tensor_memo = self.left_lazy_tensor.evaluate()
        return self._evaluated_left_lazy_tensor_memo

    @property
    def _evaluated_right_lazy_tensor(self):
        if not hasattr(self, "_evaluated_right_lazy_tensor_memo"):
            self._evaluated_right_lazy_tensor_memo = self.right_lazy_tensor.evaluate()
        return self._evaluated_right_lazy_tensor_memo

    def _matmul(self, right_lazy_tensor):
        return self.left_lazy_tensor._matmul(self.right_lazy_tensor._matmul(right_lazy_tensor))

    def _t_matmul(self, right_lazy_tensor):
        return self.right_lazy_tensor._t_matmul(self.left_lazy_tensor._t_matmul(right_lazy_tensor))

    def _quad_form_derivative(self, left_vecs, right_vecs):
        if left_vecs.ndimension() == 1:
            left_vecs = left_vecs.unsqueeze(1)
            right_vecs = right_vecs.unsqueeze(1)
        right_vecs_times_right_lazy_tensor = self.right_lazy_tensor._matmul(right_vecs)
        left_vecs_times_left_lazy_tensor_t = self.left_lazy_tensor._t_matmul(left_vecs)
        left_grad, = self.left_lazy_tensor._quad_form_derivative(left_vecs, right_vecs_times_right_lazy_tensor)
        right_grad, = self.right_lazy_tensor._quad_form_derivative(left_vecs_times_left_lazy_tensor_t, right_vecs)
        return left_grad, right_grad

    def _size(self):
        if self.left_lazy_tensor.ndimension() > 2:
            return torch.Size(
                (self.left_lazy_tensor.size(0), self.left_lazy_tensor.size(1), self.right_lazy_tensor.size(2))
            )
        else:
            return torch.Size((self.left_lazy_tensor.size(0), self.right_lazy_tensor.size(1)))

    def _transpose_nonbatch(self, *args):
        return self.__class__(self.right_lazy_tensor._transpose_nonbatch(), self.left_lazy_tensor._transpose_nonbatch())

    def _batch_get_indices(self, batch_indices, left_indices, right_indices):
        n_indices = left_indices.numel()
        if n_indices > self.size(-1) * self.size(-2) * self.size(-3):
            return self._evaluated[batch_indices, left_indices, right_indices]

        else:
            outer_size = batch_indices.size(0)
            inner_size = self.left_lazy_tensor.size(-1)
            inner_indices = torch.arange(0, inner_size, dtype=torch.long, device=self.device)

            # Repeat the indices to get all the appropriate terms
            batch_indices = _outer_repeat(batch_indices, inner_size)
            left_indices = _outer_repeat(left_indices, inner_size)
            right_indices = _outer_repeat(right_indices, inner_size)
            inner_indices = _inner_repeat(inner_indices, outer_size)

            left_vals = self.left_lazy_tensor._batch_get_indices(batch_indices, left_indices, inner_indices)
            right_vals = self.right_lazy_tensor._batch_get_indices(batch_indices, inner_indices, right_indices)

            return (left_vals.view(-1, inner_size) * right_vals.view(-1, inner_size)).sum(-1)

    def _get_indices(self, left_indices, right_indices):
        n_indices = left_indices.numel()
        if n_indices > self.size(-1) * self.size(-2):
            return self._evaluated[left_indices, right_indices]

        else:
            outer_size = left_indices.size(0)
            inner_size = self.left_lazy_tensor.size(-1)
            inner_indices = torch.arange(0, inner_size, dtype=torch.long, device=self.device)

            # Repeat the indices to get all the appropriate terms
            left_indices = _outer_repeat(left_indices, inner_size)
            right_indices = _outer_repeat(right_indices, inner_size)
            inner_indices = _inner_repeat(inner_indices, outer_size)

            left_vals = self.left_lazy_tensor._get_indices(left_indices, inner_indices)
            right_vals = self.right_lazy_tensor._get_indices(inner_indices, right_indices)

            return (left_vals.view(-1, inner_size) * right_vals.view(-1, inner_size)).sum(-1)

    def diag(self):
        if isinstance(self.left_lazy_tensor, NonLazyTensor) and isinstance(self.right_lazy_tensor, NonLazyTensor):
            return (self.left_lazy_tensor.tensor * self.right_lazy_tensor.tensor.transpose(-1, -2)).sum(-1)
        else:
            return super(MatmulLazyTensor, self).diag()

    def evaluate(self):
        if not hasattr(self, "_evaluated_memo"):
            self._evaluated_memo = torch.matmul(self._evaluated_left_lazy_tensor, self._evaluated_right_lazy_tensor)
        return self._evaluated_memo
