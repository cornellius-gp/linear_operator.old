#!/usr/bin/env python3

import torch

from ..utils.broadcasting import _matmul_broadcast_shape, _mul_broadcast_shape, _pad_with_singletons
from ..utils.getitem import _noop_index
from ..utils.memoize import cached
from .diag_linear_operator import DiagLinearOperator
from .linear_operator import LinearOperator
from .non_linear_operator import NonLinearOperator, to_linear_operator


def _inner_repeat(tensor, amt):
    return tensor.unsqueeze(-1).repeat(amt, 1).squeeze(-1)


def _outer_repeat(tensor, amt):
    return tensor.unsqueeze(-1).repeat(1, amt).view(-1)


class MatmulLinearOperator(LinearOperator):
    def __init__(self, left_linear_operator, right_linear_operator):
        left_linear_operator = to_linear_operator(left_linear_operator)
        right_linear_operator = to_linear_operator(right_linear_operator)

        # Match batch dimensions
        batch_shape = _mul_broadcast_shape(left_linear_operator.batch_shape, right_linear_operator.batch_shape)
        if left_linear_operator.batch_shape != batch_shape:
            left_linear_operator = left_linear_operator._expand_batch(batch_shape)
        if right_linear_operator.batch_shape != batch_shape:
            right_linear_operator = right_linear_operator._expand_batch(batch_shape)

        super().__init__(left_linear_operator, right_linear_operator)
        batch_shape = _mul_broadcast_shape(left_linear_operator.batch_shape, right_linear_operator.batch_shape)
        if left_linear_operator.batch_shape != batch_shape:
            self.left_linear_operator = left_linear_operator._expand_batch(batch_shape)
        else:
            self.left_linear_operator = left_linear_operator
        if right_linear_operator.batch_shape != batch_shape:
            self.right_linear_operator = right_linear_operator._expand_batch(batch_shape)
        else:
            self.right_linear_operator = right_linear_operator

    def _expand_batch(self, batch_shape):
        return self.__class__(
            self.left_linear_operator._expand_batch(batch_shape), self.right_linear_operator._expand_batch(batch_shape)
        )

    def _get_indices(self, row_index, col_index, *batch_indices):
        row_index = row_index.unsqueeze(-1)
        col_index = col_index.unsqueeze(-1)
        batch_indices = tuple(batch_index.unsqueeze(-1) for batch_index in batch_indices)
        inner_index = torch.arange(0, self.left_linear_operator.size(-1), device=self.device)
        inner_index = _pad_with_singletons(inner_index, row_index.dim() - 1, 0)

        left_tensor = self.left_linear_operator._get_indices(
            row_index, inner_index, *batch_indices[-len(self.left_linear_operator.batch_shape) :]
        )
        right_tensor = self.right_linear_operator._get_indices(
            inner_index, col_index, *batch_indices[-len(self.right_linear_operator.batch_shape) :]
        )
        res = (left_tensor * right_tensor).sum(-1)
        return res

    def _getitem(self, row_index, col_index, *batch_indices):
        # Make sure we're not generating more memory with our "efficient" method
        if torch.is_tensor(row_index) and torch.is_tensor(col_index):
            num_indices = row_index.numel()
            if num_indices > self.matrix_shape.numel():
                return to_linear_operator(self.evaluate())._getitem(row_index, col_index, *batch_indices)

        left_tensor = self.left_linear_operator._getitem(row_index, _noop_index, *batch_indices)
        right_tensor = self.right_linear_operator._getitem(_noop_index, col_index, *batch_indices)

        res = MatmulLinearOperator(left_tensor, right_tensor)
        return res

    def _matmul(self, right_linear_operator):
        return self.left_linear_operator._matmul(self.right_linear_operator._matmul(right_linear_operator))

    def _t_matmul(self, right_linear_operator):
        return self.right_linear_operator._t_matmul(self.left_linear_operator._t_matmul(right_linear_operator))

    def _quad_form_derivative(self, left_vecs, right_vecs):
        if left_vecs.ndimension() == 1:
            left_vecs = left_vecs.unsqueeze(1)
            right_vecs = right_vecs.unsqueeze(1)
        right_vecs_times_right_linear_operator = self.right_linear_operator._matmul(right_vecs)
        left_vecs_times_left_linear_operator_t = self.left_linear_operator._t_matmul(left_vecs)
        left_grad = self.left_linear_operator._quad_form_derivative(left_vecs, right_vecs_times_right_linear_operator)
        right_grad = self.right_linear_operator._quad_form_derivative(
            left_vecs_times_left_linear_operator_t, right_vecs
        )

        left_grad = (left_grad,) if not isinstance(left_grad, tuple) else left_grad
        right_grad = (right_grad,) if not isinstance(right_grad, tuple) else right_grad
        return left_grad + right_grad

    def _permute_batch(self, *dims):
        return self.__class__(
            self.left_linear_operator._permute_batch(*dims), self.right_linear_operator._permute_batch(*dims)
        )

    def _size(self):
        return _matmul_broadcast_shape(self.left_linear_operator.shape, self.right_linear_operator.shape)

    def _transpose_nonbatch(self, *args):
        return self.__class__(
            self.right_linear_operator._transpose_nonbatch(), self.left_linear_operator._transpose_nonbatch()
        )

    def diag(self):
        if isinstance(self.left_linear_operator, NonLinearOperator) and isinstance(
            self.right_linear_operator, NonLinearOperator
        ):
            return (self.left_linear_operator.tensor * self.right_linear_operator.tensor.transpose(-1, -2)).sum(-1)
        elif isinstance(self.left_linear_operator, DiagLinearOperator) or isinstance(
            self.right_linear_operator, DiagLinearOperator
        ):
            return self.left_linear_operator.diag() * self.right_linear_operator.diag()
        else:
            return super().diag()

    @cached
    def evaluate(self):
        return torch.matmul(self.left_linear_operator.evaluate(), self.right_linear_operator.evaluate())
