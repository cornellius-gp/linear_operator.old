#!/usr/bin/env python3
from torch import Tensor

from ..utils.broadcasting import _mul_broadcast_shape
from ..utils.memoize import cached
from .linear_operator import LinearOperator
from .non_linear_operator import to_linear_operator
from .zero_linear_operator import ZeroLinearOperator

# from .broadcasted_linear_operator import BroadcastedLinearOperator


class SumLinearOperator(LinearOperator):
    def __init__(self, *linear_operators, **kwargs):
        try:
            linear_operators = tuple(to_linear_operator(lt) for lt in linear_operators)
        except TypeError:
            raise TypeError("All arguments of a SumLinearOperator should be LinearOperators or Tensors")
        batch_shape = _mul_broadcast_shape(*[lt.batch_shape for lt in linear_operators])
        linear_operators = tuple(
            lt._expand_batch(batch_shape) if lt.batch_shape != batch_shape else lt for lt in linear_operators
        )
        super(SumLinearOperator, self).__init__(*linear_operators, **kwargs)

        self.linear_operators = linear_operators

    def _expand_batch(self, batch_shape):
        expanded_tensors = [linear_operator._expand_batch(batch_shape) for linear_operator in self.linear_operators]
        return self.__class__(*expanded_tensors)

    def _get_indices(self, row_index, col_index, *batch_indices):
        results = [linear_operator._get_indices(row_index, col_index, *batch_indices) for linear_operator in self.linear_operators]
        return sum(results)

    def _getitem(self, row_index, col_index, *batch_indices):
        results = [linear_operator._getitem(row_index, col_index, *batch_indices) for linear_operator in self.linear_operators]
        return SumLinearOperator(*results)

    def _matmul(self, rhs):
        return sum(linear_operator._matmul(rhs) for linear_operator in self.linear_operators)

    def _quad_form_derivative(self, left_vecs, right_vecs):
        return tuple(
            var for linear_operator in self.linear_operators for var in linear_operator._quad_form_derivative(left_vecs, right_vecs)
        )

    def _size(self):
        return _mul_broadcast_shape(*[lt.shape for lt in self.linear_operators])

    def _sum_batch(self, dim):
        return self.__class__(*(linear_operator._sum_batch(dim) for linear_operator in self.linear_operators))

    def _t_matmul(self, rhs):
        return sum(linear_operator._t_matmul(rhs) for linear_operator in self.linear_operators)

    def _transpose_nonbatch(self):
        linear_operators_t = [linear_operator.transpose(-1, -2) for linear_operator in self.linear_operators]
        return self.__class__(*linear_operators_t)

    @cached
    def evaluate(self):
        return sum(linear_operator.evaluate() for linear_operator in self.linear_operators)

    def __add__(self, other):
        from .diag_linear_operator import DiagLinearOperator
        from .added_diag_linear_operator import AddedDiagLinearOperator

        if isinstance(other, ZeroLinearOperator):
            return self
        elif isinstance(other, DiagLinearOperator):
            return AddedDiagLinearOperator(self, other)
        elif isinstance(other, SumLinearOperator):
            return SumLinearOperator(*(list(self.linear_operators) + list(other.linear_operators)))
        elif isinstance(other, LinearOperator):
            return SumLinearOperator(*(list(self.linear_operators) + [other]))
        elif isinstance(other, Tensor):
            # get broadcast shape, assuming mul broadcasting the same as add broadcasting
            broadcasted_shape = _mul_broadcast_shape(self.shape, other.shape)

            # to_linear_operator + broadcast other
            broadcasted_other = to_linear_operator(other.expand(broadcasted_shape))

            # update the linear operators' shape as well
            new_self = self if broadcasted_shape == self.shape else self._expand_batch(broadcasted_shape[:-2])

            return SumLinearOperator(*(list(new_self.linear_operators) + [broadcasted_other]))
        else:
            raise AttributeError("other must be a LinearOperator")

    def diag(self):
        return sum(linear_operator.diag().contiguous() for linear_operator in self.linear_operators)
