#!/usr/bin/env python3

import operator
from functools import reduce
from typing import Optional, Tuple

import torch
from torch import Tensor

from .. import settings
from ..utils.broadcasting import _matmul_broadcast_shape, _mul_broadcast_shape
from ..utils.memoize import cached
from .diag_linear_operator import ConstantDiagLinearOperator, DiagLinearOperator
from .linear_operator import LinearOperator
from .non_linear_operator import to_linear_operator
from .triangular_linear_operator import TriangularLinearOperator


def _kron_diag(*lts) -> Tensor:
    """Compute diagonal of a KroneckerProductLinearOperator from the diagonals of the constituiting tensors"""
    lead_diag = lts[0].diag()
    if len(lts) == 1:  # base case:
        return lead_diag
    trail_diag = _kron_diag(*lts[1:])
    diag = lead_diag.unsqueeze(-2) * trail_diag.unsqueeze(-1)
    return diag.transpose(-1, -2).reshape(*diag.shape[:-2], -1)


def _prod(iterable):
    return reduce(operator.mul, iterable, 1)


def _matmul(linear_operators, kp_shape, rhs):
    output_shape = _matmul_broadcast_shape(kp_shape, rhs.shape)
    output_batch_shape = output_shape[:-2]

    res = rhs.contiguous().expand(*output_batch_shape, *rhs.shape[-2:])
    num_cols = rhs.size(-1)
    for linear_operator in linear_operators:
        res = res.view(*output_batch_shape, linear_operator.size(-1), -1)
        factor = linear_operator._matmul(res)
        factor = factor.view(*output_batch_shape, linear_operator.size(-2), -1, num_cols).transpose(-3, -2)
        res = factor.reshape(*output_batch_shape, -1, num_cols)
    return res


def _t_matmul(linear_operators, kp_shape, rhs):
    kp_t_shape = (*kp_shape[:-2], kp_shape[-1], kp_shape[-2])
    output_shape = _matmul_broadcast_shape(kp_t_shape, rhs.shape)
    output_batch_shape = torch.Size(output_shape[:-2])

    res = rhs.contiguous().expand(*output_batch_shape, *rhs.shape[-2:])
    num_cols = rhs.size(-1)
    for linear_operator in linear_operators:
        res = res.view(*output_batch_shape, linear_operator.size(-2), -1)
        factor = linear_operator._t_matmul(res)
        factor = factor.view(*output_batch_shape, linear_operator.size(-1), -1, num_cols).transpose(-3, -2)
        res = factor.reshape(*output_batch_shape, -1, num_cols)
    return res


class KroneckerProductLinearOperator(LinearOperator):
    r"""
    Returns the Kronecker product of the given linear operators

    Args:
        :`linear_operators`: List of linear operators
    """

    def __init__(self, *linear_operators):
        try:
            linear_operators = tuple(to_linear_operator(linear_operator) for linear_operator in linear_operators)
        except TypeError:
            raise RuntimeError("KroneckerProductLinearOperator is intended to wrap linear operators.")
        for prev_linear_operator, curr_linear_operator in zip(linear_operators[:-1], linear_operators[1:]):
            if prev_linear_operator.batch_shape != curr_linear_operator.batch_shape:
                raise RuntimeError(
                    "KroneckerProductLinearOperator expects linear operators with the "
                    "same batch shapes. Got {}.".format([lv.batch_shape for lv in linear_operators])
                )
        super().__init__(*linear_operators)
        self.linear_operators = linear_operators

    def __add__(self, other):
        if isinstance(other, DiagLinearOperator):
            return self.add_diag(other.diag())
        else:
            return super().__add__(other)

    def add_diag(self, diag):
        r"""
        Adds a diagonal to a KroneckerProductLinearOperator
        """

        from .kronecker_product_added_diag_linear_operator import KroneckerProductAddedDiagLinearOperator

        if not self.is_square:
            raise RuntimeError("add_diag only defined for square matrices")

        diag_shape = diag.shape
        if len(diag_shape) == 0 or diag_shape[-1] == 1:
            # interpret scalar tensor or single-trailing element as constant diag
            diag_tensor = ConstantDiagLinearOperator(diag, diag_shape=self.shape[-1])
        else:
            try:
                expanded_diag = diag.expand(self.shape[:-1])
            except RuntimeError:
                raise RuntimeError(
                    "add_diag for LinearOperator of size {} received invalid diagonal of size {}.".format(
                        self.shape, diag_shape
                    )
                )
            diag_tensor = DiagLinearOperator(expanded_diag)

        return KroneckerProductAddedDiagLinearOperator(self, diag_tensor)

    def diag(self):
        r"""
        As :func:`torch.diag`, returns the diagonal of the matrix :math:`K` this LinearOperator represents as a vector.

        :rtype: torch.tensor
        :return: The diagonal of :math:`K`. If :math:`K` is :math:`n \times n`, this will be a length
            n vector. If this LinearOperator represents a batch (e.g., is :math:`b \times n \times n`), this will be a
            :math:`b \times n` matrix of diagonals, one for each matrix in the batch.
        """
        if settings.debug.on():
            if not self.is_square:
                raise RuntimeError("Diag works on square matrices (or batches)")
        return _kron_diag(*self.linear_operators)

    @cached
    def inverse(self):
        # here we use that (A \kron B)^-1 = A^-1 \kron B^-1
        # TODO: Investigate under what conditions computing individual individual inverses makes sense
        inverses = [lt.inverse() for lt in self.linear_operators]
        return self.__class__(*inverses)

    def inv_matmul(self, right_tensor, left_tensor=None):
        # TODO: Investigate under what conditions computing individual individual inverses makes sense
        # For now, retain existing behavior
        return super().inv_matmul(right_tensor=right_tensor, left_tensor=left_tensor)

    @cached(name="cholesky")
    def _cholesky(self, upper=False):
        chol_factors = [lt.cholesky(upper=upper) for lt in self.linear_operators]
        return KroneckerProductTriangularLinearOperator(*chol_factors, upper=upper)

    def _expand_batch(self, batch_shape):
        return self.__class__(
            *[linear_operator._expand_batch(batch_shape) for linear_operator in self.linear_operators]
        )

    def _get_indices(self, row_index, col_index, *batch_indices):
        row_factor = self.size(-2)
        col_factor = self.size(-1)

        res = None
        for linear_operator in self.linear_operators:
            sub_row_size = linear_operator.size(-2)
            sub_col_size = linear_operator.size(-1)

            row_factor //= sub_row_size
            col_factor //= sub_col_size
            sub_res = linear_operator._get_indices(
                (row_index // row_factor).fmod(sub_row_size),
                (col_index // col_factor).fmod(sub_col_size),
                *batch_indices,
            )
            res = sub_res if res is None else (sub_res * res)

        return res

    def _inv_matmul(self, right_tensor, left_tensor=None):
        # Computes inv_matmul by exploiting the identity (A \kron B)^-1 = A^-1 \kron B^-1
        tsr_shapes = [q.size(-1) for q in self.linear_operators]
        n_rows = right_tensor.size(-2)
        batch_shape = _mul_broadcast_shape(self.shape[:-2], right_tensor.shape[:-2])
        perm_batch = tuple(range(len(batch_shape)))
        y = right_tensor.clone().expand(*batch_shape, *right_tensor.shape[-2:])
        for n, q in zip(tsr_shapes, self.linear_operators):
            # for KroneckerProductTriangularLinearOperator this inv_matmul is very cheap
            y = q.inv_matmul(y.reshape(*batch_shape, n, -1))
            y = y.reshape(*batch_shape, n, n_rows // n, -1).permute(*perm_batch, -2, -3, -1)
        res = y.reshape(*batch_shape, n_rows, -1)
        if left_tensor is not None:
            res = left_tensor @ res
        return res

    def _matmul(self, rhs):
        is_vec = rhs.ndimension() == 1
        if is_vec:
            rhs = rhs.unsqueeze(-1)

        res = _matmul(self.linear_operators, self.shape, rhs.contiguous())

        if is_vec:
            res = res.squeeze(-1)
        return res

    @cached(name="size")
    def _size(self):
        left_size = _prod(linear_operator.size(-2) for linear_operator in self.linear_operators)
        right_size = _prod(linear_operator.size(-1) for linear_operator in self.linear_operators)
        return torch.Size((*self.linear_operators[0].batch_shape, left_size, right_size))

    @cached(name="svd")
    def _svd(self) -> Tuple[LinearOperator, Tensor, LinearOperator]:
        U, S, V = [], [], []
        for lt in self.linear_operators:
            U_, S_, V_ = lt.svd()
            U.append(U_)
            S.append(S_)
            V.append(V_)
        S = KroneckerProductLinearOperator(*[DiagLinearOperator(S_) for S_ in S]).diag()
        U = KroneckerProductLinearOperator(*U)
        V = KroneckerProductLinearOperator(*V)
        return U, S, V

    def _symeig(self, eigenvectors: bool = False) -> Tuple[Tensor, Optional[LinearOperator]]:
        evals, evecs = [], []
        for lt in self.linear_operators:
            evals_, evecs_ = lt.symeig(eigenvectors=eigenvectors)
            evals.append(evals_)
            evecs.append(evecs_)
        evals = KroneckerProductLinearOperator(*[DiagLinearOperator(evals_) for evals_ in evals]).diag()
        if eigenvectors:
            evecs = KroneckerProductLinearOperator(*evecs)
        else:
            evecs = None
        return evals, evecs

    def _t_matmul(self, rhs):
        is_vec = rhs.ndimension() == 1
        if is_vec:
            rhs = rhs.unsqueeze(-1)

        res = _t_matmul(self.linear_operators, self.shape, rhs.contiguous())

        if is_vec:
            res = res.squeeze(-1)
        return res

    def _transpose_nonbatch(self):
        return self.__class__(
            *(linear_operator._transpose_nonbatch() for linear_operator in self.linear_operators), **self._kwargs
        )


class KroneckerProductTriangularLinearOperator(KroneckerProductLinearOperator):
    def __init__(self, *linear_operators, upper=False):
        if not all(isinstance(lt, TriangularLinearOperator) for lt in linear_operators):
            raise RuntimeError(
                "Components of KroneckerProductTriangularLinearOperator must be TriangularLinearOperator."
            )
        super().__init__(*linear_operators)
        self.upper = upper

    @cached
    def inverse(self):
        # here we use that (A \kron B)^-1 = A^-1 \kron B^-1
        inverses = [lt.inverse() for lt in self.linear_operators]
        return self.__class__(*inverses, upper=self.upper)

    def inv_matmul(self, right_tensor, left_tensor=None):
        # For triangular components, using triangular-triangular substition should generally be good
        return self._inv_matmul(right_tensor=right_tensor, left_tensor=left_tensor)

    @cached(name="cholesky")
    def _cholesky(self, upper=False):
        raise NotImplementedError("_cholesky not applicable to triangular linear operators")

    def _cholesky_solve(self, rhs, upper=False):
        if upper:
            # res = (U.T @ U)^-1 @ v = U^-1 @ U^-T @ v
            w = self._transpose_nonbatch().inv_matmul(rhs)
            res = self.inv_matmul(w)
        else:
            # res = (L @ L.T)^-1 @ v = L^-T @ L^-1 @ v
            w = self.inv_matmul(rhs)
            res = self._transpose_nonbatch().inv_matmul(w)
        return res

    def _symeig(self, eigenvectors: bool = False) -> Tuple[Tensor, Optional[LinearOperator]]:
        raise NotImplementedError("_symeig not applicable to triangular linear operators")
