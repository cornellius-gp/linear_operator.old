#!/usr/bin/env python3

from __future__ import annotations

import torch

from .added_diag_linear_operator import AddedDiagLinearOperator
from .diag_linear_operator import DiagLinearOperator


class KroneckerProductAddedDiagLinearOperator(AddedDiagLinearOperator):
    def __init__(self, *linear_operators, preconditioner_override=None):
        # TODO: implement the woodbury formula for diagonal tensors that are non constants.

        super(KroneckerProductAddedDiagLinearOperator, self).__init__(
            *linear_operators, preconditioner_override=preconditioner_override
        )
        if len(linear_operators) > 2:
            raise RuntimeError("An AddedDiagLinearOperator can only have two components")
        elif isinstance(linear_operators[0], DiagLinearOperator):
            self.diag_tensor = linear_operators[0]
            self.linear_operator = linear_operators[1]
        elif isinstance(linear_operators[1], DiagLinearOperator):
            self.diag_tensor = linear_operators[1]
            self.linear_operator = linear_operators[0]
        else:
            raise RuntimeError(
                "One of the LinearOperators input to AddedDiagLinearOperator must be a DiagLinearOperator!"
            )

    def inv_quad_logdet(self, inv_quad_rhs=None, logdet=False, reduce_inv_quad=True):
        # we want to call the standard InvQuadLogDet to easily get the probe vectors and do the
        # solve but we only want to cache the probe vectors for the backwards
        inv_quad_term, _ = super().inv_quad_logdet(
            inv_quad_rhs=inv_quad_rhs, logdet=False, reduce_inv_quad=reduce_inv_quad
        )

        if logdet is not False:
            logdet_term = self._logdet()
        else:
            logdet_term = None

        return inv_quad_term, logdet_term

    def _logdet(self):
        # symeig requires computing the eigenvectors so that it's differentiable
        evals, _ = self.linear_operator.symeig(eigenvectors=True)
        evals_plus_diag = evals + self.diag_tensor.diag()
        return torch.log(evals_plus_diag).sum(dim=-1)

    def _preconditioner(self):
        # solves don't use CG so don't waste time computing it
        return None, None, None

    def _solve(self, rhs, preconditioner=None, num_tridiag=0):
        # we do the solve in double for numerical stability issues
        # TODO: Use fp64 registry once #1213 is addressed

        rhs_dtype = rhs.dtype
        rhs = rhs.double()

        evals, q_matrix = self.linear_operator.symeig(eigenvectors=True)
        evals, q_matrix = evals.double(), q_matrix.double()

        evals_plus_diagonal = evals + self.diag_tensor.diag()
        evals_root = evals_plus_diagonal.pow(0.5)
        inv_mat_sqrt = DiagLinearOperator(evals_root.reciprocal())

        res = q_matrix.transpose(-2, -1).matmul(rhs)
        res2 = inv_mat_sqrt.matmul(res)

        lhs = q_matrix.matmul(inv_mat_sqrt)
        return lhs.matmul(res2).type(rhs_dtype)

    def _root_decomposition(self):
        evals, q_matrix = self.linear_operator.symeig(eigenvectors=True)
        updated_evals = DiagLinearOperator((evals + self.diag_tensor.diag()).pow(0.5))
        matrix_root = q_matrix.matmul(updated_evals)
        return matrix_root

    def _root_inv_decomposition(self, initial_vectors=None):
        evals, q_matrix = self.linear_operator.symeig(eigenvectors=True)
        inv_sqrt_evals = DiagLinearOperator((evals + self.diag_tensor.diag()).pow(-0.5))
        matrix_inv_root = q_matrix.matmul(inv_sqrt_evals)
        return matrix_inv_root
