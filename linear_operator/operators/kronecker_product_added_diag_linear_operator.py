#!/usr/bin/env python3

from __future__ import annotations

import torch

from .added_diag_linear_operator import AddedDiagLinearOperator
from .diag_linear_operator import ConstantDiagLinearOperator, DiagLinearOperator
from .matmul_linear_operator import MatmulLinearOperator


class KroneckerProductAddedDiagLinearOperator(AddedDiagLinearOperator):
    def __init__(self, *linear_operators, preconditioner_override=None):
        super().__init__(*linear_operators, preconditioner_override=preconditioner_override)
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
        logdet_term = self._logdet() if logdet else None
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
        if isinstance(self.diag_tensor, ConstantDiagLinearOperator):
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

        # TODO: implement woodbury formula for non-constant Kronecker-structured diagonal operators

        return super()._solve(rhs, preconditioner=None, num_tridiag=0)

    def _root_decomposition(self):
        if isinstance(self.diag_tensor, ConstantDiagLinearOperator):
            # we can be use eigendecomposition and shift the eigenvalues
            evals, q_matrix = self.linear_operator.symeig(eigenvectors=True)
            updated_evals = DiagLinearOperator((evals + self.diag_tensor.diag()).pow(0.5))
            return MatmulLinearOperator(q_matrix, updated_evals)
        return super()._root_decomposition()

    def _root_inv_decomposition(self, initial_vectors=None):
        if isinstance(self.diag_tensor, ConstantDiagLinearOperator):
            evals, q_matrix = self.linear_operator.symeig(eigenvectors=True)
            inv_sqrt_evals = DiagLinearOperator((evals + self.diag_tensor.diag()).pow(-0.5))
            return MatmulLinearOperator(q_matrix, inv_sqrt_evals)
        return super()._root_inv_decomposition(initial_vectors=initial_vectors)
