#!/usr/bin/env python3

import torch

from .added_diag_lazy_tensor import AddedDiagLazyTensor
from .diag_lazy_tensor import DiagLazyTensor
from .lazy_tensor import LazyTensor
from .root_lazy_tensor import RootLazyTensor


class KroneckerProductAddedDiagLazyTensor(AddedDiagLazyTensor):
    def __init__(self, *lazy_tensors, preconditioner_override=None):
        # TODO: implement the woodbury formula for diagonal tensors that are non constants.

        super(KroneckerProductAddedDiagLazyTensor, self).__init__(
            *lazy_tensors, preconditioner_override=preconditioner_override
        )
        if len(lazy_tensors) > 2:
            raise RuntimeError("An AddedDiagLazyTensor can only have two components")
        elif isinstance(lazy_tensors[0], DiagLazyTensor):
            self.diag_tensor = lazy_tensors[0]
            self.lazy_tensor = lazy_tensors[1]
        elif isinstance(lazy_tensors[1], DiagLazyTensor):
            self.diag_tensor = lazy_tensors[1]
            self.lazy_tensor = lazy_tensors[0]
        else:
            raise RuntimeError("One of the LazyTensors input to AddedDiagLazyTensor must be a DiagLazyTensor!")

    def inv_quad_logdet(self, inv_quad_rhs=None, logdet=False, reduce_inv_quad=True):

        if logdet is not None:
            logdet_term = self._logdet()
        else:
            logdet_term = None

        if inv_quad_rhs is not None:
            inv_quad_term, _ = super().inv_quad_logdet(
                inv_quad_rhs=inv_quad_rhs, logdet=False, reduce_inv_quad=reduce_inv_quad
            )
        else:
            inv_quad_term = None

        return inv_quad_term, logdet_term

    def _logdet(self):
        eig_matrix, _ = self.lazy_tensor._symeig()
        evals_plus_diag = eig_matrix.diag() + self.diag_tensor.diag()
        return torch.log(evals_plus_diag).sum(dim=-1)

    def _preconditioner(self):
        # solves don't use CG so don't waste time computing it
        return None, None, None

    def _solve(self, rhs, preconditioner=None, num_tridiag=0):
        eig_matrix, q_matrix = self.lazy_tensor._symeig()
        inv_mat_sqrt = DiagLazyTensor(1.0 / (eig_matrix.diag() + self.diag_tensor.diag()) ** 0.5)

        res = q_matrix.transpose(-2, -1).matmul(rhs)
        res2 = inv_mat_sqrt.matmul(res)

        lazy_lhs = q_matrix.matmul(inv_mat_sqrt)
        return lazy_lhs.matmul(res2)

    def root_decomposition(self):
        # TODO: use _root_decomposition instead
        eig_matrix, q_matrix = self.lazy_tensor._symeig()
        eigs_sqrt = DiagLazyTensor((eig_matrix.diag() + self.diag_tensor.diag()) ** 0.5)

        matrix_root = q_matrix.matmul(eigs_sqrt)

        return RootLazyTensor(matrix_root)

    def _quad_form_derivative(self, left_vecs, right_vecs):
        return LazyTensor._quad_form_derivative(self, left_vecs, right_vecs)
