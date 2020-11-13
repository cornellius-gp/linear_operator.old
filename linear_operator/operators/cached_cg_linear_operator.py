#!/usr/bin/env python3

import warnings

import torch

from .. import settings
from ..utils.warnings import ExtraComputationWarning
from .chol_linear_operator import CholLinearOperator
from .linear_operator import LinearOperator
from .triangular_linear_operator import TriangularLinearOperator


class CachedCGLinearOperator(LinearOperator):
    """
    A LinearOperator wrapper that eagerly computes many CG calls in batch.
    This maximizes CG parallelism for fast inference.
    Used primarily for variational inference with GPs.

    Args:
        :attr:`base_linear_operator` (:class:`LinearOperator`):
            the LinearOperator to wrap
        :attr:`eager_rhss` (list of :class:`LinearOperator`):
            list of right-hand sides with eagerly-computed solves
        :attr:`solves` (list of :class:`LinearOperator`):
            list of solves associated with :attr:`eager_rhss`
        :attr:`probe_vectors` (:class:`LinearOperator`, optional):
            normalized probe vectors (for computing logdet with SLQ)
        :attr:`probe_vector_norms` (:class:`LinearOperator`, optional):
            norms associated with :attr:`probe_vectors` that will return :attr:`probe_vectors`
            to having identity covariance (for computing logdet with SLQ)
        :attr:`probe_vector_solves` (:class:`LinearOperator`, optional):
            solves associated with :attr:`probe_vectors` (for computing logdet with SLQ)
        :attr:`probe_vector_tmats` (:class:`LinearOperator`, optional):
            Lanczos tridiagonal matrices associated with :attr:`probe_vectors`
            (for computing logdet with SLQ)
    """

    @classmethod
    def precompute_terms(cls, base_linear_operator, eager_rhs, logdet_terms=True, include_tmats=True):
        """
        Computes the solves, probe vectors, probe_vector norms, probe vector solves, and probe vector
        tridiagonal matrices to construct a CachedCGLinearOperator

        Set logdet_terms to False if you are not going to compute the logdet of the LinearOperator
        """
        with torch.no_grad():
            if logdet_terms:
                # Generate probe vectors
                num_random_probes = settings.num_trace_samples.value()
                probe_vectors = torch.empty(
                    base_linear_operator.matrix_shape[-1],
                    num_random_probes,
                    dtype=base_linear_operator.dtype,
                    device=base_linear_operator.device,
                )
                probe_vectors.bernoulli_().mul_(2).add_(-1)
                probe_vectors = probe_vectors.expand(
                    *base_linear_operator.batch_shape, base_linear_operator.matrix_shape[-1], num_random_probes
                )
                probe_vector_norms = torch.norm(probe_vectors, 2, dim=-2, keepdim=True)
                probe_vectors = probe_vectors.div(probe_vector_norms)

                # Compute solves
                if include_tmats:
                    all_solves, probe_vector_tmats, = base_linear_operator._solve(
                        torch.cat([probe_vectors, eager_rhs], -1),
                        preconditioner=base_linear_operator._preconditioner()[0],
                        num_tridiag=probe_vectors.size(-1),
                    )
                else:
                    all_solves = base_linear_operator._solve(
                        torch.cat([probe_vectors, eager_rhs], -1), preconditioner=base_linear_operator._preconditioner()[0]
                    )
                    probe_vector_tmats = torch.tensor([])
                probe_vector_solves = all_solves[..., : probe_vectors.size(-1)].detach()
                solves = all_solves[..., probe_vectors.size(-1) :]

                return (
                    solves.detach(),
                    probe_vectors.detach(),
                    probe_vector_norms.detach(),
                    probe_vector_solves.detach(),
                    probe_vector_tmats.detach(),
                )

            else:
                # Compute solves
                if settings.fast_computations.log_prob.on():
                    solves = base_linear_operator._solve(eager_rhs, preconditioner=base_linear_operator._preconditioner()[0])
                else:
                    solves = base_linear_operator.cholesky()._cholesky_solve(eager_rhs)
                dtype = solves.dtype
                device = solves.device
                return (
                    solves.detach(),
                    torch.tensor([], dtype=dtype, device=device),
                    torch.tensor([], dtype=dtype, device=device),
                    torch.tensor([], dtype=dtype, device=device),
                    torch.tensor([], dtype=dtype, device=device),
                )

    def __init__(
        self,
        base_linear_operator,
        eager_rhss=[],
        solves=[],
        probe_vectors=torch.tensor([]),
        probe_vector_norms=torch.tensor([]),
        probe_vector_solves=torch.tensor([]),
        probe_vector_tmats=torch.tensor([]),
    ):
        super().__init__(
            base_linear_operator,
            eager_rhss=eager_rhss,
            solves=solves,
            probe_vectors=probe_vectors,
            probe_vector_norms=probe_vector_norms,
            probe_vector_solves=probe_vector_solves,
            probe_vector_tmats=probe_vector_tmats,
        )
        self.base_linear_operator = base_linear_operator
        self.eager_rhss = [eager_rhs.detach() for eager_rhs in eager_rhss]
        self.solves = [solve.detach() for solve in solves]
        self.probe_vectors = probe_vectors.detach()
        self.probe_vector_norms = probe_vector_norms.detach()
        self.probe_vector_solves = probe_vector_solves.detach()
        self.probe_vector_tmats = probe_vector_tmats.detach()

    @property
    def requires_grad(self):
        return self.base_linear_operator.requires_grad

    @requires_grad.setter
    def requires_grad(self, val):
        self.base_linear_operator.requires_grad = val

    def _cholesky(self, upper=False):
        from .triangular_linear_operator import TriangularLinearOperator

        res = self.__class__(
            self.base_linear_operator.cholesky(upper=upper),
            eager_rhss=self.eager_rhss,
            solves=self.solves,
            probe_vectors=self.probe_vectors,
            probe_vector_norms=self.probe_vector_norms,
            probe_vector_solves=self.probe_vector_solves,
            probe_vector_tmats=self.probe_vector_tmats,
        )
        return TriangularLinearOperator(res, upper=upper)

    def _cholesky_solve(self, rhs, upper: bool = False):
        # Here we check to see what solves we've already performed
        for eager_rhs, solve in zip(self.eager_rhss, self.solves):
            if torch.equal(rhs, eager_rhs):
                return solve

        if settings.debug.on():
            warnings.warn(
                "CachedCGLinearOperator had to run CG on a tensor of size {}. For best performance, this "
                "LinearOperator should pre-register all vectors to run CG against.".format(rhs.shape),
                ExtraComputationWarning,
            )
        return torch.cholesky_solve(rhs, self.evaluate(), upper=upper)

    def _expand_batch(self, batch_shape):
        return self.base_linear_operator._expand_batch(batch_shape)

    def _get_indices(self, row_index, col_index, *batch_indices):
        return self.base_linear_operator._get_indices(row_index, col_index, *batch_indices)

    def _getitem(self, row_index, col_index, *batch_indices):
        return self.base_linear_operator._getitem(row_index, col_index, *batch_indices)

    def _matmul(self, tensor):
        return self.base_linear_operator._matmul(tensor)

    def _probe_vectors_and_norms(self):
        return self.probe_vectors, self.probe_vector_norms

    def _quad_form_derivative(self, left_vecs, right_vecs):
        return self.base_linear_operator._quad_form_derivative(left_vecs, right_vecs)

    def _solve(self, rhs, preconditioner, num_tridiag=0):
        if num_tridiag:
            probe_vectors = rhs[..., :num_tridiag].detach()
            if torch.equal(probe_vectors, self.probe_vectors):
                probe_vector_solves = self.probe_vector_solves
                tmats = self.probe_vector_tmats
            else:
                if settings.debug.on():
                    warnings.warn(
                        "CachedCGLinearOperator did not recognize the supplied probe vectors for tridiagonalization.",
                        ExtraComputationWarning,
                    )
                return super()._solve(rhs, preconditioner, num_tridiag=num_tridiag)

        # Here we check to see what solves we've already performed
        truncated_rhs = rhs[..., (num_tridiag or 0) :]
        for eager_rhs, solve in zip(self.eager_rhss, self.solves):
            if torch.equal(truncated_rhs, eager_rhs):
                if num_tridiag:
                    return torch.cat([probe_vector_solves, solve], -1), tmats
                else:
                    return solve

        if settings.debug.on():
            warnings.warn(
                "CachedCGLinearOperator had to run CG on a tensor of size {}. For best performance, this "
                "LinearOperator should pre-register all vectors to run CG against.".format(rhs.shape),
                ExtraComputationWarning,
            )
        return super()._solve(rhs, preconditioner, num_tridiag=num_tridiag)

    def _size(self):
        return self.base_linear_operator._size()

    def _t_matmul(self, tensor):
        return self.base_linear_operator._t_matmul(tensor)

    def _transpose_nonbatch(self):
        return self.base_linear_operator._transpose_nonbatch()

    def detach_(self):
        self.base_linear_operator.detach_()
        return self

    def inv_matmul(self, right_tensor, left_tensor=None):
        if isinstance(self.base_linear_operator, TriangularLinearOperator):
            return self.base_linear_operator.inv_matmul(right_tensor, left_tensor=left_tensor)

        if not isinstance(self.base_linear_operator, CholLinearOperator):
            return super().inv_matmul(right_tensor, left_tensor=left_tensor)

        with settings.fast_computations(solves=False):
            return super().inv_matmul(right_tensor, left_tensor=left_tensor)

    def inv_quad_logdet(self, inv_quad_rhs=None, logdet=False, reduce_inv_quad=True):
        if not isinstance(self.base_linear_operator, CholLinearOperator):
            return super().inv_quad_logdet(inv_quad_rhs=inv_quad_rhs, logdet=logdet, reduce_inv_quad=reduce_inv_quad)

        if not self.is_square:
            raise RuntimeError(
                "inv_quad_logdet only operates on (batches of) square (positive semi-definite) LinearOperators. "
                "Got a {} of size {}.".format(self.__class__.__name__, self.size())
            )

        if inv_quad_rhs is not None:
            if self.dim() == 2 and inv_quad_rhs.dim() == 1:
                if self.shape[-1] != inv_quad_rhs.numel():
                    raise RuntimeError(
                        "LinearOperator (size={}) cannot be multiplied with right-hand-side Tensor (size={}).".format(
                            self.shape, inv_quad_rhs.shape
                        )
                    )
            elif self.dim() != inv_quad_rhs.dim():
                raise RuntimeError(
                    "LinearOperator (size={}) and right-hand-side Tensor (size={}) should have the same number "
                    "of dimensions.".format(self.shape, inv_quad_rhs.shape)
                )
            elif self.shape[-1] != inv_quad_rhs.shape[-2]:
                raise RuntimeError(
                    "LinearOperator (size={}) cannot be multiplied with right-hand-side Tensor (size={}).".format(
                        self.shape, inv_quad_rhs.shape
                    )
                )

        inv_quad_term = None
        logdet_term = None

        if inv_quad_rhs is not None:
            inv_quad_term = self.inv_quad(inv_quad_rhs, reduce_inv_quad=reduce_inv_quad)

        if logdet:
            logdet_term = self.base_linear_operator._chol_diag.pow(2).log().sum(-1)

        return inv_quad_term, logdet_term


__all__ = ["ExtraComputationWarning", "CachedCGLinearOperator"]
