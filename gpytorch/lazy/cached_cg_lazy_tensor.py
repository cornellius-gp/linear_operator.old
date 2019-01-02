#!/usr/bin/env python3

import torch
import warnings
from .lazy_tensor import LazyTensor
from .. import settings


class CachedCGLazyTensor(LazyTensor):
    """
    A LazyTensor wrapper that eagerly computes many CG calls in batch.
    This maximizes CG parallelism for fast inference.
    Used primarily for variational inference with GPs.

    Args:
        :attr:`base_lazy_tensor` (:class:`gpytorch.lazy.LazyTensor`): the LazyTensor to wrap
    """

    @classmethod
    def precompute_terms(cls, base_lazy_tensor, eager_rhs):
        """
        Computes the solves, probe vectors, probe_vector norms, probe vector solves, and probe vector
        tridiagonal matrices to construct a CachedCGLazyTensor
        """
        with torch.no_grad():
            # Generate probe vectors
            num_random_probes = settings.num_trace_samples.value()
            probe_vectors = torch.empty(
                base_lazy_tensor.matrix_shape[-1], num_random_probes, dtype=base_lazy_tensor.dtype,
                device=base_lazy_tensor.device
            )
            probe_vectors.bernoulli_().mul_(2).add_(-1)
            probe_vectors = probe_vectors.expand(
                *base_lazy_tensor.batch_shape, base_lazy_tensor.matrix_shape[-1], num_random_probes
            )
            probe_vector_norms = torch.norm(probe_vectors, 2, dim=-2, keepdim=True)
            probe_vectors = probe_vectors.div(probe_vector_norms)

            # Compute solves
            all_solves, probe_vector_tmats, = base_lazy_tensor._solve(
                torch.cat([probe_vectors, eager_rhs], -1),
                preconditioner=base_lazy_tensor._preconditioner()[0],
                num_tridiag=probe_vectors.size(-1)
            )
            probe_vector_solves = all_solves[..., :probe_vectors.size(-1)].detach()
            solves = all_solves[..., probe_vectors.size(-1):]

            return solves.detach(), probe_vectors.detach(), probe_vector_norms.detach(), \
                probe_vector_solves.detach(), probe_vector_tmats.detach()

    def __init__(
        self, base_lazy_tensor, eager_rhss=[], solves=[], probe_vectors=torch.tensor([]),
        probe_vector_norms=torch.tensor([]), probe_vector_solves=torch.tensor([]), probe_vector_tmats=torch.tensor([])
    ):
        super(CachedCGLazyTensor, self).__init__(
            base_lazy_tensor, eager_rhss=eager_rhss, solves=solves, probe_vectors=probe_vectors,
            probe_vector_norms=probe_vector_norms, probe_vector_solves=probe_vector_solves,
            probe_vector_tmats=probe_vector_tmats,
        )
        self.base_lazy_tensor = base_lazy_tensor
        self.eager_rhss = [eager_rhs.requires_grad_(False) for eager_rhs in eager_rhss]
        self.solves = [solve.requires_grad_(False) for solve in solves]
        self.probe_vectors = probe_vectors.requires_grad_(False)
        self.probe_vector_norms = probe_vector_norms.requires_grad_(False)
        self.probe_vector_solves = probe_vector_solves.requires_grad_(False)
        self.probe_vector_tmats = probe_vector_tmats.requires_grad_(False)

    @property
    def requires_grad(self):
        return self.base_lazy_tensor.requires_grad

    @requires_grad.setter
    def requires_grad(self, val):
        self.base_lazy_tensor.requires_grad = val

    def _get_indices(self, left_indices, right_indices, *batch_indices):
        return self.base_lazy_tensor._get_indices(left_indices, right_indices, *batch_indices)

    def _getitem(self, *indices):
        return self.base_lazy_tensor._getitem(*indices)

    def _matmul(self, tensor):
        return self.base_lazy_tensor._matmul(tensor)

    def _probe_vectors_and_norms(self):
        return self.probe_vectors, self.probe_vector_norms

    def _quad_form_derivative(self, left_vecs, right_vecs):
        return self.base_lazy_tensor._quad_form_derivative(left_vecs, right_vecs)

    def _solve(self, rhs, preconditioner, num_tridiag=None):
        if num_tridiag:
            probe_vectors = rhs[..., :num_tridiag].detach()
            if torch.equal(probe_vectors, self.probe_vectors):
                probe_vector_solves = self.probe_vector_solves
                tmats = self.probe_vector_tmats
            else:
                if settings.debug.on():
                    warnings.warn(
                        "CachedCGLazyTensor did not recognize the supplied probe vectors for tridiagonalization."
                    )
                return super(CachedCGLazyTensor, self)._solve(rhs, preconditioner, num_tridiag=num_tridiag)

        # Here we check to see what solves we've already performed
        truncated_rhs = rhs[..., (num_tridiag or 0):]
        for eager_rhs, solve in zip(self.eager_rhss, self.solves):
            if torch.equal(truncated_rhs, eager_rhs):
                if num_tridiag:
                    return torch.cat([probe_vector_solves, solve], -1), tmats
                else:
                    return solve

        if settings.debug.on():
            warnings.warn(
                "CachedCGLazyTensor had to run CG on a tensor of size {}. For best performance, this "
                "LazyTensor should pre-register all vectors to run CG against.".format(rhs.shape)
            )
        return super(CachedCGLazyTensor, self)._solve(rhs, preconditioner, num_tridiag=num_tridiag)

    def _size(self):
        return self.base_lazy_tensor._size()

    def _t_matmul(self, tensor):
        return self.base_lazy_tensor._t_matmul(tensor)

    def _transpose_nonbatch(self):
        return self.base_lazy_tensor._transpose_nonbatch()

    def detach_(self):
        self.base_lazy_tensor.detach_()
        return self
