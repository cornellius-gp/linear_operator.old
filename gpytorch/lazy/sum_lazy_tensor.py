from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from .lazy_tensor import LazyTensor
from .non_lazy_tensor import NonLazyTensor
from .zero_lazy_tensor import ZeroLazyTensor


class SumLazyTensor(LazyTensor):
    def __init__(self, *lazy_tensors):
        lazy_tensors = list(lazy_tensors)
        for i, lazy_tensor in enumerate(lazy_tensors):
            if not isinstance(lazy_tensor, LazyTensor):
                if torch.is_tensor(lazy_tensor):
                    lazy_tensors[i] = NonLazyTensor(lazy_tensor)
                else:
                    raise RuntimeError("All arguments of a SumLazyTensor should be LazyTensors or Tensors")
        super(SumLazyTensor, self).__init__(*lazy_tensors)

        self.lazy_tensors = lazy_tensors

    def _matmul(self, rhs):
        return sum(lazy_tensor._matmul(rhs) for lazy_tensor in self.lazy_tensors)

    def _t_matmul(self, rhs):
        return sum(lazy_tensor._t_matmul(rhs) for lazy_tensor in self.lazy_tensors)

    def _quad_form_derivative(self, left_vecs, right_vecs):
        return tuple(
            var for lazy_tensor in self.lazy_tensors for var in lazy_tensor._quad_form_derivative(left_vecs, right_vecs)
        )

    def _size(self):
        return self.lazy_tensors[0].size()

    def _transpose_nonbatch(self):
        lazy_tensors_t = [lazy_tensor.transpose(-1, -2) for lazy_tensor in self.lazy_tensors]
        return SumLazyTensor(*lazy_tensors_t)

    def _batch_get_indices(self, batch_indices, left_indices, right_indices):
        return sum(
            lazy_tensor._batch_get_indices(batch_indices, left_indices, right_indices)
            for lazy_tensor in self.lazy_tensors
        )

    def _get_indices(self, left_indices, right_indices):
        return sum(lazy_tensor._get_indices(left_indices, right_indices) for lazy_tensor in self.lazy_tensors)

    def add_jitter(self):
        lazy_tensors = list(self.lazy_tensors[:-1])
        lazy_tensors.append(self.lazy_tensors[-1].add_jitter())
        return SumLazyTensor(*lazy_tensors)

    def _exact_predictive_covar_inv_quad_form_cache(self, train_train_covar_inv_root, test_train_covar):
        return tuple(
            lazy_tensor._exact_predictive_covar_inv_quad_form_cache(
                train_train_covar_inv_root, test_train_covar_comp
            ).detach()
            for lazy_tensor, test_train_covar_comp in zip(self.lazy_tensors, test_train_covar.lazy_tensors)
        )

    def _exact_predictive_covar_inv_quad_form_root(self, precomputed_cache, test_train_covar):
        # Here the precomputed cache is a list
        # where each component in the list is the precomputed cache for each component lazy tensor
        return sum(
            lazy_tensor._exact_predictive_covar_inv_quad_form_root(cache_comp, test_train_covar_comp)
            for lazy_tensor, cache_comp, test_train_covar_comp in zip(
                self.lazy_tensors, precomputed_cache, test_train_covar.lazy_tensors
            )
        )

    def evaluate(self):
        return sum(lazy_tensor.evaluate() for lazy_tensor in self.lazy_tensors)

    def __add__(self, other):
        if isinstance(other, ZeroLazyTensor):
            return self
        if isinstance(other, SumLazyTensor):
            return SumLazyTensor(*(list(self.lazy_tensors) + list(other.lazy_tensors)))
        elif isinstance(other, LazyTensor):
            return SumLazyTensor(*(list(self.lazy_tensors) + [other]))
        else:
            raise AttributeError("other must be a LazyTensor")

    def diag(self):
        diags = [lazy_tensor.diag().contiguous() for lazy_tensor in self.lazy_tensors]
        size = diags[0].size()
        res = sum(diag.view(-1) for diag in diags)
        res = res.view(size)
        return res

    def repeat(self, *sizes):
        return self.__class__(*(lazy_tensor.repeat(*sizes) for lazy_tensor in self.lazy_tensors))

    def sum_batch(self, sum_batch_size=None):
        return self.__class__(*(lazy_tensor.sum_batch(sum_batch_size) for lazy_tensor in self.lazy_tensors))

    def __getitem__(self, index):
        results = tuple(lazy_tensor.__getitem__(index) for lazy_tensor in self.lazy_tensors)
        if isinstance(results[0], LazyTensor):
            return SumLazyTensor(*results)
        else:
            return sum(results)
