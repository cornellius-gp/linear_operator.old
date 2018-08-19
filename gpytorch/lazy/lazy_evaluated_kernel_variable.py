from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .lazy_variable import LazyVariable
from .non_lazy_variable import NonLazyVariable
from .lazy_variable_representation_tree import LazyVariableRepresentationTree


LAZY_KERNEL_TENSOR_WARNING = (
    "A LazyEvaluatedKernelVariable is not intended to be used directly " "as a tensor! Call evaluate() first."
)


class LazyEvaluatedKernelVariable(LazyVariable):
    def __init__(self, kernel, x1, x2, squeeze_row=False, squeeze_col=False, **params):
        super(LazyEvaluatedKernelVariable, self).__init__(kernel, x1, x2, **params)
        self.kernel = kernel
        self.x1 = x1
        self.x2 = x2
        self.squeeze_row = squeeze_row
        self.squeeze_col = squeeze_col
        self.is_batch = (self.x1.ndimension() == 3 or (self.x1.ndimension() == 2 and self.squeeze_row))
        self.params = params

    def _matmul(self, rhs):
        raise RuntimeError(LAZY_KERNEL_TENSOR_WARNING)

    def _t_matmul(self, rhs):
        raise RuntimeError(LAZY_KERNEL_TENSOR_WARNING)

    def _quad_form_derivative(self, left_vecs, right_vecs):
        raise RuntimeError(LAZY_KERNEL_TENSOR_WARNING)

    def _transpose_nonbatch(self):
        return self.__class__(self.kernel, self.x2, self.x1, **self.params)

    def _get_indices(self, left_indices, right_indices):
        from ..kernels import Kernel

        x1 = self.x1[left_indices, :].unsqueeze(0)
        x2 = self.x2[right_indices, :].unsqueeze(0)
        res = super(Kernel, self.kernel).__call__(x1.transpose(0, 1), x2.transpose(0, 1))
        if isinstance(res, LazyVariable):
            res = res.evaluate()
        res = res.view(-1)
        return res

    def diag(self):
        """
        Getting the diagonal of a kernel can be handled more efficiently by
        transposing the batch and data dimension before calling the kernel.
        Implementing it this way allows us to compute predictions more efficiently
        in cases where only the variances are required.
        """
        if hasattr(self, "_cached_kernel_diag"):
            return self._cached_kernel_diag
        elif hasattr(self, "_cached_kernel_eval"):
            return self._cached_kernel_eval.diag()
        else:
            if not self.is_batch:
                x1 = self.x1.unsqueeze(0)
                x2 = self.x2.unsqueeze(0)
            else:
                x1 = self.x1
                x2 = self.x2

            # If x1 or x2 only has one data point, make sure to unsqueeze the data-size dimension
            if x1.dim() == 2:  # We only have a single data point
                x1 = x1.unsqueeze(1)
            if x2.dim() == 2:  # We only have a single data point
                x2 = x2.unsqueeze(1)

            res = self.kernel.forward_diag(x1, x2, **self.params)
            if isinstance(res, LazyVariable):
                res = res.evaluate()
            self._cached_kernel_diag = res.transpose(-3, -2).squeeze()
            return self._cached_kernel_diag

    def evaluate_kernel(self):
        """
        NB: This is a meta LazyVariable, in the sense that evaluate can return
        a LazyVariable if the kernel being evaluated does so.
        """
        from ..kernels import Kernel

        if hasattr(self, "_cached_kernel_eval"):
            return self._cached_kernel_eval
        else:
            if not self.is_batch:
                x1 = self.x1.unsqueeze(0)
                x2 = self.x2.unsqueeze(0)
            else:
                x1 = self.x1
                x2 = self.x2

            self._cached_kernel_eval = super(Kernel, self.kernel).__call__(x1, x2, **self.params)
            if self.squeeze_row:
                self._cached_kernel_eval.squeeze_(-2)
            if self.squeeze_col:
                self._cached_kernel_eval.squeeze_(-1)

            if not self.is_batch and self._cached_kernel_eval.ndimension() == 3:
                self._cached_kernel_eval = self._cached_kernel_eval[0]
            if not isinstance(self._cached_kernel_eval, LazyVariable):
                self._cached_kernel_eval = NonLazyVariable(self._cached_kernel_eval)
            return self._cached_kernel_eval

    def representation(self):
        return self.evaluate_kernel().representation()

    def representation_tree(self):
        return LazyVariableRepresentationTree(self.evaluate_kernel())

    def evaluate(self):
        return self.evaluate_kernel().evaluate()

    def exact_predictive_mean(self, full_mean, train_labels, n_train, likelihood, precomputed_cache=None):
        if self.kernel.has_custom_exact_predictions:
            return self.evaluate_kernel().exact_predictive_mean(
                full_mean, train_labels, n_train, likelihood, precomputed_cache
            )
        else:
            return super(LazyEvaluatedKernelVariable, self).exact_predictive_mean(
                full_mean, train_labels, n_train, likelihood, precomputed_cache
            )

    def exact_predictive_covar(self, n_train, likelihood, precomputed_cache=None):
        if self.kernel.has_custom_exact_predictions:
            return self.evaluate_kernel().exact_predictive_covar(n_train, likelihood, precomputed_cache)
        else:
            return super(LazyEvaluatedKernelVariable, self).exact_predictive_covar(
                n_train, likelihood, precomputed_cache
            )

    def repeat(self, *sizes):
        if self.squeeze_row or self.squeeze_col:
            raise RuntimeError('Can\'t repeat a row/col of a LazyEvaluatedKernelVariable')
        elif len(sizes) == 3:
            x1 = self.x1.repeat(sizes[0], sizes[1], 1)
            x2 = self.x2.repeat(sizes[0], sizes[1], 1)
        elif len(sizes) == 2 and x1.ndim() == 2:
            x1 = self.x1.repeat(sizes[0], 1)
            x2 = self.x2.repeat(sizes[0], 1)
        else:
            raise RuntimeError('Invalid number of sizes (expected 2 or 3)')

        return LazyEvaluatedKernelVariable(
            self.kernel, x1, x2, **self.params
        )

    def __getitem__(self, index):
        index = list(index) if isinstance(index, tuple) else [index]
        ndimension = self.ndimension()
        index += [slice(None, None, None)] * (ndimension - len(index))
        if self.is_batch:
            batch_index = index[0]
            left_index = index[1]
            right_index = index[2]
            squeeze_row = self.squeeze_row
            squeeze_col = self.squeeze_col

            x1 = self.x1[batch_index, left_index, :]
            if x1.dim() == 2 and not isinstance(batch_index, int):
                x1 = x1.unsqueeze(1)
                squeeze_row = True
            x2 = self.x2[batch_index, right_index, :]
            if x2.dim() == 2 and not isinstance(batch_index, int):
                x2 = x2.unsqueeze(1)
                squeeze_col = True

            return LazyEvaluatedKernelVariable(
                self.kernel, x1, x2, squeeze_row=squeeze_row, squeeze_col=squeeze_col, **self.params
            )
        else:
            left_index = index[0]
            right_index = index[1]
            squeeze_row = self.squeeze_row
            squeeze_col = self.squeeze_col

            x1 = self.x1[left_index, :]
            if x1.dim() == 1:
                x1 = x1.unsqueeze(1)
                squeeze_row = True
            x2 = self.x2[right_index, :]
            if x2.dim() == 1:
                x2 = x2.unsqueeze(1)
                squeeze_col = True

            return LazyEvaluatedKernelVariable(
                self.kernel, x1, x2, squeeze_row=squeeze_row, squeeze_col=squeeze_col, **self.params
            )

    def _size(self):
        return self.kernel.size(self.x1, self.x2)
