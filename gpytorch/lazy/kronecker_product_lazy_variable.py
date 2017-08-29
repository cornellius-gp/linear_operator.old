import torch
import math
import gpytorch
from .lazy_variable import LazyVariable
from .toeplitz_lazy_variable import ToeplitzLazyVariable
from torch.autograd import Variable
from ..utils import sparse_eye
from ..utils.kronecker_product import sym_kronecker_product_toeplitz_mul, kp_interpolated_toeplitz_mul, \
    kp_sym_toeplitz_derivative_quadratic_form, list_of_indices_and_values_to_sparse, kronecker_product
from gpytorch.utils import toeplitz
from gpytorch.functions.lazy_kronecker_product import KPInterpolatedToeplitzGPMarginalLogLikelihood


class KroneckerProductLazyVariable(LazyVariable):
    def __init__(self, columns, J_lefts=None, C_lefts=None, J_rights=None, C_rights=None, added_diag=None):
        if not isinstance(columns, Variable):
            raise RuntimeError('KroneckerProductLazyVariable is intended to wrap Variable versions of \
                                the first column and row.')

        self.columns = columns
        self.J_lefts = J_lefts
        self.C_lefts = C_lefts
        self.J_rights = J_rights
        self.C_rights = C_rights
        self.added_diag = added_diag
        self.kronecker_product_size = int(math.pow(self.columns.size()[1], self.columns.size()[0]))
        if J_lefts is not None:
            self.size = (self.J_lefts.size()[1], self.J_rights.size()[1])
        else:
            self.size = (self.kronecker_product_size, self.kronecker_product_size)

    def _mm_closure_factory(self, *args):
        if len(args) == 1:
            columns, = args
            return lambda mat2: sym_kronecker_product_toeplitz_mul(columns, mat2)
        elif len(args) == 4:
            columns, W_lefts, W_rights, added_diag = args
            return lambda mat2: kp_interpolated_toeplitz_mul(columns, W_lefts, W_rights, added_diag)
        else:
            raise AttributeError('Invalid number of arguments')

    def _derivative_quadratic_form_factory(*args):
        columns, = args
        return lambda left_vector, right_vector: (kp_sym_toeplitz_derivative_quadratic_form(left_vector, right_vector),)

    def add_diag(self, diag):
        if self.J_lefts is not None:
            kronecker_product_diag = diag.expand(self.size[0])
        else:
            kronecker_product_diag = diag.expand_as(self.kronecker_product_size)

        return KroneckerProductLazyVariable(self.columns, self.J_lefts, self.C_lefts,
                                            self.J_rights, self.C_rights, kronecker_product_diag)

    def add_jitter(self):
        jitter = torch.zeros(self.columns.size())
        jitter[:, 0] = 1e-4
        return KroneckerProductLazyVariable(self.columns.add(Variable(jitter)), self.J_lefts, self.C_lefts,
                                            self.J_rights, self.C_rights, self.added_diag)

    # def diag(self):
    #     """
    #     Gets the diagonal of the Toeplitz matrix wrapped by this object.

    #     By definition of a Toeplitz matrix, every element along the diagonal is equal
    #     to c[0] == r[0]. Therefore, we return a vector of length len(self.c) with
    #     each element equal to c[0].

    #     If the interpolation matrices exist, then the diagonal of WTW^{T} is simply
    #     W(T_diag)W^{T}.
    #     """
    #     if self.J_lefts is not None:
    #         if len(self.J_lefts[0]) != len(self.J_rights[0]):
    #             raise RuntimeError('diag not supported for non-square interpolated Toeplitz matrices.')
    #         WTW_diag = self.c[0].expand(len(self.J_left))
    #     else:
    #         WTW_diag = self.c[0].expand_as(self.c)

    #     if self.added_diag is not None:
    #         if len(self.added_diag) > len(WTW_diag):
    #             raise RuntimeError('Additional diagonal component length does not \
    #                                 match the rest of this implicit tensor.')
    #         WTW_diag = WTW_diag + self.added_diag

    #     return WTW_diag

    def explicit_interpolate_K(self, Js, Cs):
        """
        Multiplies the Kronecker Product matrix K this object represents (by columns and rows)
        by an interpolation matrix W, to get WK, without explicitly forming the Kronecker Product
        matrix K. This is a much more space-efficient approach.

        Args:
            - Js (matrix d x n x k) - d Index matrices for interpolation matrix W
            - Cs (matrix d x n x k) - d Coefficients matrices for interpolation matrix W
        Returns:
            - Matrix (n x m) - The result of the multiplication WK
        """
        m = self.kronecker_product_size
        d, n, num_coefficients = Js.size()
        m0 = self.columns.size()[1]

        result_matrix = Variable(torch.zeros(n, m))

        result_dim = Variable(torch.zeros(d, n, m0))
        for i in range(d):
            result_dim[i] = ToeplitzLazyVariable(self.columns[i]).explicit_interpolate_T(Js[i], Cs[i])

        for i in range(n):
            result_matrix[i] = kronecker_product(result_dim[:, i, :].unsqueeze(1))    

        return result_matrix

    def evaluate(self):
        """
        Explicitly evaluate and return the Kronecer Product matrix this object wraps as a float Tensor.
        To do this, we explicitly compute W_{left}TW_{right}^{T} and return it.

        Warning: as implicitly stored by this LazyVariable, W is very sparse and T requires O(m)
        storage, where as the full matrix requires O(m^2) storage. Calling evaluate can very easily
        lead to memory issues. As a result, using it should be a last resort.
        """

        if self.J_lefts is not None:
            n_left, n_right = self.size
            W_left = list_of_indices_and_values_to_sparse(self.J_lefts, self.C_lefts, self.columns)
            W_right = list_of_indices_and_values_to_sparse(self.J_rights, self.C_rights, self.columns)
            if n_left <= n_right:
                W_left_T = self.explicit_interpolate_K(self.J_lefts, self.C_lefts)
                WTW = gpytorch.dsmm(Variable(W_right), W_left_T.t()).t()
            else:
                W_right_T = self.explicit_interpolate_T(self.J_rights, self.C_rights)
                WTW = gpytorch.dsmm(Variable(W_left), W_right_T.t())
        else:
            WTW = KroneckerProductLazyVariable(self.columns).mm(Variable(torch.eye(self.kronecker_product_size)))

        if self.added_diag is not None:
            WTW = WTW + torch.diag(self.added_diag)

        return WTW

    def exact_gp_marginal_log_likelihood(self, target):
        W_left = Variable(list_of_indices_and_values_to_sparse(self.J_lefts,
                                                               self.C_lefts,
                                                               self.columns))
        W_right = Variable(list_of_indices_and_values_to_sparse(self.J_rights,
                                                                self.C_rights,
                                                                self.columns))
        noise_diag = self.added_diag
        return KPInterpolatedToeplitzGPMarginalLogLikelihood(W_left, W_right)(self.columns, target, noise_diag)

    def monte_carlo_log_likelihood(self, log_probability_func, train_y, variational_mean, chol_var_covar, num_samples):
        epsilon = Variable(torch.randn(self.kronecker_product_size, num_samples))
        samples = chol_var_covar.mm(epsilon)
        samples = samples + variational_mean.unsqueeze(1).expand_as(samples)
        W_left = Variable(list_of_indices_and_values_to_sparse(self.J_lefts, self.C_lefts, self.columns))
        samples = gpytorch.dsmm(W_left, samples)
        log_likelihood = log_probability_func(samples, train_y)
        return log_likelihood

    def mul(self, constant):
        """
        Multiplies this interpolated Toeplitz matrix elementwise by a constant. To accomplish this,
        we multiply the first Toeplitz component of this KroneckerProductLazyVariable by the constant.

        Args:
            - constant (broadcastable with self.columns[0]) - Constant to multiply by.
        Returns:
            - KroneckerProductLazyVariable with columns[0] = columns[0]*(constant) and columns[i] = columns[i] for i>0
        """
        columns = self.columns.clone()
        columns[0] = columns[0].mul(constant)
        return KroneckerProductLazyVariable(columns, self.J_lefts, self.C_lefts,
                                            self.J_rights, self.C_rights, self.added_diag)

    def mul_(self, constant):
        """
        In-place version of mul.
        """
        self.columns[0].mul_(constant)
        return self

    def representation(self):
        if self.J_lefts is None and self.C_lefts is None and self.J_rights is None \
                and self.C_rights is None and self.added_diag is None:
            return self.columns,

        if self.J_lefts is None and self.C_lefts is None and self.J_rights is None \
                and self.C_rights is None and self.added_diag is None:
            return self.columns,

        if self.J_lefts is not None and self.C_lefts is not None:
            W_left = Variable(list_of_indices_and_values_to_sparse(self.J_lefts,
                                                                   self.C_lefts,
                                                                   self.columns))
        else:
            W_left = Variable(sparse_eye(self.kronecker_product_size))
        if self.J_rights is not None and self.C_rights is not None:
            W_right = Variable(list_of_indices_and_values_to_sparse(self.J_rights,
                                                                    self.C_rights,
                                                                    self.columns))
        else:
            W_right = Variable(sparse_eye(self.kronecker_product_size))
        if self.added_diag is not None:
            added_diag = self.added_diag
        else:
            added_diag = Variable(torch.zeros(1))
        return self.columns, W_left, W_right, added_diag

    def trace_log_det_quad_form(self, mu_diffs, chol_covar_1, num_samples):
        if self.J_left is None and self.J_right is None:
            return super(KroneckerProductLazyVariable, self).trace_log_det_quad_form(mu_diffs, chol_covar_1, num_samples)
        else:
            return ToeplitzLazyVariable(self.c).trace_log_det_quad_form(mu_diffs, chol_covar_1, num_samples)

    def exact_posterior_alpha(self, train_mean, train_y):
        train_residual = (train_y - train_mean).unsqueeze(1)
        alpha = self.invmm(train_residual)
        W_train_right = Variable(list_of_indices_and_values_to_sparse(self.J_rights,
                                                                      self.C_rights,
                                                                      self.columns))
        alpha = gpytorch.dsmm(W_train_right.t(), alpha)
        alpha = KroneckerProductLazyVariable(self.columns).mm(alpha)
        return alpha.squeeze()

    def exact_posterior_mean(self, test_mean, alpha):
        alpha = alpha.unsqueeze(1)
        W_test_left = list_of_indices_and_values_to_sparse(self.J_lefts,
                                                           self.C_lefts,
                                                           self.columns)
        return test_mean.add(gpytorch.dsmm(W_test_left, alpha).squeeze())

    def variational_posterior_mean(self, alpha):
        """
        Assumes self is the covariance matrix between test and inducing points

        Returns the mean of the posterior GP on test points, given
        prior means/covars

        Args:
            - alpha (Variable m) - alpha vector, computed from exact_posterior_alpha
        """
        W_left = list_of_indices_and_values_to_sparse(self.J_lefts, self.C_lefts, self.columns)
        return gpytorch.dsmm(W_left, alpha.unsqueeze(1)).squeeze()

    def variational_posterior_covar(self, chol_variational_covar):
        """
        Assumes self is the covariance matrix between test and inducing points

        Returns the covar of the posterior GP on test points, given
        prior covars

        Args:
            - chol_variational_covar (Variable nxn) - Cholesky decomposition of variational covar
        """
        W_left = list_of_indices_and_values_to_sparse(self.J_lefts, self.C_lefts, self.columns)
        W_right = W_left.t()

        covar_right = gpytorch.dsmm(W_right.t(), chol_variational_covar.t()).t()
        covar_left = gpytorch.dsmm(W_left, chol_variational_covar.t())
        return covar_left.mm(covar_right)

    def __getitem__(self, i):
        if isinstance(i, tuple):
            if self.J_lefts is not None:
                # J[i[0], :], C[i[0], :]
                J_lefts_new = self.J_lefts[:, i[0], :]
                C_lefts_new = self.C_lefts[:, i[0], :]

                # J[i[1], :] C[i[1], :]
                J_rights_new = self.J_rights[:, i[1], :]
                C_rights_new = self.C_rights[:, i[1], :]

                if self.added_diag is not None:
                    if len(J_lefts_new[0]) != len(J_rights_new[0]):
                        raise RuntimeError('Slicing in to interpolated Kronecker Product matrix that has an additional \
                                            diagonal component to make it non-square is probably not intended.\
                                            It is ambiguous which diagonal elements to choose')

                    diag_new = self.added_diag[i[0]]
                else:
                    diag_new = None

                return KroneckerProductLazyVariable(self.columns, J_lefts_new, C_lefts_new,
                                                    J_rights_new, C_rights_new, diag_new)
            else:
                if i[0] != i[1]:
                    raise RuntimeError('Slicing an uninterpolated Kronecker Product matrix to be non-square is probably \
                                        unintended. If that was the intent, use evaluate() and slice the full matrix.')
                columns_new = self.columns[:, i[1], :]
                if self.added_diag is not None:
                    diag_new = self.added_diag[i[0]]
                else:
                    diag_new = None

                return KroneckerProductLazyVariable(columns_new, diag=diag_new)
        else:
            if self.J_lefts is not None:
                J_lefts_new = self.J_lefts[:, i, :]
                C_lefts_new = self.C_lefts[:, i, :]
                if self.added_diag is not None:
                    diag_new = self.added_diag[i]
                else:
                    diag_new = None

                return KroneckerProductLazyVariable(self.columns, J_lefts_new, C_lefts_new,
                                                    self.J_rights, self.C_rights, diag_new)
            else:
                raise RuntimeError('Slicing an uninterpolated Kronecker Product matrix to be non-square is probably \
                                    unintended. If that was the intent, use evaluate() and slice the full matrix.')
