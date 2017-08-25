from ..utils import function_factory


class LazyVariable(object):
    def _mm_closure_factory(self, *args):
        """
        Generates a closure that performs a *tensor* matrix multiply
        The closure will take in a *tensor* matrix (not variable) and return the
        result of a matrix multiply with the lazy variable.

        The arguments into the closure factory are the *tensors* corresponding to
        the Variables in self.representation()

        Returns:
        function(tensor (nxn)) - closure that performs a matrix multiply
        """
        raise NotImplementedError

    def _derivative_quadratic_form_factory(self, *args):
        """
        Generates a closure that computes the derivatives of uKv^t w.r.t. `args` given u, v

        K is a square matrix corresponding to the Variables in self.representation()

        Returns:
        function(vector u, vector v) - closure that computes the derivatives of uKv^t w.r.t.
        `args` given u, v
        """
        raise NotImplementedError

    def add_diag(self, diag):
        """
        Adds an element to the diagonal of the matrix.

        Args:
            - diag (Scalar Variable)
        """
        raise NotImplementedError

    def add_jitter(self):
        """
        Adds jitter (i.e., a small diagonal component) to the matrix this LazyVariable represents.
        This could potentially be implemented as a no-op, however this could lead to numerical instabilities,
        so this should only be done at the user's risk.
        """

    def evaluate(self):
        """
        Explicitly evaluates the matrix this LazyVariable represents. This
        function should return a Variable explicitly wrapping a Tensor storing
        an exact representation of this LazyVariable.
        """
        raise NotImplementedError

    def gp_marginal_log_likelihood(self, target):
        """
        Computes the marginal log likelihood of a Gaussian process whose covariance matrix
        plus the diagonal noise term (added using add_diag above) is stored as this lazy variable

        Args:
            - target (vector n) - training label vector to be used in the marginal log likelihood calculation.
        Returns:
            - scalar - The GP marginal log likelihood where (K+\sigma^{2}I) is represented by this LazyVariable.
        """
        raise NotImplementedError

    def invmm(self, rhs_mat):
        """
        Computes a linear solve (w.r.t self) with several right hand sides.

        Args:
            - rhs_mat (matrix nxk) - Matrix of k right hand side vectors.

        Returns:
            - matrix nxk - (self)^{-1} rhs_mat
        """
        if not hasattr(self, '_invmm_class'):
            grad_fn = self._grad_fn if hasattr(self, '_grad_fn') else None
            self._invmm_class = function_factory.invmm_factory(self._mm_closure_factory, grad_fn)
        args = list(self.representation()) + [rhs_mat]
        return self._invmm_class()(*args)

    def mm(self, rhs_mat):
        """
        Multiplies self by a matrix

        Args:
            - rhs_mat (matrix nxk) - Matrix to multiply with

        Returns:
            - matrix nxk
        """
        if not hasattr(self, '_mm_class'):
            grad_fn = self._grad_fn if hasattr(self, '_grad_fn') else None
            self._mm_class = function_factory.mm_factory(self._mm_closure_factory, grad_fn)
        args = list(self.representation()) + [rhs_mat]
        return self._mm_class()(*args)

    def monte_carlo_log_likelihood(self, log_probability_func, train_y, variational_mean, chol_var_covar, num_samples):
        """
        Performs Monte Carlo integration of the provided log_probability function. Typically, this should work by
        drawing samples of u from the variational posterior, transforming these in to samples of f using the information
        stored in this LazyVariable, and then calling the log_probability_func with these samples and train_y.

        Args:
            - log_probability_func (function) - Log probability function to integrate.
            - train_y (vector n) - Training label vector.
            - variational_mean (vector m) - Mean vector of the variational posterior.
            - chol_var_covar (matrix m x m) - Cholesky decomposition of the variational posterior covariance matrix.
            - num_samples (scalar) - Number of samples to use for Monte Carlo integration.
        Returns:
            - The average of calling log_probability_func on num_samples samples of f, where f is sampled from the
              current posterior.
        """
        raise NotImplementedError

    def mul(self, constant):
        """
        Multiplies this interpolated Toeplitz matrix elementwise by a constant. To accomplish this,
        we multiply the Toeplitz component by the constant. This way, the interpolation acts on the
        multiplied values in T, and the entire kernel is ultimately multiplied by this constant.

        Args:
            - constant (broadcastable with self.c) - Constant to multiply by.
        Returns:
            - ToeplitzLazyVariable with c = c*(constant)
        """
        raise NotImplementedError

    def mul_(self, constant):
        """
        In-place version of mul.
        """
        raise NotImplementedError

    def mvn_kl_divergence(self, mean_1, chol_covar_1, mean_2):
        """
        Computes the KL divergence between two multivariate Normal distributions. The first of these
        distributions is specified by mean_1 and chol_covar_1, while the second distribution is specified
        by mean_2 and this LazyVariable.

        Args:
            - mean_1 (vector n) - Mean vector of the first Gaussian distribution.
            - chol_covar_1 (matrix n x n) - Cholesky factorization of the covariance matrix of the first Gaussian
                                            distribution.
            - mean_2 (vector n) - Mean vector of the second Gaussian distribution.
        Returns:
            - KL divergence between N(mean_1, chol_covar_1) and N(mean_2, self)
        """
        raise NotImplementedError

    def representation(self, *args):
        """
        Returns the variables that are used to define the LazyVariable
        """
        raise NotImplementedError

    def trace_log_det_quad_form(self, mu_diffs, chol_covar_1, num_samples=10):
        if not hasattr(self, '_trace_log_det_quad_form_class'):
            tlqf_function_factory = function_factory.trace_logdet_quad_form_factory
            self._trace_log_det_quad_form_class = tlqf_function_factory(self._mm_closure_factory,
                                                                        self._derivative_quadratic_form_factory)
        covar2_args = self.representation()
        return self._trace_log_det_quad_form_class(num_samples)(mu_diffs, chol_covar_1, *covar2_args)

    def exact_posterior_alpha(self, train_mean, train_y):
        """
        Assumes that self represents the train-train prior covariance matrix.

        Returns alpha - a vector to memoize for calculating the
        mean of the posterior GP on test points

        Args:
            - train_mean (Variable n) - prior mean values for the test points.
            - train_y (Variable n) - alpha vector, computed from exact_posterior_alpha
        """
        raise NotImplementedError

    def exact_posterior_mean(self, test_mean, alpha):
        """
        Assumes that self represents the test-train prior covariance matrix.

        Returns the mean of the posterior GP on test points, given
        prior means/covars

        Args:
            - test_mean (Variable m) - prior mean values for the test points.
            - alpha (Variable m) - alpha vector, computed from exact_posterior_alpha
        """
        raise NotImplementedError

    def variational_posterior_mean(self, alpha):
        """
        Assumes self is the covariance matrix between test and inducing points

        Returns the mean of the posterior GP on test points, given
        prior means/covars

        Args:
            - alpha (Variable m) - alpha vector, computed from exact_posterior_alpha
        """
        raise NotImplementedError

    def variational_posterior_covar(self):
        """
        Assumes self is the covariance matrix between test and inducing points

        Returns the covar of the posterior GP on test points, given
        prior covars

        Args:
            - chol_variational_covar (Variable nxn) - Cholesky decomposition of variational covar
        """
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError
