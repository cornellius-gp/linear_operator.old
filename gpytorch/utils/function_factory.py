from torch.autograd import Function
from .lincg import LinearCG
from .lanczos_quadrature import StochasticLQ
import torch
import math


def _default_matmul_closure_factor(mat):
    return mat


def _default_grad_fn(grad_output, rhs):
    return rhs.t().matmul(grad_output),


def _default_derivative_quadratic_form_factory(mat):
    return lambda left_vector, right_vector: (left_vector.ger(right_vector),)


def _default_exact_gp_mml_grad_closure_factory(*args):
    def closure(matmul_closure, tr_inv, mat_inv_labels, labels, num_samples):
        grad = torch.ger(labels.view(-1), mat_inv_labels.view(-1))
        grad.add_(-torch.eye(*grad.size()))
        grad = LinearCG().solve(matmul_closure, grad)
        return grad,
    return closure


def inv_matmul_factory(matmul_closure_factory=_default_matmul_closure_factor,
                       derivative_quadratic_form_factory=_default_derivative_quadratic_form_factory):
    class InvMatmul(Function):
        def __init__(self, *args):
            self.args = args

        def forward(self, *args):
            closure_args = self.args + args[:-1]
            rhs = args[-1]
            res = LinearCG().solve(matmul_closure_factory(*closure_args), rhs)
            self.save_for_backward(*(list(args) + [res]))
            return res

        def backward(self, grad_output):
            if derivative_quadratic_form_factory is None:
                raise NotImplementedError
            args = self.saved_tensors[:-2]
            closure_args = self.args + args
            res = self.saved_tensors[-1]

            arg_grads = [None] * len(args)
            rhs_grad = None

            # input_1 gradient
            if any(self.needs_input_grad[:-1]):
                lhs_matrix_grad = LinearCG().solve(matmul_closure_factory(*closure_args), grad_output)
                lhs_matrix_grad = lhs_matrix_grad.mul_(-1)
                if res.ndimension() == 1:
                    res = res.unsqueeze(1)
                if lhs_matrix_grad.ndimension() == 1:
                    lhs_matrix_grad = lhs_matrix_grad.unsqueeze(1)

                for i in range(len(args)):
                    if self.needs_input_grad[i]:
                        arg_grads[i] = torch.zeros(args[i].size())

                for i in range(lhs_matrix_grad.size()[1]):
                    quad_derivative = derivative_quadratic_form_factory(*args)(lhs_matrix_grad[:, i], res[:, i])
                    for j in range(len(args)):
                        if arg_grads[j] is not None:
                            arg_grads[j].add_(quad_derivative[j])

            # input_2 gradient
            if self.needs_input_grad[-1]:
                rhs_grad = LinearCG().solve(matmul_closure_factory(*closure_args), grad_output)

            return tuple(arg_grads + [rhs_grad])

    return InvMatmul


def matmul_factory(matmul_closure_factory=_default_matmul_closure_factor,
                   derivative_quadratic_form_factory=_default_derivative_quadratic_form_factory):
    class Matmul(Function):
        def __init__(self, *args):
            self.args = args

        def forward(self, *args):
            closure_args = self.args + args[:-1]
            rhs = args[-1]
            res = matmul_closure_factory(*closure_args)(rhs)
            self.save_for_backward(*args)
            return res

        def backward(self, grad_output):
            if derivative_quadratic_form_factory is None:
                raise NotImplementedError
            args = self.saved_tensors[:-1]
            rhs = self.saved_tensors[-1]
            closure_args = self.args + args

            arg_grads = [None] * len(args)
            rhs_grad = None

            # input_1 gradient
            if any(self.needs_input_grad[:-1]):
                if rhs.ndimension() == 1:
                    rhs = rhs.unsqueeze(1)
                if grad_output.ndimension() == 1:
                    grad_output_matrix = grad_output.unsqueeze(1)
                else:
                    grad_output_matrix = grad_output

                for i in range(len(args)):
                    if self.needs_input_grad[i]:
                        arg_grads[i] = torch.zeros(args[i].size())

                for i in range(grad_output_matrix.size()[1]):
                    quad_derivative = derivative_quadratic_form_factory(*args)(grad_output_matrix[:, i], rhs[:, i])
                    for j in range(len(args)):
                        if arg_grads[j] is not None:
                            arg_grads[j].add_(quad_derivative[j])

            # input_2 gradient
            if self.needs_input_grad[-1]:
                rhs_grad = matmul_closure_factory(*closure_args)(grad_output)

            return tuple(arg_grads + [rhs_grad])

    return Matmul


def trace_logdet_quad_form_factory(matmul_closure_factory=_default_matmul_closure_factor,
                                   derivative_quadratic_form_factory=_default_derivative_quadratic_form_factory):
    class TraceLogDetQuadForm(Function):
        def __init__(self, num_samples=10):
            self.num_samples = num_samples

        def forward(self, mu_diff, chol_covar1, *covar2_args):
            covar2_matmul_closure = matmul_closure_factory(*covar2_args)

            def quad_form_closure(z):
                rhs_vector = chol_covar1.t().mv(chol_covar1.mv(z))
                mat_inv_vector = LinearCG().solve(covar2_matmul_closure, rhs_vector)
                return z.dot(mat_inv_vector)

            # log |K2|
            log_det_covar2, = StochasticLQ(num_random_probes=10).evaluate(covar2_matmul_closure,
                                                                          len(mu_diff),
                                                                          [lambda x: x.log()])

            # Tr(K2^{-1}K1)
            sample_matrix = torch.sign(torch.randn(self.num_samples, len(mu_diff)))
            trace = 0
            for z in sample_matrix:
                trace = trace + quad_form_closure(z)
            trace = trace / self.num_samples

            # Inverse quad form
            mat_inv_y = LinearCG().solve(covar2_matmul_closure, mu_diff)
            inv_quad_form = mat_inv_y.dot(mu_diff)

            res = log_det_covar2 + trace + inv_quad_form

            self.save_for_backward(*([mu_diff] + [chol_covar1] + list(covar2_args)))
            self.covar2_matmul_closure = covar2_matmul_closure
            self.mat_inv_y = mat_inv_y

            return mu_diff.new().resize_(1).fill_(res)

        def backward(self, grad_output):
            if derivative_quadratic_form_factory is None:
                raise NotImplementedError
            grad_output_value = grad_output.squeeze()[0]

            args = self.saved_tensors

            mu_diff = args[0]
            chol_covar1 = args[1]
            covar2_args = args[2:]

            mat_inv_y = self.mat_inv_y
            covar2_matmul_closure = self.covar2_matmul_closure

            grad_mu_diff = None
            grad_cholesky_factor = None
            grad_covar2_args = [None] * len(covar2_args)

            if self.needs_input_grad[0]:
                # Need gradient with respect to mu_diff
                grad_mu_diff = mat_inv_y.mul(2 * grad_output_value)

            if self.needs_input_grad[1]:
                # Compute gradient with respect to the Cholesky factor L
                grad_cholesky_factor = 2 * LinearCG().solve(matmul_closure_factory(*covar2_args), chol_covar1)
                grad_cholesky_factor.mul_(grad_output_value)

            if any(self.needs_input_grad[2:]):
                # Compute gradient with respect to covar2
                quad_part = derivative_quadratic_form_factory(*covar2_args)(mat_inv_y, mat_inv_y)

                sample_matrix = torch.sign(torch.randn(self.num_samples, len(mu_diff)))

                for i in range(len(covar2_args)):
                    if self.needs_input_grad[i + 2]:
                        grad_covar2_args[i] = torch.zeros(covar2_args[i].size())

                def deriv_quad_form_closure(z):
                    rhs_vec = chol_covar1.t().mv(chol_covar1.mv(z))
                    I_minus_Tinv_M_z = z - LinearCG().solve(covar2_matmul_closure, rhs_vec)
                    Tinv_z = LinearCG().solve(covar2_matmul_closure, z)
                    return derivative_quadratic_form_factory(*covar2_args)(Tinv_z, I_minus_Tinv_M_z)

                for z in sample_matrix:
                    quad_derivative = deriv_quad_form_closure(z)
                    for i in range(len(covar2_args)):
                        if grad_covar2_args[i] is not None:
                            grad_covar2_args[i].add_(quad_derivative[i])

                for i in range(len(covar2_args)):
                    if grad_covar2_args[i] is not None:
                        grad_covar2_args[i].div_(self.num_samples)
                        grad_covar2_args[i].add_(-quad_part[i])
                        grad_covar2_args[i].mul_(grad_output_value)

            return tuple([grad_mu_diff] + [grad_cholesky_factor] + grad_covar2_args)

    return TraceLogDetQuadForm


def exact_gp_mll_factory(matmul_closure_factory=_default_matmul_closure_factor,
                         exact_gp_mml_grad_closure_factory=_default_exact_gp_mml_grad_closure_factory):
    class ExactGPMLL(Function):
        def __init__(self, num_samples=10):
            self.num_samples = num_samples

        def forward(self, *args):
            closure_args = args[:-1]
            labels = args[-1]

            matmul_closure = matmul_closure_factory(*closure_args)
            mat_inv_labels = LinearCG().solve(matmul_closure, labels)
            # Inverse quad form
            res = mat_inv_labels.dot(labels)
            # Log determinant
            logdet, tr_inv = StochasticLQ(num_random_probes=10).evaluate(matmul_closure, len(labels),
                                                                         [lambda x: x.log(), lambda x: x.pow(-1)])

            res += logdet
            res += math.log(2 * math.pi) * len(labels)
            res *= -0.5

            self.mat_inv_labels = mat_inv_labels
            self.tr_inv = tr_inv
            self.matmul_closure = matmul_closure
            self.save_for_backward(*args)
            return labels.new().resize_(1).fill_(res)

        def backward(self, grad_output):
            if exact_gp_mml_grad_closure_factory is None:
                raise NotImplementedError

            closure_args = self.saved_tensors[:-1]
            labels = self.saved_tensors[-1]
            mat_inv_labels = self.mat_inv_labels
            grad_output_value = grad_output.squeeze()[0]

            matmul_closure = self.matmul_closure
            tr_inv = self.tr_inv
            closure_arg_grads = [None] * len(closure_args)
            labels_grad = None

            # input_1 gradient
            if any(self.needs_input_grad[:-1]):
                grad_closure = exact_gp_mml_grad_closure_factory(*(list(closure_args)))
                closure_arg_grads = list(grad_closure(matmul_closure, tr_inv, mat_inv_labels, labels, self.num_samples))
                for i, closure_arg_grad in enumerate(closure_arg_grads):
                    if closure_arg_grad is not None:
                        closure_arg_grad.mul_(0.5 * grad_output_value)

            # input_2 gradient
            if self.needs_input_grad[-1]:
                # Need gradient with respect to labels
                labels_grad = mat_inv_labels.mul_(-grad_output_value)

            return tuple(closure_arg_grads + [labels_grad])

    return ExactGPMLL
