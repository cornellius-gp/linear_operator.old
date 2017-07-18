import torch
from functools import reduce
from torch.optim import Optimizer
from math import isinf

import pdb

class LBFGS(Optimizer):
    """Implements L-BFGS algorithm.

.. warning::
    This optimizer doesn't support per-parameter options and parameter
    groups (there can be only one).

.. warning::
    Right now all parameters have to be on a single device. This will be
    improved in the future.

.. note::
    This is a very memory intensive optimizer (it requires additional
    ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
    try reducing the history size, or use a different algorithm.

Arguments:
    lr (float): learning rate (default: 1)
    max_iter (int): maximal number of iterations per optimization step
        (default: 20)
    max_eval (int): maximal number of function evaluations per optimization
        step (default: max_iter * 1.25).
    tolerance_grad (float): termination tolerance on first order optimality
        (default: 1e-5).
    tolerance_change (float): termination tolerance on function value/parameter
        changes (default: 1e-9).
    line_search_fn (str): line search methods, currently available
        ['backtracking', 'goldstein', 'weak_wolfe']
    bounds (list of tuples of tensor): bounds[i][0], bounds[i][1] are elementwise
        lowerbound and upperbound of param[i], respectively
    history_size (int): update history size (default: 100).
"""

    def __init__(self, params, lr=1, max_iter=20, max_eval=None,
                 tolerance_grad=1e-5, tolerance_change=1e-9, history_size=100,
                 line_search_fn=None, bounds=None):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(lr=lr, max_iter=max_iter, max_eval=max_eval,
                        tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
                        history_size=history_size, line_search_fn=line_search_fn, bounds=bounds)
        super(LBFGS, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGS doesn't support per-parameter options " +
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._bounds = [(None, None)] * len(self._params) if bounds is None else bounds
        self._numel_cache = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        return torch.cat(
            tuple(param.grad.data.view(-1) for param in self._params), 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.data.add_(step_size, update[offset:offset + numel].resize_(p.size()))
            offset += numel
        assert offset == self._numel()

    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        group = self.param_groups[0]
        lr = group['lr']
        max_iter = group['max_iter']
        max_eval = group['max_eval']
        tolerance_grad = group['tolerance_grad']
        tolerance_change = group['tolerance_change']
        line_search_fn = group['line_search_fn']
        history_size = group['history_size']

        state = self.state['global_state']
        state.setdefault('func_evals', 0)
        state.setdefault('n_iter', 0)

        # evaluate initial f(x) and df/dx
        orig_loss = closure()
        loss = orig_loss.data[0]
        current_evals = 1
        state['func_evals'] += 1

        flat_grad = self._gather_flat_grad()
        abs_grad_sum = flat_grad.abs().sum()

        if abs_grad_sum <= tolerance_grad:
            return loss

        # variables cached in state (for tracing)
        d = state.get('d')
        t = state.get('t')
        old_dirs = state.get('old_dirs')
        old_stps = state.get('old_stps')
        H_diag = state.get('H_diag')
        prev_flat_grad = state.get('prev_flat_grad')
        prev_loss = state.get('prev_loss')

        n_iter = 0
        # optimize for a max of max_iter iterations
        while n_iter < max_iter:
            # keep track of nb of iterations
            n_iter += 1
            state['n_iter'] += 1

            ############################################################
            # compute gradient descent direction
            ############################################################
            if state['n_iter'] == 1:
                d = flat_grad.neg()
                old_dirs = []
                old_stps = []
                H_diag = 1
            else:
                # do lbfgs update (update memory)
                y = flat_grad.sub(prev_flat_grad)
                s = d.mul(t)
                ys = y.dot(s)  # y*s
                if ys > 1e-10:
                    # updating memory
                    if len(old_dirs) == history_size:
                        # shift history by one (limited-memory)
                        old_dirs.pop(0)
                        old_stps.pop(0)

                    # store new direction/step
                    old_dirs.append(s)
                    old_stps.append(y)

                    # update scale of initial Hessian approximation
                    H_diag = ys / y.dot(y)  # (y*y)

                # compute the approximate (L-BFGS) inverse Hessian
                # multiplied by the gradient
                num_old = len(old_dirs)

                if 'ro' not in state:
                    state['ro'] = [None] * history_size
                    state['al'] = [None] * history_size
                ro = state['ro']
                al = state['al']

                for i in range(num_old):
                    ro[i] = 1. / old_stps[i].dot(old_dirs[i])

                # iteration in L-BFGS loop collapsed to use just one buffer
                q = flat_grad.neg()
                for i in range(num_old - 1, -1, -1):
                    al[i] = old_dirs[i].dot(q) * ro[i]
                    q.add_(-al[i], old_stps[i])

                # multiply by initial Hessian
                # r/d is the final direction
                d = r = torch.mul(q, H_diag)
                for i in range(num_old):
                    be_i = old_stps[i].dot(r) * ro[i]
                    r.add_(al[i] - be_i, old_dirs[i])

            if prev_flat_grad is None:
                prev_flat_grad = flat_grad.clone()
            else:
                prev_flat_grad.copy_(flat_grad)
            prev_loss = loss

            ############################################################
            # compute step length
            ############################################################
            # directional derivative
            gtd = flat_grad.dot(d)  # g * d

            # check that progress can be made along that direction
            if gtd > -tolerance_change:
                break

            # reset initial guess for step size
            if state['n_iter'] == 1:
                t = min(1., 1. / abs_grad_sum) * lr
            else:
                t = lr

            # optional line search: user function
            ls_func_evals = 0
            if line_search_fn is not None:
                # perform line search, using user function
                # raise RuntimeError("line search function is not supported yet")
                if line_search_fn == 'weak_wolfe':
                    t = self._weak_wolfe(closure, d)
                elif line_search_fn == 'goldstein':
                    t = self._goldstein(closure, d)
                elif line_search_fn == 'backtracking':
                    t = self._backtracking(closure, d)
                self._add_grad(t, d)
            else:
                # no line search, simply move with fixed-step
                self._add_grad(t, d)
            if n_iter != max_iter:
                # re-evaluate function only if not in last iteration
                # the reason we do this: in a stochastic setting,
                # no use to re-evaluate that function here
                loss = closure().data[0]
                flat_grad = self._gather_flat_grad()
                abs_grad_sum = flat_grad.abs().sum()
                ls_func_evals = 1

            # update func eval
            current_evals += ls_func_evals
            state['func_evals'] += ls_func_evals

            ############################################################
            # check conditions
            ############################################################
            if n_iter == max_iter:
                break

            if current_evals >= max_eval:
                break

            if abs_grad_sum <= tolerance_grad:
                break

            if d.mul(t).abs_().sum() <= tolerance_change:
                break

            if abs(loss - prev_loss) < tolerance_change:
                break

        state['d'] = d
        state['t'] = t
        state['old_dirs'] = old_dirs
        state['old_stps'] = old_stps
        state['H_diag'] = H_diag
        state['prev_flat_grad'] = prev_flat_grad
        state['prev_loss'] = prev_loss

        return orig_loss

    def _copy_param(self):
        original_param_data_list = []
        for p in self._params:
            param_data = p.data.new(p.size())
            param_data.copy_(p.data)
            original_param_data_list.append(param_data)
        return original_param_data_list

    def _set_param(self, param_data_list):
        for i in range(len(param_data_list)):
            self._params[i].data.copy_(param_data_list[i])

    def _set_param_incremental(self, alpha, d):
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.data.copy_(p.data + alpha * d[offset:offset + numel].resize_(p.size()))
            offset += numel
        assert offset == self._numel()

    def _directional_derivative(self, d):
        deriv = 0.0
        offset = 0
        for p in self._params:
            numel = p.numel()
            deriv += torch.sum(p.grad.data * d[offset:offset + numel].resize_(p.size()))
            offset += numel
        assert offset == self._numel()
        return deriv

    def _max_alpha(self, d):
        offset = 0
        max_alpha = float('inf')
        min_l_bnd = float('inf')
        min_u_bnd = float('inf')
        for p, bnd in zip(self._params, self._bounds):
#            pdb.set_trace()
            numel = p.numel()
            l_bnd, u_bnd = bnd
            p_grad = d[offset:offset + numel].resize_(p.size())
            if l_bnd is not None:
                from_l_bnd = ((l_bnd-p.data)/p_grad)[p_grad<0]
                min_l_bnd = torch.min(from_l_bnd) if from_l_bnd.numel() > 0 else max_alpha
            if u_bnd is not None:
                from_u_bnd = ((u_bnd-p.data)/p_grad)[p_grad>0]
                min_u_bnd = torch.min(from_u_bnd) if from_u_bnd.numel() > 0 else max_alpha
            max_alpha = min(max_alpha, min_l_bnd, min_u_bnd)
            offset = offset + numel

        return max_alpha


>>>>>>> Parameters now require bounds for optimization
    def _backtracking(self, closure, d):
        # 0 < rho < 0.5 and 0 < w < 1
        rho = 1e-4
        w = 0.5

        original_param_data_list = self._copy_param()
        phi_0 = closure().data[0]
        phi_0_prime = self._directional_derivative(d)
        alpha_k = 1.0
        while True:
            self._set_param_incremental(alpha_k, d)
            phi_k = closure().data[0]
            self._set_param(original_param_data_list)
            if phi_k <= phi_0 + rho * alpha_k * phi_0_prime:
                break
            else:
                alpha_k *= w
        return alpha_k


    def _goldstein(self, closure, d):
        # 0 < rho < 0.5 and t > 1
        rho = 1e-4
        t = 2.0

        original_param_data_list = self._copy_param()
        phi_0 = closure().data[0]
        phi_0_prime = self._directional_derivative(d)
        a_k = 0.0
        b_k = self._max_alpha(d)
        alpha_k = min(1e4, (a_k + b_k) / 2.0)
        while True:
            self._set_param_incremental(alpha_k, d)
            phi_k = closure().data[0]
            self._set_param(original_param_data_list)
            if phi_k <= phi_0 + rho*alpha_k*phi_0_prime:
                if phi_k >= phi_0 + (1-rho)*alpha_k*phi_0_prime:
                    break
                else:
                    a_k = alpha_k
                    alpha_k = t*alpha_k if isinf(b_k) else (a_k + b_k) / 2.0
            else:
                b_k = alpha_k
                alpha_k = (a_k + b_k)/2.0
            if torch.sum(torch.abs(alpha_k * d)) < self.param_groups[0]['tolerance_grad']:
                break
            if abs(b_k-a_k) < 1e-6:
                break
        return alpha_k


    def _weak_wolfe(self, closure, d):
        # 0 < rho < 0.5 and rho < sigma < 1
        rho = 1e-4
        sigma = 0.9

        original_param_data_list = self._copy_param()
        phi_0 = closure().data[0]
        phi_0_prime = self._directional_derivative(d)
        a_k = 0.0
        b_k = self._max_alpha(d)
        alpha_k = min(1e4, (a_k + b_k) / 2.0)
        while True:
            self._set_param_incremental(alpha_k, d)
            phi_k = closure().data[0]
            phi_k_prime = self._directional_derivative(d)
            self._set_param(original_param_data_list)
            if phi_k <= phi_0 + rho*alpha_k*phi_0_prime:
                if phi_k_prime >= sigma*phi_0_prime:
                    break
                else:
                    alpha_hat = alpha_k + (alpha_k - a_k) * phi_k_prime / (phi_0_prime - phi_k_prime)
                    a_k = alpha_k
                    phi_0 = phi_k
                    phi_0_prime = phi_k_prime
                    alpha_k = alpha_hat
            else:
                alpha_hat = a_k + 0.5*(alpha_k-a_k)/(1+(phi_0-phi_k)/((alpha_k-a_k)*phi_0_prime))
                b_k = alpha_k
                alpha_k = alpha_hat
            if torch.sum(torch.abs(alpha_k * d)) < self.param_groups[0]['tolerance_grad']:
                break
            if abs(b_k-a_k) < 1e-6:
                break
        return alpha_k
