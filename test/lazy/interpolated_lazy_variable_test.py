import gpytorch
import torch
from torch.autograd import Variable
from gpytorch.lazy import NonLazyVariable, InterpolatedLazyVariable
from gpytorch.utils import approx_equal


def test_matmul():
    left_interp_indices = Variable(torch.LongTensor([[2, 3], [3, 4], [4, 5]])).repeat(3, 1)
    left_interp_values = Variable(torch.Tensor([[1, 2], [0.5, 1], [1, 3]])).repeat(3, 1)
    right_interp_indices = Variable(torch.LongTensor([[0, 1], [1, 2], [2, 3]])).repeat(3, 1)
    right_interp_values = Variable(torch.Tensor([[1, 2], [2, 0.5], [1, 3]])).repeat(3, 1)

    base_lazy_variable_mat = torch.randn(6, 6)
    base_lazy_variable_mat = base_lazy_variable_mat.t().matmul(base_lazy_variable_mat)
    base_lazy_variable = NonLazyVariable(Variable(base_lazy_variable_mat))

    test_matrix = torch.randn(9, 4)

    interp_lazy_var = InterpolatedLazyVariable(base_lazy_variable, left_interp_indices, left_interp_values,
                                               right_interp_indices, right_interp_values)
    res = interp_lazy_var.matmul(Variable(test_matrix)).data

    left_matrix = torch.Tensor([
        [0, 0, 1, 2, 0, 0],
        [0, 0, 0, 0.5, 1, 0],
        [0, 0, 0, 0, 1, 3],
        [0, 0, 1, 2, 0, 0],
        [0, 0, 0, 0.5, 1, 0],
        [0, 0, 0, 0, 1, 3],
        [0, 0, 1, 2, 0, 0],
        [0, 0, 0, 0.5, 1, 0],
        [0, 0, 0, 0, 1, 3],
    ])
    right_matrix = torch.Tensor([
        [1, 2, 0, 0, 0, 0],
        [0, 2, 0.5, 0, 0, 0],
        [0, 0, 1, 3, 0, 0],
        [1, 2, 0, 0, 0, 0],
        [0, 2, 0.5, 0, 0, 0],
        [0, 0, 1, 3, 0, 0],
        [1, 2, 0, 0, 0, 0],
        [0, 2, 0.5, 0, 0, 0],
        [0, 0, 1, 3, 0, 0],
    ])
    actual = left_matrix.matmul(base_lazy_variable_mat).matmul(right_matrix.t()).matmul(test_matrix)
    assert approx_equal(res, actual)


def pending_test_inv_matmul():
    left_interp_indices = Variable(torch.LongTensor([[2, 3], [3, 4], [4, 5]]))
    left_interp_values = Variable(torch.Tensor([[1, 2], [0.5, 1], [1, 3]]))
    right_interp_indices = Variable(torch.LongTensor([[2, 3], [3, 4], [4, 5]]))
    right_interp_values = Variable(torch.Tensor([[1, 2], [0.5, 1], [1, 3]]))

    base_lazy_variable_mat = torch.randn(6, 6)
    base_lazy_variable_mat = base_lazy_variable_mat.t().matmul(base_lazy_variable_mat)
    base_lazy_variable = NonLazyVariable(Variable(base_lazy_variable_mat))
    test_matrix = torch.randn(3, 4)

    interp_lazy_var = InterpolatedLazyVariable(base_lazy_variable, left_interp_indices, left_interp_values,
                                               right_interp_indices, right_interp_values)
    res = interp_lazy_var.inv_matmul(Variable(test_matrix)).data

    left_matrix = torch.Tensor([
        [0, 0, 1, 2, 0, 0],
        [0, 0, 0, 0.5, 1, 0],
        [0, 0, 0, 0, 1, 3],
    ])
    right_matrix = torch.Tensor([
        [0, 0, 1, 2, 0, 0],
        [0, 0, 0, 0.5, 1, 0],
        [0, 0, 0, 0, 1, 3],
    ])
    actual_mat = Variable(left_matrix.matmul(base_lazy_variable_mat).matmul(right_matrix.t()))
    actual = gpytorch.inv_matmul(actual_mat, Variable(test_matrix)).data
    assert approx_equal(res, actual)


def test_matmul_batch():
    left_interp_indices = Variable(torch.LongTensor([[2, 3], [3, 4], [4, 5]])).repeat(5, 3, 1)
    left_interp_values = Variable(torch.Tensor([[1, 2], [0.5, 1], [1, 3]])).repeat(5, 3, 1)
    right_interp_indices = Variable(torch.LongTensor([[0, 1], [1, 2], [2, 3]])).repeat(5, 3, 1)
    right_interp_values = Variable(torch.Tensor([[1, 2], [2, 0.5], [1, 3]])).repeat(5, 3, 1)

    base_lazy_variable_mat = torch.randn(5, 6, 6)
    base_lazy_variable_mat = base_lazy_variable_mat.transpose(1, 2).matmul(base_lazy_variable_mat)
    test_matrix = Variable(torch.randn(1, 9, 4))

    base_lazy_variable = NonLazyVariable(Variable(base_lazy_variable_mat, requires_grad=True))
    interp_lazy_var = InterpolatedLazyVariable(base_lazy_variable, left_interp_indices, left_interp_values,
                                               right_interp_indices, right_interp_values)
    res = interp_lazy_var.matmul(test_matrix)

    left_matrix = torch.Tensor([
        [0, 0, 1, 2, 0, 0],
        [0, 0, 0, 0.5, 1, 0],
        [0, 0, 0, 0, 1, 3],
        [0, 0, 1, 2, 0, 0],
        [0, 0, 0, 0.5, 1, 0],
        [0, 0, 0, 0, 1, 3],
        [0, 0, 1, 2, 0, 0],
        [0, 0, 0, 0.5, 1, 0],
        [0, 0, 0, 0, 1, 3],
    ]).repeat(5, 1, 1)

    right_matrix = torch.Tensor([
        [1, 2, 0, 0, 0, 0],
        [0, 2, 0.5, 0, 0, 0],
        [0, 0, 1, 3, 0, 0],
        [1, 2, 0, 0, 0, 0],
        [0, 2, 0.5, 0, 0, 0],
        [0, 0, 1, 3, 0, 0],
        [1, 2, 0, 0, 0, 0],
        [0, 2, 0.5, 0, 0, 0],
        [0, 0, 1, 3, 0, 0],
    ]).repeat(5, 1, 1)
    actual = left_matrix.matmul(base_lazy_variable_mat).matmul(right_matrix.transpose(-1, -2)).matmul(test_matrix.data)

    assert approx_equal(res.data, actual)


def test_derivatives():
    left_interp_indices = Variable(torch.LongTensor([[2, 3], [3, 4], [4, 5]])).repeat(5, 3, 1)
    left_interp_values = Variable(torch.Tensor([[1, 2], [0.5, 1], [1, 3]])).repeat(5, 3, 1)
    right_interp_indices = Variable(torch.LongTensor([[2, 3], [3, 4], [4, 5]])).repeat(5, 3, 1)
    right_interp_values = Variable(torch.Tensor([[1, 2], [0.5, 1], [1, 3]])).repeat(5, 3, 1)

    base_lazy_variable_mat = torch.randn(5, 6, 6)
    base_lazy_variable_mat = base_lazy_variable_mat.transpose(1, 2).matmul(base_lazy_variable_mat)
    test_matrix = Variable(torch.randn(1, 9, 4))

    base_lazy_variable = NonLazyVariable(Variable(base_lazy_variable_mat, requires_grad=True))
    interp_lazy_var = InterpolatedLazyVariable(base_lazy_variable, left_interp_indices, left_interp_values,
                                               right_interp_indices, right_interp_values)
    res = interp_lazy_var.matmul(test_matrix)
    res.sum().backward()

    base_lazy_variable2 = Variable(base_lazy_variable_mat, requires_grad=True)
    left_matrix = torch.Tensor([
        [0, 0, 1, 2, 0, 0],
        [0, 0, 0, 0.5, 1, 0],
        [0, 0, 0, 0, 1, 3],
        [0, 0, 1, 2, 0, 0],
        [0, 0, 0, 0.5, 1, 0],
        [0, 0, 0, 0, 1, 3],
        [0, 0, 1, 2, 0, 0],
        [0, 0, 0, 0.5, 1, 0],
        [0, 0, 0, 0, 1, 3],
    ]).repeat(5, 1, 1)
    actual = Variable(left_matrix).matmul(base_lazy_variable2).matmul(Variable(left_matrix).transpose(-1, -2))
    actual = actual.matmul(test_matrix)
    actual.sum().backward()

    assert approx_equal(base_lazy_variable.var.grad.data, base_lazy_variable2.grad.data)


def test_getitem_batch():
    left_interp_indices = Variable(torch.LongTensor([[2, 3], [3, 4], [4, 5]]).repeat(5, 1, 1))
    left_interp_values = Variable(torch.Tensor([[1, 1], [1, 1], [1, 1]]).repeat(5, 1, 1))
    right_interp_indices = Variable(torch.LongTensor([[0, 1], [1, 2], [2, 3]]).repeat(5, 1, 1))
    right_interp_values = Variable(torch.Tensor([[1, 1], [1, 1], [1, 1]]).repeat(5, 1, 1))

    base_lazy_variable_mat = torch.randn(5, 6, 6)
    base_lazy_variable_mat = base_lazy_variable_mat.transpose(1, 2).matmul(base_lazy_variable_mat)

    base_lazy_variable = NonLazyVariable(Variable(base_lazy_variable_mat, requires_grad=True))
    interp_lazy_var = InterpolatedLazyVariable(base_lazy_variable, left_interp_indices, left_interp_values,
                                               right_interp_indices, right_interp_values)

    actual = (base_lazy_variable[:, 2:5, 0:3] + base_lazy_variable[:, 2:5, 1:4] +
              base_lazy_variable[:, 3:6, 0:3] + base_lazy_variable[:, 3:6, 1:4]).evaluate()

    assert approx_equal(interp_lazy_var[2].evaluate().data, actual[2].data)
    assert approx_equal(interp_lazy_var[0:2].evaluate().data, actual[0:2].data)
    assert approx_equal(interp_lazy_var[:, 2:3].evaluate().data, actual[:, 2:3].data)
    assert approx_equal(interp_lazy_var[:, 0:2].evaluate().data, actual[:, 0:2].data)
    assert approx_equal(interp_lazy_var[1, :1, :2].evaluate().data, actual[1, :1, :2].data)
    assert approx_equal(interp_lazy_var[1, 1, :2].data, actual[1, 1, :2].data)
    assert approx_equal(interp_lazy_var[1, :1, 2].data, actual[1, :1, 2].data)


def test_diag():
    left_interp_indices = Variable(torch.LongTensor([[2, 3], [3, 4], [4, 5]]))
    left_interp_values = Variable(torch.Tensor([[1, 1], [1, 1], [1, 1]]))
    right_interp_indices = Variable(torch.LongTensor([[0, 1], [1, 2], [2, 3]]))
    right_interp_values = Variable(torch.Tensor([[1, 1], [1, 1], [1, 1]]))

    base_lazy_variable_mat = torch.randn(6, 6)
    base_lazy_variable_mat = base_lazy_variable_mat.t().matmul(base_lazy_variable_mat)

    base_lazy_variable = NonLazyVariable(Variable(base_lazy_variable_mat, requires_grad=True))
    interp_lazy_var = InterpolatedLazyVariable(base_lazy_variable, left_interp_indices, left_interp_values,
                                               right_interp_indices, right_interp_values)

    actual = interp_lazy_var.evaluate()
    assert approx_equal(actual.diag().data, interp_lazy_var.diag().data)


def test_batch_diag():
    left_interp_indices = Variable(torch.LongTensor([[2, 3], [3, 4], [4, 5]]).repeat(5, 1, 1))
    left_interp_values = Variable(torch.Tensor([[1, 1], [1, 1], [1, 1]]).repeat(5, 1, 1))
    right_interp_indices = Variable(torch.LongTensor([[0, 1], [1, 2], [2, 3]]).repeat(5, 1, 1))
    right_interp_values = Variable(torch.Tensor([[1, 1], [1, 1], [1, 1]]).repeat(5, 1, 1))

    base_lazy_variable_mat = torch.randn(5, 6, 6)
    base_lazy_variable_mat = base_lazy_variable_mat.transpose(1, 2).matmul(base_lazy_variable_mat)

    base_lazy_variable = NonLazyVariable(Variable(base_lazy_variable_mat, requires_grad=True))
    interp_lazy_var = InterpolatedLazyVariable(base_lazy_variable, left_interp_indices, left_interp_values,
                                               right_interp_indices, right_interp_values)

    actual = interp_lazy_var.evaluate()
    actual_diag = torch.stack([actual[0].diag(), actual[1].diag(), actual[2].diag(),
                               actual[3].diag(), actual[4].diag()])

    assert approx_equal(actual_diag.data, interp_lazy_var.diag().data)
