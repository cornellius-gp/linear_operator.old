#!/usr/bin/env python3

import warnings
import functools


class DeprecationError(Exception):
    pass


def _deprecated_function_for(old_function_name, function):
    @functools.wraps(function)
    def _deprecated_function(*args, **kwargs):
        warnings.warn(
            "The `{}` function is deprecated. Use `{}` instead".format(old_function_name, function.__name__),
            DeprecationWarning
        )
        return function(*args, **kwargs)

    return _deprecated_function


def _deprecate_kwarg(kwargs, old_kw, new_kw, new_kw_value):
    old_kwarg = kwargs.get(old_kw)
    if old_kwarg is not None:
        warnings.warn("The `{}` argument is deprecated. Use `{}` instead.".format(old_kw, new_kw), DeprecationWarning)
        if new_kw_value is not None:
            raise ValueError("Cannot set both `{}` and `{}`".format(old_kw, new_kw))
        return old_kwarg
    return new_kw_value


def _deprecated_renamed_method(cls, old_method_name, new_method_name):
    def _deprecated_method(self, *args, **kwargs):
        warnings.warn(
            "The `{}` method is deprecated. Use `{}` instead".format(old_method_name, new_method_name),
            DeprecationWarning
        )
        return getattr(self, new_method_name)(*args, **kwargs)

    _deprecated_method.__name__ = old_method_name
    setattr(cls, old_method_name, _deprecated_method)
    return cls


def _deprecate_renamed_methods(cls, **renamed_methods):
    for old_method_name, new_method_name in renamed_methods.items():
        _deprecated_renamed_method(cls, old_method_name, new_method_name)
    return cls
