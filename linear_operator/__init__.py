#!/usr/bin/env python3
from . import operators, settings, utils
from .functions import (  # Deprecated
    add_diag,
    add_jitter,
    dsmm,
    inv_matmul,
    inv_quad,
    inv_quad_logdet,
    logdet,
    matmul,
    root_decomposition,
    root_inv_decomposition,
)
from .operators import LinearOperator, cat, to_dense, to_linear_operator

__version__ = "0.0.1"

__all__ = [
    # Submodules
    "operators",
    "utils",
    # Linear operators,
    "LinearOperator",
    # Functions
    "add_diag",
    "add_jitter",
    "cat",
    "to_dense",
    "dsmm",
    "inv_matmul",
    "inv_quad",
    "inv_quad_logdet",
    "to_linear_operator",
    "logdet",
    "matmul",
    "root_decomposition",
    "root_inv_decomposition",
    # Context managers
    "settings",
    # Other
    "__version__",
]
