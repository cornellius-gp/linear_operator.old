#!/usr/bin/env python3
from . import (
    lazy,
    settings,
    utils,
)

from .functions import (  # Deprecated
    add_diag,
    add_jitter,
    dsmm,
    inv_matmul,
    inv_quad,
    inv_quad_logdet,
    log_normal_cdf,
    logdet,
    matmul,
    root_decomposition,
    root_inv_decomposition,
)
from .lazy import cat, delazify, lazify

__version__ = "0.0.1"

__all__ = [
    # Submodules
    "lazy",
    "utils",
    # Functions
    "add_diag",
    "add_jitter",
    "cat",
    "delazify",
    "dsmm",
    "inv_matmul",
    "inv_quad",
    "inv_quad_logdet",
    "lazify",
    "logdet",
    "log_normal_cdf",
    "matmul",
    "root_decomposition",
    "root_inv_decomposition",
    # Context managers
    "settings",
    # Other
    "__version__",
]
