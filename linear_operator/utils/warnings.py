#!/usr/bin/env python3


class ExtraComputationWarning(UserWarning):
    """
    Warning thrown when a LineawrOperator does extra computation that it is not designed to do.
    """

    pass


class NumericalWarning(RuntimeWarning):
    """
    Warning thrown when convergence criteria are not met, or when comptuations require extra stability.
    """

    pass
