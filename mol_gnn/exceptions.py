from typing import Iterable


def pretty_shape(shape: Iterable[int]) -> str:
    """Make a pretty string from an input shape

    Example
    -------
    >>> X = np.random.rand(10, 4)
    >>> X.shape
    (10, 4)
    >>> pretty_shape(X.shape)
    '10 x 4'
    """
    return " x ".join(map(str, shape))


class InvalidShapeError(ValueError):
    def __init__(self, var_name: str, received: Iterable[int], expected: Iterable[int]):
        message = (
            f"arg '{var_name}' has incorrect shape! "
            f"got: `{pretty_shape(received)}`. expected: `{pretty_shape(expected)}`"
        )

        super().__init__(message)


class ClosedDatabaseError(ValueError):
    def __init__(self, clz: type) -> None:
        clz_name = clz.__name__
        message = (
            f"tried to retrieve an item from a closed database! An `{clz_name}`"
            + f" must be used inside a context manager, i.e., `with {clz_name}(): ...`"
        )

        super().__init__(message)
