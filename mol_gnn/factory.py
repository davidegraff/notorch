from typing import Callable, Concatenate


def meta_factory[S, T, **P](clz: Callable[Concatenate[S, P], T], *args: P.args, **kwargs: P.kwargs):
    """Build a parameterized factory from a class factory.

    It's essentially just a typed version of a :class:`~functools.partial` object that unpacks the
    first positional argument:

    .. code-block:: python
        from torch.optim import Adam
        adam_factory = meta_factory(Adam)
        # adam_factory : ParamsT -> Adam

    Parameters
    ----------
    clz : Callable[Concatenate[S, P], T]
        The class that will be built
    *args, **kwargs
        additional positional and keyword arguments to supply.

        .. note::
            in contrast to typical ``partial`` object, the positional arguments will be _appended_
            the input argument rather than prepended.
    """

    def fun(s: S) -> T:
        return clz(s, *args, **kwargs)

    return fun
