from typing import Callable, Generic, Protocol, TypedDict, TypeVar, Self

T = TypeVar("T")


class HParamsDict(TypedDict, Generic[T]):
    """A dictionary containing a object's constructor and keyword arguments

    Using this type allows for initializing an object via::

        obj = hparams.pop("from_hparams")(**hparams)
    """

    from_hparams: Callable[..., T]


class HasHParams(Protocol):
    """:class:`HasHParams` is a protocol for clases which possess an :attr:`hparams` attribute of
    type :class:`HParamsDict`.

    That is, any object which implements :class:`HasHParams` can be initialized via::

        class Foo(HasHParams):
            def __init__(self, *args, **kwargs):
                ...

        foo1 = Foo(...)
        foo_from_hparams = foo1.hparams['from_hparams']
        foo1_kwargs = {k: v for k, v in foo1.hparams.items() if k != "from_hparams"}
        foo2 = foo_from_hparams(**foo1_kwargs)
        # code to compare foo1 and foo2 goes here and they should be equal
    """

    hparams: HParamsDict[Self]
