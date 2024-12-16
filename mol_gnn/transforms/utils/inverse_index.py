from __future__ import annotations

from collections.abc import Collection, Hashable, Iterable, Mapping
from typing import Literal, TypeVar, overload

KT = TypeVar("KT", bound=Hashable)


class InverseIndex[KT](Mapping[KT, int]):
    """
    A :class:`InverseIndex` is the logical inverse of a :class:`list`. That is, a list is a mapping
    from integer index to the respective item. In contrast, an ``IndexMap`` is a mapping from an
    item to its index in the corresponding list.
    """

    def __init__(self, keys: Iterable[KT]):
        self.__k2i = dict((x, i) for i, x in enumerate(keys))

    def __getitem__(self, key: KT) -> int:
        return self.__k2i[key]

    @overload
    def get(self, key: KT, default: None) -> int | None: ...

    @overload
    def get[T](self, key: KT, default: T) -> int | T: ...

    def get[T](self, key: KT, /, default: T | None = None) -> int | T:
        return self.__k2i.get(key, default)

    def __len__(self) -> int:
        return len(self.__k2i)

    def __iter__(self):
        return iter(self.__k2i)

    def __repr__(self):
        return str(list(map(str, self.__k2i.keys()))).replace("'", "")


class InverseIndexWithUnknown[KT](InverseIndex[KT]):
    """
    A :class:`InverseIndexWithUnknown` is like a :class:`InverseIndex`, with the only difference
    being that querying the map for an unknown item will always return padding index, i.e., the
    number of items in the map.
    """

    def __getitem__(self, key: KT):
        return super().get(key, len(self) - 1)

    def __len__(self) -> int:
        return super().__len__() + 1

    def __repr__(self):
        return super().__repr__() + " + <UNK>"

@overload
def build(choices: None, unknown_pad: bool) -> None: ...


@overload
def build(choices: Collection[KT], unknown_pad: Literal[True]) -> InverseIndexWithUnknown[KT]: ...


@overload
def build(choices: Collection[KT], unknown_pad: Literal[False]) -> InverseIndex[KT]: ...


def build(choices: Collection[KT] | None, unknown_pad: bool = True):
    if choices is not None and len(choices) == 0 and not unknown_pad:
        raise ValueError(
            "arg 'choices' was empty but arg 'unknown_pad' is False! "
            "The resuting `InverseIndex` will have no valid keys!"
        )

    if choices is None:
        return None
    elif unknown_pad:
        return InverseIndexWithUnknown(choices)

    return InverseIndex(choices)
