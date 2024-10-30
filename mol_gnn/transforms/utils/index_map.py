from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Literal, overload, TypeVar

KT = TypeVar("KT", bound=Hashable)
T = TypeVar("T")


class IndexMap[KT](Mapping[KT, int]):
    """
    A :class:`IndexMap` is the logical inverse of a :class:`list`. That is, a
    list is a mapping from integer index to the respective item. In contrast,
    an ``IndexMap`` is a mapping from an item to its index in the corresponding
    list.
    """

    def __init__(self, keys: Sequence[KT]):
        self.__k2i = dict((x, i) for i, x in enumerate(keys))

    def __getitem__(self, key: KT) -> int:
        return self.__k2i[key]

    @overload
    def get(self, key: KT, default: None) -> int | None: ...

    @overload
    def get(self, key: KT, default: T) -> int | T: ...

    def get(self, key: KT, default: T | None = None) -> int | T | None:
        return self.__k2i.get(key, default)

    def __len__(self) -> int:
        return len(self.__k2i)

    def __iter__(self):
        return iter(self.__k2i)

    def __repr__(self):
        return repr(self.__k2i)


class IndexMapWithUnknown(IndexMap[KT]):
    """
    A :class:`IndexMapWithUnknown` is like a :class:`IndexMap`, with the only
    difference being that querying the map for an unknown item will always
    return padding index, i.e., the number of items in the map.
    """

    def __getitem__(self, key: KT):
        return super().get(key, len(self) - 1)

    def __len__(self) -> int:
        return super().__len__() + 1

    def __repr__(self):
        items_str = ", ".join(
            [f"{repr(k)}: {repr(i)}" for k, i in self.items()] + [f"UNK: {len(self)-1}"]
        )

        return "".join(["{", items_str, "}"])


@overload
def build(choices: None, unknown_pad: bool) -> None: ...


@overload
def build(choices: Sequence[KT], unknown_pad: Literal[True]) -> IndexMapWithUnknown: ...


@overload
def build(choices: Sequence[KT], unknown_pad: Literal[False]) -> IndexMap: ...


def build(choices: Sequence[KT] | None, unknown_pad: bool = True):
    if choices is not None and len(choices) == 0 and not unknown_pad:
        raise ValueError(
            "arg 'choices' was empty but arg 'unknown_pad' is False! "
            "The resuting `IndexMap` will have no valid keys!"
        )

    if choices is None:
        return None
    elif unknown_pad:
        return IndexMapWithUnknown(choices)

    return IndexMap(choices)
