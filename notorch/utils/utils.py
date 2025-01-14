from __future__ import annotations

from copy import copy
from enum import StrEnum
from typing import Iterator, Self


class EnumMapping(StrEnum):
    @classmethod
    def get(cls, name: str | Self) -> Self:
        if isinstance(name, cls):
            return name

        try:
            return cls[name.upper()]
        except KeyError:
            raise KeyError(
                f"Unsupported {cls.__name__} member! got: '{name}'. expected one of: {cls.keys()}"
            )

    @classmethod
    def keys(cls) -> Iterator[str]:
        return (e.name for e in cls)

    @classmethod
    def values(cls) -> Iterator[str]:
        return (e.value for e in cls)

    @classmethod
    def items(cls) -> Iterator[tuple[str, str]]:
        return zip(cls.keys(), cls.values())


class UpdateMixin:
    def update(self, in_place: bool = False, **kwargs) -> Self:
        other = self if in_place else copy(self)
        for key, val in kwargs:
            setattr(other, key, val)

        return other
