from abc import abstractmethod
from collections.abc import Collection, Mapping
# from contextlib import AbstractContextManager


class Database[KT, VT, VT_batched](Mapping[KT, VT]):
    """A :class:`Database` is a mapping from keys to values that is intended to
    be used as a context manager.

    .. note::
        The specification does not _require_ that a given database be used
        within a context, just that the the class supports it.
    """

    @abstractmethod
    def collate(self, values: Collection[VT]) -> VT_batched: ...
