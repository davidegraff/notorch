from collections.abc import Mapping
from contextlib import AbstractContextManager


class Database[KT, VT](Mapping[KT, VT], AbstractContextManager):
    """A :class:`Database` is a mapping from keys to values that is intended to
    be used as a context manager.

    .. note::
        The specification does not _require_ that a given database be used
        within a context, just that the the class supports it.
    """
