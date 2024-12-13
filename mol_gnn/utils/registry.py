import inspect
from typing import Any, Iterable, Type, TypeVar

T = TypeVar("T")


class ClassRegistry(dict[str, Type[T]]):
    def register(self, alias: Any | Iterable[Any] | None = None):
        def decorator(cls):
            match alias:
                case None:
                    keys = [cls.__name__.lower()]
                case str():
                    keys = [alias]
                case _:
                    keys = alias

            cls.alias = keys[0]
            for k in keys:
                self[k] = cls

            return cls

        return decorator

    __call__ = register

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}: {super().__repr__()}"

    def __str__(self) -> str:  # pragma: no cover
        INDENT = 4
        items = [f"{' ' * INDENT}{repr(k)}: {repr(v)}" for k, v in self.items()]

        return "\n".join([f"{self.__class__.__name__} {'{'}", ",\n".join(items), "}"])
