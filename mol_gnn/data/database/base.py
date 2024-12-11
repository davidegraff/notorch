from typing import Protocol


class Database[KT: (int, str), VT](Protocol):
    in_key: str | None
    out_key: str

    def __getitem__(self, key: KT) -> VT: ...
    def __len__(self) -> int: ...
