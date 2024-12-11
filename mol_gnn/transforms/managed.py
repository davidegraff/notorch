from collections.abc import Collection
from dataclasses import dataclass

from mol_gnn.transforms.base import Transform


@dataclass
class ManagedTransform[S, T, T_batched]:
    """A :class:`ManagedTransform` wraps around an input :class:`Transform` that reads and writes
    from to an input dictionary.

    It's like a :class:`~tensordict.nn.TensorDictModule` analog for :class:`Transform`s.
    """

    transform: Transform[S, T, T_batched]
    in_key: str
    out_key: str

    def collate(self, samples: Collection[dict]) -> T_batched:
        inputs = [sample[self.out_key] for sample in samples]

        return self.transform.collate(inputs)
        # return {self.out_key: self.transform.collate(inputs)}

    def __call__(self, sample: dict) -> dict:
        sample[self.out_key] = self.transform(sample[self.in_key])

        return sample
