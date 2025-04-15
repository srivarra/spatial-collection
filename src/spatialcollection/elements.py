from __future__ import annotations

from collections.abc import ItemsView, Iterator, KeysView, ValuesView
from typing import Any

from spatialdata import SpatialData
from spatialdata._core._elements import Elements


class SpatialDatas(Elements):
    """A collection of SpatialData objects."""

    def __setitem__(self, key: str, value: Any) -> None:
        self._check_key(key, self.keys(), self._shared_keys)
        if not isinstance(value, SpatialData):
            raise TypeError(f"{key} is not a SpatialData object: {type(value)}.")
        super().__setitem__(key, value)

    def keys(self) -> KeysView[str]:
        """Return the keys of the SpatialData objects."""
        return self.data.keys()

    def values(self) -> ValuesView[SpatialData]:
        """Return the values of the SpatialData objects."""
        return self.data.values()

    def items(self) -> ItemsView[str, SpatialData]:
        """Return the items of the SpatialData objects."""
        return self.data.items()

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)
