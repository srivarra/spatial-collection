from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from json import JSONDecodeError
from os import PathLike
from typing import TYPE_CHECKING

import spatialdata as sd
import zarr
import zarr.storage
from ome_zarr.io import parse_url
from upath import UPath
from zarr.errors import GroupNotFoundError, MetadataError

if TYPE_CHECKING:
    from spatialcollection.spatialcollection import SpatialCollection

from spatialdata._io._utils import BadFileHandleMethod, handle_read_errors


@contextmanager
def _manage_zarr_group(
    path: PathLike | str | None,
    group: zarr.Group | None,
    mode: str,
    overwrite: bool = False,
) -> Iterator[zarr.Group]:
    """Context manager to get a Zarr group, opening from path if needed."""
    store_to_close: zarr.storage.BaseStore | None = None
    if group is not None:
        # Group provided, just yield it, don't manage store
        yield group
    else:
        if path is None:
            raise ValueError("Path must be provided if group is None.")
        current_path = UPath(path)
        location = parse_url(current_path, mode=mode)  # type: ignore
        if location is None:
            raise ValueError(f"Cannot open Zarr store at {current_path} with mode '{mode}'.")

        store_to_close = location.store
        try:
            opened_group = zarr.group(store=store_to_close, overwrite=overwrite)
            yield opened_group
        finally:
            if store_to_close is not None:
                store_to_close.close()


def read_zarr(path: PathLike | str) -> SpatialCollection:
    """Read a SpatialCollection from a Zarr store."""
    from spatialcollection import SpatialCollection

    path = UPath(path)
    sdatas = {}
    collection_key_col: str | None = None

    with (
        _manage_zarr_group(path, None, "r") as group,
        handle_read_errors(
            BadFileHandleMethod.ERROR,
            location=str(path),
            exc_types=(
                JSONDecodeError,
                MetadataError,
                KeyError,
                ValueError,
                GroupNotFoundError,
            ),
        ),
    ):
        # Read individual SpatialData objects
        for k in group.group_keys():
            sdatas[k] = sd.read_zarr(path / k)

        # Read collection-level attributes
        if "spatialcollection_attrs" in group.attrs:
            collection_attrs = group.attrs["spatialcollection_attrs"]

            collection_key_col = collection_attrs.get("internal_sdata_key", "sdata_key")

    # Pass the read metadata and key column directly to the constructor
    sc = SpatialCollection(
        sdatas=sdatas,
        collection_metadata=None,
        collection_key_col=collection_key_col,
        attrs=collection_attrs,
    )
    sc.path = path
    return sc
