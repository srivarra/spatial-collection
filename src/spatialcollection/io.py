from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from json import JSONDecodeError
from os import PathLike
from typing import TYPE_CHECKING

import polars as pl
import spatialdata as sd
import zarr
import zarr.storage
from ome_zarr.io import parse_url
from upath import UPath
from zarr.errors import GroupNotFoundError, MetadataError

if TYPE_CHECKING:
    from spatialcollection.spatialcollection import SpatialCollection
from loguru import logger
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
    from spatialcollection.spatialcollection import SpatialCollection

    path = UPath(path)
    sdatas = {}
    collection_metadata = None
    collection_key_col = None
    collection_attrs = {}

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

        # Read the stored metadata table and key info
        if "spatialcollection_metadata_table" in group.attrs:
            metadata_list = group.attrs["spatialcollection_metadata_table"]
            # Convert list of dicts back to DataFrame
            collection_metadata = pl.DataFrame(metadata_list)
            # Ensure the key column is categorical
            key_info = group.attrs.get("spatialcollection_info", {})
            internal_key = key_info.get("internal_sdata_key", "sdata_key")  # Default if info missing
            collection_key_col = key_info.get("user_sdata_key", internal_key)  # Default if info missing
            if internal_key in collection_metadata.columns:
                collection_metadata = collection_metadata.with_columns(pl.col(internal_key).cast(pl.Categorical))
            else:
                logger.warning(f"Internal key '{internal_key}' not found in stored metadata table.")
        else:
            logger.warning(
                "'spatialcollection_metadata_table' not found in Zarr attributes. Metadata will be reconstructed from sdata keys."
            )
            # Fallback: If table not stored, initialize with keys only (old behavior, less robust)
            collection_metadata = None
            collection_key_col = "sdata_key"  # Assume default

    # Pass the read metadata and key column directly to the constructor
    return SpatialCollection(
        sdatas=sdatas,
        collection_metadata=collection_metadata,
        collection_key_col=collection_key_col,
        attrs=collection_attrs,
    )
