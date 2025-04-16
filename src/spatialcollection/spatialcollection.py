from __future__ import annotations

import sys
from collections.abc import Mapping
from itertools import chain
from os import PathLike
from typing import Any, overload

import more_itertools as mit
import narwhals as nw
import polars as pl
import richuru
import spatialdata as sd
import zarr
import zarr.storage
from loguru import logger
from narwhals.typing import IntoDataFrame
from ome_zarr.io import parse_url
from polars.exceptions import (
    ColumnNotFoundError,
    ComputeError,
    SchemaError,
    ShapeError,
)
from spatialdata import SpatialData
from spatialdata._core.validation import raise_validation_errors
from spatialdata._io.format import SpatialDataFormat
from upath import UPath

from .elements import SpatialDatas
from .io import _manage_zarr_group
from .models import SpatialCollectionMetadataModel
from .utils import metadata_preserving_merge
from .views import SpatialComponentView

richuru.install()
# Remove default handler and set level to WARNING to ignore INFO messages
logger.remove()
logger.add(sys.stderr, level="WARNING")

pl.enable_string_cache()


class SpatialCollection:
    """A collection of SpatialData objects."""

    def __init__(
        self,
        sdatas: Mapping[str, SpatialData],
        collection_metadata: IntoDataFrame | None = None,
        collection_key_col: str | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        self._path: UPath | None = None
        self._shared_keys: set[str | None] = set()
        self._collection_key_col = collection_key_col or "sdata_key"
        self._model_key_col = "sdata_key"

        self._sdatas: SpatialDatas = SpatialDatas(self._shared_keys)
        self.attrs = attrs if attrs else {}
        # Process and validate metadata during initialization

        sdata_names = list(chain.from_iterable([sdatas.keys()]))
        if len(sdata_names) != len(set(sdata_names)):
            duplicates = list(mit.duplicates_everseen(sdata_names))
            raise KeyError(
                f"SpatialData names must be unique. The following names are used multiple times: {duplicates}"
            )
        with raise_validation_errors(
            title="Cannot construct SpatialCollection, input contains invalid elements.",
            exc_type=(ValueError, KeyError),
        ) as collect_error:
            if sdatas is not None:
                for k, v in sdatas.items():
                    with collect_error(location=("sdatas", k)):
                        self.sdatas[k] = v

        # Pass sdatas to the processing function
        self._metadata: pl.DataFrame = self._process_and_validate_metadata(
            sdatas=self.sdatas,
            metadata_input=collection_metadata,
            collection_key_col=self._collection_key_col,
        )
        self._add_attrs_to_sdatas()

    @property
    def path(self) -> UPath | None:
        """The path to the Zarr store backing the SpatialCollection."""
        return self._path

    @path.setter
    def path(self, value: PathLike | str) -> None:
        match value:
            case None:
                self._path = None
            case str() | PathLike():
                self._path = UPath(value)
            case _:
                raise TypeError(f"Path must be a PathLike object: {type(value)}.")

    @property
    def sdatas(self) -> SpatialDatas:
        """Return the SpatialData objects as a dictionary of name to SpatialData object.

        Returns
        -------
            The SpatialData objects as a dictionary of name to SpatialData object.
        """
        return self._sdatas

    @sdatas.setter
    def sdatas(self, sdatas: Mapping[str, SpatialData]) -> None:
        self._shared_keys = self._shared_keys - set(self._sdatas.keys())
        self._sdatas = SpatialDatas(self._shared_keys)
        for k, v in sdatas.items():
            self._sdatas[k] = v

    @property
    def metadata(self) -> pl.DataFrame:
        """The metadata DataFrame for the SpatialCollection."""
        return self._metadata

    @metadata.setter
    def metadata(self, value: IntoDataFrame) -> None:
        # Use the helper method to process and set metadata
        self._metadata = self._process_and_validate_metadata(
            sdatas=self.sdatas, metadata_input=value, collection_key_col=self._collection_key_col
        )
        # We should re-sync sdatas attributes after setting new metadata
        self._add_attrs_to_sdatas()

    def _process_and_validate_metadata(
        self,
        sdatas: Mapping[str, SpatialData],  # Added sdatas param
        metadata_input: IntoDataFrame | None,
        collection_key_col: str,
    ) -> pl.DataFrame:
        """Processes, validates, and returns the collection metadata, integrating sdata.attrs."""
        # 1. Create base metadata DataFrame
        match metadata_input:
            case None:
                collection_keys = list(sdatas.keys())
                _data = {self._model_key_col: collection_keys} if collection_keys else {}
                nw_df = nw.from_dict(
                    data=_data,
                    schema={self._model_key_col: nw.Categorical()},
                    backend="polars",
                )
                base_metadata_df = nw_df.to_polars()
            case _:
                nw_df = nw.from_native(metadata_input)
                if collection_key_col not in nw_df.columns:
                    logger.error(
                        f"The collection key column '{collection_key_col}' is not present in the metadata. "
                        f"Please ensure the column exists or provide the correct 'collection_key_col'."
                    )
                    raise ValueError(
                        f"The collection key column '{collection_key_col}' is not present in the metadata."
                    )
                else:
                    nw_df_processed = nw_df.with_columns(
                        nw.col(collection_key_col).alias(self._model_key_col).cast(nw.Categorical()),
                    )
                    base_metadata_df = nw_df_processed.to_polars()

        # 2. Collect metadata from sdata.attrs
        all_sdata_metadata_dicts = []
        all_new_columns = set()
        current_cols = set(base_metadata_df.columns)

        for k, v in sdatas.items():
            sdata_attrs_meta = v.attrs.get("spatialcollection_metadata")
            if isinstance(sdata_attrs_meta, dict):
                # Ensure key column is present
                sdata_attrs_meta[self._model_key_col] = k
                # Add the original collection key if needed
                if self._collection_key_col != self._model_key_col and self._collection_key_col in current_cols:
                    sdata_attrs_meta[self._collection_key_col] = k

                all_sdata_metadata_dicts.append(sdata_attrs_meta)
                all_new_columns.update(set(sdata_attrs_meta.keys()) - current_cols)

        # 3. Add new columns if any were found
        if all_new_columns:
            logger.debug(f"Adding new columns found in sdata.attrs: {all_new_columns}")
            base_metadata_df = base_metadata_df.with_columns(
                # Use Unknown dtype initially, Polars update might coerce later
                [pl.lit(None, dtype=pl.Unknown).alias(c) for c in all_new_columns]
            )
            # Update current columns set for the next step
            current_cols.update(all_new_columns)

        # 4. Create update DataFrame and merge
        final_metadata_df = base_metadata_df
        if all_sdata_metadata_dicts:
            # Create DF with potentially new columns. Ensure schema compatibility.
            try:
                # Let Polars infer the schema from the dictionaries directly
                updates_df = pl.DataFrame(all_sdata_metadata_dicts)
                # Ensure key column is categorical for the update operation
                updates_df = updates_df.with_columns(pl.col(self._model_key_col).cast(pl.Categorical))

                # Perform the update
                final_metadata_df = base_metadata_df.update(updates_df, on=self._model_key_col)
            except (ShapeError, SchemaError, ColumnNotFoundError, ComputeError) as e:
                logger.error(
                    f"Failed to merge metadata from sdata.attrs due to incompatible data/schema: {e}. Using base metadata."
                )
                # Fallback or re-raise depending on desired strictness
                final_metadata_df = base_metadata_df  # Fallback to base

        # 5. Validate using the Patito model
        validated_df = SpatialCollectionMetadataModel.validate(final_metadata_df, allow_superfluous_columns=True)
        return validated_df

    def _sync_metadata(self) -> None:
        """Synchronizes the _metadata DataFrame with the keys in _sdatas."""
        set_sdata_keys = set(self.sdatas.keys())
        try:
            # Use is_empty() check before accessing columns
            set_metadata_keys = (
                set(self._metadata.get_column(self._model_key_col).to_list())
                if not self._metadata.is_empty()
                else set()
            )
        except pl.exceptions.ColumnNotFoundError:
            # Handle case where metadata is missing the key column (should ideally not happen after init)
            set_metadata_keys = set()

        keys_to_add = set_sdata_keys - set_metadata_keys
        keys_to_remove = set_metadata_keys - set_sdata_keys

        metadata_updated = False

        # Remove extra rows from metadata
        if keys_to_remove:
            self._metadata = self._metadata.filter(~pl.col(self._model_key_col).is_in(list(keys_to_remove)))
            logger.debug(f"Removed keys from metadata: {keys_to_remove}")
            metadata_updated = True

        # Add missing rows to metadata
        if keys_to_add:
            if self._metadata.is_empty():
                # Initialize metadata if it was empty
                new_rows_df = pl.DataFrame({self._model_key_col: list(keys_to_add)}).with_columns(
                    pl.col(self._model_key_col).cast(pl.Categorical)
                )
                self._metadata = new_rows_df
            else:
                # Create new rows matching the existing schema
                target_schema = self._metadata.schema
                existing_columns = set(target_schema.keys())
                new_rows_data = []
                for key in keys_to_add:
                    row: dict[str, Any | None] = {}
                    # Always set the internal model key
                    row[self._model_key_col] = key

                    # Set the original user-provided key if it's different and exists
                    original_key_col = self._collection_key_col
                    if original_key_col != self._model_key_col and original_key_col in existing_columns:
                        row[original_key_col] = key

                    # Fill remaining columns with None
                    filled_columns = set(row.keys())
                    for col_name in existing_columns:
                        if col_name not in filled_columns:
                            row[col_name] = None
                    new_rows_data.append(row)

                # Create DataFrame with the exact target schema
                new_rows_df = pl.DataFrame(new_rows_data, schema=target_schema)

                # Concatenate with matching schemas
                concat_df = pl.concat([self._metadata, new_rows_df], how="vertical")
                self._metadata = SpatialCollectionMetadataModel.validate(concat_df, allow_superfluous_columns=True)

            logger.debug(f"Added keys to metadata: {keys_to_add}")

            metadata_updated = True

        if not metadata_updated:
            logger.trace("Metadata already in sync.")

    def _validate_all_sdatas(self) -> None:
        with raise_validation_errors(
            title="A SpatialData object contains elements with invalid names.\n"
            "For renaming, please see the discussion here https://github.com/scverse/spatialdata/discussions/707 .",
            exc_type=ValueError,
        ) as collect_error:
            for k, v in self.sdatas.items():
                with collect_error(location=("sdatas", k)):
                    v._validate_all_elements()

    def is_backed(self) -> bool:
        """Whether the SpatialCollection is backed by a Zarr store."""
        return self.path is not None

    def sdatas_are_backed(self) -> bool:
        """Whether all SpatialData objects in the SpatialCollection are backed by a Zarr store."""
        return all(sdata.is_backed() for sdata in self.sdatas.values())

    def write_consolidated_metadata(self) -> None:
        """Write the consolidated metadata to the Zarr store."""
        store = parse_url(self.path, mode="r+").store  # type: ignore

        zarr.consolidate_metadata(store, metadata_key=".zmetadata")
        store.close()

    def _add_attrs_to_sdatas(self) -> None:
        for k, v in self.sdatas.items():
            v.attrs["spatialcollection_metadata"] = mit.one(
                self.metadata.filter(pl.col(self._model_key_col) == k).to_dicts()
            )

    def write_attrs(self, sdata_format: SpatialDataFormat | None = None, zarr_group: zarr.Group | None = None) -> None:
        """Write the attributes to each SpatialData object."""
        from spatialcollection import __version__

        path_to_open = self.path if zarr_group is None else None
        if path_to_open is None and zarr_group is None:
            assert self.is_backed(), (
                "The SpatialCollection must be backed to write attrs if zarr_group is not provided."
            )
            path_to_open = self.path

        with _manage_zarr_group(path_to_open, zarr_group, mode="r+", overwrite=False) as group_to_use:
            attrs_to_write = {
                "spatialcollection_attrs": {
                    "spatialcollection_software_version": __version__,
                    **self.attrs,
                    "internal_sdata_key": self._model_key_col,
                    "user_sdata_key": self._collection_key_col,
                },
            }

            group_to_use.attrs.put(attrs_to_write)
            self._add_attrs_to_sdatas()
            for k, v in self.sdatas.items():
                sub_group = group_to_use.require_group(k)
                v.write_attrs(zarr_group=sub_group)

    def write(self, collection_store: PathLike | str, overwrite: bool = False) -> None:
        """Write the SpatialCollection to a Zarr store."""
        collection_path: UPath = UPath(collection_store)
        collection_path.mkdir(parents=True, exist_ok=True)

        # Open the root group using the context manager for the entire write operation
        with _manage_zarr_group(collection_path, None, mode="w", overwrite=overwrite) as root_group:
            # Update self.path *inside* the context if writing to a new location
            if self.path != collection_path:
                old_path = self.path
                self.path = collection_path
                logger.info(
                    f"The Zarr backing store has been changed from {old_path} to the new file path: {collection_path}",
                    style="bold",
                )

            for k, v in self.sdatas.items():
                sdata_path = collection_path / k
                v.write(file_path=sdata_path, consolidate_metadata=False)

            self.write_attrs(zarr_group=root_group)

        self.write_consolidated_metadata()

    def to_sdata(self) -> SpatialData:
        """Convert the SpatialCollection to a SpatialData object."""
        sdata = sd.concatenate(
            sdatas=dict(self.sdatas.items()),
            concatenate_tables=True,
            obs_names_make_unique=True,
            attrs_merge=metadata_preserving_merge,
        )
        sdata.attrs = metadata_preserving_merge(attrs_list=[x.attrs for x in self.sdatas.values()])
        return sdata

    def __contains__(self, key: str) -> bool:
        # Check primary storage, assuming _sync_metadata keeps things consistent.
        return key in self._sdatas

    @overload
    def __getitem__(self, key: str) -> SpatialData: ...

    @overload
    def __getitem__(self, key: list[str]) -> SpatialCollection: ...

    def __getitem__(self, key: str | list[str]) -> SpatialData | SpatialCollection:
        match key:
            case str():
                return self.sdatas[key]
            case list():
                subset_sdatas = {k: self.sdatas[k] for k in key}
                filtered_metadata = self.metadata.filter(pl.col(self._model_key_col).is_in(key))
                return SpatialCollection(
                    sdatas=subset_sdatas,
                    collection_metadata=filtered_metadata,
                    collection_key_col=self._collection_key_col,
                    attrs=self.attrs.copy(),
                )
            case _:
                raise TypeError("Key must be a string or a list of strings.")

    def __setitem__(self, key: str, value: SpatialData) -> None:
        """Add or update a SpatialData object, synchronize metadata, and integrate attrs."""
        if not isinstance(value, SpatialData):
            raise TypeError(f"Value must be a SpatialData object, got {type(value)}.")

        # 1. Add/update in the primary storage
        self._sdatas[key] = value

        # 2. Check incoming sdata for metadata and new columns
        sdata_attrs_meta = value.attrs.get("spatialcollection_metadata")
        new_cols_from_attrs = set()
        if isinstance(sdata_attrs_meta, dict):
            current_cols = set(self._metadata.columns)
            new_cols_from_attrs = (
                set(sdata_attrs_meta.keys()) - current_cols - {self._model_key_col, self._collection_key_col}
            )
            if new_cols_from_attrs:
                logger.debug(f"Adding new columns found in attrs of '{key}': {new_cols_from_attrs}")
                self._metadata = self._metadata.with_columns(
                    [pl.lit(None, dtype=pl.Unknown).alias(c) for c in new_cols_from_attrs]
                )

        # 3. Synchronize the metadata DataFrame (ensure row exists)
        self._sync_metadata()

        # 4. Update the metadata row with values from sdata.attrs if they existed
        if isinstance(sdata_attrs_meta, dict):
            try:
                # Prepare the update dictionary, ensuring keys are present
                update_dict = sdata_attrs_meta.copy()
                update_dict[self._model_key_col] = key
                if (
                    self._collection_key_col != self._model_key_col
                    and self._collection_key_col in self._metadata.columns
                ):
                    update_dict[self._collection_key_col] = key

                # Create a one-row DataFrame, letting Polars infer the schema from the dict
                update_df = pl.DataFrame([update_dict])
                # Ensure the key column is categorical for the join/update
                update_df = update_df.with_columns(pl.col(self._model_key_col).cast(pl.Categorical))

                # Perform the update. Polars will upcast types in self._metadata if needed.
                self._metadata = self._metadata.update(update_df, on=self._model_key_col)
                logger.trace(f"Updated metadata for '{key}' using its attrs.")
            except (ShapeError, SchemaError, ColumnNotFoundError, ComputeError) as e:
                logger.error(
                    f"Failed to update metadata for '{key}' from its attrs due to incompatible data/schema: {e}"
                )

        # 5. Ensure ALL SpatialData objects have attrs reflecting the latest metadata schema
        self._add_attrs_to_sdatas()  # Update all sdatas

    def __delitem__(self, key: str) -> None:
        """Remove a SpatialData object and synchronize metadata."""
        if key not in self._sdatas:
            raise KeyError(f"Key '{key}' not found in SpatialCollection.")

        # Remove from primary storage
        del self._sdatas[key]

        # Synchronize the metadata DataFrame
        self._sync_metadata()

    def filter(self, *predicates: pl.Expr) -> SpatialCollection:
        """Filter the SpatialCollection based on the predicates."""
        filtered_metadata: pl.DataFrame = self.metadata.filter(*predicates)
        filtered_sdata_keys: list[str] = filtered_metadata.get_column(self._model_key_col).to_list()
        return SpatialCollection(
            sdatas={k: self.sdatas[k] for k in filtered_sdata_keys},
            collection_metadata=filtered_metadata,
            collection_key_col=self._collection_key_col,
            attrs=self.attrs.copy(),
        )

    def sdatas_paths_in_memory(self) -> list[str]:
        """Return the paths of the SpatialData objects in memory."""
        return list(self.sdatas.keys())

    def sdatas_paths_on_disk(self) -> list[str]:
        """Return the paths of the SpatialData objects on disk."""
        sdatas_in_zarr = []
        with _manage_zarr_group(self.path, None, mode="r", overwrite=False) as group_to_use:
            sdatas_in_zarr = list(group_to_use.group_keys())
        return sdatas_in_zarr

    def _sdatas_symmetric_difference_with_zarr_store(self) -> tuple[list[str], list[str]]:
        sdatas_in_collection = self.sdatas_paths_in_memory()
        sdatas_in_zarr = self.sdatas_paths_on_disk()

        sdatas_only_in_collection = list(set(sdatas_in_collection).difference(set(sdatas_in_zarr)))
        sdatas_only_in_zarr = list(set(sdatas_in_zarr).difference(set(sdatas_in_collection)))
        return sdatas_only_in_collection, sdatas_only_in_zarr

    def elements_paths_on_disk(self) -> list[str]:
        """Return the paths of the elements on disk across all SpatialData objects."""

        def get_elements_on_disk(item: tuple[str, SpatialData]) -> list[str] | None:
            match item:
                case (k, v) if v.is_backed():
                    return [f"{k}/{elem_path}" for elem_path in v.elements_paths_on_disk()]
                case _:
                    return None

        elements_on_disk = mit.filter_map(
            get_elements_on_disk,
            self.sdatas.items(),
        )
        return list(mit.collapse(elements_on_disk))

    def elements_paths_in_memory(self) -> list[str]:
        """Return the paths of the elements in memory across all SpatialData objects."""
        elements_in_memory = [
            f"{k}/{elem_path}" for k, v in self.sdatas.items() for elem_path in v.elements_paths_in_memory()
        ]
        return elements_in_memory

    def _elements_symmetric_difference_with_zarr_store(self) -> tuple[list[str], list[str]]:
        elements_in_collection = self.elements_paths_in_memory()
        elements_in_zarr = self.elements_paths_on_disk()

        elements_only_in_collection = list(set(elements_in_collection).difference(set(elements_in_zarr)))
        elements_only_in_zarr = list(set(elements_in_zarr).difference(set(elements_in_collection)))
        return elements_only_in_collection, elements_only_in_zarr

    @property
    def images(self) -> SpatialComponentView:
        """Return the images of the SpatialCollection."""
        return SpatialComponentView(self, "images")

    @property
    def labels(self) -> SpatialComponentView:
        """Return the labels of the SpatialCollection."""
        return SpatialComponentView(self, "labels")

    @property
    def shapes(self) -> SpatialComponentView:
        """Return the shapes of the SpatialCollection."""
        return SpatialComponentView(self, "shapes")

    @property
    def points(self) -> SpatialComponentView:
        """Return the points of the SpatialCollection."""
        return SpatialComponentView(self, "points")

    @property
    def tables(self) -> SpatialComponentView:
        """Return the tables of the SpatialCollection."""
        return SpatialComponentView(self, "tables")

    @property
    def coordinate_systems(self):
        """Return the coordinate systems of the SpatialCollection."""
        cs = {}
        for sdata_key, sdata in self.sdatas.items():
            cs[sdata_key] = list(filter(lambda x: x != "global", sdata.coordinate_systems))
        return cs

    def update_from_sdata(self, sdata: SpatialData) -> None:
        """Update the SpatialCollection from a SpatialData object."""
        from spatialcollection.utils import _update_sdatas

        self.sdatas.update(_update_sdatas(self, sdata))
        self._sync_metadata()
        self._add_attrs_to_sdatas()
