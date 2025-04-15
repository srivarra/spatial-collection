from typing import Any

import more_itertools as mit
import narwhals as nw
import spatialdata as sd
from narwhals.typing import DataFrameT


def metadata_preserving_merge(attrs_list: list[dict[Any, Any]]) -> dict[Any, Any]:
    """
    Custom merge function that preserves metadata during SpatialData concatenation.

    This merge function preserves metadata from the original objects rather than
    dropping metadata fields when they don't match or are not present in all objects.
    When concatenating SpatialData objects with different metadata, this ensures the
    metadata is preserved rather than being replaced with NaNs.

    Parameters
    ----------
    attrs_list : list[dict]
        List of attribute dictionaries to merge

    Returns
    -------
    dict
        Merged attributes dictionary
    """
    if not attrs_list:
        return {}

    inner_dicts_iter = (
        attrs["spatialcollection_metadata"] for attrs in attrs_list if attrs and "spatialcollection_metadata" in attrs
    )

    result = dict(
        mit.map_reduce(
            inner_dicts_iter,
            keyfunc=lambda inner_dict: inner_dict["sdata_key"],
            reducefunc=mit.first,
        )
    )
    return {"spatialcollection_metadata": result}


def sc_col(*names: str) -> nw.Expr:
    """A narwhals expression for column access.

    Returns
    -------
        A narwhals expression for column access.
    """
    return nw.col(*names)


@nw.narwhalify
def nw_filter(df: DataFrameT, *exprs: nw.Expr) -> DataFrameT:
    """Filter a DataFrame via narwhals expressions."""
    return df.filter(*exprs)


def _non_global_coordinate_systems(sdata: sd.SpatialData) -> list[str]:
    return [cs for cs in sdata.coordinate_systems if cs != "global"]


def _filter_attrs_by_sdata_key(sdata: sd.SpatialData, sdata_key: str) -> dict:
    sdata_metadata: dict[str, Any] = sdata.attrs["spatialcollection_metadata"][sdata_key]
    return sdata_metadata
