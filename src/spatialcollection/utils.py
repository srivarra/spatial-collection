from __future__ import annotations

from typing import TYPE_CHECKING, Any

import anndata as ad
import more_itertools as mit
import narwhals as nw
import natsort as ns
import spatialdata as sd
from narwhals.typing import DataFrameT

if TYPE_CHECKING:
    from .spatialcollection import SpatialCollection


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
    return ns.natsorted([cs for cs in sdata.coordinate_systems if cs != "global"])


def _filter_attrs_by_sdata_key(sdata: sd.SpatialData, sdata_key: str) -> dict:
    sdata_metadata: dict[str, Any] = sdata.attrs["spatialcollection_metadata"][sdata_key]
    return sdata_metadata


def _update_table(table: ad.AnnData, sdata_key: str) -> ad.AnnData:
    from spatialdata.models import TableModel

    region_key: str = table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]

    updated_regions = mit.first(
        mit.flatten([x.split(f"-{sdata_key}") for x in table.uns[TableModel.ATTRS_KEY][region_key]])
    )
    first_column = table.obs[region_key].str.split(f"-{sdata_key}", expand=True).iloc[:, 0]

    table.obs[region_key] = first_column
    sd.SpatialData._change_table_annotation_target(table, region=updated_regions, region_key=region_key)
    return table


def _update_sdatas(
    sc: SpatialCollection,
    sdata: sd.SpatialData,
):
    sdatas = {}
    sdata_non_global_cs = _non_global_coordinate_systems(sdata)
    sc_sdata_keys_in_sdata = list(sdata.attrs["spatialcollection_metadata"].keys())

    if set(sc_sdata_keys_in_sdata) == set(sdata_non_global_cs):
        for sdata_key in sc_sdata_keys_in_sdata:
            _sdata = sdata.filter_by_coordinate_system(sdata_key)
            sdata_metadata_attrs = _filter_attrs_by_sdata_key(_sdata, sdata_key)
            elements = {}
            for element_type, element_name, element in _sdata.gen_elements():
                en = mit.first(element_name.split(f"-{sdata_key}"))
                if element_type == "tables" and isinstance(element, ad.AnnData):
                    elements[en] = _update_table(element, sdata_key)
                else:
                    elements[en] = element
            __sdata = sd.SpatialData.init_from_elements(elements, attrs=sdata_metadata_attrs)
            sdatas[sdata_key] = __sdata
    else:
        raise ValueError(
            f"The coordinate systems in the SpatialCollection {sc_sdata_keys_in_sdata} do not match the coordinate systems in the SpatialData {sdata_non_global_cs}"
        )
    return sdatas
