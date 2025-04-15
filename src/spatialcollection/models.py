import patito as pt
import polars as pl
from pydantic import ConfigDict


class SpatialCollectionMetadataModel(pt.Model):
    """
    A model for collection metadata.

    Requires 'sdata_key' to uniquely identify each SpatialData entry.
    Allows any other arbitrary columns/fields to be present in the metadata.
    """

    sdata_key: str = pt.Field(
        unique=True,
        dtype=pl.Categorical,
        description="Unique identifier matching the keys in the sdatas dictionary.",
    )
    model_config = ConfigDict(extra="allow")
