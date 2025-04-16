from importlib.metadata import version

from .io import read_zarr
from .spatialcollection import SpatialCollection

__all__ = ["SpatialCollection", "read_zarr"]

__version__ = version("spatial-collection")
