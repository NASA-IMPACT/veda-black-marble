"""Common type definitions and data structures used across all phases.

This module provides shared types that enable flexible data handling
throughout the Black Marble pipeline, supporting both file-based and
in-memory processing.
"""

from dataclasses import dataclass, field
from typing import Any
import numpy.typing as npt
from enum import Enum

# Type aliases used throughout the codebase
ArrayLike = npt.NDArray[Any]
BBox = tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
Transform = Any  # rasterio.transform.Affine
CRS = Any  # rasterio.crs.CRS
Window = Any  # rasterio.windows.Window


@dataclass
class SceneData:
    """Container for scene data that can be path or array.
    
    This flexible container supports both file-based and in-memory workflows,
    allowing the pipeline to optimize memory usage while maintaining a
    consistent interface.
    
    Attributes:
        path: Path to file on disk (for lazy loading)
        array: In-memory array (for small data or intermediate results)
        transform: Affine transform for georeferencing
        crs: Coordinate reference system
        bounds: Bounding box in CRS coordinates
        metadata: Additional metadata (acquisition date, sensor, etc.)
    """
    path: str | None = None
    array: ArrayLike | None = None
    transform: Transform | None = None
    crs: CRS | None = None
    bounds: BBox | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def load(self, window: Window | None = None) -> ArrayLike:
        """Load data from path or return array.
        
        Args:
            window: Optional window for partial reads
            
        Returns:
            Data array
            
        Raises:
            ValueError: If neither path nor array is available
        """
        if self.array is not None:
            return self.array
            
        if self.path is None:
            raise ValueError("No data available: both path and array are None")
            
        import rasterio
        with rasterio.open(self.path) as src:
            if window:
                return src.read(window=window)
            return src.read()
    
    def is_loaded(self) -> bool:
        """Check if data is currently in memory."""
        return self.array is not None
    
    def has_data(self) -> bool:
        """Check if any data is available (path or array)."""
        return self.path is not None or self.array is not None


# Common enumerations for pipeline options
class CompositingMethod(str, Enum):
    """Methods for temporal compositing."""
    MEDIAN = "median"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    PERCENTILE = "percentile"
    QUALITY_WEIGHTED = "quality_weighted"


class ResamplingMethod(str, Enum):
    """Methods for spatial resampling."""
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    CUBIC = "cubic"
    CUBIC_SPLINE = "cubic_spline"
    LANCZOS = "lanczos"
    AVERAGE = "average"
    MODE = "mode"


class ContrastMethod(str, Enum):
    """Methods for contrast enhancement."""
    LINEAR = "linear"
    PERCENTILE = "percentile"
    ADAPTIVE = "adaptive"
    HISTOGRAM = "histogram"
    RETINEX = "retinex"


class ColorMap(str, Enum):
    """Available color maps for visualization."""
    GRAYSCALE = "grayscale"
    INFERNO = "inferno"
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    TURBO = "turbo"


# Base exception classes
class BlackMarbleError(Exception):
    """Base exception for all Black Marble processing errors."""
    pass


class AcquisitionError(BlackMarbleError):
    """Error during data acquisition phase."""
    pass


class ProcessingError(BlackMarbleError):
    """Error during data processing phases."""
    pass


class ValidationError(BlackMarbleError):
    """Error during validation or quality checks."""
    pass


class ConfigurationError(BlackMarbleError):
    """Error in configuration or parameters."""
    pass