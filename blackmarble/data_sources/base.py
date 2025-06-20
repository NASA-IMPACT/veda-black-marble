"""Base data source interface."""

from typing import Protocol, Dict, Any, Tuple, List, Optional
from datetime import datetime
import numpy as np
from pathlib import Path


class DataSource(Protocol):
    """Interface for data sources.
    
    Data sources are responsible for fetching raw data from external
    sources and returning it in a standardized format.
    """
    
    def fetch(
        self,
        bbox: Tuple[float, float, float, float],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Fetch data for given spatial and temporal bounds.
        
        Args:
            bbox: Bounding box (west, south, east, north)
            start_date: Start of temporal range
            end_date: End of temporal range
            **kwargs: Additional source-specific parameters
            
        Returns:
            Dictionary mapping data layer names to arrays
            
        Raises:
            DataSourceError: If data fetch fails
        """
        ...
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the data source.
        
        Returns:
            Dictionary with source metadata including:
            - name: Source name
            - description: Source description
            - bands: Available bands/layers
            - resolution: Native resolution
            - crs: Coordinate reference system
        """
        ...
    
    def validate_request(
        self,
        bbox: Tuple[float, float, float, float],
        start_date: datetime,
        end_date: datetime
    ) -> None:
        """Validate that a data request is valid.
        
        Args:
            bbox: Bounding box
            start_date: Start date
            end_date: End date
            
        Raises:
            DataSourceError: If request is invalid
        """
        ...


class DataSourceError(Exception):
    """Base exception for data source errors."""
    pass


class DataNotAvailableError(DataSourceError):
    """Raised when requested data is not available."""
    pass


class InvalidRequestError(DataSourceError):
    """Raised when data request parameters are invalid."""
    pass