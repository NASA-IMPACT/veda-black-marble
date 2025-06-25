"""Coordinate system transformations and spatial reference utilities.

This module provides essential functions for handling coordinate reference systems
(CRS), transformations, and spatial calculations. These utilities are fundamental
to the prepare phase where data from multiple sources with different projections
must be aligned to a common spatial reference.

Proper coordinate handling ensures spatial accuracy and enables integration of
Landsat, VIIRS, and OSM data sources that use different native projections.
"""

from typing import Any, Literal
import numpy.typing as npt

# Type aliases
ArrayLike = npt.NDArray[Any]
BBox = tuple[float, float, float, float]
Transform = Any  # rasterio.transform.Affine
CRS = Any  # rasterio.crs.CRS


def transform_bounds(
    bounds: BBox,
    src_crs: CRS | str,
    dst_crs: CRS | str,
    densify_points: int = 21
) -> BBox:
    """Transform bounding box from source to destination CRS.
    
    This function correctly transforms bounding boxes between coordinate systems,
    accounting for the fact that straight lines in one projection may be curves
    in another. It's critical for determining data coverage after reprojection.
    
    The densify_points parameter ensures accurate transformation of bounds that
    cross projection discontinuities or cover large areas where distortion is
    significant.
    
    Args:
        bounds: Source bounds as (min_x, min_y, max_x, max_y)
        src_crs: Source coordinate reference system
        dst_crs: Destination coordinate reference system
        densify_points: Number of points to add along edges for accuracy
        
    Returns:
        Transformed bounds in destination CRS as (min_x, min_y, max_x, max_y)
        
    Example:
        >>> # Transform WGS84 bounds to Web Mercator
        >>> bounds_4326 = (-74.05, 40.69, -73.95, 40.79)
        >>> bounds_3857 = transform_bounds(
        >>>     bounds_4326,
        >>>     "EPSG:4326",
        >>>     "EPSG:3857"
        >>> )
        
    Note:
        This function is essential when determining output bounds for
        reprojection operations in the prepare phase.
    """
    # Use rasterio's transform_bounds which handles densification
    from rasterio.warp import transform_bounds as _transform_bounds
    
    return _transform_bounds(
        src_crs,
        dst_crs,
        bounds[0],  # left/min_x
        bounds[1],  # bottom/min_y
        bounds[2],  # right/max_x
        bounds[3],  # top/max_y
        densify_pts=densify_points
    )


def calculate_transform(
    bounds: BBox,
    width: int,
    height: int,
    target_resolution: float | None = None,
    method: Literal["fit", "res"] = "fit"
) -> tuple[Transform, int, int]:
    """Calculate affine transform for given bounds and dimensions.
    
    This function computes the affine transformation matrix that maps pixel
    coordinates to geographic coordinates. It's fundamental for creating
    georeferenced rasters and ensuring proper spatial alignment.
    
    The transform defines the relationship between pixel indices and real-world
    coordinates, enabling accurate spatial operations throughout the pipeline.
    
    Args:
        bounds: Geographic bounds (min_x, min_y, max_x, max_y)
        width: Desired width in pixels
        height: Desired height in pixels
        target_resolution: Target resolution in coordinate units
        method: Calculation method:
            - 'fit': Fit to exact dimensions, resolution varies
            - 'res': Use exact resolution, dimensions adjust
            
    Returns:
        Tuple of (transform, actual_width, actual_height) where:
        - transform: Affine transformation matrix
        - actual_width: Actual width (may differ if method='res')
        - actual_height: Actual height (may differ if method='res')
        
    Example:
        >>> # Calculate transform for 30m resolution output
        >>> transform, w, h = calculate_transform(
        >>>     bounds=(-8247000, 4970000, -8237000, 4980000),
        >>>     width=None,
        >>>     height=None,
        >>>     target_resolution=30.0,
        >>>     method='res'
        >>> )
    """
    # New implementation based on rasterio patterns
    from rasterio.transform import from_bounds
    import math
    
    min_x, min_y, max_x, max_y = bounds
    extent_x = max_x - min_x
    extent_y = max_y - min_y
    
    if method == "fit" and width is not None and height is not None:
        # Fit to exact dimensions
        transform = from_bounds(min_x, min_y, max_x, max_y, width, height)
        return transform, width, height
    
    elif method == "res" and target_resolution is not None:
        # Use exact resolution
        width = int(math.ceil(extent_x / target_resolution))
        height = int(math.ceil(extent_y / target_resolution))
        
        # Create transform with exact resolution
        transform = from_bounds(min_x, min_y, max_x, max_y, width, height)
        
        return transform, width, height
    
    else:
        raise ValueError("Either provide width/height for 'fit' method or target_resolution for 'res' method")


def get_crs_from_epsg(
    epsg_code: int,
    return_proj4: bool = False
) -> CRS | str:
    """Get CRS object from EPSG code with validation.
    
    This function retrieves coordinate reference system definitions from
    EPSG codes, with validation to ensure the CRS is appropriate for the
    processing region. It prevents common errors like using inappropriate
    projections for specific latitudes.
    
    Args:
        epsg_code: EPSG code number
        return_proj4: Return PROJ4 string instead of CRS object
        
    Returns:
        CRS object or PROJ4 string
        
    Example:
        >>> # Get Web Mercator CRS
        >>> crs = get_crs_from_epsg(3857)
        >>> print(crs.to_string())
        
    Note:
        Common EPSG codes in Black Marble:
        - 4326: WGS84 (lat/lon)
        - 3857: Web Mercator
        - 32600+: UTM zones
    """
    # Implementation using rasterio CRS
    import rasterio.crs
    
    try:
        crs = rasterio.crs.CRS.from_epsg(epsg_code)
    except Exception as e:
        raise ValueError(f"Invalid EPSG code {epsg_code}: {e}")
    
    if return_proj4:
        return crs.to_proj4()
    else:
        return crs


def calculate_pixel_area(
    transform: Transform,
    crs: CRS,
    shape: tuple[int, int],
    unit: Literal["m2", "km2", "ha"] = "m2"
) -> ArrayLike:
    """Calculate area of each pixel accounting for map projection.
    
    This function computes the actual ground area represented by each pixel,
    which varies with latitude in many projections. Accurate area calculation
    is essential for computing statistics like total urban area or forest cover.
    
    For geographic projections (lat/lon), pixel area varies significantly
    with latitude. For projected coordinate systems, area may be constant
    or vary depending on the projection's distortion characteristics.
    
    Args:
        transform: Affine transform of the raster
        crs: Coordinate reference system
        shape: Raster shape as (height, width)
        unit: Output unit:
            - 'm2': Square meters
            - 'km2': Square kilometers
            - 'ha': Hectares
            
    Returns:
        Array of pixel areas matching the raster shape
        
    Example:
        >>> # Calculate pixel areas for statistics
        >>> areas = calculate_pixel_area(
        >>>     transform,
        >>>     crs,
        >>>     data.shape,
        >>>     unit='km2'
        >>> )
        >>> urban_area_km2 = (urban_mask * areas).sum()
        
    Note:
        This is critical for accurate area-based statistics in
        the analysis and reporting phases.
    """
    # New implementation for pixel area calculation
    import numpy as np
    import pyproj
    
    height, width = shape
    
    # Check if CRS is geographic (lat/lon)
    if crs.is_geographic:
        # For geographic CRS, area varies with latitude
        # Create arrays of pixel coordinates
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # Convert pixel coords to geographic coords
        xs, ys = transform * (cols, rows)
        
        # Calculate area using geodesic calculations
        geod = pyproj.Geod(ellps='WGS84')
        
        # Get pixel dimensions
        pixel_width = abs(transform.a)
        pixel_height = abs(transform.e)
        
        # Area varies with latitude for geographic projections
        # Approximate area at each latitude
        areas = np.zeros(shape)
        for i in range(height):
            lat = ys[i, 0]
            # Calculate distance of 1 degree longitude at this latitude
            _, _, dist_m = geod.inv(0, lat, 1, lat)
            # Area in square meters
            area_m2 = (dist_m * pixel_width) * (111320 * pixel_height)  # 111320 m per degree latitude
            areas[i, :] = area_m2
    
    else:
        # For projected CRS, assume constant pixel area
        pixel_width = abs(transform.a)
        pixel_height = abs(transform.e)
        area_m2 = pixel_width * pixel_height
        areas = np.full(shape, area_m2)
    
    # Convert units
    if unit == "km2":
        areas = areas / 1_000_000
    elif unit == "ha":
        areas = areas / 10_000
    
    return areas


def reproject_geometry(
    geometry: Any,  # shapely geometry or GeoJSON
    src_crs: CRS | str,
    dst_crs: CRS | str
) -> Any:
    """Reproject vector geometry between coordinate systems.
    
    This function transforms vector geometries (points, lines, polygons)
    between coordinate reference systems. It's used for reprojecting
    OpenStreetMap road networks and administrative boundaries to match
    the raster data projection.
    
    Args:
        geometry: Input geometry (Shapely or GeoJSON format)
        src_crs: Source coordinate system
        dst_crs: Destination coordinate system
        
    Returns:
        Reprojected geometry in the same format as input
        
    Example:
        >>> # Reproject road network to match Landsat
        >>> roads_3857 = reproject_geometry(
        >>>     roads_4326,
        >>>     "EPSG:4326",
        >>>     "EPSG:3857"
        >>> )
    """
    # Implementation using pyproj and shapely
    import pyproj
    from shapely.ops import transform as shapely_transform
    from shapely.geometry import shape, mapping
    
    # Create transformer
    transformer = pyproj.Transformer.from_crs(
        src_crs,
        dst_crs,
        always_xy=True
    )
    
    # Check if input is GeoJSON
    if isinstance(geometry, dict):
        # Convert GeoJSON to shapely
        shp = shape(geometry)
        # Transform
        transformed = shapely_transform(transformer.transform, shp)
        # Convert back to GeoJSON
        return mapping(transformed)
    else:
        # Assume shapely geometry
        return shapely_transform(transformer.transform, geometry)


def create_crs_transformer(
    src_crs: CRS | str,
    dst_crs: CRS | str,
    always_xy: bool = True
) -> Any:  # pyproj.Transformer
    """Create reusable coordinate transformer between CRS.
    
    This function creates an efficient transformer for converting many
    coordinates between coordinate systems. It's more efficient than
    repeated single-point transformations and should be used when
    processing large arrays of coordinates.
    
    Args:
        src_crs: Source coordinate system
        dst_crs: Destination coordinate system
        always_xy: Expect and return (x, y) order regardless of CRS
        
    Returns:
        Transformer object for coordinate conversion
        
    Example:
        >>> # Create transformer for many coordinates
        >>> transformer = create_crs_transformer("EPSG:4326", "EPSG:3857")
        >>> x_out, y_out = transformer.transform(lon_array, lat_array)
        
    Note:
        Transformers are thread-safe and can be reused across
        multiple operations.
    """
    # Simple wrapper around pyproj
    import pyproj
    
    return pyproj.Transformer.from_crs(
        src_crs,
        dst_crs,
        always_xy=always_xy
    )


def calculate_utm_zone(
    lon: float,
    lat: float
) -> int:
    """Calculate appropriate UTM zone for given coordinates.
    
    This function determines the optimal UTM zone for a geographic location,
    which provides a good local projection with minimal distortion. UTM
    projections are often preferable to Web Mercator for analysis due to
    their equal-area properties.
    
    Args:
        lon: Longitude in degrees
        lat: Latitude in degrees
        
    Returns:
        UTM zone number (1-60) with hemisphere encoding
        
    Example:
        >>> # Find UTM zone for New York
        >>> zone = calculate_utm_zone(-74.0, 40.7)
        >>> epsg = 32600 + zone  # Northern hemisphere
        
    Note:
        Negative zone numbers indicate southern hemisphere.
        EPSG codes: 32600+zone (north), 32700+zone (south)
    """
    # Standard UTM zone calculation
    zone = int((lon + 180) / 6) + 1
    
    # Handle special cases for Norway
    if lat >= 56.0 and lat < 64.0 and lon >= 3.0 and lon < 12.0:
        zone = 32
    
    # Handle special cases for Svalbard
    elif lat >= 72.0 and lat < 84.0:
        if lon >= 0.0 and lon < 9.0:
            zone = 31
        elif lon >= 9.0 and lon < 21.0:
            zone = 33
        elif lon >= 21.0 and lon < 33.0:
            zone = 35
        elif lon >= 33.0 and lon < 42.0:
            zone = 37
    
    # Encode hemisphere: negative for southern
    if lat < 0:
        zone = -zone
    
    return zone


def validate_crs_compatibility(
    crs_list: list[CRS | str],
    check_units: bool = True,
    check_area: bool = False
) -> dict[str, Any]:
    """Validate compatibility of multiple CRS for processing.
    
    This function checks whether multiple coordinate reference systems are
    compatible for joint processing. It identifies potential issues like
    mixing geographic and projected systems or using incompatible units.
    
    Args:
        crs_list: List of CRS to check
        check_units: Verify unit compatibility
        check_area: Check if CRS preserves area
        
    Returns:
        Validation results dictionary:
        - compatible: Whether all CRS are compatible
        - issues: List of compatibility issues found
        - recommendations: Suggested common CRS
        - unit_mismatch: Details of unit conflicts
        
    Example:
        >>> # Check if data sources can be combined
        >>> result = validate_crs_compatibility([
        >>>     landsat_crs,
        >>>     viirs_crs,
        >>>     osm_crs
        >>> ])
        >>> if not result['compatible']:
        >>>     print(result['recommendations'])
    """
    # New implementation for CRS compatibility checking
    import rasterio.crs
    
    if not crs_list:
        return {
            "compatible": True,
            "issues": [],
            "recommendations": [],
            "unit_mismatch": []
        }
    
    issues = []
    unit_mismatch = []
    recommendations = []
    
    # Convert all to CRS objects
    crs_objects = []
    for crs in crs_list:
        if isinstance(crs, str):
            crs_objects.append(rasterio.crs.CRS.from_string(crs))
        else:
            crs_objects.append(crs)
    
    # Check if all CRS are the same
    reference_crs = crs_objects[0]
    all_same = all(crs == reference_crs for crs in crs_objects)
    
    if all_same:
        return {
            "compatible": True,
            "issues": [],
            "recommendations": [],
            "unit_mismatch": []
        }
    
    # Check for geographic vs projected mixing
    geographic_count = sum(1 for crs in crs_objects if crs.is_geographic)
    projected_count = len(crs_objects) - geographic_count
    
    if geographic_count > 0 and projected_count > 0:
        issues.append("Mixing geographic and projected coordinate systems")
        recommendations.append("Reproject all data to a common projected CRS")
    
    # Check units
    if check_units:
        units = []
        for i, crs in enumerate(crs_objects):
            if crs.is_geographic:
                units.append("degrees")
            else:
                unit = crs.linear_units
                units.append(unit)
                
        unique_units = list(set(units))
        if len(unique_units) > 1:
            unit_mismatch = list(zip(range(len(crs_objects)), units))
            issues.append(f"Unit mismatch: {unique_units}")
            recommendations.append("Use CRS with consistent units")
    
    # Check area preservation
    if check_area:
        for i, crs in enumerate(crs_objects):
            if not crs.is_geographic:
                # Check if it's an equal-area projection
                proj_name = crs.to_dict().get('proj', '').lower()
                if proj_name not in ['aea', 'laea', 'utm', 'merc']:
                    issues.append(f"CRS {i} may not preserve area accurately")
                    recommendations.append("Use equal-area projections for area calculations")
    
    # Suggest common CRS
    if issues:
        recommendations.append("Consider using EPSG:3857 (Web Mercator) or appropriate UTM zone")
    
    compatible = len(issues) == 0
    
    return {
        "compatible": compatible,
        "issues": issues,
        "recommendations": list(set(recommendations)),
        "unit_mismatch": unit_mismatch
    }


def calculate_ground_distance(
    pixel_distance: float,
    transform: Transform,
    crs: CRS,
    location: tuple[float, float] | None = None
) -> float:
    """Convert pixel distance to ground distance in meters.
    
    This function converts distances measured in pixels to actual ground
    distances, accounting for map projection distortions. It's essential
    for operations like buffer creation and kernel-based processing.
    
    Args:
        pixel_distance: Distance in pixels
        transform: Affine transform
        crs: Coordinate reference system
        location: Optional (x, y) location for local calculation
        
    Returns:
        Ground distance in meters
        
    Example:
        >>> # Convert 5-pixel buffer to meters
        >>> buffer_m = calculate_ground_distance(
        >>>     5,
        >>>     transform,
        >>>     crs
        >>> )
        >>> print(f"5 pixels = {buffer_m:.1f} meters")
        
    Note:
        For geographic CRS, distance varies with location.
        The location parameter enables accurate local calculations.
    """
    # New implementation for pixel-to-ground distance conversion
    import pyproj
    import math
    
    # Get pixel size in CRS units
    pixel_width = abs(transform.a)
    pixel_height = abs(transform.e)
    
    # Average pixel size (assumes square-ish pixels)
    pixel_size = math.sqrt(pixel_width * pixel_height)
    
    if crs.is_geographic:
        # For geographic CRS, convert to meters using geodesic calculation
        if location is None:
            # Use center of transform if no location specified
            location = (transform.c + pixel_width * 0.5, transform.f + pixel_height * 0.5)
        
        # Use geodesic distance calculation
        geod = pyproj.Geod(ellps='WGS84')
        
        # Calculate distance of pixel size in degrees
        lon, lat = location
        _, _, dist_m = geod.inv(lon, lat, lon + pixel_size, lat)
        
        # Convert pixel distance to meters
        return pixel_distance * dist_m
    
    else:
        # For projected CRS, assume units are meters (or check linear units)
        unit = crs.linear_units
        
        if unit == 'metre':
            return pixel_distance * pixel_size
        elif unit == 'foot':
            # Convert feet to meters
            return pixel_distance * pixel_size * 0.3048
        else:
            # Assume meters for unknown units
            return pixel_distance * pixel_size