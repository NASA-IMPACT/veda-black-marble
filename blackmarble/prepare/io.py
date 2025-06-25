"""File I/O utilities for data preparation phase.

This module provides robust file input/output functions that handle various
raster formats and ensure data is properly loaded for processing. These utilities
form the foundation of the prepare phase, managing the complexity of different
data formats while providing a consistent interface.

The I/O functions handle format detection, metadata extraction, and efficient
data loading strategies including windowed reads and virtual datasets.
"""

from typing import Any, Literal, Callable
import numpy.typing as npt

# Type aliases
ArrayLike = npt.NDArray[Any]
Transform = Any  # rasterio.transform.Affine
CRS = Any  # rasterio.crs.CRS
BBox = tuple[float, float, float, float]
Window = Any  # rasterio.windows.Window


def detect_format(
    file_path: str,
    check_subdatasets: bool = True
) -> dict[str, Any]:
    """Detect raster file format and structure.
    
    This function analyzes a file to determine its format, internal structure,
    and the best method for reading it. It's essential for handling diverse
    data sources in the Black Marble pipeline, from simple GeoTIFFs to complex
    HDF5 files with multiple subdatasets.
    
    Format detection ensures the appropriate reader is used and helps identify
    potential issues before processing begins.
    
    Args:
        file_path: Path to file to analyze
        check_subdatasets: Whether to check for HDF/NetCDF subdatasets
        
    Returns:
        Format information dictionary containing:
        - format: File format name ('GTiff', 'HDF5', 'NetCDF', etc.)
        - driver: Appropriate GDAL/rasterio driver
        - subdatasets: List of subdatasets if present
        - compression: Compression type if any
        - tile_structure: Internal tiling information
        - is_cog: Whether file is a Cloud-Optimized GeoTIFF
        - reader: Recommended reader function
        
    Example:
        >>> # Detect format before reading
        >>> info = detect_format("VNP46A2.h5")
        >>> if info['subdatasets']:
        >>>     print(f"Found {len(info['subdatasets'])} subdatasets")
        >>> reader = info['reader']
        >>> data = reader(file_path)
        
    Note:
        This function helps avoid common errors when reading complex
        formats like HDF5 files with multiple layers.
    """
    # New implementation for format detection
    import os
    import rasterio
    from osgeo import gdal
    
    result = {
        "format": None,
        "driver": None,
        "subdatasets": [],
        "compression": None,
        "tile_structure": None,
        "is_cog": False,
        "reader": load_with_rasterio  # Default reader
    }
    
    # Check file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Try with rasterio first
    try:
        with rasterio.open(file_path) as src:
            result["format"] = src.driver
            result["driver"] = src.driver
            
            # Check for compression
            if hasattr(src, 'compression'):
                result["compression"] = src.compression
            
            # Check for tiling
            if src.is_tiled:
                result["tile_structure"] = {
                    "tiled": True,
                    "blockxsize": src.block_shapes[0][1],
                    "blockysize": src.block_shapes[0][0]
                }
            
            # Check if COG (basic check)
            if src.driver == "GTiff" and src.is_tiled:
                # More sophisticated COG detection would check overviews
                result["is_cog"] = len(src.overviews(1)) > 0
    except:
        # Fallback to GDAL for complex formats
        pass
    
    # Use GDAL for subdataset detection
    if check_subdatasets:
        gdal.PushErrorHandler('CPLQuietErrorHandler')
        ds = gdal.Open(file_path, gdal.GA_ReadOnly)
        gdal.PopErrorHandler()
        
        if ds is not None:
            # Get format if not already detected
            if result["format"] is None:
                driver = ds.GetDriver()
                result["format"] = driver.ShortName if driver else "Unknown"
                result["driver"] = result["format"]
            
            # Check for subdatasets
            subdatasets = ds.GetSubDatasets()
            if subdatasets:
                result["subdatasets"] = [
                    {"path": sd[0], "description": sd[1]} 
                    for sd in subdatasets
                ]
                # Use GDAL reader for HDF/NetCDF with subdatasets
                result["reader"] = load_with_gdal
            
            ds = None
    
    return result


def load_with_gdal(
    file_path: str,
    band: int | str | None = None,
    window: Window | None = None,
    overview_level: int | None = None,
    as_masked: bool = True
) -> tuple[ArrayLike, dict[str, Any]]:
    """Load raster data using GDAL with comprehensive metadata.
    
    This function provides a robust interface to GDAL for reading raster data,
    handling various edge cases and format-specific quirks. It's the primary
    method for loading data when GDAL's capabilities are needed.
    
    GDAL is particularly useful for complex formats and when specific
    subdatasets need to be accessed by name.
    
    Args:
        file_path: Path to raster file
        band: Band number (1-indexed) or subdataset name
        window: Window for partial reads
        overview_level: Overview level to read (0=full resolution)
        as_masked: Return masked array handling nodata
        
    Returns:
        Tuple of (data_array, metadata_dict) where metadata includes:
        - transform: Affine transformation matrix
        - crs: Coordinate reference system
        - nodata: NoData value
        - dtype: Data type
        - bounds: Geographic bounds
        - format_metadata: Format-specific metadata
        
    Example:
        >>> # Load specific HDF5 subdataset
        >>> data, meta = load_with_gdal(
        >>>     "VNP46A2.h5",
        >>>     band="HDFEOS/GRIDS/VNP_Grid_DNB/Data_Fields/DNB_At_Sensor_Radiance"
        >>> )
        
    Note:
        GDAL is used as fallback when rasterio cannot handle
        specific format requirements.
    """
    # Implementation adapted from io.py:16 get_geo()
    # WATCHPOINT: io.py get_geo only handles band numbers, not subdatasets
    from osgeo import gdal
    import numpy as np
    from rasterio.transform import Affine
    
    # Handle subdataset access
    if isinstance(band, str):
        # Band is a subdataset name
        file_path = f"HDF5:\"{file_path}\"://{band}"
        band = 1  # Read first band of subdataset
    elif band is None:
        band = 1
    
    # Open dataset
    ds = gdal.Open(file_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Could not open raster file: {file_path}")
    
    # Get dimensions
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    
    # Handle windowed reads
    if window is not None:
        # Window should have col_off, row_off, width, height
        col_off = int(window.col_off) if hasattr(window, 'col_off') else 0
        row_off = int(window.row_off) if hasattr(window, 'row_off') else 0
        width = int(window.width) if hasattr(window, 'width') else cols
        height = int(window.height) if hasattr(window, 'height') else rows
    else:
        col_off, row_off = 0, 0
        width, height = cols, rows
    
    # Get band and read data
    band_obj = ds.GetRasterBand(band)
    
    # Handle overview level
    if overview_level is not None and overview_level > 0:
        if band_obj.GetOverviewCount() >= overview_level:
            band_obj = band_obj.GetOverview(overview_level - 1)
    
    # Read data
    array = band_obj.ReadAsArray(col_off, row_off, width, height)
    nodata = band_obj.GetNoDataValue()
    
    # Handle case where read fails (e.g., window outside bounds)
    if array is None:
        raise RuntimeError(f"Failed to read data from window: {col_off}, {row_off}, {width}, {height}")
    
    # Handle nodata and masking
    if as_masked and nodata is not None:
        array = array.astype(np.float32)
        array = np.where(array == nodata, np.nan, array)
    
    # Get georeferencing
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    
    # Adjust transform for window
    if window is not None:
        gt = list(geotransform)
        gt[0] += col_off * gt[1] + row_off * gt[2]  # X origin
        gt[3] += col_off * gt[4] + row_off * gt[5]  # Y origin
        geotransform = tuple(gt)
    
    # Create Affine transform
    transform = Affine.from_gdal(*geotransform)
    
    # Calculate bounds
    west = geotransform[0]
    north = geotransform[3]
    east = west + geotransform[1] * width
    south = north + geotransform[5] * height
    bounds = (west, south, east, north)
    
    # Build metadata
    metadata = {
        "transform": transform,
        "crs": projection,
        "nodata": nodata,
        "dtype": array.dtype,
        "bounds": bounds,
        "shape": array.shape,
        "format_metadata": {
            "driver": ds.GetDriver().ShortName if ds.GetDriver() else None,
            "band_count": ds.RasterCount,
            "metadata": ds.GetMetadata()
        }
    }
    
    ds = None  # Close dataset
    
    return array, metadata


def load_with_rasterio(
    file_path: str,
    band: int | list[int] | None = None,
    window: Window | None = None,
    masked: bool = True,
    indexes: Any = None,
    resampling: Any = None
) -> tuple[ArrayLike, dict[str, Any]]:
    """Load raster data using rasterio with modern Python interface.
    
    This function leverages rasterio's Pythonic API for efficient raster I/O,
    providing better performance for standard formats and seamless integration
    with numpy. It's the preferred method for most raster reading operations.
    
    Rasterio excels at handling GeoTIFFs, especially Cloud-Optimized GeoTIFFs,
    with efficient partial reads and overview access.
    
    Args:
        file_path: Path to raster file
        band: Band number(s) to read (1-indexed)
        window: Window for partial reads
        masked: Return numpy masked array
        indexes: Alternative to band parameter
        resampling: Resampling method for overviews
        
    Returns:
        Tuple of (data_array, metadata_dict) where metadata includes:
        - transform: Affine transformation
        - crs: Coordinate reference system
        - count: Number of bands
        - dtype: Data type
        - nodata: NoData value
        - bounds: Geographic bounds
        - profile: Complete rasterio profile
        
    Example:
        >>> # Efficient windowed read
        >>> data, meta = load_with_rasterio(
        >>>     "landsat_B5.tif",
        >>>     window=Window(1000, 1000, 512, 512)
        >>> )
        
    Note:
        Rasterio is preferred for standard formats and when
        advanced features like windowed reads are needed.
    """
    # Implementation adapted from io_rasterio.py:12 get_geo()
    import rasterio
    import numpy as np
    
    # Determine indexes to read
    if indexes is not None:
        band = indexes
    elif band is None:
        band = 1
    
    with rasterio.open(file_path) as src:
        # Read data with optional window
        if masked:
            if resampling is not None:
                data = src.read(band, window=window, masked=True, resampling=resampling)
            else:
                data = src.read(band, window=window, masked=True)
        else:
            if resampling is not None:
                data = src.read(band, window=window, resampling=resampling)
            else:
                data = src.read(band, window=window)
        
        # Get transform (adjusted for window if needed)
        if window:
            transform = src.window_transform(window)
        else:
            transform = src.transform
        
        # Get bounds
        if window:
            bounds = src.window_bounds(window)
        else:
            bounds = src.bounds
        
        # Build metadata
        metadata = {
            "transform": transform,
            "crs": src.crs,
            "count": src.count,
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
            "bounds": bounds,
            "shape": data.shape,
            "profile": src.profile.copy()
        }
        
        # Handle single band vs multi-band
        if isinstance(band, int) and data.ndim == 3 and data.shape[0] == 1:
            data = data[0]  # Remove band dimension for single band
    
    return data, metadata


def create_virtual_raster(
    file_paths: list[str],
    output_vrt: str | None = None,
    resolution: float | None = None,
    crs: CRS | str | None = None,
    extent: BBox | None = None
) -> str:
    """Create GDAL Virtual Raster (VRT) from multiple files.
    
    This function creates a virtual mosaic of multiple raster files without
    copying data, enabling efficient processing of large datasets. VRTs are
    essential for handling multi-tile datasets like VIIRS where data is
    distributed across multiple files.
    
    Virtual rasters allow the pipeline to treat multiple files as a single
    dataset, dramatically improving memory efficiency and processing speed.
    
    Args:
        file_paths: List of input raster paths
        output_vrt: Path for VRT file (temporary if None)
        resolution: Target resolution (uses first file if None)
        crs: Target CRS (uses first file if None)
        extent: Limit VRT to specific extent
        
    Returns:
        Path to created VRT file
        
    Example:
        >>> # Create mosaic of VIIRS tiles
        >>> vrt = create_virtual_raster(
        >>>     ["VNP46A2_h08v05.tif", "VNP46A2_h09v05.tif"],
        >>>     output_vrt="viirs_mosaic.vrt"
        >>> )
        >>> # Read mosaic as single dataset
        >>> data, meta = load_with_rasterio(vrt)
        
    Note:
        VRTs are particularly useful for temporal stacks and
        spatial mosaics in the Black Marble pipeline.
    """
    # New implementation for VRT creation
    from osgeo import gdal
    import tempfile
    import os
    
    if not file_paths:
        raise ValueError("No input files provided for VRT creation")
    
    # Generate temp file if no output path
    if output_vrt is None:
        temp_file = tempfile.NamedTemporaryFile(suffix=".vrt", delete=False)
        output_vrt = temp_file.name
        temp_file.close()
    
    # Build VRT options
    vrt_options = gdal.BuildVRTOptions(
        resolution='user' if resolution else 'average',
        xRes=resolution,
        yRes=resolution,
        outputBounds=extent,
        outputSRS=str(crs) if crs else None
    )
    
    # Create VRT
    vrt_ds = gdal.BuildVRT(output_vrt, file_paths, options=vrt_options)
    
    if vrt_ds is None:
        raise RuntimeError("Failed to create VRT")
    
    # Close dataset to write to disk
    vrt_ds = None
    
    return output_vrt


def read_metadata(
    file_path: str,
    metadata_type: Literal["all", "spatial", "bands", "tags"] = "all"
) -> dict[str, Any]:
    """Extract comprehensive metadata from raster file.
    
    This function reads metadata without loading raster data, providing
    essential information for processing decisions. It extracts spatial
    properties, band information, and embedded metadata tags.
    
    Metadata extraction is crucial for validation, compatibility checking,
    and processing parameter selection.
    
    Args:
        file_path: Path to raster file
        metadata_type: Type of metadata to extract:
            - 'all': Complete metadata
            - 'spatial': CRS, bounds, resolution
            - 'bands': Band count, names, data types
            - 'tags': Embedded metadata tags
            
    Returns:
        Metadata dictionary with requested information:
        - spatial: Transform, CRS, bounds, shape
        - bands: Count, descriptions, data types, nodata
        - tags: File-level and band-level tags
        - format: Format-specific metadata
        
    Example:
        >>> # Check file before processing
        >>> meta = read_metadata("landsat_scene.tif", "spatial")
        >>> print(f"Resolution: {meta['pixel_size']} {meta['units']}")
        >>> print(f"CRS: {meta['crs']}")
    """
    # New implementation for metadata extraction
    import rasterio
    
    metadata = {}
    
    with rasterio.open(file_path) as src:
        # Spatial metadata
        if metadata_type in ["all", "spatial"]:
            # Calculate pixel size
            pixel_x = abs(src.transform.a)
            pixel_y = abs(src.transform.e)
            
            metadata["spatial"] = {
                "transform": src.transform,
                "crs": src.crs,
                "bounds": src.bounds,
                "shape": (src.height, src.width),
                "pixel_size": (pixel_x, pixel_y),
                "units": src.crs.linear_units if src.crs else "unknown"
            }
        
        # Band metadata
        if metadata_type in ["all", "bands"]:
            band_info = []
            for i in range(1, src.count + 1):
                band_info.append({
                    "index": i,
                    "dtype": src.dtypes[i-1],
                    "nodata": src.nodatavals[i-1],
                    "description": src.descriptions[i-1] if src.descriptions else None
                })
            
            metadata["bands"] = {
                "count": src.count,
                "info": band_info
            }
        
        # Tags/metadata
        if metadata_type in ["all", "tags"]:
            metadata["tags"] = {
                "file": dict(src.tags()),
                "bands": {i: dict(src.tags(i)) for i in range(1, src.count + 1)}
            }
        
        # Format info
        metadata["format"] = {
            "driver": src.driver,
            "compression": src.compression if hasattr(src, 'compression') else None,
            "tiled": src.is_tiled,
            "interleave": src.interleaving.value if hasattr(src, 'interleaving') else None
        }
    
    return metadata


def setup_output_profile(
    template_path: str | None = None,
    template_meta: dict[str, Any] | None = None,
    modifications: dict[str, Any] | None = None,
    driver: str = "GTiff",
    dtype: str | None = None
) -> dict[str, Any]:
    """Setup rasterio profile for output file creation.
    
    This function creates proper output profiles for writing rasters,
    ensuring compatibility and optimal settings. It handles the complexity
    of rasterio profiles while providing sensible defaults.
    
    Proper output profiles are essential for creating valid, efficient
    output files that maintain georeferencing and optimize performance.
    
    Args:
        template_path: Path to template raster
        template_meta: Template metadata dictionary
        modifications: Profile modifications to apply
        driver: Output format driver
        dtype: Output data type
        
    Returns:
        Complete rasterio profile for output creation
        
    Example:
        >>> # Setup profile for processed output
        >>> profile = setup_output_profile(
        >>>     template_path="input.tif",
        >>>     modifications={
        >>>         'dtype': 'float32',
        >>>         'compress': 'deflate',
        >>>         'tiled': True
        >>>     }
        >>> )
        >>> with rasterio.open("output.tif", 'w', **profile) as dst:
        >>>     dst.write(processed_data)
        
    Note:
        This function ensures outputs maintain spatial reference
        while optimizing for the specific data type and use case.
    """
    # New implementation for output profile setup
    import rasterio
    
    # Start with template or empty profile
    if template_path:
        with rasterio.open(template_path) as src:
            profile = src.profile.copy()
    elif template_meta:
        profile = template_meta.copy()
    else:
        # Default profile
        profile = {
            'driver': driver,
            'width': 512,
            'height': 512,
            'count': 1,
            'dtype': 'float32',
            'crs': 'EPSG:4326',
            'transform': rasterio.transform.Affine.identity()
        }
    
    # Apply driver
    profile['driver'] = driver
    
    # Apply dtype if specified
    if dtype:
        profile['dtype'] = dtype
    
    # Apply modifications
    if modifications:
        profile.update(modifications)
    
    # Set sensible defaults for GTiff
    if driver == 'GTiff':
        # Add compression if not specified
        if 'compress' not in profile:
            profile['compress'] = 'deflate'
        
        # Add tiling for larger files
        if profile.get('width', 0) > 512 or profile.get('height', 0) > 512:
            if 'tiled' not in profile:
                profile['tiled'] = True
            if 'blockxsize' not in profile:
                profile['blockxsize'] = 256
            if 'blockysize' not in profile:
                profile['blockysize'] = 256
    
    return profile


def validate_raster_stack(
    file_paths: list[str],
    check_alignment: bool = True,
    check_resolution: bool = True,
    check_crs: bool = True,
    tolerance: float = 1e-6
) -> dict[str, Any]:
    """Validate multiple rasters for stacking compatibility.
    
    This function checks whether multiple raster files can be stacked for
    temporal analysis or multi-band processing. It identifies misalignments
    and incompatibilities that would prevent successful stacking.
    
    Stack validation is critical before temporal compositing to ensure
    all inputs are properly aligned and compatible.
    
    Args:
        file_paths: List of raster files to validate
        check_alignment: Verify spatial alignment
        check_resolution: Verify matching resolution
        check_crs: Verify matching CRS
        tolerance: Tolerance for alignment checks
        
    Returns:
        Validation results containing:
        - valid: Whether stack is valid
        - issues: List of compatibility issues
        - alignment_errors: Specific alignment problems
        - recommendations: How to fix issues
        - common_extent: Intersection of all extents
        
    Example:
        >>> # Validate before temporal composite
        >>> result = validate_raster_stack(viirs_files)
        >>> if not result['valid']:
        >>>     raise ValueError(f"Stack validation failed: {result['issues']}")
    """
    # New implementation for stack validation
    import rasterio
    import numpy as np
    
    if not file_paths:
        return {
            "valid": False,
            "issues": ["No files provided"],
            "alignment_errors": [],
            "recommendations": [],
            "common_extent": None
        }
    
    issues = []
    alignment_errors = []
    recommendations = []
    
    # Read metadata from all files
    metadata_list = []
    for fp in file_paths:
        try:
            with rasterio.open(fp) as src:
                metadata_list.append({
                    "file": fp,
                    "crs": src.crs,
                    "transform": src.transform,
                    "shape": (src.height, src.width),
                    "bounds": src.bounds,
                    "res": (abs(src.transform.a), abs(src.transform.e))
                })
        except Exception as e:
            issues.append(f"Cannot read {fp}: {e}")
            continue
    
    if not metadata_list:
        return {
            "valid": False,
            "issues": issues,
            "alignment_errors": [],
            "recommendations": ["Ensure all files are valid rasters"],
            "common_extent": None
        }
    
    # Use first file as reference
    ref_meta = metadata_list[0]
    
    # Check CRS
    if check_crs:
        for meta in metadata_list[1:]:
            if meta["crs"] != ref_meta["crs"]:
                issues.append(f"CRS mismatch: {meta['file']} has {meta['crs']}, expected {ref_meta['crs']}")
                recommendations.append("Reproject all files to common CRS")
    
    # Check resolution
    if check_resolution:
        for meta in metadata_list[1:]:
            res_diff_x = abs(meta["res"][0] - ref_meta["res"][0])
            res_diff_y = abs(meta["res"][1] - ref_meta["res"][1])
            if res_diff_x > tolerance or res_diff_y > tolerance:
                issues.append(f"Resolution mismatch: {meta['file']} has {meta['res']}, expected {ref_meta['res']}")
                recommendations.append("Resample all files to common resolution")
    
    # Check alignment
    if check_alignment:
        for meta in metadata_list[1:]:
            # Check if grids align
            # Compare origins modulo resolution
            origin_diff_x = (meta["transform"].c - ref_meta["transform"].c) % ref_meta["res"][0]
            origin_diff_y = (meta["transform"].f - ref_meta["transform"].f) % ref_meta["res"][1]
            
            if abs(origin_diff_x) > tolerance or abs(origin_diff_y) > tolerance:
                alignment_errors.append({
                    "file": meta["file"],
                    "offset": (origin_diff_x, origin_diff_y)
                })
                recommendations.append("Align all rasters to common grid")
    
    # Calculate common extent (intersection)
    bounds_list = [m["bounds"] for m in metadata_list]
    common_extent = (
        max(b.left for b in bounds_list),    # max left
        max(b.bottom for b in bounds_list),  # max bottom  
        min(b.right for b in bounds_list),   # min right
        min(b.top for b in bounds_list)      # min top
    )
    
    # Check if common extent is valid
    if common_extent[2] <= common_extent[0] or common_extent[3] <= common_extent[1]:
        issues.append("No spatial overlap between files")
        common_extent = None
    
    valid = len(issues) == 0 and len(alignment_errors) == 0
    
    return {
        "valid": valid,
        "issues": issues,
        "alignment_errors": alignment_errors,
        "recommendations": list(set(recommendations)),  # Remove duplicates
        "common_extent": common_extent
    }


def create_memory_efficient_reader(
    file_path: str,
    chunk_size: tuple[int, int] = (1024, 1024),
    overlap: int = 0
) -> Callable:
    """Create memory-efficient reader for large rasters.
    
    This function returns a reader that processes large rasters in chunks,
    enabling processing of datasets larger than available memory. It's
    essential for operational processing of large satellite scenes.
    
    The reader handles chunk boundaries and overlap, ensuring seamless
    processing while maintaining memory efficiency.
    
    Args:
        file_path: Path to large raster
        chunk_size: Size of chunks to read
        overlap: Pixel overlap between chunks
        
    Returns:
        Reader function that yields (chunk_data, chunk_window) tuples
        
    Example:
        >>> # Process large raster in chunks
        >>> reader = create_memory_efficient_reader(
        >>>     "large_mosaic.tif",
        >>>     chunk_size=(2048, 2048)
        >>> )
        >>> for chunk, window in reader():
        >>>     processed = process_chunk(chunk)
        >>>     write_chunk(processed, window)
        
    Note:
        Overlap is necessary for operations like convolution
        that require neighborhood context.
    """
    # New implementation for chunked reading
    import rasterio
    from rasterio.windows import Window
    
    def chunk_reader():
        """Generator function that yields chunks."""
        with rasterio.open(file_path) as src:
            height, width = src.height, src.width
            chunk_height, chunk_width = chunk_size
            
            # Calculate number of chunks
            n_chunks_y = (height + chunk_height - 1) // chunk_height
            n_chunks_x = (width + chunk_width - 1) // chunk_width
            
            for row in range(n_chunks_y):
                for col in range(n_chunks_x):
                    # Calculate window with overlap
                    row_start = max(0, row * chunk_height - overlap)
                    col_start = max(0, col * chunk_width - overlap)
                    
                    # Calculate actual size (handle edges)
                    actual_height = min(chunk_height + 2 * overlap, height - row_start)
                    actual_width = min(chunk_width + 2 * overlap, width - col_start)
                    
                    # Create window
                    window = Window(
                        col_off=col_start,
                        row_off=row_start,
                        width=actual_width,
                        height=actual_height
                    )
                    
                    # Read chunk
                    chunk_data = src.read(window=window)
                    
                    # Remove band dimension if single band
                    if chunk_data.shape[0] == 1:
                        chunk_data = chunk_data[0]
                    
                    yield chunk_data, window
    
    return chunk_reader