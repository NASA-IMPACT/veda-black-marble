"""
Raster I/O utilities:
  - get_geo: read a raster band into a NumPy array + geo info
  - save_image: write a single-band GeoTIFF
  - save_geotiff_single: uint16 single-band GeoTIFF writer
  - save_geotiff_rgb: 3-band RGB GeoTIFF writer
  - scale_image: clamp & scale array to uint8
"""
# Imports
from osgeo import gdal
import numpy as np
from typing import Union

def get_geo(filepath: str, band: int) -> tuple[np.ndarray, tuple[float, ...], str]:
    """
    Reads a specific band from a raster file and returns its data along with georeferencing information.

    Args:
        filepath (str): The path to the raster file (e.g., GeoTIFF).
        band (int): The band number to read (1-based index).

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: A 2D NumPy array representing the raster band data.
            - tuple: The GDAL GeoTransform tuple (6 elements).
            - str: The projection information in WKT format.

    Raises:
        RuntimeError: If the GDAL dataset cannot be opened.
    """
    ds = gdal.Open(filepath, gdal.GA_ReadOnly)
    if ds is None:
         raise RuntimeError(f"Could not open raster file: {filepath}")
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    array = ds.GetRasterBand(band).ReadAsArray(0, 0, cols, rows)

    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjectionRef()
    ds = None
    return array, geotransform, projection

def save_image(array: np.ndarray, geotransform: tuple[float, ...], projection: str, gdal_type: int, out_path: str) -> None:
    """
    Saves a NumPy array as a single-band GeoTIFF file.

    Args:
        array (numpy.ndarray): The 2D NumPy array containing the image data.
        geotransform (tuple): The GDAL GeoTransform tuple (6 elements).
        projection (str): The projection information in WKT format.
        gdal_type (int): The GDAL data type constant (e.g., gdal.GDT_Byte, gdal.GDT_UInt16).
        out_path (str): The file path where the GeoTIFF will be saved.

    Raises:
        RuntimeError: If the GDAL 'GTiff' driver is not found.
    """
    driver = gdal.GetDriverByName('GTiff')
    if driver is None:
        raise RuntimeError("GDAL driver 'GTiff' not found")
    rows, cols = array.shape
    dst = driver.Create(out_path, cols, rows, 1, gdal_type)
    dst.SetProjection(projection)
    dst.SetGeoTransform(geotransform)

    band = dst.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.WriteArray(array, 0, 0)
    dst = None

def save_geotiff_rgb(out_path: str, r: np.ndarray, g: np.ndarray, b: np.ndarray, projection: str, geotransform: tuple[float, ...]) -> None:
    rows, cols = r.shape
    driver = gdal.GetDriverByName('GTiff')
    dst = driver.Create(out_path, cols, rows, 3, gdal.GDT_Byte)
    dst.SetProjection(projection)
    dst.SetGeoTransform(geotransform)
    dst.GetRasterBand(1).SetNoDataValue(0)
    dst.GetRasterBand(2).SetNoDataValue(0)
    dst.GetRasterBand(3).SetNoDataValue(0)
    dst.GetRasterBand(1).WriteArray(r)
    dst.GetRasterBand(2).WriteArray(g)
    dst.GetRasterBand(3).WriteArray(b)
    dst = None

def save_geotiff_single(out_path: str, array: np.ndarray, projection: str, geotransform: tuple[float, ...]) -> None:
    """
    Saves a NumPy array as a single-band GeoTIFF file.

    Args:
        out_path (str): The file path where the GeoTIFF will be saved.
        array (numpy.ndarray): The 2D NumPy array containing the image data.
        projection (str): The projection information in WKT format.
        geotransform (tuple): The GDAL GeoTransform tuple (6 elements).

    Raises:
        RuntimeError: If the GDAL 'GTiff' driver is not found.
    """
    rows, cols = array.shape
    driver = gdal.GetDriverByName('GTiff')
    if driver is None:
        raise RuntimeError("GDAL driver 'GTiff' not found")
    
    # Determine data type based on array dtype
    if array.dtype == np.float32 or array.dtype == np.float64:
        gdal_dtype = gdal.GDT_Float32
        # Convert NaN to a specific NoData value for better compatibility
        nodata_value = -9999.0
        array_copy = array.copy()
        array_copy[np.isnan(array)] = nodata_value
    else:
        gdal_dtype = gdal.GDT_UInt16
        nodata_value = 0
        array_copy = array
    
    dst = driver.Create(out_path, cols, rows, 1, gdal_dtype)
    dst.SetProjection(projection)
    dst.SetGeoTransform(geotransform)
    band = dst.GetRasterBand(1)
    band.SetNoDataValue(nodata_value)
    band.WriteArray(array_copy)
    dst = None

def scale_image(input_array: np.ndarray, min_value: Union[int, float], max_value: Union[int, float]) -> np.ndarray:
    """
    Clamps the values of a NumPy array to a specified range [min_value, max_value]
    and then linearly scales the result to an unsigned 8-bit integer range [0, 255].

    Args:
        input_array (numpy.ndarray): The input NumPy array to scale.
        min_value (numeric): The minimum value to clamp to. Values below this will be set to this value.
        max_value (numeric): The maximum value to clamp to. Values above this will be set to this value.

    Returns:
        numpy.ndarray: A new NumPy array with dtype 'uint8', containing the clamped and scaled values.
                       Returns an array of zeros if min_value equals max_value after clamping.
    """
    arr = np.where(input_array > max_value, max_value, input_array)
    arr = np.where(arr < min_value, min_value, arr)

    # Handle potential division by zero if min == max after clamping
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max == arr_min:
        # Return an array of zeros or a constant value, depending on desired behavior
        return np.zeros_like(arr, dtype='uint8')

    scaled = ((arr - arr_min) * (255.0 / (arr_max - arr_min))).astype('uint8')
    return scaled