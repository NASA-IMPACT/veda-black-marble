# Black Marble Pipeline Walkthrough (`docker/bm.py`)

This document outlines the high-level workflow of the Black Marble processing pipeline as implemented in the `docker/bm.py` script. The script takes geographic coordinates (bounding box) and a date as input (via environment variables) and produces a Cloud-Optimized GeoTIFF (COG) representing an urban index (NDUI), which is then uploaded to an S3 bucket.

## Pipeline Steps

1.  **Initialization:**
    *   Reads configuration from environment variables:
        *   Bounding box coordinates (`LAT1`, `LONG1`, `LAT2`, `LONG2`)
        *   Date (`YEAR`, `MONTH`, `DAY`)
        *   AWS Credentials (`AWS_ACCESS_KEY`, `AWS_ACCESS_SECRET`, `AWS_WEB_IDENTITY_TOKEN_FILE`, `AWS_WEB_IDENTITY_TOKEN`)
        *   NASA Earthdata Login Token (`EARTH_DATA_TOKEN`)
        *   S3 Upload details (`UPLOAD_BUCKET`, `UPLOAD_FILE`)
    *   Loads necessary Python libraries (GDAL, Rasterio, NumPy, Boto3, OSMnx, etc.).
    *   Loads helper functions for raster manipulation, data download, etc.

2.  **Landsat Data Acquisition & Initial Processing:**
    *   **Determine Path/Row:** Uses the input bounding box coordinates and a WRS-2 shapefile (`WRS2_descending_0`) to identify the required Landsat Path/Row tiles covering the area of interest (AOI).
    *   **Download Data:** For each required Path/Row:
        *   Finds the Landsat 8/9 Collection 2 Level-2 scene directory on AWS OpenData (usgs-landsat S3 bucket) corresponding to the input date.
        *   Downloads Surface Reflectance (SR) bands B3 (Green), B4 (Red), B5 (NIR), and Quality Assessment (QA) bands (Pixel, Radsat, Aerosol) for the target date.
        *   **Compositing:** Downloads the same bands for previous months (up to `composite_history_months`, default 12) to use for filling gaps later.
    *   **Merge Tiles:** If the AOI spans multiple Landsat tiles, merges the downloaded tiles for each band and date into single files using `gdal.Warp`. Creates separate merged files for the target date and each composite date.

3.  **Landsat Data Cropping & Masking:**
    *   **Convert Coordinates:** Converts the input Lat/Lon bounding box to the appropriate UTM zone coordinates.
    *   **Crop:** Crops each merged Landsat band (B3, B4, B5, QA bands for all dates) to the UTM bounding box extent using `gdal.Translate`. This focuses processing on the precise AOI.
    *   **Masking:** For each date's set of cropped bands:
        *   Reads the QA Pixel band and creates a mask for clouds/cloud shadows.
        *   Reads the QA Aerosol band and creates a mask for high aerosol levels.
        *   Reads the QA Radsat band and creates masks for saturated pixels in B3, B4, B5.
        *   Applies the combined cloud, aerosol, and saturation masks to the corresponding B3, B4, B5 bands, setting masked pixels to 0 or NaN.

4.  **Landsat Compositing:**
    *   Stacks the masked B3, B4, and B5 bands from all downloaded dates (target date + historical composites).
    *   Calculates the median pixel value across the time series for each band (B3, B4, B5) using `np.nanmedian`. This fills gaps caused by clouds/masking in the target date image with data from other dates.
    *   Saves the final composite B3, B4, B5 bands as intermediate GeoTIFFs.

5.  **Reprojection:**
    *   Converts the composite B3, B4, B5 GeoTIFFs from their native UTM projection to WGS84 (EPSG:4326) using `gdal.Warp`. This aligns them with the NTL and OSM data.

6.  **VIIRS Nighttime Lights (NTL) Data Acquisition & Processing:**
    *   **Determine Tile:** Calculates the VIIRS VNP46A2 tile coordinates (h, v) covering the AOI.
    *   **Download:** Uses the NASA LAADS DAAC API and the Earthdata token to find and download the VNP46A2 HDF5 file for the target date and tile.
    *   **Convert to GeoTIFF:** Extracts the DNB radiance, Lunar Irradiance, and Mandatory Quality Flag layers from the HDF5 file and converts them to individual GeoTIFFs using `gdal.Translate`, setting the correct geographic bounds.
    *   **Subset & Resample:** Crops the NTL GeoTIFFs (radiance, lunar irradiance, quality) to the AOI bounding box and resamples them to a target resolution using `gdal.Translate` and `gdal.Warp`.
    *   **Adjust NTL:** Creates an adjusted NTL band by subtracting lunar irradiance from the DNB radiance and applying the mandatory quality flags (keeping only high-quality, cloud-free pixels).

7.  **OpenStreetMap (OSM) Data Acquisition & Processing:**
    *   **Download:** Uses `osmnx` to download road network data (primary, secondary, tertiary, motorway, residential, trunk, service, etc.) within the AOI bounding box.
    *   **Rasterize:** Burns the road vector data into a raster matching the NTL raster's dimensions and projection. Different road types can optionally be assigned different pixel values. Creates an intermediate OSM GeoTIFF.

8.  **Data Alignment & Resampling:**
    *   Resamples the WGS84 Landsat bands (B3, B4, B5), the adjusted NTL band, and the OSM road raster to a final, common spatial resolution (e.g., 0.0001 degrees) using `gdal.Warp`. This ensures all datasets align pixel-to-pixel.

9.  **Index Calculation:**
    *   **Rescale Landsat:** Applies scaling factors to the Landsat B3, B4, B5 bands to convert digital numbers to surface reflectance values (typically 0-1).
    *   **Calculate NDVI:** Computes the Normalized Difference Vegetation Index: `(B5 - B4) / (B5 + B4)`.
    *   **Calculate NDWI:** Computes the Normalized Difference Water Index: `(B3 - B5) / (B3 + B5)`.
    *   **Scale NTL:** Scales the adjusted NTL radiance values (e.g., dividing by 10 and capping values).
    *   **Calculate NDUI:** Computes the Normalized Difference Urban Index: `(NTL_scaled - NDVI) / (NTL_scaled + NDVI)`. Applies a water mask using NDWI (setting NDUI to -1 where water is detected) and potentially adjusts values based on proximity to roads (from the OSM raster). Normalizes the result (e.g., to 0-1 range).

10. **Final Output Generation:**
    *   **Colorize:** Applies a colormap (e.g., 'inferno', loaded from `.npy` files) to the final NDUI raster based on its values, scaling the range for visual representation.
    *   **Create COG:** Saves the colorized RGB image as a Cloud-Optimized GeoTIFF (COG) using `rio-cogeo`. This format is optimized for cloud storage and web access. The output file is typically named `outputs/final/ndui-co.tif`.

11. **Upload to S3:**
    *   Uses `boto3` and the provided AWS credentials to upload the final COG (`outputs/final/ndui-co.tif`) to the specified `UPLOAD_BUCKET` with the specified `UPLOAD_FILE` key.

This completes the pipeline, resulting in a processed urban index map stored in S3.