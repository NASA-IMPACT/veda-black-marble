# Black Marble Command Line Interface

The Black Marble pipeline provides a comprehensive CLI for processing nighttime lights data with advanced temporal compositing and quality control features.

## Basic Usage

```bash
blackmarble run --lat1 40.7 --lon1 -74.0 --lat2 40.6 --lon2 -73.9 \
                --year 2023 --month 6 --day 15 \
                --earthdata-token YOUR_TOKEN
```

## Complete Parameter Reference

### Required Geographic Parameters

#### `--lat1` (required)
- **Type**: Float
- **Description**: Northern latitude boundary of the area of interest
- **Example**: `--lat1 40.7128`
- **Range**: -90 to 90 degrees

#### `--lon1` (required)
- **Type**: Float  
- **Description**: Western longitude boundary of the area of interest
- **Example**: `--lon1 -74.0060`
- **Range**: -180 to 180 degrees

#### `--lat2` (required)
- **Type**: Float
- **Description**: Southern latitude boundary of the area of interest
- **Example**: `--lat2 40.6`
- **Range**: -90 to 90 degrees

#### `--lon2` (required)
- **Type**: Float
- **Description**: Eastern longitude boundary of the area of interest  
- **Example**: `--lon2 -73.9`
- **Range**: -180 to 180 degrees

### Required Temporal Parameters

#### `--year` (required)
- **Type**: Integer
- **Description**: Target year for data acquisition
- **Example**: `--year 2023`
- **Range**: 2012 to present (VIIRS availability)

#### `--month` (required)
- **Type**: Integer
- **Description**: Target month for data acquisition
- **Example**: `--month 6`
- **Range**: 1-12

#### `--day` (required)
- **Type**: Integer
- **Description**: Target day for data acquisition
- **Example**: `--day 15`
- **Range**: 1-31 (valid for given month)

### Authentication

#### `--earthdata-token` / `-t`
- **Type**: String
- **Description**: NASA Earthdata authentication token
- **Environment Variable**: `EARTHDATA_TOKEN`
- **Example**: `--earthdata-token abc123def456`
- **Note**: Can be set via environment variable to avoid command line exposure

### Output Configuration

#### `--output-dir` / `-o`
- **Type**: String
- **Default**: `"outputs"`
- **Description**: Base directory for all output and temporary files
- **Example**: `--output-dir /data/blackmarble/outputs`

#### `--upload-bucket` / `-b`
- **Type**: String (optional)
- **Description**: S3 bucket name for uploading final COG output
- **Example**: `--upload-bucket my-bucket`
- **Requires**: AWS credentials configured

#### `--upload-key` / `-k`
- **Type**: String (optional)
- **Description**: S3 key (path/filename) for uploaded file
- **Example**: `--upload-key processed/nyc_2023_06_15.tif`

### Temporal Processing Parameters (Phase 1)

#### `--use-advanced-gap-filling`
- **Type**: Boolean flag
- **Default**: False
- **Description**: Enable scipy-based nearest neighbor gap filling for higher quality results. Uses Euclidean distance transform to find the nearest valid pixel for each gap.
- **Example**: `--use-advanced-gap-filling`
- **Impact**: Significantly improves quality in areas with data gaps

#### `--composite-method`
- **Type**: Choice
- **Default**: `"median"`
- **Choices**: `median`, `mean`, `max`, `min`, `percentile`
- **Description**: Method for combining multiple temporal observations
  - `median`: Robust to outliers, good for removing clouds
  - `mean`: Smooth average of all observations
  - `max`: Maximum value (brightest lights)
  - `min`: Minimum value (darkest stable lights)
  - `percentile`: Custom percentile (requires --composite-percentile)
- **Example**: `--composite-method max`

#### `--composite-percentile`
- **Type**: Integer (0-100)
- **Default**: None
- **Description**: Percentile value when using percentile composite method
- **Example**: `--composite-percentile 85`
- **Required when**: `--composite-method percentile`

#### `--min-valid-observations`
- **Type**: Integer
- **Default**: 1
- **Description**: Minimum number of valid observations required for a pixel to be included in the composite
- **Example**: `--min-valid-observations 3`
- **Impact**: Higher values increase quality but may create more gaps

#### `--composite-months`
- **Type**: Integer
- **Default**: 12
- **Description**: Number of months to look back for temporal compositing
- **Example**: `--composite-months 6`
- **Range**: 1-24

### Scene Merging Parameters (Phase 2)

#### `--blend-method`
- **Type**: Choice
- **Default**: `"feather"`
- **Choices**: `feather`, `max`, `min`, `mean`
- **Description**: Method for blending overlapping scenes
  - `feather`: Smooth transitions using distance-based weights (requires scipy)
  - `max`: Take maximum value at each pixel
  - `min`: Take minimum value at each pixel
  - `mean`: Average overlapping pixels
- **Example**: `--blend-method feather`

#### `--feather-distance`
- **Type**: Integer
- **Default**: 10
- **Description**: Distance in pixels for feather blending transitions
- **Example**: `--feather-distance 20`
- **Range**: 1-50 (larger = smoother but slower)
- **Only used when**: `--blend-method feather`

### Quality Control Parameters (Phase 3)

#### `--cloud-mask-strategy`
- **Type**: Choice
- **Default**: `"conservative"`
- **Choices**: `conservative`, `permissive`
- **Description**: Strategy for cloud masking
  - `conservative`: Masks more pixels, safer but may lose valid data
  - `permissive`: Keeps more pixels, may include some clouds
- **Example**: `--cloud-mask-strategy permissive`

#### `--qa-dilation-radius`
- **Type**: Integer
- **Default**: 3
- **Description**: Radius in pixels to dilate quality masks (clouds, shadows)
- **Example**: `--qa-dilation-radius 5`
- **Impact**: Larger values provide more buffer around clouds

#### `--max-cloud-cover`
- **Type**: Float (0-100)
- **Default**: None (no filtering)
- **Description**: Maximum acceptable cloud cover percentage for scenes
- **Example**: `--max-cloud-cover 20`
- **Impact**: Filters out heavily clouded scenes

#### `--min-valid-pixels`
- **Type**: Float (0-100)
- **Default**: 10.0
- **Description**: Minimum percentage of valid pixels required in output
- **Example**: `--min-valid-pixels 25`
- **Impact**: Quality threshold for final output

### Processing Configuration (Phase 4)

#### `--water-threshold`
- **Type**: Float
- **Default**: 0.0
- **Description**: NDWI threshold for water body detection
- **Example**: `--water-threshold 0.1`
- **Range**: -1.0 to 1.0

#### `--ndui-floor`
- **Type**: Float
- **Default**: 0.02
- **Description**: Minimum NDUI value threshold for urban detection
- **Example**: `--ndui-floor 0.05`
- **Range**: 0.0 to 1.0

#### `--road-enhancement-major`
- **Type**: Float
- **Default**: 1.3
- **Description**: Enhancement factor for major roads in NDUI calculation
- **Example**: `--road-enhancement-major 1.5`
- **Range**: 1.0 to 2.0

#### `--road-enhancement-minor`
- **Type**: Float
- **Default**: 1.1
- **Description**: Enhancement factor for minor roads in NDUI calculation
- **Example**: `--road-enhancement-minor 1.2`
- **Range**: 1.0 to 2.0

### Output Options (Phase 5)

#### `--cog-profile`
- **Type**: Choice
- **Default**: `"deflate"`
- **Choices**: `deflate`, `lzw`, `webp`, `zstd`
- **Description**: Cloud-Optimized GeoTIFF compression profile
  - `deflate`: Good compression, widely compatible
  - `lzw`: Lossless, good for scientific data
  - `webp`: Lossy, smaller files
  - `zstd`: Modern compression, good ratio
- **Example**: `--cog-profile lzw`

#### `--colormap`
- **Type**: Choice
- **Default**: `"inferno"`
- **Choices**: `inferno`, `viridis`, `plasma`, `magma`, `cividis`
- **Description**: Colormap for visualization
- **Example**: `--colormap viridis`

#### `--no-parallel`
- **Type**: Boolean flag
- **Default**: False (parallelization enabled)
- **Description**: Disable parallel processing
- **Example**: `--no-parallel`
- **Use when**: Debugging or limited resources

#### `--quiet`
- **Type**: Boolean flag
- **Default**: False (verbose output)
- **Description**: Reduce output verbosity
- **Example**: `--quiet`

## Common Use Cases

### High Quality Urban Mapping
```bash
blackmarble run --lat1 40.7 --lon1 -74.0 --lat2 40.6 --lon2 -73.9 \
                --year 2023 --month 6 --day 15 \
                --use-advanced-gap-filling \
                --composite-method median \
                --min-valid-observations 5 \
                --blend-method feather \
                --cloud-mask-strategy conservative \
                --qa-dilation-radius 5
```

### Quick Processing (Lower Quality)
```bash
blackmarble run --lat1 40.7 --lon1 -74.0 --lat2 40.6 --lon2 -73.9 \
                --year 2023 --month 6 --day 15 \
                --composite-months 3 \
                --composite-method max \
                --blend-method max \
                --cloud-mask-strategy permissive
```

### Maximum Brightness Composite
```bash
blackmarble run --lat1 40.7 --lon1 -74.0 --lat2 40.6 --lon2 -73.9 \
                --year 2023 --month 6 --day 15 \
                --composite-method max \
                --min-valid-observations 1 \
                --water-threshold -0.1 \
                --ndui-floor 0.01
```

### Custom Percentile with Strict Quality
```bash
blackmarble run --lat1 40.7 --lon1 -74.0 --lat2 40.6 --lon2 -73.9 \
                --year 2023 --month 6 --day 15 \
                --composite-method percentile \
                --composite-percentile 85 \
                --min-valid-observations 10 \
                --max-cloud-cover 15 \
                --min-valid-pixels 80
```

## Environment Variables

The following environment variables can be used:

- `EARTHDATA_TOKEN`: NASA Earthdata authentication token
- `AWS_ACCESS_KEY_ID`: AWS access key for S3 uploads
- `AWS_SECRET_ACCESS_KEY`: AWS secret key for S3 uploads
- `AWS_SESSION_TOKEN`: AWS session token (if using temporary credentials)

## Exit Codes

- `0`: Success
- `1`: General error (check error message)
- `2`: Invalid parameters
- `3`: Authentication failure
- `4`: Data download failure
- `5`: Processing failure
- `6`: Upload failure

## Getting Help

```bash
# Show all available options
blackmarble run --help

# Show version
blackmarble --version
```