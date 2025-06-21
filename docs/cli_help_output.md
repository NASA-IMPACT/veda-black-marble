# Black Marble CLI Help Output

This document shows what users will see when they run `blackmarble run --help` with the enhanced CLI.

```
$ blackmarble run --help

 Usage: blackmarble run [OPTIONS]                                              
                                                                                
 Run the Black Marble processing pipeline with advanced temporal processing.    
                                                                                
 This command processes nighttime lights data for a specified area and date,   
 with options for advanced gap filling, temporal compositing, and quality      
 control.                                                                       
                                                                                
 Example:                                                                       
     blackmarble run --lat1 40.7 --lon1 -74.0 --lat2 40.6 --lon2 -73.9 \       
                    --year 2023 --month 6 --day 15 \                           
                    --use-advanced-gap-filling \                                
                    --composite-method median \                                 
                    --min-valid-observations 3                                  

╭─ Required Options ───────────────────────────────────────────────────────────╮
│ *  --lat1         FLOAT    Northern latitude boundary [required]             │
│ *  --lon1         FLOAT    Western longitude boundary [required]             │
│ *  --lat2         FLOAT    Southern latitude boundary [required]             │
│ *  --lon2         FLOAT    Eastern longitude boundary [required]             │
│ *  --year         INTEGER  Target year for data [required]                   │
│ *  --month        INTEGER  Target month for data [required]                  │
│ *  --day          INTEGER  Target day for data [required]                    │
╰──────────────────────────────────────────────────────────────────────────────╯

╭─ Authentication ─────────────────────────────────────────────────────────────╮
│ --earthdata-token  -t  TEXT  NASA Earthdata token                            │
│                              (or set EARTHDATA_TOKEN env var)                │
│                              [default: None]                                  │
╰──────────────────────────────────────────────────────────────────────────────╯

╭─ Output Configuration ───────────────────────────────────────────────────────╮
│ --output-dir  -o  TEXT  Base directory for outputs [default: outputs]        │
│ --upload-bucket   TEXT  S3 bucket name for uploading final COG               │
│ --upload-key      TEXT  S3 key (path/filename) for uploaded file             │
╰──────────────────────────────────────────────────────────────────────────────╯

╭─ Temporal Processing ────────────────────────────────────────────────────────╮
│ --use-advanced-gap-filling                                                   │
│     Enable scipy-based nearest neighbor gap filling for higher quality       │
│     results. Uses Euclidean distance transform to find the nearest valid     │
│     pixel for each gap.                                                      │
│     [default: False]                                                         │
│                                                                              │
│ --composite-method [median|mean|max|min|percentile]                         │
│     Method for combining multiple temporal observations:                     │
│     • median: Robust to outliers, good for removing clouds                  │
│     • mean: Smooth average of all observations                              │
│     • max: Maximum value (brightest lights)                                 │
│     • min: Minimum value (darkest stable lights)                            │
│     • percentile: Custom percentile (requires --composite-percentile)        │
│     [default: median]                                                        │
│                                                                              │
│ --composite-percentile INTEGER RANGE [0<=x<=100]                            │
│     Percentile value (0-100) when using percentile composite method          │
│     [default: None]                                                          │
│                                                                              │
│ --min-valid-observations INTEGER RANGE [x>=1]                               │
│     Minimum number of valid observations required for a pixel.               │
│     Higher values increase quality but may create more gaps.                 │
│     [default: 1]                                                             │
│                                                                              │
│ --composite-months INTEGER RANGE [1<=x<=24]                                 │
│     Number of months to look back for temporal compositing                   │
│     [default: 12]                                                            │
╰──────────────────────────────────────────────────────────────────────────────╯

╭─ Scene Merging ──────────────────────────────────────────────────────────────╮
│ --blend-method [feather|max|min|mean]                                       │
│     Method for blending overlapping scenes:                                  │
│     • feather: Smooth transitions using distance-based weights               │
│     • max: Take maximum value at each pixel                                 │
│     • min: Take minimum value at each pixel                                 │
│     • mean: Average overlapping pixels                                       │
│     [default: feather]                                                       │
│                                                                              │
│ --feather-distance INTEGER RANGE [1<=x<=50]                                 │
│     Distance in pixels for feather blending transitions.                     │
│     Larger values create smoother but slower transitions.                    │
│     [default: 10]                                                            │
╰──────────────────────────────────────────────────────────────────────────────╯

╭─ Quality Control ────────────────────────────────────────────────────────────╮
│ --cloud-mask-strategy [conservative|permissive]                             │
│     Cloud masking strategy:                                                  │
│     • conservative: Masks more pixels, safer but may lose valid data         │
│     • permissive: Keeps more pixels, may include some clouds                │
│     [default: conservative]                                                  │
│                                                                              │
│ --qa-dilation-radius INTEGER                                                │
│     Radius in pixels to dilate quality masks (clouds, shadows)              │
│     [default: 3]                                                             │
│                                                                              │
│ --max-cloud-cover FLOAT RANGE [0<=x<=100]                                   │
│     Maximum acceptable cloud cover percentage for scenes                     │
│     [default: None]                                                          │
│                                                                              │
│ --min-valid-pixels FLOAT RANGE [0<=x<=100]                                  │
│     Minimum percentage of valid pixels required in output                    │
│     [default: 10.0]                                                          │
╰──────────────────────────────────────────────────────────────────────────────╯

╭─ Processing Configuration ───────────────────────────────────────────────────╮
│ --water-threshold FLOAT                                                      │
│     NDWI threshold for water body detection                                  │
│     [default: 0.0]                                                           │
│                                                                              │
│ --ndui-floor FLOAT                                                          │
│     Minimum NDUI value threshold for urban detection                         │
│     [default: 0.02]                                                          │
│                                                                              │
│ --road-enhancement-major FLOAT                                              │
│     Enhancement factor for major roads in NDUI calculation                   │
│     [default: 1.3]                                                           │
│                                                                              │
│ --road-enhancement-minor FLOAT                                              │
│     Enhancement factor for minor roads in NDUI calculation                   │
│     [default: 1.1]                                                           │
╰──────────────────────────────────────────────────────────────────────────────╯

╭─ Output Options ─────────────────────────────────────────────────────────────╮
│ --cog-profile [deflate|lzw|webp|zstd]                                       │
│     Cloud-Optimized GeoTIFF compression profile                              │
│     [default: deflate]                                                       │
│                                                                              │
│ --colormap [inferno|viridis|plasma|magma|cividis]                          │
│     Colormap for visualization                                               │
│     [default: inferno]                                                       │
│                                                                              │
│ --no-parallel                                                               │
│     Disable parallel processing                                              │
│     [default: False]                                                         │
│                                                                              │
│ --quiet                                                                     │
│     Reduce output verbosity                                                  │
│     [default: False]                                                         │
╰──────────────────────────────────────────────────────────────────────────────╯

╭─ Other Options ──────────────────────────────────────────────────────────────╮
│ --help  Show this message and exit.                                         │
╰──────────────────────────────────────────────────────────────────────────────╯

 See full documentation at: https://github.com/nasa-veda/blackmarble          
```

## Key Features of the Enhanced Help:

1. **Organized Panels**: Options are grouped into logical sections with rich formatting
2. **Clear Descriptions**: Each parameter has detailed help text
3. **Value Constraints**: Shows valid ranges and choices
4. **Examples**: Includes a working example command
5. **Defaults**: Shows default values for all optional parameters
6. **Rich Formatting**: Uses Unicode boxes and formatting for clarity

## Interactive Features:

When users make errors, they get helpful feedback:

```bash
$ blackmarble run --composite-method percentile --lat1 40 --lon1 -74 ...

Error: --composite-percentile is required when using percentile method
```

## Tab Completion:

Users can tab-complete options:
```bash
$ blackmarble run --composite-method <TAB>
max      mean     median   min      percentile

$ blackmarble run --blend-method <TAB>  
feather  max      mean     min
```