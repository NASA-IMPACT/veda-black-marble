# veda-black-marble
Veda Implementation of Black Marble Data Processing

## Docker image build and push commands
```
cd docker
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 018923174646.dkr.ecr.us-west-2.amazonaws.com
docker build --platform linux/amd64 -t black-marble .
docker tag black-marble:latest 018923174646.dkr.ecr.us-west-2.amazonaws.com/black-marble:latest
docker push 018923174646.dkr.ecr.us-west-2.amazonaws.com/black-marble:latest
```

## Local Usage

You can run the pipeline locally (no Docker) once dependencies are installed.

1. Set environment variables for NASA Earthdata and AWS (if downloading Landsat data):
   ```bash
   export EARTH_DATA_TOKEN="<YOUR_EARTHDATA_TOKEN>"
   export AWS_ACCESS_KEY_ID="<YOUR_AWS_KEY_ID>"
   export AWS_SECRET_ACCESS_KEY="<YOUR_AWS_SECRET>"
   # Optional: AWS_SESSION_TOKEN if using temporary credentials
   ```

2. Set geographic and date parameters:
   ```bash
   export LAT1=<lat1> LONG1=<lon1>
   export LAT2=<lat2> LONG2=<lon2>
   export YEAR=<year> MONTH=<month> DAY=<day>
   ```

3. Run the pipeline:
   Option A: as a module
   ```bash
   python -m blackmarble.pipeline
   ```

   Option B: using the CLI wrapper
   ```bash
   python cli.py
   ```

Outputs will land in the `outputs/` directory. If `UPLOAD_BUCKET` and `UPLOAD_FILE` are set, the final COG will also be uploaded to S3.

## Advanced Features

The Black Marble pipeline includes advanced image processing features using scipy:

### Available Features

1. **Advanced Gap Filling** - Uses scipy's Euclidean distance transform to fill gaps with nearest neighbor interpolation:
   ```python
   from blackmarble.components.temporal import TemporalCompositor
   
   compositor = TemporalCompositor(
       method="median",
       use_scipy_fill_gaps=True  # Enable advanced gap filling
   )
   ```

2. **Feather Blending** - Creates seamless mosaics with smooth transitions between overlapping scenes:
   ```python
   from blackmarble.components.temporal import SceneMerger
   
   merger = SceneMerger(
       blend_method="feather",
       feather_distance=10
   )
   ```

These features are disabled by default for backward compatibility.

## Module Organization

The Black Marble codebase is organized to follow the natural flow of the data processing pipeline. If you look at the original `bm.py` script, you'll see it can be thought of as following a clear sequence: acquire data, prepare it for analysis, perform calculations, enhance the results, and export the final product. Modules are organized around that structure:

```
blackmarble/
├── acquire/
│   ├── landsat.py      # Download and tile selection for Landsat data
│   ├── viirs.py        # VIIRS nighttime lights data acquisition
│   ├── osm.py          # OpenStreetMap road network fetching
├── prepare/
│   ├── scaling.py      # Convert DN to reflectance, apply lunar correction
│   ├── quality.py      # Cloud masks, aerosol masks, quality filtering
│   ├── spatial.py      # Reprojection, alignment, cropping to bounds
├── analyze/
│   ├── indices.py      # Calculate NDVI, NDWI, NDUI indices
│   ├── temporal.py     # Create composites, merge scenes
├── enhance/
│   ├── contrast.py     # Dynamic range adjustments
│   ├── features.py     # Road enhancement, urban feature highlighting
│   ├── visualize.py    # Apply colormaps, prepare for display
├── export/
│   ├── cog.py          # Generate Cloud-Optimized GeoTIFFs
│   ├── metadata.py     # Create metadata and statistics
│   ├── validate.py     # Validate output quality
```

This makes it easy to find what you need - if you're working on data acquisition, look in `acquire/`. If you need to calculate indices, check `analyze/indices.py`. Each module contains functions that work together for that stage of processing, keeping related functionality close while maintaining clear boundaries between pipeline stages.
```