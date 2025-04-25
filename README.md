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
```