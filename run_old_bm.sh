#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
DOCKER_IMAGE_NAME="blackmarble-processor"
DOCKERFILE_DIR="docker"
# Define the size of the bounding box around the center point (in degrees)
# 0.1 degrees is roughly 11km
# BOX_DELTA=0.1 # No longer needed

# --- Argument Parsing ---
LAT1=""
LONG1=""
LAT2=""
LONG2=""
YEAR=""
MONTH=""
DAY=""

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --lat1)
      LAT1="$2"
      shift # past argument
      shift # past value
      ;;
    --long1)
      LONG1="$2"
      shift # past argument
      shift # past value
      ;;
    --lat2)
      LAT2="$2"
      shift # past argument
      shift # past value
      ;;
    --long2)
      LONG2="$2"
      shift # past argument
      shift # past value
      ;;
    --year)
      YEAR="$2"
      shift # past argument
      shift # past value
      ;;
    --month)
      MONTH="$2"
      shift # past argument
      shift # past value
      ;;
    --day)
      DAY="$2"
      shift # past argument
      shift # past value
      ;;
    *)    # unknown option
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# --- Validate Input Arguments ---
if [ -z "$LAT1" ] || [ -z "$LONG1" ] || [ -z "$LAT2" ] || [ -z "$LONG2" ] || [ -z "$YEAR" ] || [ -z "$MONTH" ] || [ -z "$DAY" ]; then
  echo "Usage: $0 --lat1 <lat1> --long1 <long1> --lat2 <lat2> --long2 <long2> --year <year> --month <month> --day <day>"
  echo "Example: $0 --lat1 33.03 --long1 21.28 --lat2 32.47 --long2 22.92 --year 2023 --month 1 --day 24"
  exit 1
fi

# --- Check Required Environment Variables ---
if [ -z "$AWS_ACCESS_KEY" ]; then echo "Error: AWS_ACCESS_KEY environment variable not set."; exit 1; fi
# Use AWS_ACCESS_SECRET instead of AWS_SECRET_KEY
if [ -z "$AWS_ACCESS_SECRET" ]; then echo "Error: AWS_ACCESS_SECRET environment variable not set."; exit 1; fi
if [ -z "$EARTH_DATA_TOKEN" ]; then echo "Error: EARTH_DATA_TOKEN environment variable not set."; exit 1; fi
if [ -z "$UPLOAD_BUCKET" ]; then echo "Error: UPLOAD_BUCKET environment variable not set."; exit 1; fi

# --- Create Local Output Directory ---
LOCAL_OUTPUT_DIR="bm_outputs"
mkdir -p "$LOCAL_OUTPUT_DIR"
echo "Local output will be stored in: $(pwd)/$LOCAL_OUTPUT_DIR"

# --- Define Upload File Name ---
# Format: blackmarble_output/ndui_LAT1_LON1_YYYYMMDD.tif
# Use LAT1/LONG1 for filename consistency, replace dots and minus signs
LAT_STR=$(printf "%.2f" $LAT1 | sed 's/\./p/' | sed 's/-/m/')
LON_STR=$(printf "%.2f" $LONG1 | sed 's/\./p/' | sed 's/-/m/')
DATE_STR=$(printf "%04d%02d%02d" $YEAR $MONTH $DAY)
UPLOAD_FILE="blackmarble_output/ndui_${LAT_STR}N_${LON_STR}E_${DATE_STR}.tif"
echo "Output S3 Key: s3://$UPLOAD_BUCKET/$UPLOAD_FILE"

# --- Build Docker Image ---
echo "Building Docker image '$DOCKER_IMAGE_NAME' from '$DOCKERFILE_DIR'..."
docker build -t "$DOCKER_IMAGE_NAME" "$DOCKERFILE_DIR"

# --- Run Docker Container ---
echo "Running Black Marble processing in Docker..."

# Add volume to store intermediate files + outputs locally
docker run --rm \
  -v "$(pwd)/$LOCAL_OUTPUT_DIR:/root/outputs" \
  -e LAT1="$LAT1" \
  -e LONG1="$LONG1" \
  -e LAT2="$LAT2" \
  -e LONG2="$LONG2" \
  -e YEAR="$YEAR" \
  -e MONTH="$MONTH" \
  -e DAY="$DAY" \
  -e AWS_ACCESS_KEY="$AWS_ACCESS_KEY" \
  -e AWS_ACCESS_SECRET="$AWS_ACCESS_SECRET" \
  -e EARTH_DATA_TOKEN="$EARTH_DATA_TOKEN" \
  -e UPLOAD_BUCKET="$UPLOAD_BUCKET" \
  -e UPLOAD_FILE="$UPLOAD_FILE" \
  -e AWS_WEB_IDENTITY_TOKEN_FILE="/tmp/aws_token" \
  -e AWS_WEB_IDENTITY_TOKEN="dummy-token-for-script" \
  "$DOCKER_IMAGE_NAME"

echo "Docker container finished."
echo "Output should be available at: s3://$UPLOAD_BUCKET/$UPLOAD_FILE"