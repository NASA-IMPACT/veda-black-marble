import os
import re
import pytest
from typing import Dict, Union

class FilenameParseError(ValueError):
    """Custom exception for filename parsing errors."""
    pass

def extract_params_from_filename(filename: str) -> Dict[str, Union[float, int]]:
    """
    Extracts bounding box coordinates and date from a Black Marble output filename.

    Expected format: ndui_LAT_STR_LAT_HEMI_LON_STR_LON_HEMI_YYYYMMDD.tif
    LAT_STR/LON_STR: Can start with 'm' for negative, uses 'p' for decimal.
    LAT_HEMI: 'N' or 'S'.
    LON_HEMI: 'E' or 'W'.

    Args:
        filename (str): The Black Marble output filename.

    Returns:
        Dict[str, Union[float, int]]: A dictionary containing 'lat1', 'lon1',
                                      'year', 'month', 'day'. Note: The filename
                                      only encodes the *start* coordinates (lat1, lon1).

    Raises:
        FilenameParseError: If the filename format does not match.
    """
    base_name = os.path.basename(filename)
    # Regex to match the pattern: ndui_LATSTR(N|S)_LONSTR(E|W)_YYYYMMDD.tif(f)?
    # LATSTR/LONSTR can contain digits, 'p', and optionally start with 'm'
    match = re.match(r'^ndui_([mp\d]+)([NS])_([mp\d]+)([EW])_(\d{8})\.tif(?:f)?$', base_name)

    if not match:
        raise FilenameParseError(f"Filename does not match expected format: {base_name}")

    lat_str, lat_hemi, lon_str, lon_hemi, date_str = match.groups()

    # Parse Latitude
    lat_value_str = lat_str.replace('p', '.')
    try:
        lat1 = float(lat_value_str.lstrip('m')) # Convert to float, ignoring 'm' for now
    except ValueError:
         raise FilenameParseError(f"Could not parse latitude value from filename: {base_name}")

    if lat_str.startswith('m'):
        lat1 *= -1.0
    if lat_hemi == 'S':
        lat1 = -abs(lat1) # Ensure negative if South

    # Parse Longitude
    lon_value_str = lon_str.replace('p', '.')
    try:
        lon1 = float(lon_value_str.lstrip('m')) # Convert to float, ignoring 'm' for now
    except ValueError:
         raise FilenameParseError(f"Could not parse longitude value from filename: {base_name}")

    if lon_str.startswith('m'):
        lon1 *= -1.0
    if lon_hemi == 'W':
        lon1 = -abs(lon1) # Ensure negative if West

    # Parse Date
    try:
        year = int(date_str[:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
    except ValueError:
         raise FilenameParseError(f"Could not parse date from filename: {base_name}")


    return {
        'lat1': lat1,
        'lon1': lon1,
        'year': year,
        'month': month,
        'day': day
        # Note: lat2 and lon2 are not encoded in the filename
    }

# --- Unit Tests for extract_params_from_filename ---

def test_extract_params_from_filename_valid():
    """Test extraction with a valid filename."""
    filename = "ndui_33p03N_21p28E_20230124.tif"
    params = extract_params_from_filename(filename)
    assert params == {'lat1': 33.03, 'lon1': 21.28, 'year': 2023, 'month': 1, 'day': 24}

def test_extract_params_from_filename_valid_negative():
    """Test extraction with negative coordinates."""
    # Assuming 'm' prefix for negative and S/W for hemisphere
    filename = "ndui_m30p50S_m100p01W_20221215.tif"
    params = extract_params_from_filename(filename)
    assert params == {'lat1': -30.50, 'lon1': -100.01, 'year': 2022, 'month': 12, 'day': 15}

def test_extract_params_from_filename_valid_mixed_sign():
    """Test extraction with mixed positive/negative coordinates."""
    filename = "ndui_40p12N_m75p34W_20210701.tif"
    params = extract_params_from_filename(filename)
    assert params == {'lat1': 40.12, 'lon1': -75.34, 'year': 2021, 'month': 7, 'day': 1}

    filename = "ndui_m15p67S_50p89E_20200420.tif"
    params = extract_params_from_filename(filename)
    assert params == {'lat1': -15.67, 'lon1': 50.89, 'year': 2020, 'month': 4, 'day': 20}

def test_extract_params_from_filename_invalid_format():
    """Test extraction with invalid filenames."""
    with pytest.raises(FilenameParseError, match="Filename does not match expected format"):
        extract_params_from_filename("not_ndui_file.tif")
    with pytest.raises(FilenameParseError, match="Filename does not match expected format"):
        extract_params_from_filename("ndui_33p03N_20230124.tif") # Missing lon part
    with pytest.raises(FilenameParseError, match="Could not parse date from filename"):
        extract_params_from_filename("ndui_33p03N_21p28E_202301.tif") # Invalid date format
    with pytest.raises(FilenameParseError, match="Filename does not match expected format"):
        extract_params_from_filename("ndui_33.03N_21.28E_20230124.tif") # Uses dot instead of p
    with pytest.raises(FilenameParseError, match="Filename does not match expected format"):
        extract_params_from_filename("ndui_33p03X_21p28E_20230124.tif") # Invalid hemisphere
    with pytest.raises(FilenameParseError, match="Could not parse latitude value from filename"):
         extract_params_from_filename("ndui_33p.03N_21p28E_20230124.tif") # Invalid number format

def test_extract_params_from_filename_different_extension():
    """Test extraction with .tiff extension."""
    filename = "ndui_33p03N_21p28E_20230124.tiff"
    params = extract_params_from_filename(filename)
    assert params == {'lat1': 33.03, 'lon1': 21.28, 'year': 2023, 'month': 1, 'day': 24}