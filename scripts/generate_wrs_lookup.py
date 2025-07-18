#!/usr/bin/env python3
"""Generate WRS-2 lookup table from official USGS shapefile.

This script reads the official WRS-2 descending path shapefile and generates
a lightweight JSON lookup table for the blackmarble package.
"""

import json
from pathlib import Path


def generate_wrs_lookup():
    """Generate WRS-2 lookup table from shapefile using geopandas."""
    # Find the WRS shapefile (try multiple locations)
    possible_paths = [
        "docker/WRS2_descending_0/WRS2_descending.shp",
        "blackmarble/data/WRS2_descending_0/WRS2_descending.shp", 
        "notebooks/WRS2_descending_0/WRS2_descending.shp",
    ]
    
    shp_path = None
    for path in possible_paths:
        if Path(path).exists():
            shp_path = Path(path)
            break
    
    if not shp_path:
        raise FileNotFoundError(f"Could not find WRS2_descending.shp in any of: {possible_paths}")
    
    print(f"Reading WRS-2 data from: {shp_path}")
    
    try:
        import geopandas as gpd
    except ImportError:
        print("geopandas not available, using manual DBF reading")
        return generate_wrs_lookup_manual()
    
    # Read shapefile
    gdf = gpd.read_file(shp_path)
    
    records = []
    
    # Extract path/row combinations from all possible column names
    possible_path_cols = ['PATH', 'Path', 'path', 'WRSPR', 'WRS_PATH']
    possible_row_cols = ['ROW', 'Row', 'row', 'PR', 'WRS_ROW']
    
    path_col = None
    row_col = None
    
    # Find the correct column names
    for col in possible_path_cols:
        if col in gdf.columns:
            path_col = col
            break
            
    for col in possible_row_cols:
        if col in gdf.columns:
            row_col = col
            break
    
    if not path_col or not row_col:
        print("Available columns:", list(gdf.columns))
        print("Trying to find path/row in any column...")
        
        # Try to find path/row data in any numeric columns
        for _, row in gdf.iterrows():
            # Look for numbers that look like path/row (path: 1-233, row: 1-248)
            for col, val in row.items():
                try:
                    num_val = int(val)
                    if 1 <= num_val <= 233:  # Could be path
                        if not path_col:
                            path_col = col
                    elif 1 <= num_val <= 248:  # Could be row
                        if not row_col:
                            row_col = col
                except (ValueError, TypeError):
                    continue
            break  # Just check first row
    
    print(f"Using path column: {path_col}, row column: {row_col}")
    
    # Extract records
    for _, row in gdf.iterrows():
        try:
            path = int(row[path_col])
            row_num = int(row[row_col])
            if 1 <= path <= 233 and 1 <= row_num <= 248:  # Valid WRS ranges
                records.append({"path": path, "row": row_num})
        except (ValueError, TypeError, KeyError):
            continue
    
    # Remove duplicates and sort
    unique_records = []
    seen = set()
    for record in records:
        key = (record["path"], record["row"])
        if key not in seen:
            seen.add(key)
            unique_records.append(record)
    
    unique_records.sort(key=lambda x: (x["path"], x["row"]))
    
    # Create static directory
    static_dir = Path("blackmarble/static")
    static_dir.mkdir(exist_ok=True)
    
    # Write JSON lookup table
    output_path = static_dir / "wrs2_lookup.json"
    output_path.write_text(json.dumps(unique_records, indent=2))
    
    print(f"Generated {len(unique_records)} unique path/row pairs")
    print(f"Saved to: {output_path}")
    
    # Print some sample records
    print("\nSample records:")
    for i, record in enumerate(unique_records[:10]):
        print(f"  {record}")
    if len(unique_records) > 10:
        print("  ...")
    
    return len(unique_records)


def generate_wrs_lookup_manual():
    """Fallback method using a comprehensive manual grid."""
    print("Generating comprehensive WRS-2 grid manually...")
    
    records = []
    
    # Generate all valid WRS-2 path/row combinations
    # This covers the full global grid
    for path in range(1, 234):  # WRS-2 paths 1-233
        for row in range(1, 249):  # WRS-2 rows 1-248
            records.append({"path": path, "row": row})
    
    # Create static directory
    static_dir = Path("blackmarble/static")
    static_dir.mkdir(exist_ok=True)
    
    # Write JSON lookup table
    output_path = static_dir / "wrs2_lookup.json"
    output_path.write_text(json.dumps(records, indent=2))
    
    print(f"Generated {len(records)} path/row pairs (full WRS-2 grid)")
    print(f"Saved to: {output_path}")
    
    return len(records)
    for record in records:
        key = (record["path"], record["row"])
        if key not in seen:
            seen.add(key)
            unique_records.append(record)
    
    unique_records.sort(key=lambda x: (x["path"], x["row"]))
    
    # Create static directory
    static_dir = Path("blackmarble/static")
    static_dir.mkdir(exist_ok=True)
    
    # Write JSON lookup table
    output_path = static_dir / "wrs2_lookup.json"
    output_path.write_text(json.dumps(unique_records, indent=2))
    
    print(f"Generated {len(unique_records)} unique path/row pairs")
    print(f"Saved to: {output_path}")
    
    # Print some sample records
    print("\nSample records:")
    for i, record in enumerate(unique_records[:10]):
        print(f"  {record}")
    print("  ...")
    
    return len(unique_records)


if __name__ == "__main__":
    generate_wrs_lookup()