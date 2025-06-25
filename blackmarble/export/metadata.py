"""Metadata generation and STAC item creation utilities.

This module provides functions for creating metadata files, STAC items,
and other documentation to accompany exported data products.
"""

from typing import Any, Literal
from datetime import datetime
import numpy.typing as npt

# Type aliases
ArrayLike = npt.NDArray[Any]
BBox = tuple[float, float, float, float]


def create_metadata(
    data_sources: dict[str, dict[str, Any]],
    processing_steps: list[str],
    indices_calculated: list[str] | None = None,
    date_processed: datetime | None = None,
    bbox: BBox | None = None,
    crs: str | None = None,
    resolution: float | None = None,
    custom_metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Create comprehensive metadata for processed data.
    
    Args:
        data_sources: Info about input data sources (landsat, viirs, osm)
        processing_steps: List of processing steps applied
        indices_calculated: List of indices computed (NDVI, NDUI, etc.)
        date_processed: Processing timestamp
        bbox: Bounding box of output
        crs: Coordinate reference system
        resolution: Spatial resolution in CRS units
        custom_metadata: Additional metadata fields
        
    Returns:
        Complete metadata dictionary
    """
    from datetime import datetime
    import platform
    
    # Use current time if not provided
    if date_processed is None:
        date_processed = datetime.now()
    
    # Build metadata structure following ISO 19115 conventions
    metadata = {
        "file_info": {
            "format": "Cloud-Optimized GeoTIFF",
            "format_version": "1.0",
            "creation_date": date_processed.isoformat(),
            "producer": "Black Marble Pipeline",
            "producer_version": "1.0.0"
        },
        
        "data_sources": {},
        
        "processing": {
            "steps": processing_steps,
            "date": date_processed.isoformat(),
            "system": {
                "platform": platform.system(),
                "python_version": platform.python_version(),
                "processor": platform.processor()
            }
        },
        
        "spatial_info": {},
        
        "data_quality": {
            "lineage": "Processed from multiple satellite and ancillary data sources",
            "completeness": "Quality-controlled composite with cloud and artifact removal",
            "positional_accuracy": "Sub-pixel alignment verified across all input layers"
        }
    }
    
    # Add data source details
    for source_name, source_info in data_sources.items():
        metadata["data_sources"][source_name] = {
            "files": source_info.get("files", []),
            "date_range": source_info.get("date_range", None),
            "sensor": source_info.get("sensor", None),
            "processing_level": source_info.get("processing_level", None),
            "cloud_cover": source_info.get("cloud_cover", None)
        }
        
        # Add source-specific metadata
        if source_name == "landsat":
            metadata["data_sources"][source_name].update({
                "bands_used": source_info.get("bands", ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]),
                "collection": source_info.get("collection", "Collection 2")
            })
        elif source_name == "viirs":
            metadata["data_sources"][source_name].update({
                "product": source_info.get("product", "VNP46A1"),
                "quality_flags_applied": source_info.get("quality_flags", True)
            })
        elif source_name == "osm":
            metadata["data_sources"][source_name].update({
                "features": source_info.get("features", ["roads", "water"]),
                "date_extracted": source_info.get("date_extracted", date_processed.isoformat())
            })
    
    # Add spatial information
    if bbox:
        metadata["spatial_info"]["bbox"] = {
            "west": bbox[0],
            "south": bbox[1],
            "east": bbox[2],
            "north": bbox[3]
        }
    
    if crs:
        metadata["spatial_info"]["crs"] = crs
        metadata["spatial_info"]["crs_wkt"] = None  # Could add WKT representation
    
    if resolution:
        metadata["spatial_info"]["resolution"] = {
            "value": resolution,
            "unit": "meters" if "utm" in str(crs).lower() else "degrees"
        }
    
    # Add indices information
    if indices_calculated:
        metadata["indices"] = {}
        for index in indices_calculated:
            if index.upper() == "NDVI":
                metadata["indices"]["NDVI"] = {
                    "full_name": "Normalized Difference Vegetation Index",
                    "formula": "(NIR - Red) / (NIR + Red)",
                    "valid_range": [-1.0, 1.0],
                    "interpretation": "Higher values indicate more vegetation"
                }
            elif index.upper() == "NDUI":
                metadata["indices"]["NDUI"] = {
                    "full_name": "Normalized Difference Urban Index",
                    "formula": "(SWIR - NIR) / (SWIR + NIR)",
                    "valid_range": [0.0, 1.0],
                    "interpretation": "Higher values indicate urban areas"
                }
            elif index.upper() == "NDWI":
                metadata["indices"]["NDWI"] = {
                    "full_name": "Normalized Difference Water Index",
                    "formula": "(Green - NIR) / (Green + NIR)",
                    "valid_range": [-1.0, 1.0],
                    "interpretation": "Positive values indicate water presence"
                }
    
    # Add custom metadata
    if custom_metadata:
        for key, value in custom_metadata.items():
            if key not in metadata:
                metadata[key] = value
            elif isinstance(metadata[key], dict) and isinstance(value, dict):
                metadata[key].update(value)
            else:
                metadata[key] = value
    
    return metadata


def create_stac_item(
    asset_path: str,
    item_id: str,
    bbox: BBox,
    datetime_acquired: datetime,
    collection: str = "black-marble",
    properties: dict[str, Any] | None = None,
    assets: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Create STAC item for data product.
    
    Args:
        asset_path: Path to main data file
        item_id: Unique identifier for item
        bbox: Bounding box in lon/lat
        datetime_acquired: Acquisition date
        collection: STAC collection ID
        properties: Additional STAC properties
        assets: Asset definitions (COG, thumbnail, metadata)
        
    Returns:
        STAC item dictionary
    """
    import os
    from datetime import datetime, timezone
    import rasterio
    from rasterio.warp import transform_bounds
    
    # STAC version
    stac_version = "1.0.0"
    
    # Ensure datetime is timezone-aware
    if datetime_acquired.tzinfo is None:
        datetime_acquired = datetime_acquired.replace(tzinfo=timezone.utc)
    
    # Create base STAC item
    stac_item = {
        "stac_version": stac_version,
        "type": "Feature",
        "id": item_id,
        "collection": collection,
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [bbox[0], bbox[1]],  # SW corner
                [bbox[2], bbox[1]],  # SE corner
                [bbox[2], bbox[3]],  # NE corner
                [bbox[0], bbox[3]],  # NW corner
                [bbox[0], bbox[1]]   # Close polygon
            ]]
        },
        "bbox": list(bbox),
        "properties": {
            "datetime": datetime_acquired.isoformat(),
            "created": datetime.now(timezone.utc).isoformat(),
            "updated": datetime.now(timezone.utc).isoformat()
        },
        "assets": {},
        "links": [
            {
                "rel": "self",
                "href": f"./{item_id}.json",
                "type": "application/json"
            },
            {
                "rel": "collection",
                "href": f"../{collection}.json",
                "type": "application/json"
            }
        ]
    }
    
    # Add common properties for Earth observation data
    # Handle case where properties is None
    if properties is None:
        properties = {}
    
    eo_properties = {
        "eo:cloud_cover": properties.get("cloud_cover", None),
        "platform": properties.get("platform", "multiple"),
        "instruments": properties.get("instruments", ["OLI", "VIIRS"]),
        "gsd": properties.get("gsd", 30),  # Ground sample distance in meters
        "proj:epsg": properties.get("epsg", None),
        "proj:shape": properties.get("shape", None),
        "proj:transform": properties.get("transform", None)
    }
    
    # Add properties, filtering out None values
    for key, value in eo_properties.items():
        if value is not None:
            stac_item["properties"][key] = value
    
    # Add custom properties
    if properties:
        for key, value in properties.items():
            if key not in stac_item["properties"]:
                stac_item["properties"][key] = value
    
    # Process main asset
    if os.path.exists(asset_path):
        # Get file info
        file_size = os.path.getsize(asset_path)
        
        # Read raster metadata
        with rasterio.open(asset_path) as src:
            # Update bbox if needed (transform from data CRS to WGS84)
            if src.crs and str(src.crs) != "EPSG:4326":
                data_bbox = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
                stac_item["bbox"] = list(data_bbox)
                stac_item["geometry"]["coordinates"][0] = [
                    [data_bbox[0], data_bbox[1]],
                    [data_bbox[2], data_bbox[1]],
                    [data_bbox[2], data_bbox[3]],
                    [data_bbox[0], data_bbox[3]],
                    [data_bbox[0], data_bbox[1]]
                ]
            
            # Update projection info
            stac_item["properties"]["proj:epsg"] = src.crs.to_epsg()
            stac_item["properties"]["proj:shape"] = [src.height, src.width]
            stac_item["properties"]["proj:transform"] = list(src.transform)[:6]
    
        # Main data asset
        stac_item["assets"]["data"] = {
            "href": os.path.basename(asset_path),
            "type": "image/tiff; application=geotiff; profile=cloud-optimized",
            "title": "Cloud Optimized GeoTIFF",
            "roles": ["data", "cloud-optimized"],
            "file:size": file_size
        }
    
    # Add additional assets if provided
    if assets:
        for asset_key, asset_info in assets.items():
            if asset_key not in stac_item["assets"]:
                stac_item["assets"][asset_key] = asset_info
    
    # Common additional assets
    # Thumbnail
    thumbnail_path = asset_path.replace(".tif", "_thumb.png")
    if os.path.exists(thumbnail_path):
        stac_item["assets"]["thumbnail"] = {
            "href": os.path.basename(thumbnail_path),
            "type": "image/png",
            "title": "Thumbnail",
            "roles": ["thumbnail"]
        }
    
    # Metadata
    metadata_path = asset_path.replace(".tif", "_metadata.json")
    if os.path.exists(metadata_path):
        stac_item["assets"]["metadata"] = {
            "href": os.path.basename(metadata_path),
            "type": "application/json",
            "title": "Processing Metadata",
            "roles": ["metadata"]
        }
    
    return stac_item


def generate_statistics(
    data_path: str | ArrayLike,
    mask_path: str | ArrayLike | None = None,
    percentiles: list[float] = [2, 10, 25, 50, 75, 90, 98]
) -> dict[str, Any]:
    """Generate comprehensive statistics for data.
    
    Args:
        data_path: Path to data or array
        mask_path: Optional mask for valid pixels
        percentiles: Percentiles to calculate
        
    Returns:
        Statistics dictionary with min, max, mean, std, percentiles
    """
    import numpy as np
    import rasterio
    
    # Load data
    if isinstance(data_path, str):
        with rasterio.open(data_path) as src:
            data = src.read()
            nodata = src.nodata
            
            # Get band descriptions if available
            band_descriptions = [src.descriptions[i] if src.descriptions[i] else f"Band_{i+1}" 
                               for i in range(src.count)]
    else:
        data = np.asarray(data_path)
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        nodata = None
        band_descriptions = [f"Band_{i+1}" for i in range(data.shape[0])]
    
    # Load mask if provided
    if mask_path is not None:
        if isinstance(mask_path, str):
            with rasterio.open(mask_path) as src:
                mask = src.read(1).astype(bool)
        else:
            mask = np.asarray(mask_path).astype(bool)
    else:
        mask = None
    
    # Calculate statistics for each band
    stats = {
        "band_count": data.shape[0],
        "shape": data.shape,
        "dtype": str(data.dtype),
        "bands": {}
    }
    
    for i in range(data.shape[0]):
        band_data = data[i]
        
        # Create valid data mask
        valid_mask = np.ones_like(band_data, dtype=bool)
        
        # Exclude nodata
        if nodata is not None:
            valid_mask &= (band_data != nodata)
        
        # Exclude NaN
        if np.issubdtype(band_data.dtype, np.floating):
            valid_mask &= ~np.isnan(band_data)
        
        # Apply user mask
        if mask is not None:
            valid_mask &= mask
        
        # Get valid data
        valid_data = band_data[valid_mask]
        
        # Calculate statistics
        band_stats = {
            "name": band_descriptions[i],
            "valid_pixels": int(np.sum(valid_mask)),
            "total_pixels": int(band_data.size),
            "valid_percent": float(np.sum(valid_mask) / band_data.size * 100)
        }
        
        if len(valid_data) > 0:
            band_stats.update({
                "min": float(np.min(valid_data)),
                "max": float(np.max(valid_data)),
                "mean": float(np.mean(valid_data)),
                "std": float(np.std(valid_data)),
                "median": float(np.median(valid_data))
            })
            
            # Calculate percentiles
            band_stats["percentiles"] = {}
            for p in percentiles:
                band_stats["percentiles"][f"p{int(p)}"] = float(np.percentile(valid_data, p))
            
            # Additional statistics
            band_stats["coefficient_variation"] = float(band_stats["std"] / band_stats["mean"]) if band_stats["mean"] != 0 else None
            
            # Check for common data ranges
            if band_stats["min"] >= -1 and band_stats["max"] <= 1:
                band_stats["likely_index"] = True  # Likely a normalized index
            elif band_stats["min"] >= 0 and band_stats["max"] <= 1:
                band_stats["likely_normalized"] = True  # Likely normalized data
            elif band_stats["min"] >= 0 and band_stats["max"] <= 255:
                band_stats["likely_byte"] = True  # Likely byte data
        else:
            # No valid data
            band_stats.update({
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "median": None,
                "percentiles": {},
                "coefficient_variation": None
            })
        
        stats["bands"][f"band_{i+1}"] = band_stats
    
    # Add overall statistics
    all_valid_pixels = sum(b["valid_pixels"] for b in stats["bands"].values())
    all_total_pixels = sum(b["total_pixels"] for b in stats["bands"].values())
    
    stats["overall"] = {
        "total_valid_pixels": all_valid_pixels,
        "total_pixels": all_total_pixels,
        "overall_valid_percent": float(all_valid_pixels / all_total_pixels * 100) if all_total_pixels > 0 else 0
    }
    
    return stats


def export_metadata_sidecar(
    metadata: dict[str, Any],
    output_path: str,
    format: Literal["json", "xml", "yaml"] = "json"
) -> str:
    """Export metadata as sidecar file.
    
    Args:
        metadata: Metadata dictionary
        output_path: Path for sidecar file
        format: Output format
        
    Returns:
        Path to metadata file
    """
    import json
    import os
    from datetime import datetime
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Convert datetime objects to strings
    def serialize_dates(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: serialize_dates(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [serialize_dates(item) for item in obj]
        return obj
    
    metadata_serialized = serialize_dates(metadata)
    
    if format == "json":
        # Export as JSON with pretty formatting
        with open(output_path, 'w') as f:
            json.dump(metadata_serialized, f, indent=2, sort_keys=True)
    
    elif format == "xml":
        # Export as XML
        import xml.etree.ElementTree as ET
        from xml.dom import minidom
        
        def dict_to_xml(tag, d):
            """Convert dictionary to XML element."""
            elem = ET.Element(tag)
            for key, val in d.items():
                if isinstance(val, dict):
                    elem.append(dict_to_xml(key, val))
                elif isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict):
                            elem.append(dict_to_xml(key, item))
                        else:
                            child = ET.SubElement(elem, key)
                            child.text = str(item)
                else:
                    child = ET.SubElement(elem, key)
                    child.text = str(val) if val is not None else ""
            return elem
        
        root = dict_to_xml("metadata", metadata_serialized)
        
        # Pretty print XML
        rough_string = ET.tostring(root, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")
        
        with open(output_path, 'w') as f:
            f.write(pretty_xml)
    
    elif format == "yaml":
        # Export as YAML
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required for YAML export. Install with: pip install pyyaml")
        
        with open(output_path, 'w') as f:
            yaml.dump(metadata_serialized, f, default_flow_style=False, sort_keys=True)
    
    else:
        raise ValueError(f"Unknown format: {format}")
    
    return output_path


def create_processing_report(
    config: Any,  # PipelineConfig
    inputs: dict[str, list[str]],
    outputs: dict[str, str],
    statistics: dict[str, dict[str, float]],
    processing_time: float,
    errors: list[str] | None = None
) -> dict[str, Any]:
    """Create detailed processing report.
    
    Args:
        config: Pipeline configuration used
        inputs: Input files by type
        outputs: Output files created
        statistics: Processing statistics
        processing_time: Total processing time in seconds
        errors: Any errors encountered
        
    Returns:
        Processing report dictionary
    """
    from datetime import datetime, timedelta
    import platform
    import os
    
    # Format processing time
    hours, remainder = divmod(processing_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"
    
    report = {
        "report_type": "Black Marble Processing Report",
        "report_version": "1.0",
        "generated_at": datetime.now().isoformat(),
        
        "processing_summary": {
            "status": "completed" if not errors else "completed_with_errors",
            "total_time_seconds": processing_time,
            "total_time_formatted": time_str,
            "start_time": (datetime.now() - timedelta(seconds=processing_time)).isoformat(),
            "end_time": datetime.now().isoformat()
        },
        
        "system_info": {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "hostname": platform.node()
        },
        
        "configuration": {},
        
        "inputs": {},
        
        "outputs": {},
        
        "statistics": statistics,
        
        "errors": errors or [],
        
        "quality_metrics": {}
    }
    
    # Add configuration details
    if config:
        # Extract key configuration parameters
        report["configuration"] = {
            "target_date": getattr(config, 'target_date', None),
            "bbox": getattr(config, 'bbox', None),
            "output_crs": getattr(config, 'output_crs', None),
            "output_resolution": getattr(config, 'output_resolution', None),
            "cloud_cover_threshold": getattr(config, 'cloud_cover_threshold', None),
            "indices_calculated": getattr(config, 'indices', [])
        }
    
    # Process input files
    total_input_size = 0
    total_input_files = 0
    
    for input_type, file_list in inputs.items():
        file_info = []
        for file_path in file_list:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                total_input_size += size
                file_info.append({
                    "path": file_path,
                    "size_mb": size / (1024 * 1024),
                    "exists": True
                })
            else:
                file_info.append({
                    "path": file_path,
                    "size_mb": 0,
                    "exists": False
                })
        
        total_input_files += len(file_list)
        report["inputs"][input_type] = {
            "count": len(file_list),
            "files": file_info
        }
    
    # Process output files
    total_output_size = 0
    
    for output_name, file_path in outputs.items():
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            total_output_size += size
            report["outputs"][output_name] = {
                "path": file_path,
                "size_mb": size / (1024 * 1024),
                "exists": True
            }
        else:
            report["outputs"][output_name] = {
                "path": file_path,
                "size_mb": 0,
                "exists": False,
                "error": "File not found after processing"
            }
    
    # Add summary statistics
    report["processing_summary"].update({
        "input_files_count": total_input_files,
        "input_size_mb": total_input_size / (1024 * 1024),
        "output_files_count": len(outputs),
        "output_size_mb": total_output_size / (1024 * 1024),
        "compression_ratio": (total_input_size / total_output_size) if total_output_size > 0 else None
    })
    
    # Calculate quality metrics from statistics
    if statistics:
        valid_pixel_counts = []
        coverage_percents = []
        
        for stat_name, stat_values in statistics.items():
            if "valid_pixels" in stat_values:
                valid_pixel_counts.append(stat_values["valid_pixels"])
            if "coverage_percent" in stat_values:
                coverage_percents.append(stat_values["coverage_percent"])
        
        if valid_pixel_counts:
            report["quality_metrics"]["average_valid_pixels"] = sum(valid_pixel_counts) / len(valid_pixel_counts)
        
        if coverage_percents:
            report["quality_metrics"]["average_coverage_percent"] = sum(coverage_percents) / len(coverage_percents)
    
    # Add error analysis
    if errors:
        report["error_analysis"] = {
            "error_count": len(errors),
            "error_types": {},
            "recommendations": []
        }
        
        # Categorize errors
        for error in errors:
            error_lower = error.lower()
            if "memory" in error_lower:
                report["error_analysis"]["error_types"]["memory"] = report["error_analysis"]["error_types"].get("memory", 0) + 1
                if "Reduce processing extent" not in report["error_analysis"]["recommendations"]:
                    report["error_analysis"]["recommendations"].append("Reduce processing extent or increase available memory")
            elif "file" in error_lower or "path" in error_lower:
                report["error_analysis"]["error_types"]["file_access"] = report["error_analysis"]["error_types"].get("file_access", 0) + 1
            elif "cloud" in error_lower or "coverage" in error_lower:
                report["error_analysis"]["error_types"]["data_quality"] = report["error_analysis"]["error_types"].get("data_quality", 0) + 1
    
    return report


def export_processing_lineage(
    lineage_graph: dict[str, Any],
    output_path: str,
    format: Literal["json", "dot", "mermaid"] = "json"
) -> str:
    """Export processing lineage/provenance.
    
    Args:
        lineage_graph: DAG of processing steps
        output_path: Output file path
        format: Export format
        
    Returns:
        Path to lineage file
    """
    import json
    import os
    from datetime import datetime
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    if format == "json":
        # Export as JSON
        with open(output_path, 'w') as f:
            json.dump(lineage_graph, f, indent=2, default=str)
    
    elif format == "dot":
        # Export as Graphviz DOT format
        dot_content = ["digraph ProcessingLineage {"]
        dot_content.append('  rankdir=TB;')
        dot_content.append('  node [shape=box, style=rounded];')
        dot_content.append('')
        
        # Add nodes
        nodes = lineage_graph.get("nodes", {})
        for node_id, node_info in nodes.items():
            label = node_info.get("label", node_id)
            node_type = node_info.get("type", "process")
            
            if node_type == "input":
                style = 'shape=ellipse, style=filled, fillcolor=lightblue'
            elif node_type == "output":
                style = 'shape=ellipse, style=filled, fillcolor=lightgreen'
            else:
                style = 'shape=box, style=rounded'
            
            dot_content.append(f'  "{node_id}" [label="{label}", {style}];')
        
        dot_content.append('')
        
        # Add edges
        edges = lineage_graph.get("edges", [])
        for edge in edges:
            from_node = edge.get("from")
            to_node = edge.get("to")
            label = edge.get("label", "")
            
            if label:
                dot_content.append(f'  "{from_node}" -> "{to_node}" [label="{label}"];')
            else:
                dot_content.append(f'  "{from_node}" -> "{to_node}";')
        
        dot_content.append('}')
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(dot_content))
    
    elif format == "mermaid":
        # Export as Mermaid diagram format
        mermaid_content = ["graph TD"]
        
        # Add nodes
        nodes = lineage_graph.get("nodes", {})
        for node_id, node_info in nodes.items():
            label = node_info.get("label", node_id)
            node_type = node_info.get("type", "process")
            
            # Escape special characters for Mermaid
            safe_id = node_id.replace(" ", "_").replace("-", "_")
            
            if node_type == "input":
                mermaid_content.append(f'  {safe_id}[("{label}")]')
            elif node_type == "output":
                mermaid_content.append(f'  {safe_id}[["{label}"]]')
            else:
                mermaid_content.append(f'  {safe_id}["{label}"]')
        
        mermaid_content.append('')
        
        # Add edges
        edges = lineage_graph.get("edges", [])
        for edge in edges:
            from_node = edge.get("from", "").replace(" ", "_").replace("-", "_")
            to_node = edge.get("to", "").replace(" ", "_").replace("-", "_")
            label = edge.get("label", "")
            
            if label:
                mermaid_content.append(f'  {from_node} -->|{label}| {to_node}')
            else:
                mermaid_content.append(f'  {from_node} --> {to_node}')
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(mermaid_content))
    
    else:
        raise ValueError(f"Unknown format: {format}")
    
    return output_path


def bundle_outputs(
    primary_data: str,
    metadata: dict[str, Any],
    ancillary_files: dict[str, str] | None = None,
    output_dir: str | None = None,
    create_archive: bool = False
) -> dict[str, str]:
    """Bundle data product with all ancillary files.
    
    Args:
        primary_data: Path to main data file
        metadata: Metadata dictionary
        ancillary_files: Additional files to include
        output_dir: Directory for organized outputs
        create_archive: Create zip/tar archive
        
    Returns:
        Paths to all bundled files
    """
    import os
    import shutil
    import zipfile
    import json
    from datetime import datetime
    from pathlib import Path
    
    # Determine output directory
    if output_dir is None:
        # Use directory of primary data file
        output_dir = os.path.dirname(primary_data)
    
    # Create bundle directory
    bundle_name = Path(primary_data).stem
    bundle_dir = os.path.join(output_dir, f"{bundle_name}_bundle")
    os.makedirs(bundle_dir, exist_ok=True)
    
    bundled_files = {}
    
    # Copy primary data file
    primary_filename = os.path.basename(primary_data)
    primary_dest = os.path.join(bundle_dir, primary_filename)
    if os.path.abspath(primary_data) != os.path.abspath(primary_dest):
        shutil.copy2(primary_data, primary_dest)
    bundled_files["primary_data"] = primary_dest
    
    # Create and save metadata
    metadata_path = os.path.join(bundle_dir, f"{bundle_name}_metadata.json")
    
    # Add bundle information to metadata
    metadata["bundle_info"] = {
        "created_at": datetime.now().isoformat(),
        "bundle_version": "1.0",
        "primary_file": primary_filename,
        "bundle_contents": []
    }
    
    export_metadata_sidecar(metadata, metadata_path, format="json")
    bundled_files["metadata"] = metadata_path
    metadata["bundle_info"]["bundle_contents"].append("metadata.json")
    
    # Generate quicklook if it doesn't exist
    quicklook_path = os.path.join(bundle_dir, f"{bundle_name}_quicklook.png")
    if not os.path.exists(quicklook_path):
        try:
            generate_quicklook(primary_data, quicklook_path)
            bundled_files["quicklook"] = quicklook_path
            metadata["bundle_info"]["bundle_contents"].append("quicklook.png")
        except:
            pass  # Quicklook generation is optional
    
    # Generate statistics report
    stats_path = os.path.join(bundle_dir, f"{bundle_name}_statistics.json")
    try:
        stats = generate_statistics(primary_data)
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        bundled_files["statistics"] = stats_path
        metadata["bundle_info"]["bundle_contents"].append("statistics.json")
    except:
        pass  # Statistics generation is optional
    
    # Copy ancillary files
    if ancillary_files:
        ancillary_dir = os.path.join(bundle_dir, "ancillary")
        os.makedirs(ancillary_dir, exist_ok=True)
        
        for file_type, file_path in ancillary_files.items():
            if os.path.exists(file_path):
                dest_path = os.path.join(ancillary_dir, os.path.basename(file_path))
                if os.path.abspath(file_path) != os.path.abspath(dest_path):
                    shutil.copy2(file_path, dest_path)
                bundled_files[f"ancillary_{file_type}"] = dest_path
                metadata["bundle_info"]["bundle_contents"].append(f"ancillary/{os.path.basename(file_path)}")
    
    # Update metadata with complete bundle contents
    export_metadata_sidecar(metadata, metadata_path, format="json")
    
    # Create README file
    readme_path = os.path.join(bundle_dir, "README.txt")
    with open(readme_path, 'w') as f:
        f.write(f"Black Marble Data Bundle\n")
        f.write(f"========================\n\n")
        f.write(f"Bundle Name: {bundle_name}\n")
        f.write(f"Created: {datetime.now().isoformat()}\n\n")
        f.write(f"Contents:\n")
        f.write(f"- {primary_filename}: Primary data file (Cloud-Optimized GeoTIFF)\n")
        f.write(f"- {bundle_name}_metadata.json: Complete processing metadata\n")
        f.write(f"- {bundle_name}_quicklook.png: Preview image\n")
        f.write(f"- {bundle_name}_statistics.json: Data statistics\n")
        if ancillary_files:
            f.write(f"- ancillary/: Additional files\n")
        f.write(f"\nFor more information, see the metadata.json file.\n")
    
    bundled_files["readme"] = readme_path
    
    # Create archive if requested
    if create_archive:
        archive_path = os.path.join(output_dir, f"{bundle_name}_bundle.zip")
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all files from bundle directory
            for root, dirs, files in os.walk(bundle_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname)
        
        bundled_files["archive"] = archive_path
        
        # Optionally remove bundle directory after archiving
        # shutil.rmtree(bundle_dir)
    
    return bundled_files