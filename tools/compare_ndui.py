#!/usr/bin/env python3
"""Compare NDUI outputs between old and new calculation methods.

This tool helps evaluate the impact of the index-first calculation approach
by comparing NDUI outputs from the legacy method vs the new method.
"""

import json
from pathlib import Path

import numpy as np
import rasterio
import click


@click.command()
@click.argument('old_path', type=click.Path(exists=True))
@click.argument('new_path', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='comparison_output', 
              help='Directory to save comparison outputs')
@click.option('--verbose', '-v', is_flag=True, help='Print detailed statistics')
def compare_ndui(old_path: str, new_path: str, output_dir: str, verbose: bool):
    """Generate comparison statistics and difference raster between NDUI outputs.
    
    OLD_PATH: Path to NDUI from legacy band-first calculation
    NEW_PATH: Path to NDUI from new index-first calculation
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Comparing NDUI outputs")
    print(f"  Old (legacy): {old_path}")
    print(f"  New (index-first): {new_path}")
    print(f"  Output directory: {output_dir}")
    
    # Read the rasters
    with rasterio.open(old_path) as old_src:
        old_ndui = old_src.read(1)
        profile = old_src.profile
        
    with rasterio.open(new_path) as new_src:
        new_ndui = new_src.read(1)
        
    # Verify shapes match
    if old_ndui.shape != new_ndui.shape:
        raise ValueError(f"Shape mismatch: {old_ndui.shape} vs {new_ndui.shape}")
    
    # Calculate differences
    diff = new_ndui - old_ndui
    abs_diff = np.abs(diff)
    
    # Mask invalid pixels
    valid_mask = ~(np.isnan(old_ndui) | np.isnan(new_ndui))
    valid_pixels = np.sum(valid_mask)
    
    # Calculate statistics on valid pixels only
    if valid_pixels > 0:
        stats = {
            "total_pixels": int(old_ndui.size),
            "valid_pixels": int(valid_pixels),
            "valid_percentage": float(valid_pixels / old_ndui.size * 100),
            "mean_difference": float(np.mean(diff[valid_mask])),
            "median_difference": float(np.median(diff[valid_mask])),
            "std_difference": float(np.std(diff[valid_mask])),
            "pixels_changed_5pct": int(np.sum(abs_diff[valid_mask] > 0.05)),
            "pixels_changed_10pct": int(np.sum(abs_diff[valid_mask] > 0.10)),
            "pixels_changed_5pct_ratio": float(np.sum(abs_diff[valid_mask] > 0.05) / valid_pixels),
            "pixels_changed_10pct_ratio": float(np.sum(abs_diff[valid_mask] > 0.10) / valid_pixels),
            "max_increase": float(np.max(diff[valid_mask])),
            "max_decrease": float(np.min(diff[valid_mask])),
            "percentile_5": float(np.percentile(diff[valid_mask], 5)),
            "percentile_95": float(np.percentile(diff[valid_mask], 95)),
        }
        
        # Agricultural area analysis (areas with significant vegetation)
        # Define agricultural as areas with old NDUI < 0.3 (not urban) but > 0
        agricultural_mask = valid_mask & (old_ndui < 0.3) & (old_ndui > 0)
        ag_pixels = np.sum(agricultural_mask)
        
        if ag_pixels > 0:
            stats["agricultural_analysis"] = {
                "pixels": int(ag_pixels),
                "mean_difference": float(np.mean(diff[agricultural_mask])),
                "median_difference": float(np.median(diff[agricultural_mask])),
                "pixels_increased": int(np.sum(diff[agricultural_mask] > 0.05)),
                "pixels_decreased": int(np.sum(diff[agricultural_mask] < -0.05)),
            }
        
        # Urban area analysis (areas already identified as urban)
        urban_mask = valid_mask & (old_ndui >= 0.3)
        urban_pixels = np.sum(urban_mask)
        
        if urban_pixels > 0:
            stats["urban_analysis"] = {
                "pixels": int(urban_pixels),
                "mean_difference": float(np.mean(diff[urban_mask])),
                "median_difference": float(np.median(diff[urban_mask])),
                "pixels_increased": int(np.sum(diff[urban_mask] > 0.05)),
                "pixels_decreased": int(np.sum(diff[urban_mask] < -0.05)),
            }
    else:
        stats = {
            "error": "No valid pixels found for comparison"
        }
    
    # Print summary statistics
    print("\n📈 Comparison Statistics:")
    print(f"  Valid pixels: {stats.get('valid_pixels', 0):,} ({stats.get('valid_percentage', 0):.1f}%)")
    
    if 'mean_difference' in stats:
        print(f"\n  Overall changes:")
        print(f"    Mean difference: {stats['mean_difference']:.4f}")
        print(f"    Median difference: {stats['median_difference']:.4f}")
        print(f"    Std deviation: {stats['std_difference']:.4f}")
        print(f"    Max increase: {stats['max_increase']:.4f}")
        print(f"    Max decrease: {stats['max_decrease']:.4f}")
        
        print(f"\n  Significant changes:")
        print(f"    >5% change: {stats['pixels_changed_5pct']:,} pixels ({stats['pixels_changed_5pct_ratio']*100:.1f}%)")
        print(f"    >10% change: {stats['pixels_changed_10pct']:,} pixels ({stats['pixels_changed_10pct_ratio']*100:.1f}%)")
        
        if 'agricultural_analysis' in stats:
            ag = stats['agricultural_analysis']
            print(f"\n  Agricultural areas (NDUI < 0.3):")
            print(f"    Pixels: {ag['pixels']:,}")
            print(f"    Mean change: {ag['mean_difference']:.4f}")
            print(f"    Increased: {ag['pixels_increased']:,}")
            print(f"    Decreased: {ag['pixels_decreased']:,}")
            
        if 'urban_analysis' in stats:
            urb = stats['urban_analysis']
            print(f"\n  Urban areas (NDUI >= 0.3):")
            print(f"    Pixels: {urb['pixels']:,}")
            print(f"    Mean change: {urb['mean_difference']:.4f}")
            print(f"    Increased: {urb['pixels_increased']:,}")
            print(f"    Decreased: {urb['pixels_decreased']:,}")
    
    # Save statistics to JSON
    stats_path = output_path / "comparison_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n💾 Saved statistics to: {stats_path}")
    
    # Create difference raster with classification
    diff_classes = np.zeros_like(diff, dtype=np.uint8)
    diff_classes[diff < -0.05] = 1  # Decreased (darker)
    diff_classes[np.abs(diff) <= 0.05] = 2  # Unchanged
    diff_classes[diff > 0.05] = 3  # Increased (brighter)
    diff_classes[~valid_mask] = 0  # Invalid/no data
    
    # Save difference classes raster
    profile.update(dtype=rasterio.uint8, count=1, nodata=0)
    diff_class_path = output_path / "ndui_difference_classes.tif"
    with rasterio.open(diff_class_path, "w", **profile) as dst:
        dst.write(diff_classes, 1)
    print(f"  Saved difference classes to: {diff_class_path}")
    
    # Save raw difference raster
    profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)
    diff_raw_path = output_path / "ndui_difference_raw.tif"
    with rasterio.open(diff_raw_path, "w", **profile) as dst:
        dst.write(diff.astype(np.float32), 1)
    print(f"  Saved raw differences to: {diff_raw_path}")
    
    # Create histogram data if verbose
    if verbose:
        print("\n📊 Difference Histogram:")
        hist, bins = np.histogram(diff[valid_mask], bins=50)
        for i in range(len(hist)):
            if hist[i] > 0:
                bar_width = int(hist[i] / valid_pixels * 200)  # Scale to max 200 chars
                bar = "█" * bar_width
                print(f"  [{bins[i]:6.3f} to {bins[i+1]:6.3f}]: {bar} ({hist[i]:,})")
    
    print("\n✅ Comparison complete!")
    print("\nClass legend for difference raster:")
    print("  0 = No data")
    print("  1 = Decreased (< -0.05)")
    print("  2 = Unchanged (-0.05 to 0.05)")
    print("  3 = Increased (> 0.05)")


if __name__ == "__main__":
    compare_ndui()