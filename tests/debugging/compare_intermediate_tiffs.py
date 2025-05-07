#!/usr/bin/env python3

import argparse
import os
import rasterio
import numpy as np
from pathlib import Path

def find_tiff_files(base_dir: Path) -> dict:
    """Finds all .tif and .TIFF files, excluding 'final' subdirectories.
    Returns a dictionary mapping relative paths to absolute paths."""
    tiff_files = {}
    for p in base_dir.rglob('*.tif'):
        if 'final' not in p.relative_to(base_dir).parts:
            tiff_files[p.relative_to(base_dir)] = p
    for p in base_dir.rglob('*.TIFF'):
        if 'final' not in p.relative_to(base_dir).parts:
            tiff_files[p.relative_to(base_dir)] = p
    return tiff_files

def compare_metadata(src1: rasterio.DatasetReader, src2: rasterio.DatasetReader, rel_path: Path):
    """Compares metadata and returns a dictionary of match statuses."""
    print(f"  Metadata Comparison for: {rel_path}")
    print(f"    Property         | {'File 1'.ljust(30)} | {'File 2'.ljust(30)} | Match")
    print(f"    -----------------|--------------------------------|--------------------------------|-------")

    matches = {}
    matches['shape'] = src1.shape == src2.shape
    matches['count'] = src1.count == src2.count
    matches['crs'] = src1.crs == src2.crs
    matches['transform'] = np.allclose(np.array(src1.transform), np.array(src2.transform), atol=1e-9)
    matches['dtypes'] = src1.dtypes == src2.dtypes
    matches['nodata'] = src1.nodatavals == src2.nodatavals

    m1_shape = str(src1.shape)
    m2_shape = str(src2.shape)
    print(f"    Shape            | {m1_shape.ljust(30)} | {m2_shape.ljust(30)} | {'YES' if matches['shape'] else 'NO'}")

    m1_count = str(src1.count)
    m2_count = str(src2.count)
    print(f"    Band Count       | {m1_count.ljust(30)} | {m2_count.ljust(30)} | {'YES' if matches['count'] else 'NO'}")

    m1_crs = str(src1.crs)
    m2_crs = str(src2.crs)
    print(f"    CRS              | {m1_crs.ljust(30)} | {m2_crs.ljust(30)} | {'YES' if matches['crs'] else 'NO'}")

    print(f"    Transform        | {'(see details below)'.ljust(30)} | {'(see details below)'.ljust(30)} | {'YES' if matches['transform'] else 'NO'}")
    if not matches['transform']:
        print(f"      File 1 Transform: {src1.transform}")
        print(f"      File 2 Transform: {src2.transform}")

    m1_dtypes = str(src1.dtypes)
    m2_dtypes = str(src2.dtypes)
    print(f"    Dtypes           | {m1_dtypes.ljust(30)} | {m2_dtypes.ljust(30)} | {'YES' if matches['dtypes'] else 'NO'}")

    m1_nodata = str(src1.nodatavals)
    m2_nodata = str(src2.nodatavals)
    print(f"    NoData Values    | {m1_nodata.ljust(30)} | {m2_nodata.ljust(30)} | {'YES' if matches['nodata'] else 'NO'}")
    print("-" * 80)
    
    # Overall metadata compatibility for pixel comparison
    pixel_compare_compatible = matches['shape'] and matches['count']
    # Overall full metadata match
    full_match = all(matches.values())
    
    return matches, pixel_compare_compatible, full_match

def compare_pixel_data(src1: rasterio.DatasetReader, src2: rasterio.DatasetReader, tolerance=1e-6):
    """Compares pixel data band by band and returns detailed results."""
    if src1.count != src2.count or src1.shape != src2.shape:
        # This case should ideally be caught before calling this function
        print("    Pixel data comparison skipped due to band count or shape mismatch.")
        return []

    print("  Pixel Data Comparison:")
    band_results = []
    for i in range(1, src1.count + 1):
        print(f"    Band {i}:")
        data1 = src1.read(i, masked=True)
        data2 = src2.read(i, masked=True)

        band_result = {
            "band_num": i,
            "nodata_mask_diff_pixels": 0,
            "differing_pixels": 0,
            "comparable_pixels": 0,
            "percent_diff_pixels": 0.0,
            "mean_abs_diff": 0.0,
            "min_raw_diff": np.nan,
            "max_raw_diff": np.nan,
            "mean_raw_diff": np.nan,
            "any_pixel_diff": False,
            "any_mask_diff": False,
        }

        # Compare masks
        mask_diff_count = np.sum(data1.mask != data2.mask)
        band_result["nodata_mask_diff_pixels"] = mask_diff_count
        if mask_diff_count > 0:
            print(f"      NoData masks differ at {mask_diff_count} pixel locations.")
            band_result["any_mask_diff"] = True
        else:
            print(f"      NoData masks are identical.")

        common_mask = data1.mask | data2.mask
        
        if np.all(common_mask):
            print(f"      All pixels are masked in at least one array for band {i}. Skipping numerical comparison.")
            band_results.append(band_result)
            continue

        valid_data1 = np.ma.array(data1, mask=common_mask).compressed()
        valid_data2 = np.ma.array(data2, mask=common_mask).compressed()
        
        if valid_data1.size == 0 and valid_data2.size == 0:
            print(f"      No commonly unmasked pixels to compare for band {i}.")
            band_results.append(band_result)
            continue
        
        band_result["comparable_pixels"] = valid_data1.size

        diff_abs = np.abs(valid_data1.astype(np.float64) - valid_data2.astype(np.float64))
        raw_diff = valid_data1.astype(np.float64) - valid_data2.astype(np.float64)
        
        num_differing_pixels = np.sum(diff_abs > tolerance)
        band_result["differing_pixels"] = num_differing_pixels
        
        total_comparable_pixels = diff_abs.size
        band_result["comparable_pixels"] = total_comparable_pixels
        
        if total_comparable_pixels > 0:
            percent_diff = (num_differing_pixels / total_comparable_pixels) * 100
            band_result["percent_diff_pixels"] = percent_diff
            print(f"      Differing pixels (tol={tolerance}): {num_differing_pixels} / {total_comparable_pixels} ({percent_diff:.2f}%)")
            if num_differing_pixels > 0:
                band_result["any_pixel_diff"] = True
                band_result["mean_abs_diff"] = np.mean(diff_abs[diff_abs > tolerance]) # Mean of actual differences
                
                print(f"        Min difference (abs): {np.min(diff_abs[diff_abs > tolerance]) if np.any(diff_abs > tolerance) else 'N/A'}")
                print(f"        Max difference (abs): {np.max(diff_abs)}") # Max of all absolute differences
                print(f"        Mean difference (abs of differing): {band_result['mean_abs_diff']:.4f}")
                
                band_result["min_raw_diff"] = np.min(raw_diff)
                band_result["max_raw_diff"] = np.max(raw_diff)
                band_result["mean_raw_diff"] = np.mean(raw_diff)
                print(f"        Min raw difference: {band_result['min_raw_diff']:.4f}")
                print(f"        Max raw difference: {band_result['max_raw_diff']:.4f}")
                print(f"        Mean raw difference: {band_result['mean_raw_diff']:.4f}")
        else:
            print(f"      No commonly unmasked pixels to compare for band {i}.")
        band_results.append(band_result)
    print("-" * 80)
    return band_results

def generate_enhanced_summary(stats: list, files_only_in_dir1: list, files_only_in_dir2: list, dir1_path: Path, dir2_path: Path):
    print("\n" + "=" * 30 + " ENHANCED SUMMARY " + "=" * 30)
    
    total_compared = len(stats)
    if total_compared == 0 and not files_only_in_dir1 and not files_only_in_dir2:
        print("No files found to compare in either directory (excluding 'final' subdirectories).")
        return

    print(f"Comparison between Directory 1 ('{dir1_path.name}') and Directory 2 ('{dir2_path.name}')")
    print(f"Total unique relative paths considered from Dir1: {total_compared + len(files_only_in_dir1)}")
    print(f"Files compared head-to-head: {total_compared}")

    # Counters
    files_with_any_diff = 0
    size_diff_count = 0
    meta_shape_diff = 0
    meta_count_diff = 0
    meta_crs_diff = 0
    meta_transform_diff = 0
    meta_dtype_diff = 0
    meta_nodata_val_diff = 0
    pixel_val_diff_count = 0
    nodata_mask_diff_count = 0
    
    size_diff_pixels_match_files = []
    problematic_pixel_files = [] # (rel_path, percent_diff, mask_diff_pixels)
    merged_file_stats = {"dir1_size_sum": 0, "dir2_size_sum": 0, "count": 0, "pixel_match_size_diff_count": 0}

    for s in stats:
        file_had_difference = False
        if s["size_differs"]:
            size_diff_count += 1
            file_had_difference = True
        
        if not s["metadata_details"]["shape"]: meta_shape_diff +=1; file_had_difference = True
        if not s["metadata_details"]["count"]: meta_count_diff +=1; file_had_difference = True
        if not s["metadata_details"]["crs"]: meta_crs_diff +=1; file_had_difference = True
        if not s["metadata_details"]["transform"]: meta_transform_diff +=1; file_had_difference = True
        if not s["metadata_details"]["dtypes"]: meta_dtype_diff +=1; file_had_difference = True
        if not s["metadata_details"]["nodata"]: meta_nodata_val_diff +=1; file_had_difference = True

        current_file_pixel_diff = False
        current_file_mask_diff = False
        if s["pixel_comparison_results"]:
            for band_res in s["pixel_comparison_results"]:
                if band_res["any_pixel_diff"]:
                    current_file_pixel_diff = True
                if band_res["any_mask_diff"]:
                    current_file_mask_diff = True
            
            if current_file_pixel_diff:
                pixel_val_diff_count += 1
                file_had_difference = True
                # Get max percent diff for this file
                max_perc = max(b["percent_diff_pixels"] for b in s["pixel_comparison_results"])
                max_mask_diff = max(b["nodata_mask_diff_pixels"] for b in s["pixel_comparison_results"])
                problematic_pixel_files.append((s["rel_path"], max_perc, max_mask_diff))


            if current_file_mask_diff:
                nodata_mask_diff_count +=1 # Count files with mask diffs, not total mask diff pixels
                file_had_difference = True # Already covered if pixel_val_diff_count is also true

            if s["size_differs"] and not current_file_pixel_diff and not current_file_mask_diff and s["metadata_fully_matches"]:
                size_diff_pixels_match_files.append(s["rel_path"])
        
        if file_had_difference:
            files_with_any_diff +=1

        if "_Merged_" in str(s["rel_path"]):
            merged_file_stats["count"] += 1
            merged_file_stats["dir1_size_sum"] += s["size1"]
            merged_file_stats["dir2_size_sum"] += s["size2"]
            if s["size_differs"] and not current_file_pixel_diff and not current_file_mask_diff and s["metadata_fully_matches"]:
                 merged_file_stats["pixel_match_size_diff_count"] +=1


    print(f"\n--- Overall Difference Statistics ---")
    print(f"Files with any difference (size, metadata, or pixels): {files_with_any_diff} / {total_compared}")
    print(f"  File size differences: {size_diff_count}")
    print(f"  Metadata differences:")
    print(f"    Shape mismatches: {meta_shape_diff}")
    print(f"    Band count mismatches: {meta_count_diff}")
    print(f"    CRS mismatches: {meta_crs_diff}")
    print(f"    Transform mismatches: {meta_transform_diff}")
    print(f"    Dtype mismatches: {meta_dtype_diff}")
    print(f"    NoData value mismatches: {meta_nodata_val_diff}")
    print(f"  Pixel data differences (value): {pixel_val_diff_count}")
    print(f"  NoData mask differences: {nodata_mask_diff_count}")

    print(f"\n--- File Size Analysis ---")
    if size_diff_count > 0:
        print(f"Found {size_diff_count} files with differing sizes.")
        if merged_file_stats["count"] > 0:
            avg_size1_merged = merged_file_stats["dir1_size_sum"] / merged_file_stats["count"] if merged_file_stats["count"] > 0 else 0
            avg_size2_merged = merged_file_stats["dir2_size_sum"] / merged_file_stats["count"] if merged_file_stats["count"] > 0 else 0
            print(f"  For {merged_file_stats['count']} '*_Merged_*.TIFF' files:")
            print(f"    Avg size in Dir1 ('{dir1_path.name}'): {avg_size1_merged:,.0f} bytes")
            print(f"    Avg size in Dir2 ('{dir2_path.name}'): {avg_size2_merged:,.0f} bytes")
            if avg_size2_merged > 0 and avg_size1_merged / avg_size2_merged > 1.5:
                 print(f"    Observation: Merged files in Dir1 are significantly larger (avg {avg_size1_merged/avg_size2_merged:.1f}x) than in Dir2.")
            elif avg_size1_merged > 0 and avg_size2_merged / avg_size1_merged > 1.5:
                 print(f"    Observation: Merged files in Dir2 are significantly larger (avg {avg_size2_merged/avg_size1_merged:.1f}x) than in Dir1.")


    if size_diff_pixels_match_files:
        print(f"\n  PATTERN: Size Differs, Pixels & Masks Match ({len(size_diff_pixels_match_files)} files)")
        print(f"    These files have different disk sizes but identical pixel values and NoData masks (where metadata fully matches).")
        print(f"    This often suggests differences in compression, internal tiling, block sizes, or metadata overhead, rather than core data changes.")
        if merged_file_stats["pixel_match_size_diff_count"] > 0:
            print(f"    Notably, {merged_file_stats['pixel_match_size_diff_count']} of these are '*_Merged_*.TIFF' files.")
        #for p in size_diff_pixels_match_files[:min(5, len(size_diff_pixels_match_files))]: # Print first 5
        #    print(f"      - {p}")

    print(f"\n--- Pixel Data & NoData Mask Analysis ---")
    if pixel_val_diff_count == 0 and nodata_mask_diff_count == 0 and total_compared > 0:
        print("  No pixel value or NoData mask differences found in any comparable files.")
    
    if problematic_pixel_files:
        print(f"  Found {len(problematic_pixel_files)} files with pixel value or mask differences.")
        problematic_pixel_files.sort(key=lambda x: (x[1], x[2]), reverse=True) # Sort by %diff, then mask_diff
        print(f"  Top problematic files (by % pixel difference, then NoData mask difference):")
        for rel_path, perc_diff, mask_px_diff in problematic_pixel_files[:min(10, len(problematic_pixel_files))]:
            print(f"    - {rel_path}: {perc_diff:.2f}% differing pixels, {mask_px_diff} NoData mask diff pixels.")
        
        # Example: B4_Merged_10P.TIFF and B4_9.TIFF from user's sample
        specific_files_of_interest = {Path("B4_Merged_10P.TIFF"), Path("B4_9.TIFF")}
        for s in stats:
            if s["rel_path"] in specific_files_of_interest:
                print(f"  Detailed look at {s['rel_path']}:")
                if s["size_differs"]: print(f"    Size: Dir1={s['size1']}, Dir2={s['size2']} (DIFFER)")
                for band_res in s["pixel_comparison_results"]:
                    if band_res["any_pixel_diff"] or band_res["any_mask_diff"]:
                        print(f"      Band {band_res['band_num']}: {band_res['percent_diff_pixels']:.2f}% pixels differ, {band_res['nodata_mask_diff_pixels']} mask pixels differ.")
                        if band_res["any_pixel_diff"]:
                             print(f"        Pixel Diffs (raw): min={band_res['min_raw_diff']:.1f}, max={band_res['max_raw_diff']:.1f}, mean={band_res['mean_raw_diff']:.1f}")


    print(f"\n--- Unique Files ---")
    if files_only_in_dir1:
        print(f"  Files only in Directory 1 ('{dir1_path.name}') ({len(files_only_in_dir1)}):")
        for p in files_only_in_dir1[:min(10, len(files_only_in_dir1))]: print(f"    - {p}")
        if len(files_only_in_dir1) > 10: print(f"    ... and {len(files_only_in_dir1)-10} more.")
    else:
        print(f"  No files found only in Directory 1 ('{dir1_path.name}').")

    if files_only_in_dir2:
        print(f"  Files only in Directory 2 ('{dir2_path.name}') ({len(files_only_in_dir2)}):")
        for p in files_only_in_dir2[:min(10, len(files_only_in_dir2))]: print(f"    - {p}")
        if len(files_only_in_dir2) > 10: print(f"    ... and {len(files_only_in_dir2)-10} more.")
        # Check for ntl_raw.tif specifically based on user output
        if Path("temp/ntl_raw.tif") in [Path(f) for f in files_only_in_dir2]:
            print(f"    Note: 'temp/ntl_raw.tif' is present in Dir2 ('{dir2_path.name}') and not Dir1, indicating a potential difference in intermediate NTL processing stages or paths.")

    else:
        print(f"  No files found only in Directory 2 ('{dir2_path.name}').")
    
    print("\n" + "="*78)


def main():
    parser = argparse.ArgumentParser(description="Compare intermediate TIFF files from two directories.")
    parser.add_argument("dir1", type=Path, help="Path to the first output directory.")
    parser.add_argument("dir2", type=Path, help="Path to the second output directory.")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Tolerance for float comparisons.")
    
    args = parser.parse_args()

    if not args.dir1.is_dir():
        print(f"Error: Directory not found: {args.dir1}")
        return
    if not args.dir2.is_dir():
        print(f"Error: Directory not found: {args.dir2}")
        return

    print(f"Comparing TIFFs in Dir1 ('{args.dir1.name}') vs Dir2 ('{args.dir2.name}') (excluding 'final' subdirs)\n")

    files1_map = find_tiff_files(args.dir1) # rel_path -> abs_path
    files2_map = find_tiff_files(args.dir2)

    all_rel_paths = set(files1_map.keys()) | set(files2_map.keys())
    
    collected_stats = []
    files_only_in_dir1 = []
    files_only_in_dir2 = []

    for rel_path in sorted(list(all_rel_paths)):
        abs_path1 = files1_map.get(rel_path)
        abs_path2 = files2_map.get(rel_path)
        
        current_file_stats = {
            "rel_path": rel_path,
            "size1": None, "size2": None, "size_differs": None,
            "metadata_details": {}, "metadata_pixel_compatible": False, "metadata_fully_matches": False,
            "pixel_comparison_results": [],
            "error": None
        }

        if abs_path1 and not abs_path2:
            print(f"\nFile {rel_path} exists in Dir1 ('{args.dir1.name}') but NOT in Dir2 ('{args.dir2.name}').")
            files_only_in_dir1.append(rel_path)
            print("=" * 80)
            continue
        if not abs_path1 and abs_path2:
            print(f"\nFile {rel_path} exists in Dir2 ('{args.dir2.name}') but NOT in Dir1 ('{args.dir1.name}').")
            files_only_in_dir2.append(rel_path)
            print("=" * 80)
            continue
        
        # Both files exist, proceed with comparison
        print(f"\nComparing: {rel_path}")
        current_file_stats["size1"] = abs_path1.stat().st_size
        current_file_stats["size2"] = abs_path2.stat().st_size
        current_file_stats["size_differs"] = current_file_stats["size1"] != current_file_stats["size2"]
        
        print(f"  File Size: Dir1={current_file_stats['size1']} bytes, Dir2={current_file_stats['size2']} bytes. {'Match' if not current_file_stats['size_differs'] else 'DIFFER'}")

        try:
            with rasterio.open(abs_path1) as src1, rasterio.open(abs_path2) as src2:
                meta_matches, pixel_compatible, full_meta_match = compare_metadata(src1, src2, rel_path)
                current_file_stats["metadata_details"] = meta_matches
                current_file_stats["metadata_pixel_compatible"] = pixel_compatible
                current_file_stats["metadata_fully_matches"] = full_meta_match

                if pixel_compatible:
                    pixel_results = compare_pixel_data(src1, src2, tolerance=args.tolerance)
                    current_file_stats["pixel_comparison_results"] = pixel_results
                else:
                    print("    Pixel data comparison skipped due to fundamental metadata mismatches (shape or band count).")
        except Exception as e:
            error_msg = f"Error comparing raster data for {rel_path}: {e}"
            print(f"  {error_msg}")
            current_file_stats["error"] = error_msg
        
        collected_stats.append(current_file_stats)
        print("=" * 80)

    generate_enhanced_summary(collected_stats, files_only_in_dir1, files_only_in_dir2, args.dir1, args.dir2)

if __name__ == "__main__":
    main()