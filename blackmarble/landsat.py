"""
Landsat data download & processing:
  - LandsatUtil: wrapper for WRS-2 lookup & S3 interactions
  - download_tiles_for_band: fetch band + QA tiles for given date & bbox
  - download_tiles_for_band_with_composites: include composite history
  - process_band_data: warp & merge bands into final TIFFs
"""
# Landsat data download & processing
import os
import glob
import numpy as np
import boto3
from importlib.resources import files
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any, Union
from boto3.session import Session
from mypy_boto3_s3.client import S3Client

# Default path to WRS2 descending shapefile bundled with this package
DEFAULT_WRS_SHAPEFILE: str = str(
    files("blackmarble").joinpath("data", "WRS2_descending_0", "WRS2_descending.shp")
)
from osgeo import gdal
from osgeo import ogr
# Enable GDAL/OGR exceptions (avoids FutureWarning and uses Python exceptions)
ogr.UseExceptions()
import shapely.wkt
import shapely.geometry

# Read AWS credentials from environment
aws_access_key: Optional[str] = os.environ.get('AWS_ACCESS_KEY')
aws_secret_key: Optional[str] = os.environ.get('AWS_ACCESS_SECRET')

class LandsatUtil:
    """Perform conversion between lat/lon and WRS-2 + download bands from USGS S3."""
    dir_cache: Dict[str, List[Dict[str, str]]] = {}
    shapefile: ogr.DataSource
    layer: ogr.Layer
    polygons: List[Tuple[shapely.geometry.Polygon, int, int]]

    def __init__(self, shapefile: Optional[str] = None) -> None:
        # Use bundled default WRS-2 shapefile if none provided
        if shapefile is None:
            shapefile = DEFAULT_WRS_SHAPEFILE
        self.shapefile = ogr.Open(shapefile)
        self.layer = self.shapefile.GetLayer(0)
        self.polygons = []
        for i in range(self.layer.GetFeatureCount()):
            feat = self.layer.GetFeature(i)
            path = feat['PATH']
            row = feat['ROW']
            geom = feat.GetGeometryRef()
            shape = shapely.wkt.loads(geom.ExportToWkt())
            self.polygons.append((shape, path, row))

    def get_wrs(self, lat: float, lon: float) -> List[Dict[str, int]]:
        """Return list of {'path', 'row'} for given lat/lon."""
        pt = shapely.geometry.Point(lon, lat)
        res: List[Dict[str, int]] = []
        for poly, path, row in self.polygons:
            if pt.within(poly):
                res.append({'path': path, 'row': row})
        return res

    def filter_landsat_data_dir(self, year: int, month: int, day: int, path: str, row: str) -> Optional[str]:
        """Select the USGS S3 prefix for the given date and path/row."""
        yr: str = str(year)
        mo: str = str(month).zfill(2)
        da: str = str(day).zfill(2)
        prefix: str = f'collection02/level-2/standard/oli-tirs/{yr}/{path}/{row}/'
        if prefix in self.dir_cache:
            directories: List[Dict[str, str]] = self.dir_cache[prefix]
        else:
            # Use provided AWS keys or default credential chain
            session: Session
            if aws_access_key and aws_secret_key:
                session = boto3.Session(
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key
                )
            else:
                session = boto3.Session()
            s3: S3Client = session.client('s3')
            resp = s3.list_objects_v2(
                Bucket='usgs-landsat', Prefix=prefix,
                RequestPayer='requester', Delimiter='/'
            )
            directories = resp.get('CommonPrefixes', [])
            self.dir_cache[prefix] = directories

        target: datetime = datetime.strptime(yr + mo + da, '%Y%m%d')
        selected: Optional[str] = None
        for entry in directories:
            dr: List[str] = entry['Prefix'].split('/')[-2].split('_')[3:5]
            start: datetime = datetime.strptime(dr[0], '%Y%m%d')
            end: datetime = datetime.strptime(dr[1], '%Y%m%d')
            if start <= target <= end:
                selected = entry['Prefix']
                break
        return selected

    def download_landsat_band(self, directory: str, band: Union[int, str], output_path: str) -> None:
        """Download a specific band (or QA) file from USGS S3 to local output_path."""
        scene: str = directory.split('/')[-2]
        key: str
        if band in ('QA_PIXEL', 'QA_RADSAT', 'QA_AEROSOL'):
            # Adjust suffix for QA_AEROSOL if needed based on actual filenames
            suffix: str = {'QA_PIXEL': '_QA_PIXEL.TIF',
                      'QA_RADSAT': '_QA_RADSAT.TIF',
                      'QA_AEROSOL': '_SR_QA_AEROSOL.TIF'}[band] # Assuming SR_QA_AEROSOL
            key = directory + scene + suffix
        else:
            key = f"{directory}{scene}_SR_B{band}.TIF"
        # Use provided AWS keys or default credential chain
        session: Session
        if aws_access_key and aws_secret_key:
            session = boto3.Session(
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key
            )
        else:
            session = boto3.Session()
        s3: S3Client = session.client('s3')
        # Ensure the directory for the output file exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        s3.download_file('usgs-landsat', key, output_path,
                         ExtraArgs={'RequestPayer': 'requester'})

def download_tiles_for_band(band: Union[int, str], year: int, month: int, day: int,
                            lat1: float, lon1: float, lat2: float, lon2: float,
                            temp_dir: str, # Add temp_dir argument
                            ignore_missing: bool = False) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Download source, QA_PIXEL, QA_RADSAT, QA_AEROSOL for all overlapping tiles."""
    landsat = LandsatUtil()
    wrs1: Dict[str, int] = landsat.get_wrs(lat1, lon1)[0]
    wrs2: Dict[str, int] = landsat.get_wrs(lat2, lon2)[0]
    paths = range(min(wrs1['path'], wrs2['path']), max(wrs1['path'], wrs2['path'])+1)
    rows = range(min(wrs1['row'], wrs2['row']), max(wrs1['row'], wrs2['row'])+1)
    downloads: List[str] = []
    pixels: List[str] = []
    radsat: List[str] = []
    aerosol: List[str] = []
    for p in paths:
        for r in rows:
            d: Optional[str] = landsat.filter_landsat_data_dir(year, month, day,
                                                str(p).zfill(3), str(r).zfill(3))
            if not d:
                if ignore_missing:
                    print(f"Warning: No data found for tile P{p:03d}R{r:03d} on {year}-{month}-{day}. Skipping.")
                    continue
                else:
                    raise RuntimeError(f'No data for tile P{p:03d}R{r:03d} on {year}-{month}-{day}')
            base: str = d.split('/')[-2]
            # Construct paths within the temp_dir
            out_src: str = os.path.join(temp_dir, f'{base}_SR_B{band}.TIF')
            if not os.path.exists(out_src):
                print(f"Downloading {out_src}...")
                landsat.download_landsat_band(d, band, out_src)
            else:
                print(f"File {out_src} exists..")
            downloads.append(out_src)

            # Define QA types and their corresponding suffixes/lists
            qa_types = {
                'QA_PIXEL': ('_QA_PIXEL.TIF', pixels),
                'QA_RADSAT': ('_QA_RADSAT.TIF', radsat),
                'QA_AEROSOL': ('_SR_QA_AEROSOL.TIF', aerosol) # Assuming this suffix
            }

            for qa_key, (suffix, target_list) in qa_types.items():
                out_qa: str = os.path.join(temp_dir, f'{base}{suffix}')
                if not os.path.exists(out_qa):
                    print(f"Downloading {out_qa}...")
                    landsat.download_landsat_band(d, qa_key, out_qa)
                else:
                    print(f"File {out_qa} exists..")
                target_list.append(out_qa)

    return downloads, pixels, radsat, aerosol

def download_tiles_for_band_with_composites(band: Union[int, str], year: int, month: int, day: int,
                                            lat1: float, lon1: float, lat2: float, lon2: float,
                                            temp_dir: str, # Add temp_dir argument
                                            composite_history_months: int = 12) -> List[Tuple[List[str], List[str], List[str], List[str]]]:
    """Download current and past composites for a band, ignoring missing where needed."""
    tiles: List[Tuple[List[str], List[str], List[str], List[str]]] = [download_tiles_for_band(band, year, month, day,
                                    lat1, lon1, lat2, lon2, temp_dir)] # Pass temp_dir
    for comp in range(1, composite_history_months+1):
        y: int = year
        m: int = month - comp
        if m < 1:
            y -= 1
            m += 12
        try:
            print(f"Downloading composite for {y}-{m}-01...")
            t: Tuple[List[str], List[str], List[str], List[str]] = download_tiles_for_band(band, y, m, 1,
                                        lat1, lon1, lat2, lon2, temp_dir, # Pass temp_dir
                                        ignore_missing=True)
            # Only add if data was actually found
            if any(t): # Check if any list in the tuple is non-empty
                tiles.append(t)
            else:
                print(f"No data found for composite {y}-{m}-01. Skipping.")
        except RuntimeError as e:
            print(f"Error downloading composite for {y}-{m}-01: {e}. Skipping.")
            continue
    return tiles

def process_band_data(band: Union[int, str], year: int, month: int, day: int,
                      lat1: float, lon1: float, lat2: float, lon2: float,
                      output_prefix: str, temp_dir: str) -> Tuple[List[str], List[str], List[str], List[str]]: # Add temp_dir
    """Download, warp, and merge source + QA into multi-scene TIFFs."""
    composites: List[Tuple[List[str], List[str], List[str], List[str]]] = download_tiles_for_band_with_composites(
        band, year, month, day, lat1, lon1, lat2, lon2, temp_dir) # Pass temp_dir
    outputs: List[str] = []
    pixel_tifs: List[str] = []
    radsat_tifs: List[str] = []
    aerosol_tifs: List[str] = []
    for idx, (srcs, pxs, rxs, axs) in enumerate(composites):
        # Skip processing if no source files were found for this composite period
        if not srcs:
            print(f"Skipping processing for composite index {idx} due to missing source files.")
            continue

        print(f"Merging base {idx}")
        outb: str = f'{output_prefix}{idx}.TIFF'
        # Warp composite band to output TIFF
        warp_opts = gdal.WarpOptions(
            format='GTiff',
            creationOptions=['COMPRESS=LZW', 'TILED=YES'],
            resampleAlg='bilinear',
            multithread=True
        )
        gdal.Warp(outb, srcs, options=warp_opts)
        outputs.append(outb)

        for name, tifs, nod in (
            ('P', pxs, {'srcNodata': 1, 'dstNodata': 0}),
            ('R', rxs, {'srcNodata': 1, 'dstNodata': 0}),
            ('A', axs, {'srcNodata': 1, 'dstNodata': 0}),
        ):
            # Skip QA processing if no corresponding QA files were found
            if not tifs:
                print(f"Skipping processing for QA type {name}, index {idx} due to missing files.")
                # Append a placeholder or handle downstream logic if needed
                if name == 'P': pixel_tifs.append("") # Or None, depending on how it's handled
                elif name == 'R': radsat_tifs.append("")
                else: aerosol_tifs.append("")
                continue

            print(f"Merging {name} {idx}")
            outf: str = f'{output_prefix}{idx}{name}.TIFF'
            # Warp QA/Radsat/Aerosol bands with nodata handling
            warp_opts = gdal.WarpOptions(
                format='GTiff',
                creationOptions=['COMPRESS=LZW', 'TILED=YES'],
                resampleAlg='near', # Use nearest neighbor for QA flags
                multithread=True,
                srcNodata=nod['srcNodata'],
                dstNodata=nod['dstNodata']
            )
            gdal.Warp(outf, tifs, options=warp_opts)
            if name == 'P':
                pixel_tifs.append(outf)
            elif name == 'R':
                radsat_tifs.append(outf)
            else:
                aerosol_tifs.append(outf)

    # Filter out empty placeholders if added
    pixel_tifs = [p for p in pixel_tifs if p]
    radsat_tifs = [r for r in radsat_tifs if r]
    aerosol_tifs = [a for a in aerosol_tifs if a]

    return outputs, pixel_tifs, radsat_tifs, aerosol_tifs