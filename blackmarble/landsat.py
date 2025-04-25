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
# Default path to WRS2 descending shapefile bundled with this package
DEFAULT_WRS_SHAPEFILE = str(
    files("blackmarble").joinpath("data", "WRS2_descending_0", "WRS2_descending.shp")
)
from osgeo import gdal
from osgeo import ogr
# Enable GDAL/OGR exceptions (avoids FutureWarning and uses Python exceptions)
ogr.UseExceptions()
import shapely.wkt
import shapely.geometry

# Read AWS credentials from environment
aws_access_key = os.environ.get('AWS_ACCESS_KEY')
aws_secret_key = os.environ.get('AWS_ACCESS_SECRET')

class LandsatUtil:
    """Perform conversion between lat/lon and WRS-2 + download bands from USGS S3."""
    dir_cache = {}

    def __init__(self, shapefile=None):
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

    def get_wrs(self, lat, lon):
        """Return list of {'path', 'row'} for given lat/lon."""
        pt = shapely.geometry.Point(lon, lat)
        res = []
        for poly, path, row in self.polygons:
            if pt.within(poly):
                res.append({'path': path, 'row': row})
        return res

    def filter_landsat_data_dir(self, year, month, day, path, row):
        """Select the USGS S3 prefix for the given date and path/row."""
        yr = str(year)
        mo = str(month).zfill(2)
        da = str(day).zfill(2)
        prefix = f'collection02/level-2/standard/oli-tirs/{yr}/{path}/{row}/'
        if prefix in self.dir_cache:
            directories = self.dir_cache[prefix]
        else:
            # Use provided AWS keys or default credential chain
            if aws_access_key and aws_secret_key:
                session = boto3.Session(
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key
                )
            else:
                session = boto3.Session()
            s3 = session.client('s3')
            resp = s3.list_objects_v2(
                Bucket='usgs-landsat', Prefix=prefix,
                RequestPayer='requester', Delimiter='/'
            )
            directories = resp.get('CommonPrefixes', [])
            self.dir_cache[prefix] = directories

        target = datetime.strptime(yr + mo + da, '%Y%m%d')
        selected = None
        for entry in directories:
            dr = entry['Prefix'].split('/')[-2].split('_')[3:5]
            start = datetime.strptime(dr[0], '%Y%m%d')
            end = datetime.strptime(dr[1], '%Y%m%d')
            if start <= target <= end:
                selected = entry['Prefix']
                break
        return selected

    def download_landsat_band(self, directory, band, output_path):
        """Download a specific band (or QA) file from USGS S3 to local output_path."""
        scene = directory.split('/')[-2]
        if band in ('QA_PIXEL', 'QA_RADSAT', 'QA_AEROSOL'):
            suffix = {'QA_PIXEL': '_QA_PIXEL.TIF',
                      'QA_RADSAT': '_QA_RADSAT.TIF',
                      'QA_AEROSOL': '_SR_QA_AEROSOL.TIF'}[band]
            key = directory + scene + suffix
        else:
            key = f"{directory}{scene}_SR_B{band}.TIF"
        # Use provided AWS keys or default credential chain
        if aws_access_key and aws_secret_key:
            session = boto3.Session(
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key
            )
        else:
            session = boto3.Session()
        s3 = session.client('s3')
        s3.download_file('usgs-landsat', key, output_path,
                         ExtraArgs={'RequestPayer': 'requester'})

def download_tiles_for_band(band, year, month, day,
                            lat1, lon1, lat2, lon2,
                            ignore_missing=False):
    """Download source, QA_PIXEL, QA_RADSAT, QA_AEROSOL for all overlapping tiles."""
    landsat = LandsatUtil()
    wrs1 = landsat.get_wrs(lat1, lon1)[0]
    wrs2 = landsat.get_wrs(lat2, lon2)[0]
    paths = range(min(wrs1['path'], wrs2['path']), max(wrs1['path'], wrs2['path'])+1)
    rows = range(min(wrs1['row'], wrs2['row']), max(wrs1['row'], wrs2['row'])+1)
    downloads, pixels, radsat, aerosol = [], [], [], []
    for p in paths:
        for r in rows:
            d = landsat.filter_landsat_data_dir(year, month, day,
                                                str(p).zfill(3), str(r).zfill(3))
            if not d:
                if ignore_missing:
                    continue
                else:
                    raise RuntimeError('No data for tile')
            base = d.split('/')[-2]
            out_src = f'outputs/temp/{base}_SR_B{band}.TIF'
            if not os.path.exists(out_src):
                landsat.download_landsat_band(d, band, out_src)
            downloads.append(out_src)
            for qa in ['QA_PIXEL', 'QA_RADSAT', 'QA_AEROSOL']:
                out_qa = f'outputs/temp/{base}_{qa}.TIF'
                if not os.path.exists(out_qa):
                    landsat.download_landsat_band(d, qa, out_qa)
                {'pixel': pixels, 'radsat': radsat, 'aerosol': aerosol}[qa.lower().split('_')[-1]].append(out_qa)
    return downloads, pixels, radsat, aerosol

def download_tiles_for_band_with_composites(band, year, month, day,
                                            lat1, lon1, lat2, lon2,
                                            composite_history_months=12):
    """Download current and past composites for a band, ignoring missing where needed."""
    tiles = [download_tiles_for_band(band, year, month, day,
                                    lat1, lon1, lat2, lon2)]
    for comp in range(1, composite_history_months+1):
        y, m = year, month-comp
        if m < 1:
            y -= 1
            m += 12
        try:
            t = download_tiles_for_band(band, y, m, 1,
                                        lat1, lon1, lat2, lon2,
                                        ignore_missing=True)
            tiles.append(t)
        except RuntimeError:
            continue
    return tiles

def process_band_data(band, year, month, day,
                      lat1, lon1, lat2, lon2,
                      output_prefix):
    """Download, warp, and merge source + QA into multi-scene TIFFs."""
    composites = download_tiles_for_band_with_composites(
        band, year, month, day, lat1, lon1, lat2, lon2)
    outputs, pixel_tifs, radsat_tifs, aerosol_tifs = [], [], [], []
    for idx, (srcs, pxs, rxs, axs) in enumerate(composites):
        outb = f'{output_prefix}{idx}.TIFF'
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
            outf = f'{output_prefix}{idx}{name}.TIFF'
            # Warp QA/Radsat/Aerosol bands with nodata handling
            warp_opts = gdal.WarpOptions(
                format='GTiff',
                creationOptions=['COMPRESS=LZW', 'TILED=YES'],
                resampleAlg='bilinear',
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
    return outputs, pixel_tifs, radsat_tifs, aerosol_tifs