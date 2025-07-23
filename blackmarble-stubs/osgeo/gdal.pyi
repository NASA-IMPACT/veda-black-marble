"""Type stubs for osgeo.gdal."""

from typing import Any, Sequence

# Constants
GA_ReadOnly: int
GA_Update: int
GDT_Byte: int
GDT_UInt16: int
GDT_Int16: int
GDT_UInt32: int
GDT_Int32: int
GDT_Float32: int
GDT_Float64: int

def UseExceptions() -> None: ...
def AllRegister() -> None: ...
def Open(filename: str, mode: int = ...) -> Dataset | None: ...
def BuildVRT(
    dst_filename: str,
    src_ds_or_filenames: Sequence[str] | Sequence[Dataset],
    options: Sequence[str] | None = None,
    callback: Any | None = None,
    callback_data: Any | None = None
) -> Dataset | None: ...

def Warp(
    destNameOrDestDS: str | Dataset,
    srcDSOrSrcDSTab: Dataset | Sequence[Dataset] | str | Sequence[str],
    options: WarpOptions | None = None,
    callback: Any | None = None,
    callback_data: Any | None = None
) -> Dataset | None: ...

def WarpOptions(
    options: Sequence[str] | None = None,
    format: str | None = None,
    outputBounds: Sequence[float] | None = None,
    outputBoundsSRS: str | None = None,
    xRes: float | None = None,
    yRes: float | None = None,
    targetAlignedPixels: bool = False,
    width: int = 0,
    height: int = 0,
    srcSRS: str | None = None,
    dstSRS: str | None = None,
    srcAlpha: bool = False,
    dstAlpha: bool = False,
    warpOptions: Sequence[str] | None = None,
    errorThreshold: float | None = None,
    warpMemoryLimit: float | None = None,
    creationOptions: Sequence[str] | None = None,
    outputType: int = 0,
    workingType: int = 0,
    resampleAlg: str | None = None,
    srcNodata: float | str | None = None,
    dstNodata: float | str | None = None,
    multithread: bool = False,
    tps: bool = False,
    rpc: bool = False,
    geoloc: bool = False,
    polynomialOrder: int | None = None,
    transformerOptions: Sequence[str] | None = None,
    cutlineDSName: str | None = None,
    cutlineLayer: str | None = None,
    cutlineWhere: str | None = None,
    cutlineSQL: str | None = None,
    cutlineBlend: float | None = None,
    cropToCutline: bool = False,
    copyMetadata: bool = True,
    metadataConflictValue: str | None = None,
    setColorInterpretation: bool = False,
    callback: Any | None = None,
    callback_data: Any | None = None
) -> WarpOptions: ...

class Dataset:
    RasterXSize: int
    RasterYSize: int
    RasterCount: int
    
    def GetRasterBand(self, band: int) -> Band: ...
    def GetProjection(self) -> str: ...
    def GetGeoTransform(self) -> tuple[float, float, float, float, float, float]: ...
    def SetGeoTransform(self, geotransform: Sequence[float]) -> int: ...
    def SetProjection(self, projection: str) -> int: ...
    def FlushCache(self) -> None: ...
    def GetDriver(self) -> Driver: ...
    def GetMetadata(self, domain: str = "") -> dict[str, str]: ...
    def SetMetadata(self, metadata: dict[str, str], domain: str = "") -> int: ...
    def BuildOverviews(
        self,
        resampling: str = "NEAREST",
        overviewlist: Sequence[int] | None = None,
        callback: Any | None = None,
        callback_data: Any | None = None
    ) -> int: ...

class Band:
    XSize: int
    YSize: int
    DataType: int
    
    def ReadAsArray(
        self,
        xoff: int = 0,
        yoff: int = 0,
        win_xsize: int | None = None,
        win_ysize: int | None = None,
        buf_xsize: int | None = None,
        buf_ysize: int | None = None,
        buf_type: int | None = None,
        resample_alg: int = 0,
        callback: Any | None = None,
        callback_data: Any | None = None
    ) -> Any: ...
    def WriteArray(
        self,
        array: Any,
        xoff: int = 0,
        yoff: int = 0,
        resample_alg: int = 0,
        callback: Any | None = None,
        callback_data: Any | None = None
    ) -> int: ...
    def GetNoDataValue(self) -> float | None: ...
    def SetNoDataValue(self, value: float) -> int: ...
    def GetStatistics(self, approx_ok: int, force: int) -> tuple[float, float, float, float]: ...
    def ComputeStatistics(self, approx_ok: int) -> tuple[float, float, float, float]: ...
    def GetHistogram(
        self,
        min: float = -0.5,
        max: float = 255.5,
        buckets: int = 256,
        include_out_of_range: int = 0,
        approx_ok: int = 1,
        callback: Any | None = None,
        callback_data: Any | None = None
    ) -> list[int]: ...

class Driver:
    def Create(
        self,
        filename: str,
        xsize: int,
        ysize: int,
        bands: int = 1,
        datatype: int = GDT_Byte,
        options: Sequence[str] | None = None
    ) -> Dataset: ...
    def CreateCopy(
        self,
        filename: str,
        src: Dataset,
        strict: int = 1,
        options: Sequence[str] | None = None,
        callback: Any | None = None,
        callback_data: Any | None = None
    ) -> Dataset: ...

class WarpOptions:
    pass

__all__ = [
    "UseExceptions",
    "AllRegister", 
    "Open",
    "BuildVRT",
    "Warp",
    "WarpOptions",
    "Dataset",
    "Band",
    "Driver",
    "GA_ReadOnly",
    "GA_Update",
    "GDT_Byte",
    "GDT_UInt16",
    "GDT_Int16",
    "GDT_UInt32",
    "GDT_Int32",
    "GDT_Float32",
    "GDT_Float64",
]