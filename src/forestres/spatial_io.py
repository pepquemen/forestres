import numpy as np
import xarray as xr
import rioxarray
import geopandas as gpd
import warnings
import logging
from shapely.geometry import box

logger = logging.getLogger(__name__)


def load_and_standardize_netcdf(
    file_path: str,
    var_mapping: dict = None,
    crs: str = "EPSG:4326",
    bbox: tuple = None
) -> xr.Dataset:
    """
    Load a NetCDF file and standardize its variables and coordinate reference system.

    If a bbox is provided, the dataset is spatially clipped BEFORE loading into
    memory via index selection (isel). This dramatically reduces memory usage when
    working with large-scale indices (SPI/SPEI) that would otherwise load gigabytes
    of data before clipping.

    Parameters
    ----------
    file_path : str
        Path to the NetCDF file.
    var_mapping : dict, optional
        Dictionary for renaming variables to the library standard.
    crs : str, optional
        Default CRS to assign if the file has none defined.
    bbox : tuple, optional
        Bounding box (minx, miny, maxx, maxy) in raster coordinates for early
        spatial clipping. If None, the full dataset is loaded.

    Returns
    -------
    xr.Dataset
        Clean dataset with standardized variable names and validated CRS.
    """
    # Load with lazy evaluation to avoid filling memory
    dataset = xr.open_dataset(
        file_path,
        decode_coords="all",
        engine="netcdf4",
        chunks={"time": 100}
    )

    # Rename variables to internal standard
    if var_mapping:
        rename_dict = {k: v for k, v in var_mapping.items() if k in dataset.variables}
        dataset = dataset.rename(rename_dict)

    # Validate and assign CRS
    if not dataset.rio.crs:
        warnings.warn(f"File has no CRS defined. Assigning {crs} as default.")
        dataset = dataset.rio.write_crs(crs)

    # Standardize spatial dimensions to 'x' and 'y'
    dim_mapping = {}
    if 'lon' in dataset.dims and 'lat' in dataset.dims:
        dim_mapping = {'lon': 'x', 'lat': 'y'}
    elif 'longitude' in dataset.dims and 'latitude' in dataset.dims:
        dim_mapping = {'longitude': 'x', 'latitude': 'y'}

    if dim_mapping:
        dataset = dataset.rename_dims(dim_mapping)
        dataset = dataset.rename(dim_mapping)
        dataset = dataset.assign_coords({
            "x": dataset.x,
            "y": dataset.y
        })

    # Early spatial clip by bbox (before loading into memory).
    # Index selection over coordinates is lazy and nearly instantaneous.
    if bbox is not None:
        minx, miny, maxx, maxy = bbox
        x_vals = dataset.x.values
        y_vals = dataset.y.values

        x_mask = (x_vals >= minx) & (x_vals <= maxx)
        y_mask = (y_vals >= miny) & (y_vals <= maxy)

        x_idx = np.where(x_mask)[0]
        y_idx = np.where(y_mask)[0]

        if len(x_idx) == 0 or len(y_idx) == 0:
            raise ValueError(
                "The shapefile bbox does not intersect the raster extent. "
                "Verify that the shapefile and the data share the same CRS."
            )

        dataset = dataset.isel(x=x_idx, y=y_idx)

    # Register spatial dimensions with rioxarray
    dataset = dataset.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)

    # Validate spatial dimensions
    if "x" not in dataset.dims or "y" not in dataset.dims:
        raise ValueError("Could not standardize spatial dimensions to 'x' and 'y'.")

    return dataset


def _get_shapefile_bbox(shapefile_path: str, target_crs) -> tuple:
    """
    Read a shapefile and return its bounding box reprojected to the raster CRS.

    Parameters
    ----------
    shapefile_path : str
        Path to the vector file.
    target_crs :
        CRS of the raster to reproject the bbox into.

    Returns
    -------
    tuple
        (minx, miny, maxx, maxy) in raster coordinates.
    """
    polygons = gpd.read_file(shapefile_path)
    if polygons.crs != target_crs:
        polygons = polygons.to_crs(target_crs)
    return tuple(polygons.total_bounds)  # (minx, miny, maxx, maxy)


def _align_time_nearest(ds_source: xr.Dataset, target_times: np.ndarray) -> xr.Dataset:
    """
    Align a dataset's time dimension to reference timestamps using vectorised
    nearest-neighbour search with numpy.

    Faster than xr.interp() because it operates on integer timestamps (nanoseconds
    since epoch) without Python loops or spatial interpolation. Works with any
    input time resolution (weekly, monthly, daily, irregular) and is robust to
    non-uniform timestamps.

    Parameters
    ----------
    ds_source : xr.Dataset
        Dataset whose time dimension will be realigned.
    target_times : np.ndarray
        Reference timestamps (dtype datetime64).

    Returns
    -------
    xr.Dataset
        Dataset with the time dimension reindexed to target_times.
    """
    # Convert timestamps to integers (nanoseconds since epoch) for arithmetic
    src_times_int = ds_source.time.values.astype("datetime64[ns]").astype(np.int64)
    tgt_times_int = target_times.astype("datetime64[ns]").astype(np.int64)

    # For each target timestamp, find the index of the nearest source timestamp.
    # Distance matrix shape: (n_src, n_tgt); argmin over axis 0 gives closest source index.
    closest_idx = np.argmin(
        # Nearest-neighbour avoids averaging or interpolating standardised values,
        # preserving the statistical properties (mean=0, std=1) of the drought index.
        np.abs(src_times_int[:, None] - tgt_times_int[None, :]),
        axis=0
    )

    # Select by position and reassign target timestamps
    ds_aligned = ds_source.isel(time=closest_idx)
    ds_aligned["time"] = target_times

    return ds_aligned


def load_and_merge_datasets(
    drought_path: str,
    vegetation_path: str,
    drought_var: str,
    vegetation_var: str,
    shapefile_path: str,
    crs: str = "EPSG:4326"
) -> xr.Dataset:
    """
    Load two NetCDF files (drought index and vegetation), clip them to the study
    area, align them temporally, and merge them into a single analysis-ready Dataset.

    Optimised workflow:
        1. Read the shapefile bounding box.
        2. Clip both NetCDF files by bbox BEFORE loading into memory (isel).
        3. Trim to the common temporal period.
        4. Align the drought index to SNDVI timestamps via vectorised nearest-
           neighbour search (numpy argmin, fast vectorised operation).
        5. Merge both variables.
        6. Apply the exact shapefile polygon clip.

    Early bbox clipping is critical for performance: large-scale indices (SPI/SPEI)
    have thousands of time steps, and loading the full spatial extent into memory
    before clipping can exhaust RAM. Clipping by bbox first means all subsequent
    operations work on the study area only (typically a few MB).

    The drought index is typically in weekly resolution (~8 days). SNDVI is in
    semi-monthly resolution (1st and 15th of each month). Temporal alignment is
    performed by vectorised nearest-neighbour search, preserving the statistical
    properties of the standardised index (mean=0, std=1) without averaging or
    interpolating values.

    Parameters
    ----------
    drought_path : str
        Path to the drought index NetCDF (e.g. spei03.nc).
    vegetation_path : str
        Path to the vegetation NetCDF (e.g. SNDVI.nc).
    drought_var : str
        Variable name in the drought NetCDF (e.g. 'spei').
    vegetation_var : str
        Variable name in the vegetation NetCDF (e.g. 'SNDVI').
    shapefile_path : str
        Path to the study area shapefile. Defines the spatial clip and is
        required for efficient memory usage.
    crs : str
        Default CRS if the files have none defined.

    Returns
    -------
    xr.Dataset
        Merged dataset clipped to the study area, with standardised variables
        'drought_index' and 'ndvi', ready for the pipeline.
    """
    # Read the shapefile bounding box in the data CRS
    logger.info(f"Reading study area: {shapefile_path}")
    bbox = _get_shapefile_bbox(shapefile_path, crs)
    logger.info(f"Area bounding box: {tuple(round(b, 2) for b in bbox)}")

    # Load and clip each NetCDF by bbox (early clip)
    logger.info(f"Loading drought index: {drought_path}")
    ds_drought = load_and_standardize_netcdf(
        drought_path,
        var_mapping={drought_var: "drought_index"},
        crs=crs,
        bbox=bbox
    )

    logger.info(f"Loading vegetation: {vegetation_path}")
    ds_veg = load_and_standardize_netcdf(
        vegetation_path,
        var_mapping={vegetation_var: "ndvi"},
        crs=crs,
        bbox=bbox
    )

    # Trim to the common temporal period
    t_start = max(ds_drought.time.values[0], ds_veg.time.values[0])
    t_end   = min(ds_drought.time.values[-1], ds_veg.time.values[-1])

    ds_drought = ds_drought.sel(time=slice(t_start, t_end))
    ds_veg     = ds_veg.sel(time=slice(t_start, t_end))

    logger.info(f"Common period: {str(t_start)[:10]} to {str(t_end)[:10]}")
    logger.info(f"SNDVI time steps in common period: {len(ds_veg.time)}")

    # Align the drought index to SNDVI timestamps via vectorised nearest-neighbour search.
    # Operates on integers (ns since epoch) -- faster than xr.interp() with no spatial ops.
    logger.info("Aligning drought index to SNDVI timestamps (vectorised argmin)...")
    ds_drought = _align_time_nearest(
        ds_drought[["drought_index"]],
        ds_veg.time.values
    )

    # Verify temporal alignment
    n = min(len(ds_drought.time), len(ds_veg.time))
    if not np.all(ds_drought.time.values[:n] == ds_veg.time.values[:n]):
        warnings.warn(
            "Timestamps are not perfectly aligned after temporal alignment. "
            "Check the temporal compatibility of the input NetCDF files."
        )

    # Merge into a single Dataset (explicit compat to suppress FutureWarning)
    ds_merged = xr.merge([ds_drought, ds_veg[["ndvi"]]], compat="override")

    # Load the bbox-clipped result into memory (small)
    ds_merged = ds_merged.compute()

    # Apply the exact shapefile polygon clip to the small in-memory array
    ds_clipped = clip_dataset_to_polygon(ds_merged, shapefile_path)

    logger.info(f"Merged and clipped dataset: {dict(ds_clipped.sizes)}")
    return ds_clipped


def clip_dataset_to_polygon(dataset: xr.Dataset, shapefile_path: str) -> xr.Dataset:
    """
    Clip a multidimensional dataset to a vector polygon, validating intersection.

    This function is called internally at the end of load_and_merge_datasets(), but
    is kept public to allow clipping an already-loaded dataset to a different
    shapefile (e.g. to subdivide an analysis by zones).

    Parameters
    ----------
    dataset : xr.Dataset
        The standardised dataset.
    shapefile_path : str
        Path to the vector file.

    Returns
    -------
    xr.Dataset
        Dataset clipped to the exact polygon boundary.
    """
    # Read the vector file
    polygons = gpd.read_file(shapefile_path)

    # Align CRS
    if polygons.crs != dataset.rio.crs:
        polygons = polygons.to_crs(dataset.rio.crs)

    # Dissolve geometries to avoid errors with FeatureCollections
    polygons = polygons.dissolve().reset_index(drop=True)

    # Verify geographic overlap
    raster_bounds = box(*dataset.rio.bounds())
    try:
        union = polygons.union_all()
    except AttributeError:
        union = polygons.unary_union  # geopandas < 1.0

    if not union.intersects(raster_bounds):
        raise ValueError(
            "Topology error: the vector polygon does not overlap the raster extent."
        )

    # Clip to the exact polygon boundary
    clipped_dataset = dataset.rio.clip(
        polygons.geometry, polygons.crs, all_touched=True, drop=True
    )

    return clipped_dataset
