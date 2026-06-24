import os
import warnings
import logging
import pandas as pd
import xarray as xr

from forestres.spatial_io import load_and_merge_datasets, clip_dataset_to_polygon  # noqa: F401
from forestres.drought_detection import get_analysis_windows
from forestres.vegetation_metrics import vegetation_impact_metrics
from forestres.spatial_clustering import calculate_getis_ord_gi_star
from forestres.export import (
    export_metrics_to_geotiff,
    export_clustering_to_geotiff,
    export_zonal_statistics_to_csv,
    export_metrics_to_vector,
    plot_drought_timeseries,
    plot_metrics_individual,
    plot_hotspots,
    plot_metrics_histograms
)

logger = logging.getLogger(__name__)


def run_drought_impact_pipeline(
        dataset: xr.Dataset,
        pre_start: str,
        event_start: str,
        event_end: str,
        post_end: str,
        output_dir: str,
        drought_index_var: str = "drought_index",
        veg_var: str = "ndvi",
        vegetation_lag_periods: int = 2,
        exposure_threshold: float = 0.0,
        agg_method: str = "median",
        min_recovery_periods: int = 4,
        kernel_size: int = 3,
        min_valid_neighbors: int = 3,
        clustering_target_var: str = "accumulated_deficit",
        area_name: str = "Study area"
) -> dict:
    """
    Central orchestrator for the drought ecological impact assessment pipeline.

    Receives the already-loaded and clipped dataset and the dates defined manually
    by the user after consulting detect_drought_events() and compute_lag_correlation().

    Recommended workflow:
        1. Load and clip data (one step; the shapefile triggers the clip):
            ds_clip = load_and_merge_datasets(
                drought_path    = "scpdsi.nc",
                vegetation_path = "SNDVI.nc",
                drought_var     = "value",
                vegetation_var  = "SNDVI",
                shapefile_path  = "study_area.shp",
                crs             = "EPSG:23030"
            )

        2. Explore events and lag (decision-support functions):
            events = detect_drought_events(ds_clip)
            lag_df = compute_lag_correlation(ds_clip)

        3. Run the pipeline with the chosen dates:
            results = run_drought_impact_pipeline(
                dataset                = ds_clip,
                pre_start              = "1995-06-15",
                event_start            = "1997-01-15",
                event_end              = "2001-10-15",
                post_end               = "2003-03-01",
                vegetation_lag_periods = 2,
                min_recovery_periods   = 4,
                output_dir             = "results/",
                area_name              = "Serra de Tramuntana"
            )

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset clipped to the study area (output of load_and_merge_datasets).
    pre_start : str
        Start of the pre-drought window. Should correspond to a period of
        favourable hydric conditions (index > 0) per Run Theory.
    event_start : str
        Start of the drought event.
    event_end : str
        End of the drought event.
    post_end : str
        End of the post-drought window. Should correspond to favourable conditions.
    output_dir : str
        Root output directory for all deliverables.
    drought_index_var : str
        Internal name of the drought index variable.
    veg_var : str
        Internal name of the vegetation variable.
    vegetation_lag_periods : int
        Vegetation response lag in biweekly periods. Use compute_lag_correlation()
        to estimate the optimal value. Default 2 (~1 month).
    exposure_threshold : float
        Index threshold to exclude pixels not exposed to drought. Pixels with a
        median index >= this value are set to NaN. Default 0.0.
    agg_method : str
        Temporal aggregation method per pixel per window. 'median' recommended
        for robustness to outliers (default).
    min_recovery_periods : int
        Consecutive biweekly periods above anomaly 0 required to confirm recovery.
        Prevents transient rebounds from being counted as real recovery.
        Default 4 (~2 months). Adjust according to the vegetation type.
    kernel_size : int
        Kernel size for the Gi* analysis (must be odd: 3, 5, 7).
        At 1.1 km CSIC resolution: 3x3 ~10 km2, 5x5 ~30 km2, 7x7 ~60 km2.
    min_valid_neighbors : int
        Minimum valid neighbours for the Getis-Ord Gi* analysis.
    clustering_target_var : str
        Metric variable on which to apply Gi*. Default 'accumulated_deficit'.
        Other useful options: 'recovery_time' (slow-recovery hotspots),
        'resistance' (structural collapse cores).
    area_name : str
        Study area name for figure titles.

    Returns
    -------
    dict
        Paths to all generated deliverables:
        windows, metrics_tifs, statistics_csv, hotspots_tif,
        plot_timeseries, plot_individual, plot_hotspots,
        plot_histograms, plot_resilience.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # --- Step 1: Date validation and temporal windows
    logger.info("[1/4] Validating dates and building temporal windows...")

    data_start = pd.to_datetime(dataset.time.values[0])
    data_end   = pd.to_datetime(dataset.time.values[-1])

    pre_start_dt   = pd.to_datetime(pre_start)
    event_start_dt = pd.to_datetime(event_start)
    event_end_dt   = pd.to_datetime(event_end)
    post_end_dt    = pd.to_datetime(post_end)

    if pre_start_dt >= event_start_dt:
        raise ValueError(f"pre_start ({pre_start}) must be earlier than event_start ({event_start}).")
    if event_start_dt >= event_end_dt:
        raise ValueError(f"event_start ({event_start}) must be earlier than event_end ({event_end}).")
    if event_end_dt >= post_end_dt:
        raise ValueError(f"event_end ({event_end}) must be earlier than post_end ({post_end}).")

    if pre_start_dt < data_start:
        warnings.warn(
            f"pre_start ({pre_start}) is earlier than the dataset start ({data_start.date()}). "
            "The Pre window may be incomplete."
        )

    # Calendar-drift correction: same formula as get_analysis_windows
    months_offset = vegetation_lag_periods // 2
    extra_days    = 15 if vegetation_lag_periods % 2 != 0 else 0
    lag_offset    = pd.DateOffset(months=months_offset, days=extra_days)
    veg_post_end  = post_end_dt + lag_offset
    if veg_post_end > data_end:
        raise ValueError(
            f"The SNDVI Post window (with lag applied) ends on {veg_post_end.date()} "
            f"but the dataset ends on {data_end.date()}. "
            f"Recovery metrics would be computed on incomplete data. "
            f"Reduce 'post_end' or 'vegetation_lag_periods' so the window "
            f"does not exceed {data_end.date()}."
        )

    windows = get_analysis_windows(
        dataset                = dataset,
        pre_start              = pre_start,
        event_start            = event_start,
        event_end              = event_end,
        post_end               = post_end,
        vegetation_lag_periods = vegetation_lag_periods
    )
    results["windows"] = windows
    logger.info(f"Lag applied: {vegetation_lag_periods} biweekly periods (~{vegetation_lag_periods/2:.1f} months)")
    logger.info(f"Pre index window:    {pre_start} to {event_start}")
    logger.info(f"During index window: {event_start} to {event_end}")
    logger.info(f"Post index window:   {event_end} to {post_end}")

    # --- Step 2: Ecological impact metrics
    logger.info("[2/4] Computing ecological impact metrics (Lloret et al., 2011)...")

    metrics_ds = vegetation_impact_metrics(
        dataset               = dataset,
        veg_var               = veg_var,
        index_var             = drought_index_var,
        windows               = windows,
        exposure_threshold    = exposure_threshold,
        agg_method            = agg_method,
        min_recovery_periods  = min_recovery_periods
    )

    tifs_metrics = export_metrics_to_geotiff(metrics_ds, os.path.join(output_dir, "geotiffs"))
    csv_stats    = export_zonal_statistics_to_csv(metrics_ds, output_dir)
    gpkg_path    = export_metrics_to_vector(metrics_ds, output_dir)
    results["metrics_tifs"]   = tifs_metrics
    results["statistics_csv"] = csv_stats
    results["metrics_gpkg"]   = gpkg_path

    # --- Step 3: Getis-Ord Gi* spatial analysis
    if clustering_target_var not in metrics_ds.data_vars:
        raise KeyError(
            f"Variable '{clustering_target_var}' not found in metrics. "
            f"Available variables: {list(metrics_ds.data_vars)}"
        )

    # For metrics where more negative = higher vulnerability, invert the sign so that
    # Gi* detects those zones as Hotspots (high values) rather than Coldspots.
    _INVERT_FOR_HOTSPOT = {
        "accumulated_deficit": True,   # more negative = greater accumulated deficit
        "resistance":          True,   # more negative = greater damage during drought
        "resilience":          True,   # more negative = worse final balance
        "drought_min":         True,   # more negative = more severe drought peak
        "drought_median":      True,   # more negative = more severe typical intensity
        "recovery_time":       False,  # higher = slower recovery
        "recovery":            False,  # higher = better recovery
        "did_not_recover":     False,  # 1 = not recovered = vulnerable
    }

    target_da = metrics_ds[clustering_target_var]
    if _INVERT_FOR_HOTSPOT.get(clustering_target_var, False):
        target_da = -target_da
        logger.info(f"Sign inverted for '{clustering_target_var}': more negative values will appear as Hotspots.")

    logger.info(f"[3/4] Running Getis-Ord Gi* on '{clustering_target_var}'...")

    clustering_ds = calculate_getis_ord_gi_star(
        target_da,
        kernel_size         = kernel_size,
        min_valid_neighbors = min_valid_neighbors
    )
    tif_hotspots = export_clustering_to_geotiff(
        clustering_ds, os.path.join(output_dir, "geotiffs"),
        name = clustering_target_var
    )
    results["hotspots_tif"] = tif_hotspots

    # --- Step 4: Figures and cartography
    logger.info("[4/4] Generating figures and cartography...")

    results["plot_timeseries"] = plot_drought_timeseries(
        dataset[drought_index_var], windows, output_dir,
        index_name = drought_index_var
    )
    results["plot_individual"] = plot_metrics_individual(metrics_ds, output_dir)
    results["plot_hotspots"]   = plot_hotspots(clustering_ds, output_dir)
    results["plot_histograms"] = plot_metrics_histograms(metrics_ds, output_dir)

    logger.info(f"Pipeline complete. Deliverables in: {output_dir}")
    return results
