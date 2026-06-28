"""
Drought Impact Assessment Library
----------------------------------
Scientific library for evaluating ecological impacts, vulnerability, and
resilience of forest stands under extreme drought events, using multi-
dimensional geospatial data (Xarray) and spatial autocorrelation analysis.

Recommended workflow:
---------------------
1. Load and clip data to the study area:
    ds_clip = load_and_merge_datasets(
        drought_path    = "scpdsi.nc",
        vegetation_path = "SNDVI.nc",
        drought_var     = "value",
        vegetation_var  = "SNDVI",
        shapefile_path  = "study_area.shp",
        crs             = "EPSG:23030"
    )

2. Decision-support functions (optional but recommended):
    events = detect_drought_events(ds_clip)   # identify drought events
    lag_df = compute_lag_correlation(ds_clip) # estimate optimal lag

3. Run the pipeline with chosen dates:
    results = run_forestres_pipeline(
        dataset                = ds_clip,
        pre_start              = "1995-06-15",
        event_start            = "1997-01-15",
        event_end              = "2001-10-15",
        post_end               = "2003-03-01",
        vegetation_lag_periods = 1,
        output_dir             = "results/",
        area_name              = "My study area"
    )
"""

from forestres.core import run_forestres_pipeline

from forestres.spatial_io import (
    load_and_standardize_netcdf,
    load_and_merge_datasets,
    clip_dataset_to_polygon
)

from forestres.drought_detection import (
    detect_drought_events,
    compute_lag_correlation,
    get_analysis_windows
)

from forestres.vegetation_metrics import vegetation_impact_metrics

from forestres.spatial_clustering import calculate_getis_ord_gi_star

from forestres.export import (
    export_metrics_to_geotiff,
    export_clustering_to_geotiff,
    export_events_to_csv,
    export_zonal_statistics_to_csv,
    export_metrics_to_vector,
    plot_drought_timeseries,
    plot_metrics_individual,
    plot_hotspots,
    plot_metrics_histograms
)

__version__ = "0.1.0"

__all__ = [
    # Main orchestrator
    "run_forestres_pipeline",

    # Data loading and preparation
    "load_and_standardize_netcdf",
    "load_and_merge_datasets",
    "clip_dataset_to_polygon",

    # Decision-support functions
    "detect_drought_events",       # detect events with Run Theory + chart
    "compute_lag_correlation",     # estimate optimal lag for the index vs SNDVI

    # Temporal windows (used internally by the pipeline)
    "get_analysis_windows",

    # Ecological impact metrics
    "vegetation_impact_metrics",

    # Spatial analysis
    "calculate_getis_ord_gi_star",

    # Data export
    "export_metrics_to_geotiff",
    "export_clustering_to_geotiff",
    "export_events_to_csv",
    "export_zonal_statistics_to_csv",
    "export_metrics_to_vector",

    # Visualisation
    "plot_drought_timeseries",
    "plot_metrics_individual",
    "plot_hotspots",
    "plot_metrics_histograms",
]
