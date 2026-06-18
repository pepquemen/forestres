"""
Drought Impact Assessment Library
---------------------------------
Librería científica para la evaluación de impactos ecológicos, vulnerabilidad
y resiliencia de masas forestales frente a eventos de sequía extrema utilizando
datos geoespaciales multidimensionales (Xarray) y análisis de autocorrelación espacial.

Flujo de trabajo recomendado:
-----------------------------
1. Cargar y recortar datos al área de estudio:
    ds_clip = load_and_merge_datasets(
        drought_path    = "scpdsi.nc",
        vegetation_path = "SNDVI.nc",
        drought_var     = "value",
        vegetation_var  = "SNDVI",
        shapefile_path  = "mi_area.shp",
        crs             = "EPSG:23030"
    )

2. Funciones de apoyo a la decisión (opcionales pero recomendadas):
    events = detect_drought_events(ds_clip)   # Ver qué sequías hay
    lag_df = compute_lag_correlation(ds_clip) # Estimar lag óptimo

3. Ejecutar el pipeline con las fechas elegidas:
    results = run_drought_impact_pipeline(
        dataset                = ds_clip,
        pre_start              = "1995-06-15",
        event_start            = "1997-01-15",
        event_end              = "2001-10-15",
        post_end               = "2003-03-01",
        vegetation_lag_periods = 1,
        output_dir             = "resultados/",
        area_name              = "Mi área de estudio"
    )
"""

from drought_impact.core import run_drought_impact_pipeline

from drought_impact.spatial_io import (
    load_and_standardize_netcdf,
    load_and_merge_datasets,
    clip_dataset_to_polygon
)

from drought_impact.drought_detection import (
    detect_drought_events,
    compute_lag_correlation,
    get_analysis_windows
)

from drought_impact.vegetation_metrics import vegetation_impact_metrics

from drought_impact.spatial_clustering import calculate_getis_ord_gi_star

from drought_impact.export import (
    export_metrics_to_geotiff,
    export_clustering_to_geotiff,
    export_events_to_csv,
    export_zonal_statistics_to_csv,
    export_metrics_to_vector,
    plot_drought_timeseries,
    plot_metrics_individual,
    plot_hotspots,
    plot_metrics_histograms,
    plot_line_of_full_resilience
)

__version__ = "0.1.0"

__all__ = [
    # Orquestador principal
    "run_drought_impact_pipeline",

    # Carga y preparación de datos
    "load_and_standardize_netcdf",
    "load_and_merge_datasets",
    "clip_dataset_to_polygon",

    # Funciones de apoyo a la decisión
    "detect_drought_events",       # Detecta eventos con Run Theory + gráfico
    "compute_lag_correlation",     # Estima lag óptimo índice → SNDVI

    # Ventanas temporales (uso interno del pipeline)
    "get_analysis_windows",

    # Métricas de impacto ecológico
    "vegetation_impact_metrics",

    # Análisis espacial
    "calculate_getis_ord_gi_star",

    # Exportación de datos
    "export_metrics_to_geotiff",
    "export_clustering_to_geotiff",
    "export_events_to_csv",
    "export_zonal_statistics_to_csv",
    "export_metrics_to_vector",

    # Visualización
    "plot_drought_timeseries",
    "plot_metrics_individual",
    "plot_hotspots",
    "plot_metrics_histograms",
    "plot_line_of_full_resilience"
]
