import os
import warnings
import logging
import pandas as pd
import xarray as xr

from drought_impact.spatial_io import load_and_merge_datasets, clip_dataset_to_polygon  # noqa: F401
from drought_impact.drought_detection import get_analysis_windows
from drought_impact.vegetation_metrics import vegetation_impact_metrics
from drought_impact.spatial_clustering import calculate_getis_ord_gi_star
from drought_impact.export import (
    export_metrics_to_geotiff,
    export_clustering_to_geotiff,
    export_zonal_statistics_to_csv,
    export_metrics_to_vector,
    plot_drought_timeseries,
    plot_metrics_individual,
    plot_hotspots,
    plot_metrics_histograms,
    plot_line_of_full_resilience
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
        area_name: str = "Área de estudio"
) -> dict:
    """
    Orquestador central del pipeline de evaluación de impacto ecológico por sequía.

    Recibe el dataset ya cargado y recortado, y las fechas definidas manualmente
    por el usuario tras consultar las funciones de apoyo detect_drought_events()
    y compute_lag_correlation().

    FLUJO DE TRABAJO RECOMENDADO:
    ------------------------------
    1. Cargar y recortar datos (un solo paso, el shapefile dispara el recorte):
        ds_clip = load_and_merge_datasets(
            drought_path    = "scpdsi.nc",
            vegetation_path = "SNDVI.nc",
            drought_var     = "value",
            vegetation_var  = "SNDVI",
            shapefile_path  = "mi_area.shp",
            crs             = "EPSG:23030"
        )

    2. Explorar eventos y lag (funciones de apoyo):
        events = detect_drought_events(ds_clip)
        lag_df = compute_lag_correlation(ds_clip)

    3. Ejecutar el pipeline con las fechas elegidas:
        results = run_drought_impact_pipeline(
            dataset                = ds_clip,
            pre_start              = "1995-06-15",
            event_start            = "1997-01-15",
            event_end              = "2001-10-15",
            post_end               = "2003-03-01",
            vegetation_lag_periods = 2,
            min_recovery_periods   = 4,
            output_dir             = "resultados/",
            area_name              = "Serra de Tramuntana"
        )

    Parámetros:
    -----------
    dataset : xr.Dataset
        Dataset recortado al área de estudio (salida de load_and_merge_datasets).
    pre_start : str
        Inicio de la ventana Pre-sequía. Debe corresponder a un período de
        condiciones hídricas favorables según la Run Theory (índice > 0).
    event_start : str
        Inicio del evento de sequía.
    event_end : str
        Fin del evento de sequía.
    post_end : str
        Fin de la ventana Post-sequía. Debe corresponder a condiciones favorables.
    output_dir : str
        Directorio raíz de salida para todos los entregables.
    drought_index_var : str
        Nombre interno de la variable del índice de sequía.
    veg_var : str
        Nombre interno de la variable de vegetación.
    vegetation_lag_periods : int
        Quincenas de lag de respuesta de la vegetación al estrés hídrico.
        Usar compute_lag_correlation() para estimar el valor óptimo.
        Por defecto 2 (~1 mes), basado en análisis de correlaciones con scPDSI.
    exposure_threshold : float
        Umbral del índice para filtrar píxeles no expuestos a la sequía.
        Píxeles con mediana del índice >= este valor quedan como NaN. Default 0.0.
    agg_method : str
        Método de agregación temporal de cada ventana por píxel.
        'median' recomendado por robustez ante outliers (por defecto).
    min_recovery_periods : int
        Quincenas consecutivas por encima de anomalía 0 para confirmar recuperación.
        Evita que rebotes puntuales sean contabilizados como recuperación real.
        Por defecto 4 (~2 meses). Ajustar según tipo de vegetación del área.
    kernel_size : int
        Tamaño del kernel para el análisis Gi* (debe ser impar: 3, 5, 7).
        Con resolución CSIC de 1.1 km: 3x3 ~10 km², 5x5 ~30 km², 7x7 ~60 km².
    min_valid_neighbors : int
        Mínimo de vecinos válidos para el análisis Getis-Ord Gi*.
    clustering_target_var : str
        Variable del Dataset de métricas sobre la que aplicar el Gi*.
        Por defecto 'accumulated_deficit'. Otras opciones útiles:
        'recovery_time' (hotspots de lenta recuperación),
        'resistance' (núcleos de colapso estructural).
    area_name : str
        Nombre del área de estudio para títulos de figuras.

    Retorna:
    --------
    dict con rutas a todos los entregables generados:
        windows, metrics_tifs, statistics_csv, hotspots_tif,
        plot_timeseries, plot_panel, plot_hotspots,
        plot_histograms, plot_resilience
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # =========================================================================
    # PASO 1: Validación de fechas y ventanas temporales
    # =========================================================================
    logger.info("[1/4] Validando fechas y construyendo ventanas temporales...")

    data_start = pd.to_datetime(dataset.time.values[0])
    data_end   = pd.to_datetime(dataset.time.values[-1])

    pre_start_dt   = pd.to_datetime(pre_start)
    event_start_dt = pd.to_datetime(event_start)
    event_end_dt   = pd.to_datetime(event_end)
    post_end_dt    = pd.to_datetime(post_end)

    if pre_start_dt >= event_start_dt:
        raise ValueError(f"pre_start ({pre_start}) debe ser anterior a event_start ({event_start}).")
    if event_start_dt >= event_end_dt:
        raise ValueError(f"event_start ({event_start}) debe ser anterior a event_end ({event_end}).")
    if event_end_dt >= post_end_dt:
        raise ValueError(f"event_end ({event_end}) debe ser anterior a post_end ({post_end}).")

    if pre_start_dt < data_start:
        warnings.warn(
            f"pre_start ({pre_start}) es anterior al inicio del dataset ({data_start.date()}). "
            "La ventana Pre puede estar incompleta."
        )

    # Corrección calendar drift: misma fórmula que get_analysis_windows
    months_offset = vegetation_lag_periods // 2
    extra_days    = 15 if vegetation_lag_periods % 2 != 0 else 0
    lag_offset    = pd.DateOffset(months=months_offset, days=extra_days)
    veg_post_end  = post_end_dt + lag_offset
    if veg_post_end > data_end:
        raise ValueError(
            f"La ventana Post del SNDVI (con lag aplicado) termina en {veg_post_end.date()} "
            f"pero el dataset finaliza en {data_end.date()}. "
            f"Las métricas de recuperación se calcularían sobre datos incompletos. "
            f"Reduce 'post_end' o 'vegetation_lag_periods' para que la ventana "
            f"no exceda {data_end.date()}."
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
    logger.info(f"Lag aplicado: {vegetation_lag_periods} quincenas (~{vegetation_lag_periods/2:.1f} meses)")
    logger.info(f"Ventana Pre índice:  {pre_start} → {event_start}")
    logger.info(f"Ventana Dur índice:  {event_start} → {event_end}")
    logger.info(f"Ventana Post índice: {event_end} → {post_end}")

    # =========================================================================
    # PASO 2: Métricas de impacto ecológico
    # =========================================================================
    logger.info("[2/4] Calculando métricas de impacto ecológico (Lloret et al., 2011)...")

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

    # =========================================================================
    # PASO 3: Análisis espacial Getis-Ord Gi*
    # =========================================================================
    if clustering_target_var not in metrics_ds.data_vars:
        raise KeyError(
            f"La variable '{clustering_target_var}' no existe en las métricas. "
            f"Variables disponibles: {list(metrics_ds.data_vars)}"
        )

    # Métricas donde valores más negativos = mayor vulnerabilidad → invertir signo
    # para que el Gi* detecte esas zonas como Hotspots (valores altos) en lugar de Coldspots.
    _INVERT_FOR_HOTSPOT = {
        "accumulated_deficit": True,   # más negativo = más déficit acumulado
        "resistance":          True,   # más negativo = mayor daño durante la sequía
        "resilience":          True,   # más negativo = peor balance final
        "drought_min":         True,   # más negativo = pico de sequía más severo
        "drought_median":      True,   # más negativo = intensidad típica más severa
        "recovery_time":       False,  # más alto = más lento en recuperarse
        "recovery":            False,  # más alto = mejor recuperación
        "did_not_recover":     False,  # 1 = no recuperado = vulnerable
    }

    target_da = metrics_ds[clustering_target_var]
    if _INVERT_FOR_HOTSPOT.get(clustering_target_var, False):
        target_da = -target_da
        logger.info(f"Signo invertido para '{clustering_target_var}': valores más negativos → Hotspots.")

    logger.info(f"[3/4] Ejecutando análisis Getis-Ord Gi* sobre '{clustering_target_var}'...")

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

    # =========================================================================
    # PASO 4: Figuras y cartografía
    # =========================================================================
    logger.info("[4/4] Generando figuras y cartografía...")

    results["plot_timeseries"] = plot_drought_timeseries(
        dataset[drought_index_var], windows, output_dir,
        index_name = drought_index_var
    )
    results["plot_individual"] = plot_metrics_individual(metrics_ds, output_dir)
    results["plot_hotspots"]   = plot_hotspots(clustering_ds, output_dir)
    results["plot_histograms"] = plot_metrics_histograms(metrics_ds, output_dir)
    results["plot_resilience"] = plot_line_of_full_resilience(
        metrics_ds, output_dir, area_name=area_name
    )

    logger.info(f"¡Pipeline completado! Entregables en: {output_dir}")
    return results
