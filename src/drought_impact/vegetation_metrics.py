import xarray as xr
import numpy as np


def aggregate_temporal_window(da: xr.DataArray, method: str = "median") -> xr.DataArray:
    """
    Colapsa la dimensión temporal de una ventana de datos en un único valor escalar por píxel.

    Parámetros:
    -----------
    da : xr.DataArray
        Array con dimensión temporal.
    method : str
        Método de agregación: 'median' (recomendado para datos ecológicos por su
        robustez ante outliers), 'mean' o 'max'.

    Retorna:
    --------
    xr.DataArray 2D (y, x) con el valor agregado por píxel.
    """
    if method == "median":
        return da.median(dim="time", skipna=True)
    elif method == "mean":
        return da.mean(dim="time", skipna=True)
    elif method == "max":
        return da.max(dim="time", skipna=True)
    else:
        raise ValueError(f"Método '{method}' no soportado. Usa 'median', 'mean' o 'max'.")


def vegetation_impact_metrics(
    dataset: xr.Dataset,
    veg_var: str,
    index_var: str,
    windows: dict,
    exposure_threshold: float = 0.0,
    agg_method: str = "median",
    min_recovery_periods: int = 4
) -> xr.Dataset:
    """
    Calcula las métricas de vulnerabilidad y resiliencia del bosque frente a la sequía.

    Implementa las métricas de Lloret et al. (2011) adaptadas al uso de diferencias
    en lugar de ratios para evitar el efecto de valores positivos y negativos derivados
    del uso de un índice estandarizado como el SNDVI (Xu et al., 2024).
    Incluye además el Déficit Acumulado, el Tiempo de Recuperación sostenida y una
    máscara de exposición que excluye los píxeles que no sufrieron estrés hídrico real.

    Las ventanas temporales del SNDVI están desplazadas respecto a las del índice
    de sequía para compensar el lag de respuesta de la vegetación. Las ventanas
    del índice se usan exclusivamente para calcular la máscara de exposición.

    Parámetros:
    -----------
    dataset : xr.Dataset
        Dataset recortado al área de estudio con variables de vegetación e índice.
    veg_var : str
        Nombre de la variable de vegetación (ej. 'ndvi').
    index_var : str
        Nombre de la variable del índice de sequía (ej. 'drought_index').
    windows : dict
        Diccionario con dos subdicts devuelto por get_analysis_windows():
            'index':      ventanas temporales del índice de sequía (sin lag)
            'vegetation': ventanas temporales del SNDVI (con lag aplicado)
    exposure_threshold : float
        Umbral del índice para considerar un píxel como expuesto a sequía.
        Píxeles donde el mínimo del índice durante el evento es >= este valor
        se excluyen del análisis (NaN). Por defecto 0.0.
    agg_method : str
        Método de agregación temporal de cada ventana por píxel.
        'median' recomendado por su robustez ante outliers en datos ecológicos.
    min_recovery_periods : int
        Número mínimo de quincenas consecutivas por encima de anomalía 0
        para considerar que un píxel se ha recuperado realmente.
        Evita que rebotes puntuales sean contabilizados como recuperación.
        Por defecto 4 (~2 meses). Configurable por el usuario.

    NOTA METODOLÓGICA:
        Las métricas siguen Lloret et al. (2011) con diferencias en lugar de ratios,
        siguiendo el enfoque de Xu et al. (2024). La Resiliencia Relativa ha sido
        eliminada por su inestabilidad matemática con índices estandarizados.
        El índice de sequía actúa como filtro de exposición: define qué píxeles
        estuvieron expuestos al estrés hídrico, evaluado por el pico de severidad
        (mínimo del índice) en lugar de la mediana.
        El Déficit Acumulado y el Tiempo de Recuperación se referencian a anomalía 0
        (condiciones históricas normales del SNDVI estandarizado) en lugar del Pre,
        eliminando la dependencia de la elección de ventanas temporales.

    Retorna:
    --------
    xr.Dataset con variables:
        resistance, recovery, resilience,
        accumulated_deficit, recovery_time, did_not_recover,
        drought_min    (mínimo del índice por píxel durante el evento),
        drought_median (mediana del índice por píxel durante el evento)
    """
    veg_data   = dataset[veg_var]
    index_data = dataset[index_var]

    # =========================================================================
    # 1. EXTRAER VENTANAS TEMPORALES
    # =========================================================================
    pre_da  = veg_data.sel(time=windows["vegetation"]["pre_drought"])
    dur_da  = veg_data.sel(time=windows["vegetation"]["during_drought"])
    post_da = veg_data.sel(time=windows["vegetation"]["post_drought"])

    index_dur_da = index_data.sel(time=windows["index"]["during_drought"])

    # =========================================================================
    # 2. MÁSCARA DE EXPOSICIÓN
    # =========================================================================
    # Pico de severidad del índice por píxel durante el evento (sin lag).
    # Se usa el mínimo en lugar de la mediana porque la exposición a un riesgo
    # debe evaluarse por el extremo alcanzado, no por la tendencia central.
    drought_min    = index_dur_da.min(dim="time", skipna=True)
    drought_median = index_dur_da.median(dim="time", skipna=True)

    exposure_mask = drought_min < exposure_threshold

    # =========================================================================
    # 3. MAPAS BASE ESTÁTICOS (agregación temporal por píxel)
    # =========================================================================
    Pre  = aggregate_temporal_window(pre_da,  method=agg_method)
    Dur  = aggregate_temporal_window(dur_da,  method=agg_method)
    Post = aggregate_temporal_window(post_da, method=agg_method)

    # =========================================================================
    # 4. MÉTRICAS DE LLORET et al. (2011) — diferencias absolutas (Xu et al., 2024)
    # =========================================================================
    resistance = Dur - Pre   # Rt = Dr - PreDr
    recovery   = Post - Dur  # Rc = PostDr - Dr
    resilience = Post - Pre  # Rs = PostDr - PreDr

    # =========================================================================
    # 5. MÉTRICAS DINÁMICAS
    # =========================================================================

    # A. Déficit Acumulado referenciado a anomalía 0
    # Suma de quincenas donde el SNDVI es negativo durante el evento.
    accumulated_deficit = dur_da.where(dur_da < 0).sum(dim="time", skipna=True)

    # B. Tiempo de Recuperación con recuperación sostenida
    # Se exige que el píxel se mantenga >= 0 durante min_recovery_periods quincenas
    # consecutivas para evitar que rebotes puntuales sean contabilizados como recuperación.
    is_above_zero = (post_da >= 0).astype(float)
    rolling_mean  = is_above_zero.rolling(
        time=min_recovery_periods,
        min_periods=min_recovery_periods
    ).mean()

    is_recovered = (rolling_mean >= 1.0)
    time_axis = is_recovered.dims.index("time")
    has_recovered_yet = xr.DataArray(
        np.maximum.accumulate(is_recovered.values.astype(float), axis=time_axis),
        coords=is_recovered.coords,
        dims=is_recovered.dims
    ).astype(bool)

    recovery_time = (~has_recovered_yet).sum(dim="time", skipna=True)

    # Píxeles que no alcanzaron recuperación sostenida → NaN
    total_post_periods = len(post_da.time)
    did_not_recover_mask = recovery_time == total_post_periods
    recovery_time = xr.where(did_not_recover_mask, np.nan, recovery_time)

    # Corrección del sesgo algorítmico del rolling
    recovery_time = xr.where(
        ~did_not_recover_mask & (recovery_time > 0),
        recovery_time - (min_recovery_periods - 1),
        recovery_time
    )

    # =========================================================================
    # 6. APLICAR MÁSCARA DE EXPOSICIÓN A TODAS LAS MÉTRICAS
    # =========================================================================
    def apply_mask(da):
        return da.where(exposure_mask)

    metrics_ds = xr.Dataset(
        {
            "resistance":          apply_mask(resistance),
            "recovery":            apply_mask(recovery),
            "resilience":          apply_mask(resilience),
            "accumulated_deficit": apply_mask(accumulated_deficit),
            "recovery_time":       apply_mask(recovery_time),
            "did_not_recover":     apply_mask(did_not_recover_mask.astype(np.uint8)),
            "drought_min":         drought_min,
            "drought_median":      drought_median
        }
    )

    metrics_ds.attrs["methodology"]                   = "Lloret et al. (2011) adapted with absolute differences (Xu et al., 2024)"
    metrics_ds.attrs["aggregation_method"]            = agg_method
    metrics_ds.attrs["exposure_threshold"]            = exposure_threshold
    metrics_ds.attrs["min_recovery_periods"]          = min_recovery_periods
    metrics_ds.attrs["max_possible_recovery_periods"] = total_post_periods
    metrics_ds.attrs["deficit_reference"]             = "anomaly_0 (SNDVI historical mean)"
    metrics_ds.attrs["recovery_reference"]            = "anomaly_0 (SNDVI historical mean)"
    metrics_ds.attrs["lag_note"] = (
        "Vegetation windows are temporally shifted to compensate for the "
        "lagged response of vegetation to drought stress. "
        "drought_min is calculated on the unshifted index windows as the "
        "minimum value (peak severity). "
        "drought_median is the median value of the index during the event. "
        "Accumulated deficit and recovery time are referenced to SNDVI anomaly 0 "
        "(historical normal conditions) instead of the Pre period value."
    )

    return metrics_ds