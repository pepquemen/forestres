import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import warnings


# =============================================================================
# FUNCIONES DE APOYO A LA DECISIÓN
# =============================================================================

def detect_drought_events(
    dataset: xr.Dataset,
    index_var: str = "drought_index",
    onset_threshold: float = 0.0,
    severity_threshold: float = -1.5,
    min_duration: int = 6,
    pooling_periods: int = 4,
    plot: bool = True,
    output_path: str = None
) -> pd.DataFrame:
    """
    Detecta todos los eventos de sequía en la serie temporal usando la Teoría de
    Rachas (Yevjevich, 1967) sobre la mediana espacial del índice de sequía.

    Un evento se define como una racha consecutiva de períodos semi-mensuales
    donde el índice cae por debajo del umbral de inicio (onset_threshold), siempre
    que en algún momento alcance la severidad mínima (severity_threshold) y tenga
    una duración mínima (min_duration). Rachas próximas separadas por menos de
    pooling_periods períodos positivos se fusionan en un único evento.

    NOTA: Los datos del CSIC tienen resolución semi-mensual (quincenas).
    Todos los parámetros temporales se expresan en número de quincenas.
    1 mes ≈ 2 quincenas.

    Parámetros:
    -----------
    dataset : xr.Dataset
        Dataset recortado al área de estudio (salida de clip_dataset_to_polygon).
    index_var : str
        Nombre de la variable del índice de sequía (ej. 'drought_index').
    onset_threshold : float
        Umbral de inicio de racha. El índice debe caer por debajo de este valor
        para que empiece a contabilizarse como período de déficit (por defecto 0.0).
    severity_threshold : float
        Severidad mínima que debe alcanzar el evento en algún momento para
        considerarse sequía real (por defecto -1.5, sequía severa).
    min_duration : int
        Número mínimo de quincenas consecutivas para considerar un evento válido
        (por defecto 6, equivalente a ~3 meses).
    pooling_periods : int
        Número máximo de quincenas positivas entre dos rachas para fusionarlas
        en un único evento (por defecto 4, equivalente a ~2 meses).
    plot : bool
        Si True, genera un gráfico de la serie temporal con los eventos sombreados.
    output_path : str, opcional
        Ruta donde guardar el gráfico. Si None, se muestra en pantalla.

    Retorna:
    --------
    pd.DataFrame con columnas:
        event_id, start_date, end_date, duration_periods, duration_months,
        max_severity, accumulated_deficit
    """
    # 1. Calcular la mediana espacial del índice
    spatial_median = dataset[index_var].median(dim=["x", "y"])
    times  = pd.to_datetime(spatial_median.time.values)
    values = spatial_median.values

    # 2. Identificar períodos bajo el umbral de inicio
    below_onset = values < onset_threshold

    # 3. Identificar rachas consecutivas bajo el umbral
    events_raw = []
    i = 0
    while i < len(values):
        if below_onset[i]:
            start_i = i
            while i < len(values) and below_onset[i]:
                i += 1
            end_i = i - 1
            events_raw.append((start_i, end_i))
        else:
            i += 1

    # 4. Pooling procedure: fusionar rachas separadas por menos de pooling_periods
    if len(events_raw) > 1:
        merged = [events_raw[0]]
        for current in events_raw[1:]:
            prev = merged[-1]
            gap = current[0] - prev[1] - 1
            if gap <= pooling_periods:
                merged[-1] = (prev[0], current[1])
            else:
                merged.append(current)
        events_raw = merged

    # 5. Filtrar eventos por severidad mínima y duración mínima
    events_list = []
    event_id = 1

    for start_i, end_i in events_raw:
        duration = end_i - start_i + 1
        event_values = values[start_i:end_i + 1]
        max_severity = float(np.min(event_values))
        accumulated_deficit = float(np.sum(event_values[event_values < 0]))

        if duration >= min_duration and max_severity <= severity_threshold:
            events_list.append({
                "event_id":            event_id,
                "start_date":          str(times[start_i].date()),
                "end_date":            str(times[end_i].date()),
                "duration_periods":    duration,
                "duration_months":     round(duration / 2, 1),
                "max_severity":        max_severity,
                "accumulated_deficit": accumulated_deficit
            })
            event_id += 1

    events_df = pd.DataFrame(events_list)

    if events_df.empty:
        warnings.warn(
            "No se detectaron eventos de sequía con los parámetros proporcionados. "
            "Considera reducir severity_threshold o min_duration."
        )
        return events_df

    # 6. Gráfico de la serie temporal con eventos sombreados
    if plot:
        fig, ax = plt.subplots(figsize=(16, 5))

        ax.plot(times, values, color="#2c3e50", linewidth=1.2, zorder=3)
        ax.fill_between(times, values, 0,
                        where=(values < 0), color="#e74c3c", alpha=0.25, zorder=2)
        ax.fill_between(times, values, 0,
                        where=(values >= 0), color="#2ecc71", alpha=0.15, zorder=2)

        for _, row in events_df.iterrows():
            ax.axvspan(
                pd.to_datetime(row["start_date"]),
                pd.to_datetime(row["end_date"]),
                color="#c0392b", alpha=0.15, zorder=1
            )
            mid_date = pd.to_datetime(row["start_date"]) + (
                pd.to_datetime(row["end_date"]) - pd.to_datetime(row["start_date"])
            ) / 2
            ax.text(mid_date, values.min() * 0.85,
                    f"E{int(row['event_id'])}",
                    ha="center", fontsize=8, color="#c0392b", weight="bold")

        ax.axhline(onset_threshold,    color="#7f8c8d", linewidth=0.8,
                   linestyle="--", alpha=0.6, label=f"Umbral inicio ({onset_threshold})")
        ax.axhline(severity_threshold, color="#962d2d", linewidth=0.8,
                   linestyle="-.", alpha=0.8, label=f"Umbral severidad ({severity_threshold})")
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)

        ax.set_xlabel("Fecha", fontsize=11)
        ax.set_ylabel(index_var, fontsize=11)
        ax.set_title(
            f"Detección de eventos de sequía — Teoría de Rachas (Yevjevich, 1967)\n"
            f"{len(events_df)} eventos detectados | "
            f"Severidad mín.: {severity_threshold} | Duración mín.: {min_duration} quincenas",
            fontsize=12
        )
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    return events_df


# =============================================================================
# FUNCIÓN DE APOYO: CORRELACIÓN LAG
# =============================================================================

def compute_lag_correlation(
    dataset: xr.Dataset,
    index_var: str = "drought_index",
    veg_var: str = "ndvi",
    max_lag: int = 24,
    only_drought_periods: bool = False,
    drought_threshold: float = 0.0,
    plot: bool = False,
    output_path: str = None,
    index_name: str = "Índice de Sequía"
) -> pd.DataFrame:
    """
    Calcula la correlación de Pearson entre el índice de sequía y el SNDVI
    para diferentes lags temporales, usando medianas espaciales sobre toda
    la serie histórica disponible.

    Usar este resultado para seleccionar el vegetation_lag_periods óptimo
    antes de ejecutar el pipeline principal.

    NOTA: Los lags se expresan en quincenas (períodos semi-mensuales).
    Lag=4 equivale aproximadamente a 2 meses.

    Parámetros:
    -----------
    dataset : xr.Dataset
        Dataset recortado al área de estudio.
    index_var : str
        Nombre de la variable del índice de sequía.
    veg_var : str
        Nombre de la variable de vegetación.
    max_lag : int
        Número máximo de quincenas de lag a calcular (por defecto 24, ~12 meses).
    plot : bool
        Si True, genera un gráfico de barras con la correlación por lag.
        Por defecto False.
    output_path : str, opcional
        Ruta donde guardar el gráfico. Si None y plot=True, se muestra en pantalla.
    index_name : str
        Nombre del índice de sequía para el título del gráfico.
    only_drought_periods : bool
        Si True, calcula la correlación solo sobre períodos donde el índice
        está por debajo de drought_threshold. Útil para capturar el lag
        específico durante condiciones de estrés hídrico.
    drought_threshold : float
        Umbral del índice para filtrar períodos de estrés cuando
        only_drought_periods=True (por defecto 0.0).

    Retorna:
    --------
    pd.DataFrame con columnas: lag_periods, lag_months, correlation
    Imprime el lag óptimo recomendado.
    """
    # Correlación píxel a píxel vectorizada sobre toda la serie histórica.
    # Se calcula para cada píxel del área de estudio y se reporta la media,
    # evitando el sesgo de la mediana espacial en áreas heterogéneas.
    index_arr = dataset[index_var].values   # (time, y, x)
    veg_arr   = dataset[veg_var].values     # (time, y, x)

    records = []
    for lag in range(0, max_lag + 1):
        if lag == 0:
            idx = index_arr
            veg = veg_arr
        else:
            idx = index_arr[:-lag]
            veg = veg_arr[lag:]

        # Filtrar solo períodos de estrés hídrico si se solicita
        if only_drought_periods:
            drought_mask = idx.mean(axis=(1, 2)) < drought_threshold
            idx = idx[drought_mask]
            veg = veg[drought_mask]

        n_t = idx.shape[0]
        if n_t < 30:
            records.append({"lag_periods": lag, "lag_months": round(lag / 2, 1), "correlation": np.nan})
            continue

        # Aplanar a (time, pixels) para vectorizar la correlación
        idx_flat = idx.reshape(n_t, -1)
        veg_flat = veg.reshape(n_t, -1)

        idx_mean = np.nanmean(idx_flat, axis=0)
        veg_mean = np.nanmean(veg_flat, axis=0)
        idx_dev  = idx_flat - idx_mean
        veg_dev  = veg_flat - veg_mean

        numerator   = np.nansum(idx_dev * veg_dev, axis=0)
        denom_idx   = np.sqrt(np.nansum(idx_dev ** 2, axis=0))
        denom_veg   = np.sqrt(np.nansum(veg_dev ** 2, axis=0))
        denominator = denom_idx * denom_veg

        corr_map  = np.where(denominator > 1e-6, numerator / denominator, np.nan)
        mean_corr = float(np.nanmean(corr_map))

        records.append({
            "lag_periods": lag,
            "lag_months":  round(lag / 2, 1),
            "correlation": round(mean_corr, 4)
        })

    corr_df = pd.DataFrame(records)

    best_lag = corr_df.loc[corr_df["correlation"].idxmax()]
    mode_str = "solo períodos de estrés hídrico" if only_drought_periods else "serie histórica completa"
    print(f"\nCorrelación por lag (píxel a píxel — {mode_str}):")
    print(corr_df.to_string(index=False))
    print(
        f"\nLag óptimo recomendado: {int(best_lag['lag_periods'])} quincenas "
        f"(~{best_lag['lag_months']} meses) — correlación media: {best_lag['correlation']:.4f}"
    )
    print("Usa este valor como vegetation_lag_periods en run_drought_impact_pipeline().")

    if plot:
        import os
        out_dir = os.path.dirname(output_path) if output_path else "."
        fname   = os.path.basename(output_path) if output_path else "lag_correlation.png"
        plot_lag_correlation(corr_df, out_dir, filename=fname, index_name=index_name)

    return corr_df




# =============================================================================
# FUNCIÓN DE VENTANAS (uso interno del pipeline)
# =============================================================================

def get_analysis_windows(
    dataset: xr.Dataset,
    pre_start: str,
    event_start: str,
    event_end: str,
    post_end: str,
    vegetation_lag_periods: int = 4
) -> dict:
    """
    Construye las ventanas temporales Pre, During y Post a partir de las fechas
    definidas por el usuario, aplicando el lag de vegetación a todas las ventanas
    del SNDVI desplazándolas hacia adelante.

    El usuario define las cuatro fechas límite manualmente, idealmente tras
    consultar detect_drought_events() y compute_lag_correlation().

    El lag desplaza todas las ventanas del SNDVI hacia adelante porque la
    vegetación responde al estrés hídrico con un retardo temporal. Esto garantiza
    que el Pre, During y Post del SNDVI corresponden al estado real de la
    vegetación en cada fase, no al estado del clima.

    NOTA: vegetation_lag_periods se expresa en quincenas.
    Lag=4 equivale aproximadamente a 2 meses.

    Parámetros:
    -----------
    dataset : xr.Dataset
        Dataset recortado, necesario para validar que las fechas desplazadas
        existen en el rango temporal disponible.
    pre_start : str
        Inicio de la ventana Pre-sequía (ej. '1997-06-01').
    event_start : str
        Inicio del evento de sequía (ej. '1999-06-01').
    event_end : str
        Fin del evento de sequía (ej. '2002-03-15').
    post_end : str
        Fin de la ventana Post-sequía (ej. '2004-03-15').
    vegetation_lag_periods : int
        Número de quincenas a desplazar todas las ventanas del SNDVI hacia
        adelante (por defecto 4, ~2 meses).

    Retorna:
    --------
    dict con dos subdicts:
        'index':      slices para el índice de sequía (sin lag)
        'vegetation': slices para el SNDVI (todas las ventanas desplazadas)
    """
    pre_start_dt   = pd.to_datetime(pre_start)
    event_start_dt = pd.to_datetime(event_start)
    event_end_dt   = pd.to_datetime(event_end)
    post_end_dt    = pd.to_datetime(post_end)

    # Ventanas del índice de sequía sin lag
    index_windows = {
        "pre_drought":    slice(str(pre_start_dt.date()),   str(event_start_dt.date())),
        "during_drought": slice(str(event_start_dt.date()), str(event_end_dt.date())),
        "post_drought":   slice(str(event_end_dt.date()),   str(post_end_dt.date()))
    }

    # Desplazamiento del lag con corrección de calendario
    # Usar days=15*N produce desvíos acumulativos en meses con distinto número de días.
    # La solución correcta es usar meses completos + días extra para quincenas impares.
    months_offset = vegetation_lag_periods // 2
    extra_days    = 15 if vegetation_lag_periods % 2 != 0 else 0
    lag_offset    = pd.DateOffset(months=months_offset, days=extra_days)

    veg_pre_start    = pre_start_dt   + lag_offset
    veg_event_start  = event_start_dt + lag_offset
    veg_event_end    = event_end_dt   + lag_offset
    veg_post_end     = post_end_dt    + lag_offset

    # Validar que el Post del SNDVI no excede el dataset
    data_end = pd.to_datetime(dataset.time.values[-1])
    if veg_post_end > data_end:
        warnings.warn(
            f"La ventana Post del SNDVI termina en {veg_post_end.date()} "
            f"pero el dataset termina en {data_end.date()}. "
            f"La recuperación puede estar incompleta. "
            f"Considera reducir post_end o vegetation_lag_periods."
        )

    # Ventanas del SNDVI con lag aplicado a todas
    vegetation_windows = {
        "pre_drought":    slice(str(veg_pre_start.date()),   str(veg_event_start.date())),
        "during_drought": slice(str(veg_event_start.date()), str(veg_event_end.date())),
        "post_drought":   slice(str(veg_event_end.date()),   str(veg_post_end.date()))
    }

    return {
        "index":      index_windows,
        "vegetation": vegetation_windows
    }