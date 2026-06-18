import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from scipy import stats
from shapely.geometry import Point


# =============================================================================
# CONFIGURACIÓN INTERNA DE ESTILOS
# =============================================================================

_METRIC_STYLES = {
    "resistance":          {"cmap": "RdYlGn",   "label": "Resistencia (ΔSNDVI)",               "diverging": True},
    "recovery":            {"cmap": "RdYlGn",   "label": "Recuperación (ΔSNDVI)",              "diverging": True},
    "resilience":          {"cmap": "RdYlGn",   "label": "Resiliencia Neta (ΔSNDVI)",          "diverging": True},
    "accumulated_deficit": {"cmap": "YlOrRd_r", "label": "Déficit Acumulado (ΔSNDVI·q)",       "diverging": False},
    "recovery_time":       {"cmap": "YlOrRd",   "label": "Tiempo de Recuperación (quincenas)", "diverging": False},
    "did_not_recover":     {"cmap": "Reds",      "label": "Estado Crónico No Recuperado",       "diverging": False},
    "drought_min":         {"cmap": "RdBu",      "label": "Intensidad Sequía (mínimo índice)",  "diverging": True},
    "drought_median":      {"cmap": "RdBu",      "label": "Intensidad Sequía (mediana índice)", "diverging": True},
}

_GI_STAR_STYLES = {
    "colors": {
         3: "#b2182b",
         2: "#ef8a62",
         0: "#f7f7f7",
        -2: "#67a9cf",
        -3: "#2166ac",
    },
    "labels": {
         3: "Hotspot: Vulnerabilidad Crítica (99% conf.)",
         2: "Hotspot: Vulnerabilidad Alta (95% conf.)",
         0: "Sin significancia espacial",
        -2: "Coldspot: Recuperación Activa (95% conf.)",
        -3: "Coldspot: Refugio Climático Estable (99% conf.)",
    }
}


def _get_imshow_origin(da: xr.DataArray) -> str:
    if "y" in da.dims and len(da.y) > 1:
        return "upper" if float(da.y[0]) > float(da.y[-1]) else "lower"
    return "upper"


def _get_extent(ds_or_da) -> list:
    return [
        float(ds_or_da.x.min()),
        float(ds_or_da.x.max()),
        float(ds_or_da.y.min()),
        float(ds_or_da.y.max()),
    ]


# =============================================================================
# EXPORTACIÓN DE DATOS
# =============================================================================

def export_metrics_to_geotiff(metrics_ds: xr.Dataset, output_dir: str) -> list:
    """
    Exporta cada métrica del Dataset como un GeoTIFF individual georreferenciado.
    Un archivo por métrica facilita la carga directa en QGIS sin seleccionar bandas.
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []

    for var_name in metrics_ds.data_vars:
        da = metrics_ds[var_name]

        if "x" in da.dims and "y" in da.dims:
            da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
            if not da.rio.crs:
                warnings.warn(
                    f"La variable '{var_name}' no tiene CRS definido. "
                    "El GeoTIFF se exportará sin georreferencia explícita."
                )
            out_path = os.path.join(output_dir, f"{var_name}.tif")
            da = da.rio.write_nodata(np.nan, encoded=True)
            da.rio.to_raster(out_path, driver="GTiff")
            generated_files.append(out_path)
        else:
            warnings.warn(f"La variable '{var_name}' carece de dimensiones x/y. Omitida.")

    return generated_files


def export_clustering_to_geotiff(clustering_ds: xr.Dataset, output_dir: str, name: str = "hotspots") -> str:
    """
    Exporta el resultado del análisis Gi* (z_score y clustering) como GeoTIFF multibanda.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{name}_gi_star.tif")
    adjusted_ds = clustering_ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
    adjusted_ds.rio.to_raster(out_path, driver="GTiff")
    return out_path


def export_events_to_csv(events_df: pd.DataFrame, output_dir: str, filename: str = "drought_events.csv") -> str:
    """
    Exporta la tabla de eventos detectados por detect_drought_events() a CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    events_df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def export_zonal_statistics_to_csv(metrics_ds: xr.Dataset, output_dir: str, filename: str = "zonal_statistics.csv") -> str:
    """
    Calcula y exporta estadísticos zonales de cada métrica a CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    records = []

    for var_name in metrics_ds.data_vars:
        values = metrics_ds[var_name].values.flatten()
        values = values[~np.isnan(values)]

        if len(values) == 0:
            warnings.warn(f"La capa '{var_name}' está vacía de datos válidos. Omitida.")
            continue

        record = {
            "metric":   var_name,
            "mean":     float(np.mean(values)),
            "median":   float(np.median(values)),
            "std":      float(np.std(values)),
            "min":      float(np.min(values)),
            "max":      float(np.max(values)),
            "p10":      float(np.percentile(values, 10)),
            "p25":      float(np.percentile(values, 25)),
            "p75":      float(np.percentile(values, 75)),
            "p90":      float(np.percentile(values, 90)),
            "n_pixels": int(len(values)),
        }

        if var_name == "did_not_recover":
            record["pct_not_recovered"] = float(np.mean(values) * 100)

        records.append(record)

    stats_df = pd.DataFrame(records)
    out_path = os.path.join(output_dir, filename)
    stats_df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def export_metrics_to_vector(
    metrics_ds: xr.Dataset,
    output_dir: str,
    filename: str = "metrics_points.gpkg"
) -> str:
    """
    Exporta todas las métricas como capa vectorial de puntos en GeoPackage.

    Cada píxel válido del área de estudio se convierte en un punto con todas
    las métricas como campos en la tabla de atributos. Permite realizar consultas
    espaciales compuestas en QGIS, ArcGIS o GeoPandas, por ejemplo:

        resistance < -0.5 AND drought_min < -3.0
        recovery_time > 15 AND resistance < -0.3
        did_not_recover = 1 AND drought_min < -4.0

    Parámetros:
    -----------
    metrics_ds : xr.Dataset
        Dataset de salida de vegetation_impact_metrics().
    output_dir : str
        Directorio de salida.
    filename : str
        Nombre del archivo GeoPackage de salida.

    Retorna:
    --------
    str : Ruta al archivo GeoPackage generado.
    """
    os.makedirs(output_dir, exist_ok=True)

    x_coords = metrics_ds.x.values
    y_coords = metrics_ds.y.values
    xx, yy = np.meshgrid(x_coords, y_coords)
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()

    records = {"geometry": [Point(x, y) for x, y in zip(xx_flat, yy_flat)]}

    for var_name in metrics_ds.data_vars:
        values = metrics_ds[var_name].values.flatten().astype(np.float64)
        records[var_name] = values

    gdf = gpd.GeoDataFrame(records, crs=metrics_ds.rio.crs)

    metric_cols = list(metrics_ds.data_vars)
    gdf = gdf.dropna(subset=metric_cols, how="all").reset_index(drop=True)

    if len(gdf) == 0:
        warnings.warn("No hay píxeles válidos para exportar como vectorial.")
        return ""

    out_path = os.path.join(output_dir, filename)
    gdf.to_file(out_path, driver="GPKG")
    return out_path


# =============================================================================
# GRÁFICOS
# =============================================================================

def plot_drought_timeseries(
    drought_index_da: xr.DataArray,
    windows: dict,
    output_dir: str,
    index_name: str = "Índice de Sequía",
    filename: str = "drought_timeseries.png"
) -> str:
    """
    Genera la serie temporal del índice de sequía con las ventanas de análisis
    sombreadas. Usa la mediana espacial del índice sobre el área de estudio.
    """
    os.makedirs(output_dir, exist_ok=True)

    if "x" in drought_index_da.dims and "y" in drought_index_da.dims:
        series = drought_index_da.median(dim=["x", "y"])
    else:
        series = drought_index_da

    times  = pd.to_datetime(series.time.values)
    values = series.values

    fig, ax = plt.subplots(figsize=(14, 5))

    index_windows = windows.get("index", windows)
    window_styles = {
        "pre_drought":    {"color": "#2196F3", "alpha": 0.12, "label": "Ventana Pre-sequía (Línea Base)"},
        "during_drought": {"color": "#F44336", "alpha": 0.18, "label": "Periodo del Evento Extremo"},
        "post_drought":   {"color": "#4CAF50", "alpha": 0.12, "label": "Ventana Post-sequía (Recuperación)"},
    }

    for window_key, style in window_styles.items():
        if window_key in index_windows:
            slc = index_windows[window_key]
            mask = (times >= pd.to_datetime(slc.start)) & (times <= pd.to_datetime(slc.stop))
            if mask.any():
                ax.axvspan(times[mask][0], times[mask][-1],
                           color=style["color"], alpha=style["alpha"], label=style["label"])

    ax.plot(times, values, color="#2c3e50", linewidth=1.6, zorder=3)
    ax.fill_between(times, values, 0, where=(values < 0),  color="#e74c3c", alpha=0.20, zorder=2)
    ax.fill_between(times, values, 0, where=(values >= 0), color="#2ecc71", alpha=0.15, zorder=2)

    for threshold, ls, label in [(-1.0, "--", "Umbral: Sequía Moderada (-1.0)"),
                                  (-1.5, "-.", "Umbral: Sequía Severa (-1.5)"),
                                  (-2.0, ":",  "Umbral: Sequía Extrema (-2.0)")]:
        ax.axhline(threshold, color="#962d2d", linewidth=0.9, linestyle=ls, alpha=0.7, label=label)

    ax.axhline(0, color="black", linewidth=0.7, alpha=0.5)
    ax.set_xlabel("Fecha", fontsize=11, labelpad=8)
    ax.set_ylabel(index_name, fontsize=11)
    ax.set_title(f"Evolución del {index_name} — Ventanas de análisis (índice sin lag)",
                 fontsize=13, pad=12)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_metrics_individual(
    metrics_ds: xr.Dataset,
    output_dir: str,
    subfolder: str = "metricas_individuales"
) -> list:
    """
    Exporta cada métrica como una figura PNG individual independiente.

    Una figura por métrica facilita la inserción directa en la memoria del TFM
    o en presentaciones sin necesidad de recortar un panel compuesto.

    Parámetros:
    -----------
    metrics_ds : xr.Dataset
        Dataset de salida de vegetation_impact_metrics().
    output_dir : str
        Directorio raíz de salida.
    subfolder : str
        Subcarpeta donde se guardan las figuras individuales.
        Por defecto 'metricas_individuales'.

    Retorna:
    --------
    list : Rutas a todos los archivos PNG generados.
    """
    out_dir = os.path.join(output_dir, subfolder)
    os.makedirs(out_dir, exist_ok=True)
    generated = []

    plot_vars = [v for v in _METRIC_STYLES if v in metrics_ds.data_vars]
    extent = _get_extent(metrics_ds)

    for var_name in plot_vars:
        da     = metrics_ds[var_name]
        data   = da.values
        style  = _METRIC_STYLES[var_name]
        origin = _get_imshow_origin(da)

        if style["diverging"]:
            abs_max = float(np.nanpercentile(np.abs(data), 97))
            abs_max = abs_max if abs_max > 1e-4 else 1.0
            norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        else:
            norm = mcolors.Normalize(
                vmin=float(np.nanmin(data)),
                vmax=float(np.nanmax(data))
            )

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(data, cmap=style["cmap"], norm=norm,
                       interpolation="nearest", extent=extent, origin=origin)
        ax.set_title(style["label"], fontsize=13, pad=10, weight="bold")
        ax.set_xlabel("X (Coordenada)", fontsize=9)
        ax.set_ylabel("Y (Coordenada)", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"{var_name}.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        generated.append(out_path)

    return generated


def plot_hotspots(clustering_ds: xr.Dataset, output_dir: str, filename: str = "hotspots_map.png") -> str:
    """
    Genera la cartografía del análisis Getis-Ord Gi* con los 5 niveles de
    confianza estadística completos.
    """
    os.makedirs(output_dir, exist_ok=True)

    da = clustering_ds["clustering"]
    clustering_raw = da.values
    extent = _get_extent(clustering_ds)
    origin = _get_imshow_origin(da)

    categories = [-3, -2, 0, 2, 3]
    color_list  = [_GI_STAR_STYLES["colors"][cat] for cat in categories]
    cmap   = mcolors.ListedColormap(color_list)
    bounds = [-3.5, -2.5, -0.5, 0.5, 2.5, 3.5]
    norm   = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(clustering_raw, cmap=cmap, norm=norm,
              interpolation="nearest", extent=extent, origin=origin)

    ax.set_title("Mapa de Patrones Espaciales de Vulnerabilidad — Getis-Ord Gi*",
                 fontsize=13, pad=12, weight="bold")
    ax.set_xlabel("Coordenada X", fontsize=9)
    ax.set_ylabel("Coordenada Y", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)

    patches = [
        mpatches.Patch(color=_GI_STAR_STYLES["colors"][cat],
                       label=_GI_STAR_STYLES["labels"][cat])
        for cat in [3, 2, 0, -2, -3]
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=9, framealpha=0.95)

    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_metrics_histograms(metrics_ds: xr.Dataset, output_dir: str, filename: str = "metrics_histograms.png") -> str:
    """
    Genera histogramas de distribución para cada métrica.
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_vars = [v for v in _METRIC_STYLES if v in metrics_ds.data_vars and v != "did_not_recover"]

    n_cols = 3
    n_rows = int(np.ceil(len(plot_vars) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows))
    axes = np.array(axes).flatten()

    for i, var_name in enumerate(plot_vars):
        ax         = axes[i]
        flat_data  = metrics_ds[var_name].values.flatten()
        clean_data = flat_data[~np.isnan(flat_data)]
        style      = _METRIC_STYLES[var_name]

        if len(clean_data) == 0:
            ax.set_title(f"{style['label']} (Sin datos)")
            continue

        mean_val   = float(np.mean(clean_data))
        median_val = float(np.median(clean_data))

        ax.hist(clean_data, bins=45, color="#455a64", alpha=0.8, edgecolor="white", linewidth=0.3)
        ax.axvline(mean_val,   color="#e74c3c", linewidth=1.5, linestyle="--", label=f"Media: {mean_val:.3f}")
        ax.axvline(median_val, color="#3498db", linewidth=1.5, linestyle=":",  label=f"Mediana: {median_val:.3f}")

        ax.set_title(style["label"], fontsize=10, weight="bold", pad=6)
        ax.set_xlabel("Rango de Valores", fontsize=8)
        ax.set_ylabel("Frecuencia (Píxeles)", fontsize=8)
        ax.legend(fontsize=8, framealpha=0.8)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    for j in range(len(plot_vars), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distribución de Frecuencias de las Métricas de Impacto Ecológico",
                 fontsize=14, y=1.01, weight="bold")
    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_line_of_full_resilience(
    metrics_ds: xr.Dataset,
    output_dir: str,
    filename: str = "line_of_full_resilience.png",
    area_name: str = "Área de estudio"
) -> str:
    """
    Genera el scatter plot Resistencia vs Recuperación con la línea de resiliencia
    completa (Schwarz et al., 2020; Xu et al., 2024).
    """
    os.makedirs(output_dir, exist_ok=True)

    rt = metrics_ds["resistance"].values.flatten()
    rc = metrics_ds["recovery"].values.flatten()

    mask = ~np.isnan(rt) & ~np.isnan(rc)
    rt = rt[mask]
    rc = rc[mask]

    if len(rt) < 10:
        warnings.warn("Menos de 10 píxeles válidos para plot_line_of_full_resilience. Omitido.")
        return ""

    _, _, r_value, p_value, _ = stats.linregress(rt, rc)
    r2        = r_value ** 2
    sign_corr = np.sign(r_value)
    slope_rma = (np.std(rc) / np.std(rt)) * sign_corr
    intercept_rma = np.mean(rc) - slope_rma * np.mean(rt)

    rt_range           = np.linspace(rt.min(), rt.max(), 200)
    rc_full_resilience = -rt_range
    rc_regression      = slope_rma * rt_range + intercept_rma

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.scatter(rt, rc, alpha=0.3, s=8, color="#546e7a", zorder=2, label="Píxeles")

    ax.plot(rt_range, rc_full_resilience, color="#2c3e50", linewidth=2.0,
            linestyle="--", zorder=4, label="Resiliencia completa (Rc = −Rt)")

    p_str = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
    ax.plot(rt_range, rc_regression, color="#e74c3c", linewidth=2.0,
            linestyle="-", zorder=5,
            label=f"Regresión RMA\ny = {slope_rma:.2f}x + {intercept_rma:.2f} | R² = {r2:.2f} | {p_str}")

    ax.axline((0, 0), slope=-1, color="#95a5a6", linewidth=0.8,
              linestyle=":", alpha=0.5, zorder=1)

    ax.text(0.02, 0.97, "Rc + Rt > 0\n(Superrecuperación)",
            transform=ax.transAxes, fontsize=8, color="#27ae60",
            verticalalignment="top", alpha=0.7)
    ax.text(0.98, 0.03, "Rc + Rt < 0\n(Recuperación incompleta)",
            transform=ax.transAxes, fontsize=8, color="#c0392b",
            horizontalalignment="right", alpha=0.7)

    ax.axhline(0, color="black", linewidth=0.6, alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.6, alpha=0.3)

    ax.set_xlabel("Resistencia — Rt (ΔSNDVI)", fontsize=11)
    ax.set_ylabel("Recuperación — Rc (ΔSNDVI)", fontsize=11)
    ax.set_title(
        f"Línea de Resiliencia Completa — {area_name}\n"
        f"(Schwarz et al., 2020; Xu et al., 2024)",
        fontsize=12, pad=10, weight="bold"
    )
    ax.legend(fontsize=9, framealpha=0.9, loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
