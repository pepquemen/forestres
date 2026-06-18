import os
import logging

os.environ["PROJ_DATA"] = r"C:\Python314\Lib\site-packages\pyproj\proj_dir\share\proj"

import drought_impact

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

OUTPUT_DIR = r"C:\TFM\V3\output\mfe_1997_2001"

# =============================================================================
# PASO 1: Cargar y recortar datos
# =============================================================================
print("\n--- PASO 1: Carga de datos ---")
ds = drought_impact.load_and_merge_datasets(
    drought_path    = r"C:\TFM\V3\datos\scpdsi.nc",
    vegetation_path = r"C:\TFM\V3\datos\SNDVI.nc",
    drought_var     = "value",
    vegetation_var  = "SNDVI",
    crs             = "EPSG:23030"
)
ds_clip = drought_impact.clip_dataset_to_polygon(
    ds, r"C:\TFM\mfe_illesbalears\MFE_53.shp"
)
print(f"Dataset recortado: {dict(ds_clip.sizes)}")

# =============================================================================
# PASO 2: Funciones de apoyo
# =============================================================================
print("\n--- PASO 2a: Detección de eventos ---")
events = drought_impact.detect_drought_events(
    ds_clip,
    severity_threshold = -2.74,
    min_duration       = 6,
    pooling_periods    = 4,
    plot               = True,
    output_path        = os.path.join(OUTPUT_DIR, "eventos_detectados.png")
)
print(events)

# Exportar tabla de eventos a CSV
csv_events = drought_impact.export_events_to_csv(
    events,
    output_dir = OUTPUT_DIR,
    filename   = "drought_events.csv"
)
print(f"Eventos exportados a: {csv_events}")

print("\n--- PASO 2b: Correlación lag (píxel a píxel) ---")
lag_df = drought_impact.compute_lag_correlation(
    ds_clip,
    max_lag     = 24,
    plot        = True,
    output_path = os.path.join(OUTPUT_DIR, "lag_correlation.png"),
    index_name  = "scPDSI"
)

# =============================================================================
# PASO 3: Pipeline completo — Evento 1997-2001
# =============================================================================
print("\n--- PASO 3: Pipeline completo ---")
results = drought_impact.run_drought_impact_pipeline(
    dataset                = ds_clip,
    pre_start              = "1995-06-15",
    event_start            = "1997-01-15",
    event_end              = "2001-10-15",
    post_end               = "2003-03-01",
    output_dir             = OUTPUT_DIR,
    vegetation_lag_periods = 2,
    min_recovery_periods   = 4,
    exposure_threshold     = 0.0,
    agg_method             = "median",
    kernel_size            = 3,
    clustering_target_var  = "accumulated_deficit",
    area_name              = "Forestal Illes Balears"
)

# =============================================================================
# PASO 4: Exportar métricas individuales del panel
# =============================================================================
print("\n--- PASO 4: Exportar métricas individuales ---")
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

metrics_ds = drought_impact.vegetation_impact_metrics(
    dataset              = ds_clip,
    veg_var              = "ndvi",
    index_var            = "drought_index",
    windows              = results["windows"],
    exposure_threshold   = 0.0,
    agg_method           = "median",
    min_recovery_periods = 4
)

_METRIC_STYLES = {
    "resistance":          {"cmap": "RdYlGn",   "label": "Resistencia (ΔSNDVI)",              "diverging": True},
    "recovery":            {"cmap": "RdYlGn",   "label": "Recuperación (ΔSNDVI)",             "diverging": True},
    "resilience":          {"cmap": "RdYlGn",   "label": "Resiliencia Neta (ΔSNDVI)",         "diverging": True},
    "accumulated_deficit": {"cmap": "YlOrRd_r", "label": "Déficit Acumulado (ΔSNDVI·q)",      "diverging": False},
    "recovery_time":       {"cmap": "YlOrRd",   "label": "Tiempo de Recuperación (quincenas)","diverging": False},
    "drought_intensity":   {"cmap": "RdBu",     "label": "Intensidad Sequía (mín. índice)",   "diverging": True},
}

extent = [
    float(metrics_ds.x.min()), float(metrics_ds.x.max()),
    float(metrics_ds.y.min()), float(metrics_ds.y.max())
]

individual_dir = os.path.join(OUTPUT_DIR, "metricas_individuales")
os.makedirs(individual_dir, exist_ok=True)

for var_name, style in _METRIC_STYLES.items():
    if var_name not in metrics_ds.data_vars:
        continue

    da   = metrics_ds[var_name]
    data = da.values
    origin = "upper" if float(da.y[0]) > float(da.y[-1]) else "lower"

    if style["diverging"]:
        abs_max = float(np.nanpercentile(np.abs(data), 97))
        abs_max = abs_max if abs_max > 1e-4 else 1.0
        norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    else:
        norm = mcolors.Normalize(vmin=float(np.nanmin(data)), vmax=float(np.nanmax(data)))

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(data, cmap=style["cmap"], norm=norm,
                   interpolation="nearest", extent=extent, origin=origin)
    ax.set_title(style["label"], fontsize=13, pad=10, weight="bold")
    ax.set_xlabel("X (Coordenada)", fontsize=9)
    ax.set_ylabel("Y (Coordenada)", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
    plt.tight_layout()

    out_path = os.path.join(individual_dir, f"{var_name}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Exportada: {out_path}")

# =============================================================================
# PASO 5: Diagnóstico ventanas SNDVI
# =============================================================================
print("\n--- PASO 5: Diagnóstico ventanas SNDVI ---")
windows = results["windows"]
pre_med  = ds_clip["ndvi"].sel(time=windows["vegetation"]["pre_drought"]).median(dim=["x","y"]).mean().item()
dur_med  = ds_clip["ndvi"].sel(time=windows["vegetation"]["during_drought"]).median(dim=["x","y"]).mean().item()
post_med = ds_clip["ndvi"].sel(time=windows["vegetation"]["post_drought"]).median(dim=["x","y"]).mean().item()

print(f"Pre  mediana espaciotemporal: {pre_med:.4f}")
print(f"Dur  mediana espaciotemporal: {dur_med:.4f}")
print(f"Post mediana espaciotemporal: {post_med:.4f}")

print("\n--- Resultados ---")
for key, value in results.items():
    if key == "windows":
        continue
    if isinstance(value, list):
        print(f"  [{key}]: {len(value)} archivos")
    else:
        print(f"  [{key}]: {value}")