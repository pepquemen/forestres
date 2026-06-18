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
    severity_threshold = -1.5,
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

print("\n--- Exportando métricas individuales ---")
individual_paths = drought_impact.plot_metrics_individual(metrics_ds, OUTPUT_DIR)
for p in individual_paths:
    print(f"  Exportada: {p}")

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
