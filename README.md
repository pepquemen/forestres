# forestres

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-orange)](https://github.com/pepquemen/forestres)

Python library for ecological drought impact assessment on forest vegetation using remote sensing data (SNDVI) and drought indices (scPDSI, SPEI, SPI).

Implements resilience metrics from [Lloret et al. (2011)](https://doi.org/10.1111/j.1600-0706.2011.19372.x) adapted for standardized indices following [Xu et al. (2024)](https://doi.org/10.1002/ece3.11467), Getis-Ord Gi* spatial autocorrelation analysis, and automated cartographic output generation.

---

## Features

- **Drought event detection** using Run Theory (Yevjevich, 1967) with configurable thresholds and pooling, including automatic estimation of Pre and Post analysis windows
- **Lag correlation analysis** to estimate the optimal vegetation response delay pixel by pixel
- **Ecological resilience metrics**: Resistance, Recovery, Resilience, Accumulated Deficit, Recovery Time
- **Spatial clustering** with Getis-Ord Gi* to identify vulnerability hotspots and climatic refugia
- **Automated output generation**: GeoTIFF, GeoPackage, CSV statistics, and publication-ready figures
- **Optimized memory management** for large NetCDF files (>7 GB) via early bounding box clipping

---

## Installation

```bash
pip install -e .
```

Or clone and install manually:

```bash
git clone https://github.com/pepquemen/forestres.git
cd forestres
pip install -e .
```

### Requirements

- Python 3.10+
- xarray, rioxarray, geopandas, pandas, numpy, scipy, matplotlib, netCDF4, shapely, pyproj

---

## Usage

### Step 1 — Load and clip data to the study area

```python
import forestres as fr

ds_clip = fr.load_and_merge_datasets(
    drought_path    = "scpdsi.nc",
    vegetation_path = "SNDVI.nc",
    drought_var     = "value",
    vegetation_var  = "SNDVI",
    shapefile_path  = "study_area.shp",
    crs             = "EPSG:23030"
)
```

`load_and_merge_datasets` reads both NetCDF files, clips them to the shapefile bounding box before loading into memory, aligns their time axes via nearest-neighbour search, and applies the exact polygon clip. The result is a single `xr.Dataset` with variables `drought_index` and `ndvi`, ready for analysis.

---

### Step 2 — Explore drought events and estimate the vegetation lag

```python
# Detect drought events in the historical series
events = fr.detect_drought_events(
    ds_clip,
    severity_threshold = -2.74,   # severe drought threshold (CSIC scPDSI scale)
    min_duration       = 6,       # minimum 6 biweekly periods (~3 months)
    pooling_periods    = 4,       # merge runs separated by fewer than 4 positive periods
    plot               = True,
    output_path        = "events.png"
)
print(events)

# Estimate the optimal vegetation response lag
lag_df = fr.compute_lag_correlation(
    ds_clip,
    max_lag     = 24,             # evaluate lags from 0 to 24 biweekly periods (~12 months)
    plot        = True,
    output_path = "lag_correlation.png",
    index_name  = "scPDSI"
)
```

`detect_drought_events` applies Run Theory to the spatial median of the drought index and returns a table of detected events with their start and end dates, duration, peak severity, accumulated deficit, and suggested Pre and Post analysis windows (`pre_start`, `pre_end`, `post_start`, `post_end`). These suggested dates are estimated automatically using the same pooling criterion as event detection. Review them against the time series plot before using them in the pipeline.

`compute_lag_correlation` calculates the pixel-wise Pearson correlation between the drought index and SNDVI for lags from 0 to `max_lag` biweekly periods. The optimal lag is the one that maximises the mean correlation across pixels. Use this value as `vegetation_lag_periods` in the pipeline.

---

### Step 3 — Run the full analysis pipeline

```python
results = fr.run_forestres_pipeline(
    dataset                = ds_clip,
    pre_start              = "1995-06-15",   # start of pre-drought baseline window
    event_start            = "1997-01-15",   # start of drought event (from Step 2)
    event_end              = "2001-10-15",   # end of drought event (from Step 2)
    post_end               = "2004-07-01",   # end of post-drought recovery window
    vegetation_lag_periods = 1,              # optimal lag (from Step 2)
    output_dir             = "results/",
    area_name              = "My study area"
)
```

The pipeline computes all resilience metrics, runs the Getis-Ord Gi* spatial analysis, and generates all output files automatically. The `results` dictionary contains the paths to all generated files.

---

## Input Data

The library is designed for CSIC drought indices and SNDVI data:

| Variable | Format | Source | Resolution |
|----------|--------|--------|-----------|
| scPDSI, SPEI, SPI | NetCDF | [CSIC Monitor de Sequía](https://monitordesequia.csic.es/) | ~1.1 km, biweekly |
| SNDVI | NetCDF | [Franquesa et al. (2025)](https://doi.org/10.1038/s41597-025-04427-7) | ~1.1 km, biweekly |
| Study area | Shapefile | User-defined | — |

Both NetCDF files must cover the same spatial extent and have overlapping time periods.

---

## Output Files

After running the pipeline, the following files are generated in `output_dir/`:

| File | Description |
|------|-------------|
| `geotiffs/*.tif` | One GeoTIFF per metric (8 total) + Gi* results |
| `metrics_points.gpkg` | GeoPackage with all metrics as point layer |
| `zonal_statistics.csv` | Summary statistics per metric |
| `individual_metrics/*.png` | One map figure per metric (8 total) |
| `individual_histograms/*.png` | One histogram per metric (7 total) |
| `drought_timeseries.png` | Drought index time series with analysis windows |
| `hotspots_map.png` | Getis-Ord Gi* vulnerability map |

---

## Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| Resistance | `Rt = Dur - Pre` | Vegetation impact during drought |
| Recovery | `Rc = Post - Dur` | Capacity to recover after event |
| Resilience | `Rs = Post - Pre` | Net balance pre vs post event |
| Accumulated Deficit | `sum(SNDVI < 0)` | Total vegetation loss during event |
| Recovery Time | — | Periods until sustained recovery (N consecutive) |
| did_not_recover | — | Binary mask: pixels without sustained recovery |
| drought_min | — | Peak drought severity per pixel |
| drought_median | — | Typical drought intensity per pixel |

Metrics follow [Lloret et al. (2011)](https://doi.org/10.1111/j.1600-0706.2011.19372.x) adapted with absolute differences instead of ratios following [Xu et al. (2024)](https://doi.org/10.1002/ece3.11467).

---

## Module Structure

```
src/forestres/
├── __init__.py          # Public API (19 functions)
├── spatial_io.py        # Data loading, alignment and clipping
├── drought_detection.py # Event detection, lag correlation, analysis windows
├── vegetation_metrics.py# Lloret metrics and dynamic metrics
├── spatial_clustering.py# Getis-Ord Gi* spatial autocorrelation
├── export.py            # GeoTIFF, GeoPackage, CSV and figure export
└── core.py              # Main pipeline orchestrator
```

---

## Troubleshooting

### PROJ error on Windows

If you encounter:
```
PROJ: proj_create_from_database: DATABASE.LAYOUT.VERSION.MINOR contains X whereas >= 6 is expected
```

Add this before importing the library:
```python
import os
os.environ["PROJ_LIB"] = r"C:\path\to\pyproj\proj_dir\share\proj"
import forestres as fr
```

Or update pyproj:
```bash
pip install --upgrade pyproj
```

---

## References

- Lloret, F., Keeling, E.G. & Sala, A. (2011). Components of tree resilience: effects of successive low-growth episodes in old ponderosa pine forests. *Oikos*, 120, 1909-1920. https://doi.org/10.1111/j.1600-0706.2011.19372.x
- Xu, Z. et al. (2024). Assessing forest resilience to drought using remote sensing. *Ecology and Evolution*. https://doi.org/10.1002/ece3.11467
- Getis, A. & Ord, J.K. (1992). The analysis of spatial association by use of distance statistics. *Geographical Analysis*, 24(3), 189-206.
- Yevjevich, V. (1967). An objective approach to definitions and investigations of continental hydrologic droughts. *Hydrology Papers*, 23, Colorado State University.
- Franquesa, M. et al. (2025). SNDVI: A standardized NDVI dataset for drought impact assessment. *Scientific Data*. https://doi.org/10.1038/s41597-025-04427-7
- Wilson, G. et al. (2017). Good enough practices in scientific computing. *PLOS Computational Biology*, 13(6), e1005510.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Citation

If you use this library in your research, please cite:

```
Quevedo Méndez, Josep. (2026). forestres: A Python library for ecological drought impact
assessment on forest vegetation (v0.1.0). GitHub.
https://github.com/pepquemen/forestres
```
