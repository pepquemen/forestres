# drought_impact

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-orange)](https://github.com/pepquemen/drought-impact)

Python library for ecological drought impact assessment on forest vegetation using remote sensing data (SNDVI) and drought indices (scPDSI, SPEI, SPI).

Implements resilience metrics from [Lloret et al. (2011)](https://doi.org/10.1111/j.1600-0706.2011.19372.x) adapted for standardized indices following [Xu et al. (2024)](https://doi.org/10.1002/ece3.11467), Getis-Ord Gi* spatial autocorrelation analysis, and automated cartographic output generation.

---

## Features

- **Drought event detection** using Run Theory (Yevjevich, 1967) with configurable thresholds and pooling
- **Lag correlation analysis** to estimate the optimal vegetation response delay pixel by pixel
- **Ecological resilience metrics**: Resistance, Recovery, Net Resilience, Accumulated Deficit, Recovery Time
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
git clone https://github.com/pepquemen/drought-impact.git
cd drought-impact
pip install -e .
```

### Requirements

- Python 3.10+
- xarray, rioxarray, geopandas, pandas, numpy, scipy, matplotlib, netCDF4, shapely, pyproj

---

## Quick Start

```python
import drought_impact

# Step 1: Load and clip data to study area
ds_clip = drought_impact.load_and_merge_datasets(
    drought_path    = "scpdsi.nc",
    vegetation_path = "SNDVI.nc",
    drought_var     = "value",
    vegetation_var  = "SNDVI",
    shapefile_path  = "study_area.shp",
    crs             = "EPSG:23030"
)

# Step 2: Explore drought events and estimate optimal lag
events = drought_impact.detect_drought_events(ds_clip, severity_threshold=-2.74)
lag_df = drought_impact.compute_lag_correlation(ds_clip, plot=True)

# Step 3: Run the full pipeline
results = drought_impact.run_drought_impact_pipeline(
    dataset                = ds_clip,
    pre_start              = "1995-06-15",
    event_start            = "1997-01-15",
    event_end              = "2001-10-15",
    post_end               = "2003-03-01",
    vegetation_lag_periods = 1,
    output_dir             = "results/",
    area_name              = "My study area"
)
```

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
| `metricas_individuales/*.png` | One figure per metric (8 total) |
| `drought_timeseries.png` | Drought index time series with analysis windows |
| `hotspots_map.png` | Getis-Ord Gi* vulnerability map |
| `metrics_histograms.png` | Distribution histograms for all metrics |
| `line_of_full_resilience.png` | Resistance vs Recovery scatter plot with RMA regression |

---

## Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| Resistance | `Rt = Dur − Pre` | Vegetation impact during drought |
| Recovery | `Rc = Post − Dur` | Capacity to recover after event |
| Net Resilience | `Rs = Post − Pre` | Net balance pre vs post event |
| Accumulated Deficit | `Σ SNDVI < 0` | Total vegetation loss during event |
| Recovery Time | — | Periods until sustained recovery (N consecutive) |
| did_not_recover | — | Binary mask: pixels without sustained recovery |
| drought_min | — | Peak drought severity per pixel |
| drought_median | — | Typical drought intensity per pixel |

Metrics follow [Lloret et al. (2011)](https://doi.org/10.1111/j.1600-0706.2011.19372.x) adapted with absolute differences instead of ratios following [Xu et al. (2024)](https://doi.org/10.1002/ece3.11467).

---

## Module Structure

```
src/drought_impact/
├── __init__.py          # Public API (20 functions)
├── spatial_io.py        # Data loading, alignment and clipping
├── drought_detection.py # Event detection, lag correlation, analysis windows
├── vegetation_metrics.py# Lloret metrics + dynamic metrics
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
import drought_impact
```

Or update pyproj:
```bash
pip install --upgrade pyproj
```

---

## References

- Lloret, F., Keeling, E.G. & Sala, A. (2011). Components of tree resilience: effects of successive low-growth episodes in old ponderosa pine forests. *Oikos*, 120, 1909–1920. https://doi.org/10.1111/j.1600-0706.2011.19372.x
- Xu, Z. et al. (2024). Assessing forest resilience to drought using remote sensing. *Ecology and Evolution*. https://doi.org/10.1002/ece3.11467
- Getis, A. & Ord, J.K. (1992). The analysis of spatial association by use of distance statistics. *Geographical Analysis*, 24(3), 189–206.
- Yevjevich, V. (1967). An objective approach to definitions and investigations of continental hydrologic droughts. *Hydrology Papers*, 23, Colorado State University.
- Franquesa, M. et al. (2025). SNDVI: A standardized NDVI dataset for drought impact assessment. *Scientific Data*. https://doi.org/10.1038/s41597-025-04427-7
- Schwarz, J. et al. (2020). Quantifying forest resilience. *Global Change Biology*.
- Wilson, G. et al. (2017). Good enough practices in scientific computing. *PLOS Computational Biology*, 13(6), e1005510.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Citation

If you use this library in your research, please cite:

```
Quetglas, P. (2025). drought_impact: A Python library for ecological drought impact 
assessment on forest vegetation (v0.1.0). GitHub. 
https://github.com/pepquemen/drought-impact
```
