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


# --- Internal style configuration

_METRIC_STYLES = {
    "resistance":          {"cmap": "RdYlGn",   "label": "Resistance (ΔSNDVI)",                "diverging": True},
    "recovery":            {"cmap": "RdYlGn",   "label": "Recovery (ΔSNDVI)",                  "diverging": True},
    "resilience":          {"cmap": "RdYlGn",   "label": "Resilience (ΔSNDVI)",                "diverging": True},
    "accumulated_deficit": {"cmap": "YlOrRd_r", "label": "Accumulated Deficit (ΔSNDVI·q)",     "diverging": False},
    "recovery_time":       {"cmap": "YlOrRd",   "label": "Recovery Time (biweekly periods)",   "diverging": False},
    "did_not_recover":     {"cmap": "Reds",      "label": "Did Not Recover",                   "diverging": False},
    "drought_min":         {"cmap": "RdBu",      "label": "Drought Intensity (minimum)",       "diverging": True},
    "drought_median":      {"cmap": "RdBu",      "label": "Drought Intensity (median)",        "diverging": True},
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
         3: "Hotspot 99% confidence",
         2: "Hotspot 95% confidence",
         0: "Not significant",
        -2: "Coldspot 95% confidence",
        -3: "Coldspot 99% confidence",
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


# --- Data export

def export_metrics_to_geotiff(metrics_ds: xr.Dataset, output_dir: str) -> list:
    """
    Export each metric in the Dataset as an individual georeferenced GeoTIFF.
    One file per metric allows direct loading in QGIS without selecting bands.
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []

    for var_name in metrics_ds.data_vars:
        da = metrics_ds[var_name]

        if "x" in da.dims and "y" in da.dims:
            da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
            if not da.rio.crs:
                warnings.warn(
                    f"Variable '{var_name}' has no CRS defined. "
                    "GeoTIFF will be exported without explicit georeferencing."
                )
            out_path = os.path.join(output_dir, f"{var_name}.tif")
            da = da.rio.write_nodata(np.nan, encoded=True)
            da.rio.to_raster(out_path, driver="GTiff")
            generated_files.append(out_path)
        else:
            warnings.warn(f"Variable '{var_name}' has no x/y dimensions. Skipped.")

    return generated_files


def export_clustering_to_geotiff(clustering_ds: xr.Dataset, output_dir: str, name: str = "hotspots") -> str:
    """
    Export the Gi* result (z_score and clustering) as a multi-band GeoTIFF.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{name}_gi_star.tif")
    adjusted_ds = clustering_ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
    adjusted_ds.rio.to_raster(out_path, driver="GTiff")
    return out_path


def export_events_to_csv(events_df: pd.DataFrame, output_dir: str, filename: str = "drought_events.csv") -> str:
    """
    Export the event table produced by detect_drought_events() to CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    events_df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def export_zonal_statistics_to_csv(metrics_ds: xr.Dataset, output_dir: str, filename: str = "zonal_statistics.csv") -> str:
    """
    Compute and export zonal statistics for each metric to CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    records = []

    for var_name in metrics_ds.data_vars:
        values = metrics_ds[var_name].values.flatten()
        values = values[~np.isnan(values)]

        if len(values) == 0:
            warnings.warn(f"Layer '{var_name}' has no valid data. Skipped.")
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
    Export all metrics as a point vector layer in GeoPackage format.

    Each valid pixel in the study area becomes a point with all metrics as
    attribute fields. Enables compound spatial queries in QGIS, ArcGIS, or
    GeoPandas, for example:

        resistance < -0.5 AND drought_min < -3.0
        recovery_time > 15 AND resistance < -0.3
        did_not_recover = 1 AND drought_min < -4.0

    Parameters
    ----------
    metrics_ds : xr.Dataset
        Output of vegetation_impact_metrics().
    output_dir : str
        Output directory.
    filename : str
        Name of the output GeoPackage file.

    Returns
    -------
    str
        Path to the generated GeoPackage file.
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
        warnings.warn("No valid pixels to export as a vector layer.")
        return ""

    out_path = os.path.join(output_dir, filename)
    gdf.to_file(out_path, driver="GPKG")
    return out_path


# --- Plots

def plot_drought_timeseries(
    drought_index_da: xr.DataArray,
    windows: dict,
    output_dir: str,
    index_name: str = "Drought Index",
    filename: str = "drought_timeseries.png"
) -> str:
    """
    Plot the drought index time series with analysis windows shaded.
    Uses the spatial median of the index over the study area.
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
        "pre_drought":    {"color": "#2196F3", "alpha": 0.12, "label": "Pre-drought"},
        "during_drought": {"color": "#F44336", "alpha": 0.18, "label": "Drought event"},
        "post_drought":   {"color": "#4CAF50", "alpha": 0.12, "label": "Post-drought"},
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

    for threshold, ls, label in [(-1.0, "--", "Moderate drought"),
                                  (-1.5, "-.", "Severe drought"),
                                  (-2.0, ":",  "Extreme drought")]:
        ax.axhline(threshold, color="#962d2d", linewidth=0.9, linestyle=ls, alpha=0.7, label=label)

    ax.axhline(0, color="black", linewidth=0.7, alpha=0.5)
    ax.set_xlabel("Date", fontsize=11, labelpad=8)
    ax.set_ylabel(index_name, fontsize=11)
    ax.set_title("Drought Index Time Series", fontsize=13, pad=12)
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
    subfolder: str = "individual_metrics"
) -> list:
    """
    Export each metric as an individual PNG figure.

    Parameters
    ----------
    metrics_ds : xr.Dataset
        Output of vegetation_impact_metrics().
    output_dir : str
        Root output directory.
    subfolder : str
        Subdirectory for the individual figures.

    Returns
    -------
    list
        Paths to all generated PNG files.
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
        ax.set_xlabel("X coordinate", fontsize=9)
        ax.set_ylabel("Y coordinate", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"{var_name}.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        generated.append(out_path)

    return generated


def plot_hotspots(clustering_ds: xr.Dataset, output_dir: str, filename: str = "hotspots_map.png") -> str:
    """
    Plot the Getis-Ord Gi* analysis map with all five statistical confidence levels.
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

    ax.set_title("Spatial Vulnerability Clusters", fontsize=13, pad=12, weight="bold")
    ax.set_xlabel("X coordinate", fontsize=9)
    ax.set_ylabel("Y coordinate", fontsize=9)
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
    Generate distribution histograms for each metric.
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
            ax.set_title(f"{style['label']} (No data)")
            continue

        mean_val   = float(np.mean(clean_data))
        median_val = float(np.median(clean_data))

        ax.hist(clean_data, bins=45, color="#455a64", alpha=0.8, edgecolor="white", linewidth=0.3)
        ax.axvline(mean_val,   color="#e74c3c", linewidth=1.5, linestyle="--", label=f"Mean: {mean_val:.3f}")
        ax.axvline(median_val, color="#3498db", linewidth=1.5, linestyle=":",  label=f"Median: {median_val:.3f}")

        ax.set_title(style["label"], fontsize=10, weight="bold", pad=6)
        ax.set_xlabel("Value", fontsize=8)
        ax.set_ylabel("Pixel count", fontsize=8)
        ax.legend(fontsize=8, framealpha=0.8)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    for j in range(len(plot_vars), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Metric Distributions", fontsize=14, y=1.01, weight="bold")
    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_line_of_full_resilience(
    metrics_ds: xr.Dataset,
    output_dir: str,
    filename: str = "line_of_full_resilience.png",
    area_name: str = "Study area"
) -> str:
    """
    Generate a Resistance vs Recovery scatter plot with the line of full resilience
    and an RMA regression.
    """
    os.makedirs(output_dir, exist_ok=True)

    rt = metrics_ds["resistance"].values.flatten()
    rc = metrics_ds["recovery"].values.flatten()

    mask = ~np.isnan(rt) & ~np.isnan(rc)
    rt = rt[mask]
    rc = rc[mask]

    if len(rt) < 10:
        warnings.warn("Fewer than 10 valid pixels for plot_line_of_full_resilience. Skipped.")
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

    ax.scatter(rt, rc, alpha=0.3, s=8, color="#546e7a", zorder=2, label="Pixels")

    ax.plot(rt_range, rc_full_resilience, color="#2c3e50", linewidth=2.0,
            linestyle="--", zorder=4, label="Full resilience (Rc = -Rt)")

    p_str = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
    ax.plot(rt_range, rc_regression, color="#e74c3c", linewidth=2.0,
            linestyle="-", zorder=5,
            label=f"RMA regression\ny = {slope_rma:.2f}x + {intercept_rma:.2f} | R2 = {r2:.2f} | {p_str}")

    ax.axline((0, 0), slope=-1, color="#95a5a6", linewidth=0.8,
              linestyle=":", alpha=0.5, zorder=1)

    ax.text(0.02, 0.97, "Rc + Rt > 0\n(Over-recovery)",
            transform=ax.transAxes, fontsize=8, color="#27ae60",
            verticalalignment="top", alpha=0.7)
    ax.text(0.98, 0.03, "Rc + Rt < 0\n(Incomplete recovery)",
            transform=ax.transAxes, fontsize=8, color="#c0392b",
            horizontalalignment="right", alpha=0.7)

    ax.axhline(0, color="black", linewidth=0.6, alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.6, alpha=0.3)

    ax.set_xlabel("Resistance -- Rt (ΔSNDVI)", fontsize=11)
    ax.set_ylabel("Recovery -- Rc (ΔSNDVI)", fontsize=11)
    ax.set_title(
        f"Line of Full Resilience -- {area_name}",
        fontsize=12, pad=10, weight="bold"
    )
    ax.legend(fontsize=9, framealpha=0.9, loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
