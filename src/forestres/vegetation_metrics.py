import xarray as xr
import numpy as np


def aggregate_temporal_window(da: xr.DataArray, method: str = "median") -> xr.DataArray:
    """
    Collapse the time dimension of a data window into a single scalar per pixel.

    Parameters
    ----------
    da : xr.DataArray
        Array with a time dimension.
    method : str
        Aggregation method: 'median' (recommended for ecological data due to
        robustness to outliers), 'mean', or 'max'.

    Returns
    -------
    xr.DataArray
        2D array (y, x) with the aggregated value per pixel.
    """
    if method == "median":
        return da.median(dim="time", skipna=True)
    elif method == "mean":
        return da.mean(dim="time", skipna=True)
    elif method == "max":
        return da.max(dim="time", skipna=True)
    else:
        raise ValueError(f"Method '{method}' is not supported. Use 'median', 'mean', or 'max'.")


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
    Compute vegetation vulnerability and resilience metrics for a drought event.

    Implements the Lloret et al. (2011) metrics adapted to use absolute differences
    instead of ratios, avoiding sign instability when working with a standardised
    index such as SNDVI (Xu et al., 2024). Also includes Accumulated Deficit,
    sustained Recovery Time, and an exposure mask that excludes pixels with no
    real water stress.

    SNDVI windows are shifted relative to the drought index windows to compensate
    for the vegetation lag. Index windows are used exclusively to compute the
    exposure mask.

    Accumulated deficit and recovery time are referenced to SNDVI anomaly 0
    (historical normal conditions) rather than the Pre period value, removing
    dependence on the choice of temporal windows.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset clipped to the study area, containing both vegetation and index
        variables.
    veg_var : str
        Name of the vegetation variable (e.g. 'ndvi').
    index_var : str
        Name of the drought index variable (e.g. 'drought_index').
    windows : dict
        Dict with two sub-dicts returned by get_analysis_windows():
            'index':      drought index temporal windows (no lag)
            'vegetation': SNDVI temporal windows (lag applied)
    exposure_threshold : float
        Index threshold to classify a pixel as exposed to drought. Pixels where
        the minimum index during the event is >= this value are excluded (NaN).
        Default 0.0.
    agg_method : str
        Temporal aggregation method per pixel per window. 'median' recommended
        for robustness to outliers in ecological data.
    min_recovery_periods : int
        Minimum consecutive biweekly periods above anomaly 0 for a pixel to be
        considered truly recovered. Prevents transient rebounds from being counted
        as recovery. Default 4 (~2 months).

    Returns
    -------
    xr.Dataset
        Variables: resistance, recovery, resilience,
        accumulated_deficit, recovery_time, did_not_recover,
        drought_min (minimum index per pixel during the event),
        drought_median (median index per pixel during the event).
    """
    veg_data   = dataset[veg_var]
    index_data = dataset[index_var]

    # --- Extract temporal windows
    pre_da  = veg_data.sel(time=windows["vegetation"]["pre_drought"])
    dur_da  = veg_data.sel(time=windows["vegetation"]["during_drought"])
    post_da = veg_data.sel(time=windows["vegetation"]["post_drought"])

    index_dur_da = index_data.sel(time=windows["index"]["during_drought"])

    # --- Exposure mask
    # Peak severity per pixel during the event (no lag).
    # Uses the minimum rather than the median because exposure to a hazard
    # should be assessed by the extreme reached, not the central tendency.
    drought_min    = index_dur_da.min(dim="time", skipna=True)
    drought_median = index_dur_da.median(dim="time", skipna=True)

    # Exposure mask: pixels below onset threshold during the event
    # Also excludes coastal edge-effect pixels where drought_median == 0
    # (partially marine pixels at island boundaries with no real drought signal)
    edge_effect_mask = drought_median != 0
    exposure_mask = (drought_min < exposure_threshold) & edge_effect_mask

    # --- Static base maps (temporal aggregation per pixel)
    Pre  = aggregate_temporal_window(pre_da,  method=agg_method)
    Dur  = aggregate_temporal_window(dur_da,  method=agg_method)
    Post = aggregate_temporal_window(post_da, method=agg_method)

    # --- Lloret et al. (2011) metrics -- absolute differences (Xu et al., 2024)
    resistance = Dur - Pre   # Rt = Dr - PreDr
    recovery   = Post - Dur  # Rc = PostDr - Dr
    resilience = Post - Pre  # Rs = PostDr - PreDr

    # --- Dynamic metrics

    # Accumulated deficit referenced to anomaly 0.
    # Sum of biweekly periods where SNDVI is negative during the event.
    accumulated_deficit = dur_da.where(dur_da < 0).sum(dim="time", skipna=True)

    # Recovery time with sustained recovery requirement.
    # The pixel must remain >= 0 for min_recovery_periods consecutive periods
    # to avoid transient rebounds being counted as recovery.
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

    # Pixels that never reached sustained recovery get NaN
    total_post_periods = len(post_da.time)
    did_not_recover_mask = recovery_time == total_post_periods
    recovery_time = xr.where(did_not_recover_mask, np.nan, recovery_time)

    # Correct the rolling-window algorithmic bias
    # The rolling mean cannot produce a result until period N, introducing
    # a delay of (N-1) periods. Subtract this offset to get the true onset.
    recovery_time = xr.where(
        ~did_not_recover_mask & (recovery_time > 0),
        recovery_time - (min_recovery_periods - 1),
        recovery_time
    )

    # --- Apply exposure mask to all metrics
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
            "drought_median":      drought_median,
            "drought_min": apply_mask(drought_min),
            "drought_median": apply_mask(drought_median)
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
