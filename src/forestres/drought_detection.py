import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import warnings


# --- Decision-support functions

def _find_positive_period_start(values, from_idx, pooling_periods):
    """
    Go backwards from from_idx to find the start of the positive period,
    applying pooling for transient negatives.
    """
    pre_start_idx = from_idx
    consecutive_neg = 0
    i = from_idx

    while i >= 0:
        if values[i] >= 0:
            pre_start_idx = i
            consecutive_neg = 0
        else:
            consecutive_neg += 1
            if consecutive_neg > pooling_periods:
                pre_start_idx = i + consecutive_neg
                break
        i -= 1

    return pre_start_idx


def _find_positive_period_end(values, from_idx, pooling_periods):
    """
    Go forward from from_idx to find the end of the positive period,
    applying pooling for transient negatives.
    """
    post_end_idx = from_idx
    consecutive_neg = 0
    i = from_idx

    while i < len(values):
        if values[i] >= 0:
            post_end_idx = i
            consecutive_neg = 0
        else:
            consecutive_neg += 1
            if consecutive_neg > pooling_periods:
                post_end_idx = i - consecutive_neg
                break
        i += 1

    return post_end_idx


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
    Detect drought events in the time series using Run Theory (Yevjevich, 1967)
    applied to the spatial median of the drought index.

    An event is defined as a consecutive run of semi-monthly periods where the
    index falls below onset_threshold, provided it reaches severity_threshold at
    some point and meets the minimum duration. Runs separated by fewer than
    pooling_periods positive periods are merged into a single event.

    For each detected event, the function also estimates the surrounding positive
    periods (Pre and Post windows) using the same pooling criterion. These are
    provided as informational columns to guide the user in defining the analysis
    windows for run_drought_impact_pipeline(). The user should review these
    suggested dates against the time series plot and adjust if needed.

    All temporal parameters are expressed in biweekly periods (1 month ~ 2 periods).

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset clipped to the study area (output of clip_dataset_to_polygon).
    index_var : str
        Name of the drought index variable (e.g. 'drought_index').
    onset_threshold : float
        Run onset threshold. The index must fall below this value to start
        accumulating a deficit run (default 0.0).
    severity_threshold : float
        Minimum severity the event must reach at some point to qualify as a
        real drought (default -1.5, severe drought).
    min_duration : int
        Minimum number of consecutive biweekly periods for a valid event
        (default 6, approximately 3 months).
    pooling_periods : int
        Maximum number of positive periods between two runs for them to be
        merged into one event (default 4, approximately 2 months).
        Also used to pool transient negatives when estimating Pre and Post windows.
    plot : bool
        If True, generate a time-series plot with events shaded.
    output_path : str, optional
        Path to save the plot. If None, the plot is displayed on screen.

    Returns
    -------
    pd.DataFrame
        Columns: event_id, start_date, end_date, duration_periods,
        duration_months, max_severity, accumulated_deficit,
        pre_start, pre_end, post_start, post_end.

        pre_start / pre_end: suggested start and end of the pre-drought window
        (full positive period before the event, with pooling).
        post_start / post_end: suggested start and end of the post-drought window
        (full positive period after the event, with pooling).
        These dates are suggestions -- review them against the time series plot
        before using them in run_drought_impact_pipeline().
    """
    # Compute the spatial median of the index
    spatial_median = dataset[index_var].median(dim=["x", "y"])
    times  = pd.to_datetime(spatial_median.time.values)
    values = spatial_median.values

    # Identify periods below the onset threshold
    below_onset = values < onset_threshold

    # Identify consecutive runs below the threshold
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

    # Pooling: merge runs separated by fewer than pooling_periods positive periods
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

    # Filter events by minimum severity and minimum duration
    events_list = []
    event_id = 1

    for start_i, end_i in events_raw:
        duration = end_i - start_i + 1
        event_values = values[start_i:end_i + 1]
        max_severity = float(np.min(event_values))
        accumulated_deficit = float(np.sum(event_values[event_values < 0]))

        if duration >= min_duration and max_severity <= severity_threshold:

            # Estimate Pre window: go backwards from event start
            pre_end_idx   = start_i - 1
            pre_start_idx = _find_positive_period_start(
                values, pre_end_idx, pooling_periods
            ) if pre_end_idx >= 0 else 0

            # Estimate Post window: go forward from event end
            post_start_idx = end_i + 1
            post_end_idx   = _find_positive_period_end(
                values, post_start_idx, pooling_periods
            ) if post_start_idx < len(values) else len(values) - 1

            events_list.append({
                "event_id":            event_id,
                "start_date":          str(times[start_i].date()),
                "end_date":            str(times[end_i].date()),
                "duration_periods":    duration,
                "duration_months":     round(duration / 2, 1),
                "max_severity":        max_severity,
                "accumulated_deficit": accumulated_deficit,
                "pre_start":           str(times[pre_start_idx].date()),
                "pre_end":             str(times[pre_end_idx].date()) if pre_end_idx >= 0 else str(times[start_i].date()),
                "post_start":          str(times[post_start_idx].date()) if post_start_idx < len(times) else str(times[end_i].date()),
                "post_end":            str(times[post_end_idx].date()),
            })
            event_id += 1

    events_df = pd.DataFrame(events_list)

    if events_df.empty:
        warnings.warn(
            "No drought events detected with the given parameters. "
            "Consider reducing severity_threshold or min_duration."
        )
        return events_df

    # Print suggested windows for each event
    print("\nSuggested analysis windows (review against time series before using):")
    for _, row in events_df.iterrows():
        print(
            f"  Event {int(row['event_id'])}: "
            f"pre_start={row['pre_start']}  "
            f"event_start={row['start_date']}  "
            f"event_end={row['end_date']}  "
            f"post_end={row['post_end']}"
        )
    print("  Use these as pre_start, event_start, event_end, post_end in run_drought_impact_pipeline().\n")

    # Plot the time series with events shaded
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
                   linestyle="--", alpha=0.6, label=f"Onset threshold ({onset_threshold})")
        ax.axhline(severity_threshold, color="#962d2d", linewidth=0.8,
                   linestyle="-.", alpha=0.8, label=f"Severity threshold ({severity_threshold})")
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)

        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel(index_var, fontsize=11)
        ax.set_title(
            f"Detected Drought Events -- {len(events_df)} events",
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


# --- Lag correlation

def compute_lag_correlation(
    dataset: xr.Dataset,
    index_var: str = "drought_index",
    veg_var: str = "ndvi",
    max_lag: int = 24,
    only_drought_periods: bool = False,
    drought_threshold: float = 0.0,
    plot: bool = False,
    output_path: str = None,
    index_name: str = "Drought Index"
) -> pd.DataFrame:
    """
    Compute pixel-wise Pearson correlation between the drought index and SNDVI
    for a range of temporal lags using spatial medians over the full historical series.

    Use the result to select the optimal vegetation_lag_periods before running
    the main pipeline.

    Lags are expressed in biweekly periods. Lag=4 is approximately 2 months.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset clipped to the study area.
    index_var : str
        Name of the drought index variable.
    veg_var : str
        Name of the vegetation variable.
    max_lag : int
        Maximum lag in biweekly periods to evaluate (default 24, ~12 months).
    only_drought_periods : bool
        If True, compute correlation only over periods where the index is below
        drought_threshold. Useful for capturing the lag specific to water-stress
        conditions.
    drought_threshold : float
        Index threshold for filtering stress periods when only_drought_periods
        is True (default 0.0).
    plot : bool
        If True, generate a bar chart of correlation by lag.
    output_path : str, optional
        Path to save the plot. If None and plot=True, the plot is displayed.
    index_name : str
        Drought index name for the plot title.

    Returns
    -------
    pd.DataFrame
        Columns: lag_periods, lag_months, correlation.
        Also prints the recommended optimal lag.
    """
    # Pixel-wise vectorised correlation over the full historical series.
    # Computed per pixel across the study area and reported as the mean,
    # avoiding the bias of the spatial median in heterogeneous areas.
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

        # Filter to drought stress periods if requested
        if only_drought_periods:
            drought_mask = idx.mean(axis=(1, 2)) < drought_threshold
            idx = idx[drought_mask]
            veg = veg[drought_mask]

        n_t = idx.shape[0]
        if n_t < 30:
            records.append({"lag_periods": lag, "lag_months": round(lag / 2, 1), "correlation": np.nan})
            continue

        # Flatten to (time, pixels) to vectorise correlation
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

        # Mean across pixels (not median) to avoid bias in heterogeneous areas
        corr_map  = np.where(denominator > 1e-6, numerator / denominator, np.nan)
        mean_corr = float(np.nanmean(corr_map))

        records.append({
            "lag_periods": lag,
            "lag_months":  round(lag / 2, 1),
            "correlation": round(mean_corr, 4)
        })

    corr_df = pd.DataFrame(records)

    best_lag = corr_df.loc[corr_df["correlation"].idxmax()]
    mode_str = "drought periods only" if only_drought_periods else "full historical series"
    print(f"\nLag correlation (pixel-wise -- {mode_str}):")
    print(corr_df.to_string(index=False))
    print(
        f"\nRecommended optimal lag: {int(best_lag['lag_periods'])} biweekly periods "
        f"(~{best_lag['lag_months']} months) -- mean correlation: {best_lag['correlation']:.4f}"
    )
    print("Use this value as vegetation_lag_periods in run_drought_impact_pipeline().")

    if plot:
        import os
        from matplotlib.patches import Patch

        best_lag = corr_df.loc[corr_df["correlation"].idxmax()]
        colors = [
            "#e74c3c" if row["lag_periods"] == best_lag["lag_periods"] else "#546e7a"
            for _, row in corr_df.iterrows()
        ]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(
            corr_df["lag_periods"],
            corr_df["correlation"],
            color=colors,
            edgecolor="white",
            linewidth=0.5,
            width=0.7
        )

        for bar, (_, row) in zip(bars, corr_df.iterrows()):
            if not np.isnan(row["correlation"]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{row['correlation']:.4f}",
                    ha="left", va="bottom",
                    fontsize=7.5, color="#2c3e50",
                    rotation=45
                )

        ax.set_xticks(corr_df["lag_periods"])
        ax.set_xticklabels(
            [f"Lag {int(r['lag_periods'])}\n(~{r['lag_months']} mo)"
             for _, r in corr_df.iterrows()],
            fontsize=7, rotation=45, ha="right"
        )
        ax.set_xlabel("Lag (biweekly periods)", fontsize=11)
        ax.set_ylabel("Mean Pearson correlation (pixel-wise)", fontsize=11)
        ax.set_title(
            f"Lag Correlation -- {index_name} vs SNDVI",
            fontsize=12, pad=10
        )
        ax.set_ylim(0, corr_df["correlation"].max() * 1.12)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

        legend_elements = [
            Patch(facecolor="#e74c3c", label=f"Optimal lag ({int(best_lag['lag_periods'])} periods)"),
            Patch(facecolor="#546e7a", label="Other lags")
        ]
        ax.legend(handles=legend_elements, fontsize=9, framealpha=0.9)

        plt.tight_layout()

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
            fig.savefig(output_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    return corr_df


# --- Temporal windows (used internally by the pipeline)

def get_analysis_windows(
    dataset: xr.Dataset,
    pre_start: str,
    event_start: str,
    event_end: str,
    post_end: str,
    vegetation_lag_periods: int = 4
) -> dict:
    """
    Build Pre, During, and Post temporal windows from user-defined dates, applying
    the vegetation lag by shifting all SNDVI windows forward.

    The user defines the four boundary dates manually, ideally after consulting
    detect_drought_events() and compute_lag_correlation().

    The lag shifts all SNDVI windows forward because vegetation responds to water
    stress with a temporal delay. This ensures that the Pre, During, and Post SNDVI
    windows reflect the actual vegetation state in each phase.

    vegetation_lag_periods is expressed in biweekly periods. Lag=4 ~ 2 months.

    Parameters
    ----------
    dataset : xr.Dataset
        Clipped dataset, used to validate that shifted dates exist in the
        available time range.
    pre_start : str
        Start of the pre-drought window (e.g. '1997-06-01').
    event_start : str
        Start of the drought event (e.g. '1999-06-01').
    event_end : str
        End of the drought event (e.g. '2002-03-15').
    post_end : str
        End of the post-drought window (e.g. '2004-03-15').
    vegetation_lag_periods : int
        Number of biweekly periods to shift all SNDVI windows forward
        (default 4, ~2 months).

    Returns
    -------
    dict
        Two sub-dicts:
            'index':      slices for the drought index (no lag)
            'vegetation': slices for SNDVI (all windows shifted)
    """
    pre_start_dt   = pd.to_datetime(pre_start)
    event_start_dt = pd.to_datetime(event_start)
    event_end_dt   = pd.to_datetime(event_end)
    post_end_dt    = pd.to_datetime(post_end)

    # Drought index windows without lag
    index_windows = {
        "pre_drought":    slice(str(pre_start_dt.date()),   str(event_start_dt.date())),
        "during_drought": slice(str(event_start_dt.date()), str(event_end_dt.date())),
        "post_drought":   slice(str(event_end_dt.date()),   str(post_end_dt.date()))
    }

    # Lag offset with calendar correction.
    # Using days=15*N accumulates drift in months with different day counts.
    # Correct approach: full months + extra days for odd biweekly periods.
    months_offset = vegetation_lag_periods // 2
    extra_days    = 15 if vegetation_lag_periods % 2 != 0 else 0
    lag_offset    = pd.DateOffset(months=months_offset, days=extra_days)

    veg_pre_start    = pre_start_dt   + lag_offset
    veg_event_start  = event_start_dt + lag_offset
    veg_event_end    = event_end_dt   + lag_offset
    veg_post_end     = post_end_dt    + lag_offset

    # Warn if the lagged Post window exceeds the dataset
    data_end = pd.to_datetime(dataset.time.values[-1])
    if veg_post_end > data_end:
        warnings.warn(
            f"The SNDVI Post window ends on {veg_post_end.date()} "
            f"but the dataset ends on {data_end.date()}. "
            f"Recovery may be incomplete. "
            f"Consider reducing post_end or vegetation_lag_periods."
        )

    # SNDVI windows with lag applied to all
    vegetation_windows = {
        "pre_drought":    slice(str(veg_pre_start.date()),   str(veg_event_start.date())),
        "during_drought": slice(str(veg_event_start.date()), str(veg_event_end.date())),
        "post_drought":   slice(str(veg_event_end.date()),   str(veg_post_end.date()))
    }

    return {
        "index":      index_windows,
        "vegetation": vegetation_windows
    }